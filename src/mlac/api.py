"""Optional OIDC-authenticated FastAPI interface.

Without MLAC_API_STATE_DB the registry is process-local and non-durable. The CLI
therefore refuses non-loopback binds even when OIDC is configured. Setting
MLAC_API_STATE_DB enables file-backed SQLite metadata persistence, but does not
enable multi-worker or network deployment.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import asdict
from sqlite3 import Error as SQLiteError
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from mlac.chat_completions_backend import (
    ChatCompletionsProviderError,
    OpenAICompatibleChatBackend,
)
from mlac.evolution import EvolutionController
from mlac.ir import Architecture
from mlac.master_intelligence import MasterIntelligenceEngine
from mlac.openai_backend import OpenAIProviderError, OpenAIResponsesBackend
from mlac.registry import ModelRegistry
from mlac.rate_limit import RateLimiterUnavailable, RedisRateLimiter
from mlac.sqlite_state import SQLiteStateStore


def _configured_rate_limiter(*, namespace: str, limit: int, window_seconds: float, max_keys: int):
    """Use Redis when configured; otherwise retain the bounded local limiter."""
    redis_url = os.getenv("MLAC_RATE_LIMIT_REDIS_URL")
    if redis_url:
        return RedisRateLimiter.from_url(
            redis_url,
            namespace=namespace,
            limit=limit,
            window_seconds=window_seconds,
        )
    return _PrincipalRateLimiter(limit=limit, window_seconds=window_seconds, max_keys=max_keys)


def _default_intelligence_engine() -> MasterIntelligenceEngine:
    """Select a configured reasoning adapter without silently changing providers."""
    selected = os.environ.get("MLAC_REASONING_PROVIDER")
    if selected is None:
        selected = (
            "openai-responses"
            if os.environ.get("MLAC_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            else "rules"
        )
    if selected == "rules":
        return MasterIntelligenceEngine()
    if selected == "openai-responses":
        return MasterIntelligenceEngine(backend=OpenAIResponsesBackend.from_env(required=True))
    if selected == "openai-compatible":
        return MasterIntelligenceEngine(backend=OpenAICompatibleChatBackend.from_env(required=True))
    raise RuntimeError(
        "MLAC_REASONING_PROVIDER must be one of: rules, openai-responses, openai-compatible"
    )


registry = ModelRegistry()
evolution = EvolutionController()
intelligence = _default_intelligence_engine()
_MAX_REQUEST_BODY_BYTES = 1_048_576


class RequestBodyLimitMiddleware:
    """Reject oversized request bodies before JSON parsing buffers them."""

    def __init__(self, app: Any, *, max_bytes: int):
        """Initialize the ASGI wrapper with a positive byte limit."""
        if isinstance(max_bytes, bool) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        """Buffer at most the configured limit, then replay the bounded body."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                declared_length = int(raw_length)
            except ValueError:
                await self._reject(send)
                return
            if declared_length < 0 or declared_length > self.max_bytes:
                await self._reject(send)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_receive() -> dict:
            """Replay the already-bounded request body exactly once."""
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _reject(send: Any) -> None:
        """Send the fixed 413 response without reflecting request data."""
        body = b'{"detail":"request body too large"}'
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})


class _PrincipalRateLimiter:
    """Small process-local sliding-window limiter for the localhost API."""

    def __init__(self, *, limit: int, window_seconds: float, max_keys: int = 4096):
        """Configure request limits and a hard bound on tracked identities."""
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or isinstance(max_keys, bool)
            or not isinstance(max_keys, int)
            or max_keys < 1
            or isinstance(window_seconds, bool)
            or window_seconds <= 0
        ):
            raise ValueError("rate limit, key capacity, and window must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._events: dict[str, deque[float]] = {}
        self._lock = Lock()

    def check(self, key: str) -> None:
        """Record one request or fail closed when the bounded identity table is full."""
        now = time.monotonic()
        with self._lock:
            events = self._events.get(key)
            if events is not None:
                while events and now - events[0] >= self.window_seconds:
                    events.popleft()
                if not events:
                    del self._events[key]
                    events = None

            if events is None:
                if len(self._events) >= self.max_keys:
                    for existing_key, existing_events in tuple(self._events.items()):
                        if not existing_events or now - existing_events[-1] >= self.window_seconds:
                            del self._events[existing_key]
                if len(self._events) >= self.max_keys:
                    raise RuntimeError(str(max(1, int(self.window_seconds))))
                events = deque()
                self._events[key] = events

            if len(events) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - events[0])))
                raise RuntimeError(str(retry_after))
            events.append(now)


class PreAuthenticationRateLimitMiddleware:
    """Bound unauthenticated requests before they can trigger remote JWKS work."""

    def __init__(self, app: Any, *, rate_limiter: _PrincipalRateLimiter):
        """Attach the per-peer limiter to the downstream ASGI application."""
        self.app = app
        self.rate_limiter = rate_limiter

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        """Limit non-health requests before route authentication or JWKS access."""
        if scope["type"] != "http" or scope.get("path") in {"/health", "/ready"}:
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        peer = client[0] if client and client[0] else "unknown"
        try:
            self.rate_limiter.check(peer)
        except RateLimiterUnavailable:
            body = b'{"detail":"Shared rate-limit service is unavailable"}'
            await send({
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"retry-after", b"5"),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return
        except RuntimeError as exc:
            body = b'{"detail":"Authentication request rate limit exceeded"}'
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"retry-after", str(exc).encode("ascii")),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)


def _validate_identity_url(value: str, *, setting: str) -> None:
    """Require HTTPS identity endpoints, except for loopback development URLs."""
    parsed = urlparse(value)
    local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if not parsed.netloc or (parsed.scheme != "https" and not local_http):
        raise RuntimeError(f"{setting} must use HTTPS (HTTP is allowed only for loopback development)")


def create_app(
    *,
    model_registry: ModelRegistry | None = None,
    evolution_controller: EvolutionController | None = None,
    intelligence_engine: MasterIntelligenceEngine | None = None,
    state_store: SQLiteStateStore | None = None,
    jwks_client: Any | None = None,
    jwks_url: str | None = None,
    issuer: str | None = None,
    audience: str | None = None,
    rate_limit: int = 60,
    rate_window_seconds: float = 60,
    pre_auth_rate_limit: int = 120,
    pre_auth_window_seconds: float = 60,
    rate_limit_max_keys: int = 4096,
    pre_auth_max_keys: int = 1024,
    max_body_bytes: int = _MAX_REQUEST_BODY_BYTES,
) -> Any:
    """Create an API that resolves every non-health request to an OIDC principal."""
    try:
        import jwt
        from fastapi import Depends, FastAPI, HTTPException
        from fastapi.security import HTTPBearer
        from jwt import PyJWKClient
    except ImportError as exc:
        raise RuntimeError("Install API dependencies: pip install 'mlac[api]'") from exc

    if state_store is not None and (model_registry is not None or evolution_controller is not None):
        raise ValueError("state_store cannot be combined with injected registry/controller instances")

    if state_store is None and model_registry is None and evolution_controller is None:
        database_path = os.getenv("MLAC_API_STATE_DB")
        if database_path:
            state_store = SQLiteStateStore(database_path)

    model_store = (
        model_registry
        if model_registry is not None
        else ModelRegistry(state_store=state_store) if state_store is not None else registry
    )
    proposal_controller = (
        evolution_controller
        if evolution_controller is not None
        else EvolutionController(state_store=state_store) if state_store is not None else evolution
    )
    planning_engine = intelligence_engine if intelligence_engine is not None else intelligence
    configured_jwks_url = jwks_url or os.getenv("MLAC_API_JWKS_URL")
    configured_issuer = issuer or os.getenv("MLAC_API_ISSUER")
    configured_audience = audience or os.getenv("MLAC_API_AUDIENCE")

    if jwks_client is None and configured_jwks_url and configured_issuer and configured_audience:
        _validate_identity_url(configured_jwks_url, setting="MLAC_API_JWKS_URL")
        _validate_identity_url(configured_issuer, setting="MLAC_API_ISSUER")
        jwks_client = PyJWKClient(configured_jwks_url, cache_keys=True, timeout=5)

    auth_configured = bool(jwks_client is not None and configured_issuer and configured_audience)
    principal_rate_limiter = _configured_rate_limiter(
        namespace="principal",
        limit=rate_limit,
        window_seconds=rate_window_seconds,
        max_keys=rate_limit_max_keys,
    )
    peer_rate_limiter = _configured_rate_limiter(
        namespace="peer",
        limit=pre_auth_rate_limit,
        window_seconds=pre_auth_window_seconds,
        max_keys=pre_auth_max_keys,
    )
    bearer = HTTPBearer(auto_error=False)
    app = FastAPI(
        title="MLAC",
        version="0.8.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=max_body_bytes)
    app.add_middleware(PreAuthenticationRateLimitMiddleware, rate_limiter=peer_rate_limiter)
    app.state.auth_configured = auth_configured
    app.state.storage_mode = "sqlite" if state_store is not None else "process-local"

    def require_principal(credentials=Depends(bearer)) -> str:
        """Verify a bearer JWT and return its issuer/subject-scoped principal."""
        if not auth_configured:
            raise HTTPException(status_code=503, detail="OIDC authentication is not configured")
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Bearer authentication is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            signing_key = jwks_client.get_signing_key_from_jwt(credentials.credentials)
            claims = jwt.decode(
                credentials.credentials,
                signing_key.key,
                algorithms=["RS256", "ES256", "EdDSA"],
                issuer=configured_issuer,
                audience=configured_audience,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
                leeway=30,
            )
        except jwt.PyJWKClientConnectionError as exc:
            raise HTTPException(status_code=503, detail="Identity provider is temporarily unavailable") from exc
        except jwt.PyJWTError as exc:
            raise HTTPException(
                status_code=401,
                detail="Bearer token is invalid",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip() or len(subject) > 255:
            raise HTTPException(status_code=401, detail="Bearer token subject is invalid")
        principal = f"{configured_issuer}|{subject}"
        try:
            principal_rate_limiter.check(principal)
        except RateLimiterUnavailable as exc:
            raise HTTPException(status_code=503, detail="Shared rate-limit service is unavailable") from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=429,
                detail="Request rate limit exceeded",
                headers={"Retry-After": str(exc)},
            ) from exc
        return principal

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return process liveness without disclosing authentication configuration."""
        return {"status": "ok", "service": "mlac"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        """Report local API readiness without implying production readiness."""
        if not auth_configured:
            raise HTTPException(status_code=503, detail="OIDC authentication is not configured")
        if state_store is not None:
            try:
                state_store.health_check()
            except (OSError, SQLiteError, RuntimeError) as exc:
                raise HTTPException(status_code=503, detail="Configured state store is unavailable") from exc
        if isinstance(principal_rate_limiter, RedisRateLimiter):
            try:
                principal_rate_limiter.health_check()
                peer_rate_limiter.health_check()
            except RateLimiterUnavailable as exc:
                raise HTTPException(status_code=503, detail="Shared rate-limit service is unavailable") from exc
        return {
            "status": "ready",
            "scope": "localhost-development",
            "storage": app.state.storage_mode,
        }

    @app.post("/models")
    def register_model(payload: dict[str, Any], principal: str = Depends(require_principal)):
        """Register a model owned by the authenticated principal."""
        try:
            architecture = Architecture.from_dict(payload)
            record = model_store.register(architecture, metadata={"owner": principal})
            return {"model_id": record.model_id, "architecture": record.architecture.to_dict()}
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Model architecture is invalid") from exc

    @app.get("/models")
    def list_models(principal: str = Depends(require_principal)):
        """List only models owned by the authenticated principal."""
        return [
            {
                "model_id": record.model_id,
                "architecture": record.architecture.to_dict(),
                "status": record.status,
                "created_at": record.created_at,
            }
            for record in model_store.list(owner=principal)
        ]

    @app.get("/models/{model_id}")
    def get_model(model_id: str, principal: str = Depends(require_principal)):
        """Return a model only to its authenticated owner."""
        try:
            record = model_store.get(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Model not found") from exc
        if record.metadata.get("owner") != principal:
            raise HTTPException(status_code=404, detail="Model not found")
        return {"model_id": model_id, "architecture": record.architecture.to_dict()}

    @app.post("/models/{model_id}/proposals")
    def propose(
        model_id: str,
        payload: dict[str, Any],
        principal: str = Depends(require_principal),
    ):
        """Create a policy-validated proposal for an owned model."""
        try:
            record = model_store.get(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Model not found") from exc
        if record.metadata.get("owner") != principal:
            raise HTTPException(status_code=404, detail="Model not found")
        hypothesis = payload.get("hypothesis")
        operation = payload.get("operation")
        parameters = payload.get("parameters", {})
        if not isinstance(hypothesis, str) or not isinstance(operation, str) or not isinstance(parameters, dict):
            raise HTTPException(status_code=422, detail="Proposal fields have invalid types")
        try:
            proposal = proposal_controller.propose(
                model_id,
                record.architecture,
                hypothesis=hypothesis,
                operation=operation,
                parameters=parameters,
            )
            return asdict(proposal)
        except (ValueError, PermissionError) as exc:
            raise HTTPException(status_code=422, detail="Proposal is invalid or denied by policy") from exc

    @app.get("/proposals")
    def list_proposals(principal: str = Depends(require_principal)):
        """List proposals associated with the authenticated principal's models."""
        owned_model_ids = {
            record.model_id for record in model_store.list(owner=principal)
        }
        return proposal_controller.list_proposals(
            owner=principal,
            model_ids=owned_model_ids,
        )

    @app.get("/intelligence/provider")
    def intelligence_provider(_principal: str = Depends(require_principal)):
        """Report the configured planning provider to an authenticated caller."""
        backend = planning_engine.backend
        return {
            "provider": getattr(backend, "provider_name", "rules-only"),
            "model": getattr(backend, "model_name", None),
        }

    @app.post("/intelligence/plan")
    def intelligence_plan(
        payload: dict[str, Any],
        _principal: str = Depends(require_principal),
    ):
        """Return a validated proposal-only architecture plan."""
        model_id = payload.get("model_id")
        architecture = payload.get("architecture")
        objective = payload.get("objective")
        constraints = payload.get("constraints")
        if (
            not isinstance(model_id, str)
            or not isinstance(architecture, dict)
            or not isinstance(objective, str)
            or (constraints is not None and not isinstance(constraints, dict))
        ):
            raise HTTPException(status_code=422, detail="Planning request fields have invalid types")
        try:
            report = planning_engine.plan(
                model_id=model_id,
                architecture=architecture,
                objective=objective,
                constraints=constraints,
            )
            return report.to_dict()
        except (OpenAIProviderError, ChatCompletionsProviderError) as exc:
            raise HTTPException(status_code=503, detail="Configured planning provider is temporarily unavailable") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Planning request is invalid") from exc

    return app


app = create_app()
