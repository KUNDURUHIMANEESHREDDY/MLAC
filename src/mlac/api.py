"""Optional OIDC-authenticated FastAPI interface.

The registry is process-local and non-durable. The CLI therefore refuses
non-loopback binds even when OIDC is configured.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import asdict
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from mlac.evolution import EvolutionController
from mlac.ir import Architecture
from mlac.master_intelligence import MasterIntelligenceEngine
from mlac.registry import ModelRegistry

registry = ModelRegistry()
evolution = EvolutionController()
intelligence = MasterIntelligenceEngine()
_MAX_REQUEST_BODY_BYTES = 1_048_576


class _BodyTooLarge(Exception):
    """Raised when a streamed request exceeds the configured body limit."""


class RequestBodyLimitMiddleware:
    """Reject oversized request bodies before JSON parsing buffers them."""

    def __init__(self, app: Any, *, max_bytes: int):
        if isinstance(max_bytes, bool) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
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

        received = 0

        async def limited_receive() -> dict:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await self._reject(send)

    @staticmethod
    async def _reject(send: Any) -> None:
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
    """Small process-local per-principal limiter for the localhost API."""

    def __init__(self, *, limit: int, window_seconds: float):
        if isinstance(limit, bool) or limit < 1 or window_seconds <= 0:
            raise ValueError("rate limit and window must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = {}
        self._lock = Lock()

    def check(self, principal: str) -> None:
        now = time.monotonic()
        with self._lock:
            events = self._events.setdefault(principal, deque())
            while events and now - events[0] >= self.window_seconds:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - events[0])))
                raise RuntimeError(str(retry_after))
            events.append(now)


def _validate_identity_url(value: str, *, setting: str) -> None:
    parsed = urlparse(value)
    local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if not parsed.netloc or (parsed.scheme != "https" and not local_http):
        raise RuntimeError(f"{setting} must use HTTPS (HTTP is allowed only for loopback development)")


def create_app(
    *,
    model_registry: ModelRegistry | None = None,
    evolution_controller: EvolutionController | None = None,
    intelligence_engine: MasterIntelligenceEngine | None = None,
    jwks_client: Any | None = None,
    jwks_url: str | None = None,
    issuer: str | None = None,
    audience: str | None = None,
    rate_limit: int = 60,
    rate_window_seconds: float = 60,
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

    model_store = model_registry if model_registry is not None else registry
    proposal_controller = evolution_controller if evolution_controller is not None else evolution
    planning_engine = intelligence_engine if intelligence_engine is not None else intelligence
    configured_jwks_url = jwks_url or os.getenv("MLAC_API_JWKS_URL")
    configured_issuer = issuer or os.getenv("MLAC_API_ISSUER")
    configured_audience = audience or os.getenv("MLAC_API_AUDIENCE")

    if jwks_client is None and configured_jwks_url and configured_issuer and configured_audience:
        _validate_identity_url(configured_jwks_url, setting="MLAC_API_JWKS_URL")
        _validate_identity_url(configured_issuer, setting="MLAC_API_ISSUER")
        jwks_client = PyJWKClient(configured_jwks_url, cache_keys=True)

    auth_configured = bool(jwks_client is not None and configured_issuer and configured_audience)
    rate_limiter = _PrincipalRateLimiter(limit=rate_limit, window_seconds=rate_window_seconds)
    bearer = HTTPBearer(auto_error=False)
    app = FastAPI(
        title="MLAC",
        version="0.8.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=max_body_bytes)
    app.state.auth_configured = auth_configured
    app.state.storage_mode = "process-local"

    def require_principal(credentials=Depends(bearer)) -> str:
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
            rate_limiter.check(principal)
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
        """Report local API readiness; this is not a production-readiness signal."""
        if not auth_configured:
            raise HTTPException(status_code=503, detail="OIDC authentication is not configured")
        return {"status": "ready", "scope": "localhost-development", "storage": "process-local"}

    @app.post("/models")
    def register_model(payload: dict[str, Any], principal: str = Depends(require_principal)):
        try:
            architecture = Architecture.from_dict(payload)
            record = model_store.register(architecture, metadata={"owner": principal})
            return {"model_id": record.model_id, "architecture": record.architecture.to_dict()}
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Model architecture is invalid") from exc

    @app.get("/models")
    def list_models(principal: str = Depends(require_principal)):
        return [
            {
                "model_id": record.model_id,
                "architecture": record.architecture.to_dict(),
                "status": record.status,
                "created_at": record.created_at,
            }
            for record in model_store.list()
            if record.metadata.get("owner") == principal
        ]

    @app.get("/models/{model_id}")
    def get_model(model_id: str, principal: str = Depends(require_principal)):
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
        owned_model_ids = {
            record.model_id for record in model_store.list()
            if record.metadata.get("owner") == principal
        }
        return [
            proposal for proposal in proposal_controller.list_proposals()
            if proposal.get("model_id") in owned_model_ids
        ]

    @app.post("/intelligence/plan")
    def intelligence_plan(
        payload: dict[str, Any],
        _principal: str = Depends(require_principal),
    ):
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
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Planning request is invalid") from exc

    return app


app = create_app()
