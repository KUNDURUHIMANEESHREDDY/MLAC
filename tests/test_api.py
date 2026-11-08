from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from mlac.api import create_app
from mlac.cli import is_loopback_host
from mlac.evolution import EvolutionController
from mlac.master_intelligence import MasterIntelligenceEngine
from mlac.registry import ModelRegistry
from mlac.sqlite_state import SQLiteStateStore

ISSUER = "https://identity.example.test/"
AUDIENCE = "mlac-api"


class StaticJWKClient:
    """Return a fixed test public key without making network requests."""

    def __init__(self, public_key):
        """Store the public key used to verify test tokens."""
        self.public_key = public_key

    def get_signing_key_from_jwt(self, token):
        """Return the fixed public key for any test key identifier."""
        return SimpleNamespace(key=self.public_key)


def make_client(*, rate_limit=60, pre_auth_rate_limit=120, state_store=None):
    """Create an isolated API client and a helper for signed test tokens."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    app = create_app(
        model_registry=ModelRegistry() if state_store is None else None,
        evolution_controller=EvolutionController() if state_store is None else None,
        state_store=state_store,
        intelligence_engine=MasterIntelligenceEngine(),
        jwks_client=StaticJWKClient(private_key.public_key()),
        issuer=ISSUER,
        audience=AUDIENCE,
        rate_limit=rate_limit,
        pre_auth_rate_limit=pre_auth_rate_limit,
    )
    client = TestClient(app)

    def token(subject, *, issuer=ISSUER, audience=AUDIENCE, headers=None):
        """Sign a short-lived token for a named test principal."""
        now = int(time.time())
        token_headers = {"kid": "test-key"}
        token_headers.update(headers or {})
        return jwt.encode(
            {
                "iss": issuer,
                "aud": audience,
                "sub": subject,
                "iat": now,
                "exp": now + 300,
            },
            private_key,
            algorithm="RS256",
            headers=token_headers,
        )

    return client, token


def auth(token):
    """Build the Authorization header for a bearer token."""
    return {"Authorization": f"Bearer {token}"}


def architecture():
    """Return a valid minimal architecture payload for API tests."""
    return {
        "name": "tiny",
        "family": "mlp",
        "components": [{"name": "drop", "kind": "Dropout", "config": {"p": 0.1}}],
    }


def test_health_is_liveness_and_non_health_routes_require_bearer():
    """Keep liveness public while requiring identity for application routes."""
    client, _ = make_client()
    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["scope"] == "localhost-development"
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    response = client.get("/models")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_missing_oidc_configuration_fails_closed(monkeypatch):
    """Return unavailable for protected routes when OIDC is not configured."""
    for name in ("MLAC_API_JWKS_URL", "MLAC_API_ISSUER", "MLAC_API_AUDIENCE"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(model_registry=ModelRegistry(), evolution_controller=EvolutionController())
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/models").status_code == 503


def test_models_and_proposals_are_scoped_to_issuer_and_subject():
    """Prevent one principal from reading or mutating another principal's models."""
    client, token = make_client()
    alice = auth(token("alice"))
    bob = auth(token("bob"))

    created = client.post("/models", headers=alice, json=architecture())
    assert created.status_code == 200
    model_id = created.json()["model_id"]

    assert len(client.get("/models", headers=alice).json()) == 1
    assert client.get("/models", headers=bob).json() == []
    assert client.get(f"/models/{model_id}", headers=bob).status_code == 404

    proposal = client.post(
        f"/models/{model_id}/proposals",
        headers=alice,
        json={
            "hypothesis": "A smaller dropout value improves validation quality",
            "operation": "dropout",
            "parameters": {"probability": 0.05},
        },
    )
    assert proposal.status_code == 200
    assert len(client.get("/proposals", headers=alice).json()) == 1
    assert client.get("/proposals", headers=bob).json() == []
    assert client.post(
        f"/models/{model_id}/proposals",
        headers=bob,
        json={"hypothesis": "cross-tenant", "operation": "dropout", "parameters": {}},
    ).status_code == 404


def test_invalid_architecture_and_proposal_types_return_422():
    """Return validation errors for malformed architecture and proposal fields."""
    client, token = make_client()
    headers = auth(token("alice"))
    assert client.post("/models", headers=headers, json={}).status_code == 422
    malformed = architecture()
    malformed["name"] = 1
    assert client.post("/models", headers=headers, json=malformed).status_code == 422

    created = client.post("/models", headers=headers, json=architecture())
    model_id = created.json()["model_id"]
    invalid = client.post(
        f"/models/{model_id}/proposals",
        headers=headers,
        json={"hypothesis": [], "operation": "dropout", "parameters": {}},
    )
    assert invalid.status_code == 422


def test_invalid_issuer_or_audience_is_rejected():
    """Reject correctly signed tokens with the wrong issuer or audience."""
    client, token = make_client()
    assert client.get("/models", headers=auth(token("alice", issuer="https://attacker.example/"))).status_code == 401
    assert client.get("/models", headers=auth(token("alice", audience="other-api"))).status_code == 401


def test_unknown_critical_jwt_header_extension_is_rejected():
    """Reject signed JWTs that require an unsupported critical header extension."""
    client, token = make_client()
    unknown_critical_header = {
        "crit": ["x-mlac-policy"],
        "x-mlac-policy": "require-mfa",
    }
    response = client.get("/models", headers=auth(token("alice", headers=unknown_critical_header)))
    assert response.status_code == 401


def test_pre_authentication_rate_limit_bounds_invalid_token_requests():
    """Limit unauthenticated requests before they can trigger JWKS resolution."""
    client, _ = make_client(pre_auth_rate_limit=1)
    invalid = auth("not-a-jwt")
    assert client.get("/models", headers=invalid).status_code == 401
    limited = client.get("/models", headers=invalid)
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_per_principal_rate_limit_returns_retry_after():
    """Apply a separate authenticated request limit for each principal."""
    client, token = make_client(rate_limit=1)
    headers = auth(token("alice"))
    assert client.get("/models", headers=headers).status_code == 200
    limited = client.get("/models", headers=headers)
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_rate_limiter_bounds_identity_cardinality_and_reclaims_expired_keys(monkeypatch):
    """Keep unique peers/principals from growing the process-local limiter without bound."""
    from mlac.api import _PrincipalRateLimiter

    clock = [100.0]
    monkeypatch.setattr("mlac.api.time.monotonic", lambda: clock[0])
    limiter = _PrincipalRateLimiter(limit=2, window_seconds=10, max_keys=1)
    limiter.check("first")

    try:
        limiter.check("second")
    except RuntimeError as exc:
        assert exc.args == ("10",)
    else:
        raise AssertionError("a full identity table must reject a new identity")

    assert list(limiter._events) == ["first"]
    clock[0] = 111.0
    limiter.check("second")
    assert list(limiter._events) == ["second"]
    assert len(limiter._events) == 1

    try:
        _PrincipalRateLimiter(limit=1, window_seconds=10, max_keys=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero identity capacity must be rejected")


def test_oversized_request_is_rejected_before_route_authentication():
    """Reject a declared oversized body before parsing or route authentication."""
    client, _ = make_client()
    response = client.post(
        "/models",
        content=b"x" * (1_048_576 + 1),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_streamed_oversized_request_is_rejected_without_content_length():
    """Reject streamed bodies that exceed the cap even without Content-Length."""
    client, _ = make_client()

    def chunks():
        """Yield two chunks whose combined size exceeds the configured cap."""
        yield b"x" * 600_000
        yield b"x" * 600_000

    response = client.post(
        "/models",
        content=chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_cli_allows_only_loopback_bind_hosts():
    """Keep the CLI bind surface restricted to loopback addresses."""
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.0.2.10")


def test_cli_disables_proxy_header_trust(monkeypatch):
    """Prevent client-supplied forwarding headers from changing the peer identity."""
    from mlac import cli

    calls = {}
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *args, **kwargs: calls.update(kwargs)))
    monkeypatch.setattr(sys, "argv", ["mlac", "serve"])

    cli.main()

    assert calls["host"] == "127.0.0.1"
    assert calls["proxy_headers"] is False


def test_intelligence_provider_route_requires_identity_and_reports_provider():
    """Expose provider metadata only after the caller is authenticated."""
    client, token = make_client()
    assert client.get("/intelligence/provider").status_code == 401
    response = client.get("/intelligence/provider", headers=auth(token("alice")))
    assert response.status_code == 200
    assert response.json() == {"provider": "rules-only", "model": None}


def test_intelligence_planning_route_requires_identity_and_returns_proposals():
    """Require identity before returning proposal-only planning output."""
    client, token = make_client()
    payload = {
        "model_id": "baseline-001",
        "architecture": architecture(),
        "objective": "Reduce inference latency by 20%",
        "constraints": {"max_latency_regression_ratio": 0.0},
    }
    assert client.post("/intelligence/plan", json=payload).status_code == 401
    response = client.post("/intelligence/plan", headers=auth(token("alice")), json=payload)
    assert response.status_code == 200
    assert response.json()["planner"] == "rules"
    assert response.json()["plans"]


def test_reasoning_provider_factory_selects_compatible_adapter(monkeypatch):
    """Allow an operator to select a compatible hosted or self-hosted LLM endpoint."""
    from mlac.api import _default_intelligence_engine

    monkeypatch.setenv("MLAC_REASONING_PROVIDER", "openai-compatible")
    monkeypatch.setenv("MLAC_CHAT_API_KEY", "test-secret")
    monkeypatch.setenv("MLAC_CHAT_MODEL", "test-model")
    monkeypatch.setenv("MLAC_CHAT_BASE_URL", "https://llm.example/v1")
    engine = _default_intelligence_engine()
    assert engine.backend.provider_name == "openai-compatible-chat-completions"
    assert engine.backend.model_name == "test-model"


def test_reasoning_provider_factory_rejects_unknown_provider(monkeypatch):
    """Fail fast rather than silently selecting an unintended provider."""
    from mlac.api import _default_intelligence_engine

    monkeypatch.setenv("MLAC_REASONING_PROVIDER", "unexpected")
    with pytest.raises(RuntimeError, match="MLAC_REASONING_PROVIDER"):
        _default_intelligence_engine()


def test_sqlite_api_state_survives_restart_and_keeps_principals_isolated(tmp_path):
    """Persist model/proposal metadata across app recreation without crossing tenants."""
    database = tmp_path / "api-state.sqlite3"
    first_client, first_token = make_client(state_store=SQLiteStateStore(database))
    alice = auth(first_token("alice"))
    bob = auth(first_token("bob"))

    created = first_client.post("/models", headers=alice, json=architecture())
    assert created.status_code == 200
    model_id = created.json()["model_id"]
    proposal = first_client.post(
        f"/models/{model_id}/proposals",
        headers=alice,
        json={
            "hypothesis": "A smaller dropout value improves validation quality",
            "operation": "dropout",
            "parameters": {"probability": 0.05},
        },
    )
    assert proposal.status_code == 200
    assert first_client.get("/ready").json()["storage"] == "sqlite"

    second_client, second_token = make_client(state_store=SQLiteStateStore(database))
    assert len(second_client.get("/models", headers=auth(second_token("alice"))).json()) == 1
    assert second_client.get("/models", headers=auth(second_token("bob"))).json() == []
    assert len(second_client.get("/proposals", headers=auth(second_token("alice"))).json()) == 1
    assert second_client.get("/proposals", headers=auth(second_token("bob"))).json() == []
