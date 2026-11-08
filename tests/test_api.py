from __future__ import annotations

import time
from types import SimpleNamespace

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from mlac.api import create_app
from mlac.cli import is_loopback_host
from mlac.evolution import EvolutionController
from mlac.registry import ModelRegistry

ISSUER = "https://identity.example.test/"
AUDIENCE = "mlac-api"


class StaticJWKClient:
    def __init__(self, public_key):
        self.public_key = public_key

    def get_signing_key_from_jwt(self, token):
        return SimpleNamespace(key=self.public_key)


def make_client(*, rate_limit=60, pre_auth_rate_limit=120):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    app = create_app(
        model_registry=ModelRegistry(),
        evolution_controller=EvolutionController(),
        jwks_client=StaticJWKClient(private_key.public_key()),
        issuer=ISSUER,
        audience=AUDIENCE,
        rate_limit=rate_limit,
        pre_auth_rate_limit=pre_auth_rate_limit,
    )
    client = TestClient(app)

    def token(subject, *, issuer=ISSUER, audience=AUDIENCE, headers=None):
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
    return {"Authorization": f"Bearer {token}"}


def architecture():
    return {
        "name": "tiny",
        "family": "mlp",
        "components": [{"name": "drop", "kind": "Dropout", "config": {"p": 0.1}}],
    }


def test_health_is_liveness_and_non_health_routes_require_bearer():
    client, _ = make_client()
    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["scope"] == "localhost-development"
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    response = client.get("/models")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_missing_oidc_configuration_fails_closed(monkeypatch):
    for name in ("MLAC_API_JWKS_URL", "MLAC_API_ISSUER", "MLAC_API_AUDIENCE"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(model_registry=ModelRegistry(), evolution_controller=EvolutionController())
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/models").status_code == 503


def test_models_and_proposals_are_scoped_to_issuer_and_subject():
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
    client, token = make_client()
    assert client.get("/models", headers=auth(token("alice", issuer="https://attacker.example/"))).status_code == 401
    assert client.get("/models", headers=auth(token("alice", audience="other-api"))).status_code == 401


def test_unknown_critical_jwt_header_extension_is_rejected():
    client, token = make_client()
    unknown_critical_header = {
        "crit": ["x-mlac-policy"],
        "x-mlac-policy": "require-mfa",
    }
    response = client.get("/models", headers=auth(token("alice", headers=unknown_critical_header)))
    assert response.status_code == 401


def test_pre_authentication_rate_limit_bounds_invalid_token_requests():
    client, _ = make_client(pre_auth_rate_limit=1)
    invalid = auth("not-a-jwt")
    assert client.get("/models", headers=invalid).status_code == 401
    limited = client.get("/models", headers=invalid)
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_per_principal_rate_limit_returns_retry_after():
    client, token = make_client(rate_limit=1)
    headers = auth(token("alice"))
    assert client.get("/models", headers=headers).status_code == 200
    limited = client.get("/models", headers=headers)
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_oversized_request_is_rejected_before_route_authentication():
    client, _ = make_client()
    response = client.post(
        "/models",
        content=b"x" * (1_048_576 + 1),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_cli_allows_only_loopback_bind_hosts():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.0.2.10")


def test_intelligence_planning_route_requires_identity_and_returns_proposals():
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
