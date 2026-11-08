import base64, hashlib
from datetime import datetime, timedelta, timezone
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from mlac.runtime_identity import RuntimeIdentityVerifier, canonical_json

def public_pem(key):
    return key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo)

def make_envelope(key, now, nonce="abcdefghijklmnopqrstuvwx"):
    statement={"authority_id":"spire-prod","worker_id":"worker-7","spiffe_id":"spiffe://mlac.example/workers/worker-7",
      "image_digest":"registry.example/mlac-worker@sha256:"+"a"*64,"attestation_type":"tpm2-ima",
      "measurement_sha256":hashlib.sha256(b"measured").hexdigest(),"issued_at":now.isoformat(),
      "expires_at":(now+timedelta(minutes=2)).isoformat(),"nonce":nonce}
    return {"schema_version":1,"statement":statement,"signature":base64.b64encode(key.sign(canonical_json(statement))).decode()}

def test_verifies_authority_and_rejects_replay(tmp_path):
    now=datetime(2026,1,1,tzinfo=timezone.utc); key=Ed25519PrivateKey.generate()
    verifier=RuntimeIdentityVerifier(trusted_authorities={"spire-prod":public_pem(key)},nonce_db=tmp_path/"nonce.db",trust_domain="mlac.example",clock=lambda:now)
    proof=make_envelope(key,now)
    result=verifier.verify(proof,expected_worker_id="worker-7",expected_image_digest=proof["statement"]["image_digest"])
    assert result["statement_sha256"]
    with pytest.raises(PermissionError,match="already been used"):
        verifier.verify(proof,expected_worker_id="worker-7",expected_image_digest=proof["statement"]["image_digest"])

def test_rejects_wrong_image(tmp_path):
    now=datetime(2026,1,1,tzinfo=timezone.utc); key=Ed25519PrivateKey.generate()
    verifier=RuntimeIdentityVerifier(trusted_authorities={"spire-prod":public_pem(key)},nonce_db=tmp_path/"nonce.db",trust_domain="mlac.example",clock=lambda:now)
    proof=make_envelope(key,now)
    with pytest.raises(PermissionError,match="image digest mismatch"):
        verifier.verify(proof,expected_worker_id="worker-7",expected_image_digest="registry.example/other@sha256:"+"b"*64)

def test_rejects_wrong_spiffe_trust_domain(tmp_path):
    now=datetime(2026,1,1,tzinfo=timezone.utc); key=Ed25519PrivateKey.generate()
    verifier=RuntimeIdentityVerifier(trusted_authorities={"spire-prod":public_pem(key)},nonce_db=tmp_path/"nonce.db",trust_domain="mlac.example",clock=lambda:now)
    proof=make_envelope(key,now); proof["statement"]["spiffe_id"]="spiffe://attacker.example/worker"
    proof["signature"]=base64.b64encode(key.sign(canonical_json(proof["statement"]))).decode()
    with pytest.raises(PermissionError,match="trust domain mismatch"):
        verifier.verify(proof,expected_worker_id="worker-7",expected_image_digest=proof["statement"]["image_digest"])
