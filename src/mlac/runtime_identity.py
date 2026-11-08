"""Verify externally issued worker runtime identity and measurement statements."""
from __future__ import annotations
import base64, hashlib, json, re, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_REQUIRED = {"authority_id","worker_id","spiffe_id","image_digest","attestation_type",
             "measurement_sha256","issued_at","expires_at","nonce"}

def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def _timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("identity timestamps must be ISO-8601 strings")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("identity timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("identity timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)

def _public_key(value: bytes | str) -> Ed25519PublicKey:
    from cryptography.hazmat.primitives import serialization
    key = serialization.load_pem_public_key(value.encode() if isinstance(value, str) else value)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("runtime authority key must be Ed25519")
    return key

class RuntimeIdentityVerifier:
    """Verify signed platform-measurement claims from a separately trusted authority.

    The authority must validate TPM/TEE/orchestrator evidence before signing.
    This class verifies that assertion; it does not itself measure hardware.
    """
    def __init__(self, *, trusted_authorities: dict[str, bytes | str], nonce_db: str | Path,
                 trust_domain: str, max_age_seconds: int = 300, clock: Any | None = None):
        if not trusted_authorities:
            raise ValueError("at least one runtime authority must be trusted")
        if not trust_domain or "/" in trust_domain:
            raise ValueError("trust_domain must be a hostname-like SPIFFE trust domain")
        if max_age_seconds < 1:
            raise ValueError("max_age_seconds must be positive")
        self.keys = {name: _public_key(value) for name, value in trusted_authorities.items()}
        self.nonce_db = Path(nonce_db)
        self.nonce_db.parent.mkdir(parents=True, exist_ok=True)
        self.trust_domain, self.max_age_seconds = trust_domain, max_age_seconds
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runtime_nonces (nonce TEXT PRIMARY KEY, authority_id TEXT NOT NULL, consumed_at TEXT NOT NULL)")

    def _connect(self):
        db = sqlite3.connect(self.nonce_db, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def verify(self, envelope: dict[str, Any], *, expected_worker_id: str,
               expected_image_digest: str) -> dict[str, Any]:
        if not isinstance(expected_worker_id, str) or not expected_worker_id.strip():
            raise ValueError("expected_worker_id must be non-empty")
        if (not isinstance(expected_image_digest, str)
                or not re.fullmatch(r"[^\\s@]+@sha256:[0-9a-f]{64}", expected_image_digest)):
            raise ValueError("expected image must be pinned by sha256 digest")
        if not isinstance(envelope, dict) or set(envelope) != {"schema_version","statement","signature"}:
            raise ValueError("runtime identity envelope schema is invalid")
        if envelope["schema_version"] != 1:
            raise ValueError("unsupported runtime identity schema version")
        statement = envelope["statement"]
        if not isinstance(statement, dict) or set(statement) != _REQUIRED:
            raise ValueError("runtime identity statement schema is invalid")
        authority_id = statement["authority_id"]
        key = self.keys.get(authority_id)
        if key is None:
            raise PermissionError("runtime identity authority is not trusted")
        try:
            key.verify(base64.b64decode(envelope["signature"], validate=True), canonical_json(statement))
        except (ValueError, InvalidSignature) as exc:
            raise PermissionError("runtime identity signature verification failed") from exc
        if statement["worker_id"] != expected_worker_id:
            raise PermissionError("runtime identity worker ID mismatch")
        if statement["image_digest"] != expected_image_digest:
            raise PermissionError("runtime identity image digest mismatch")
        if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", expected_image_digest):
            raise ValueError("expected image must be pinned by sha256 digest")
        if not isinstance(statement["spiffe_id"], str) or not statement["spiffe_id"].startswith(f"spiffe://{self.trust_domain}/"):
            raise PermissionError("runtime identity SPIFFE trust domain mismatch")
        if not isinstance(statement["attestation_type"], str) or not statement["attestation_type"]:
            raise ValueError("attestation_type must be non-empty")
        if not isinstance(statement["measurement_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", statement["measurement_sha256"]):
            raise ValueError("measurement_sha256 must be a lowercase SHA-256 digest")
        nonce = statement["nonce"]
        if not isinstance(nonce, str) or not re.fullmatch(r"[A-Za-z0-9_-]{22,128}", nonce):
            raise ValueError("runtime identity nonce is invalid")
        current = self.clock()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        now = current.astimezone(timezone.utc)
        issued, expires = _timestamp(statement["issued_at"]), _timestamp(statement["expires_at"])
        if issued > now or (now-issued).total_seconds() > self.max_age_seconds:
            raise PermissionError("runtime identity evidence is stale or from the future")
        if expires <= now or expires <= issued or (expires-issued).total_seconds() > self.max_age_seconds:
            raise PermissionError("runtime identity evidence is expired or has invalid lifetime")
        try:
            with self._connect() as db:
                db.execute("INSERT INTO runtime_nonces VALUES (?,?,?)", (nonce, authority_id, now.isoformat()))
        except sqlite3.IntegrityError as exc:
            raise PermissionError("runtime identity evidence nonce has already been used") from exc
        return {**statement, "statement_sha256": hashlib.sha256(canonical_json(statement)).hexdigest(),
                "verified_at": now.isoformat()}
