"""Ed25519 worker-result attestations and input-artifact verification."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import uuid4

def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def directory_digests(candidate_dir: str | Path, trusted_dir: str | Path) -> dict[str, str]:
    """Hash direct regular files in both read-only input mounts; reject links and nested paths."""
    result: dict[str, str] = {}
    for prefix, directory in (("candidate", Path(candidate_dir)), ("trusted", Path(trusted_dir))):
        if not directory.is_dir():
            raise ValueError(f"{prefix} input must be an existing directory")
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"{prefix} input contains a non-regular artifact: {path.name}")
            if path.stat().st_size > 512 * 1024 * 1024:
                raise ValueError(f"{prefix} artifact exceeds 512 MiB: {path.name}")
            result[f"{prefix}/{path.name}"] = sha256_bytes(path.read_bytes())
    if "candidate/candidate.pt" not in result:
        raise ValueError("candidate input is missing candidate.pt")
    for name in ("manifest.json",):
        if f"trusted/{name}" not in result:
            raise ValueError(f"trusted input is missing {name}")
    return result

def key_id_for_public_key(public_key) -> str:
    from cryptography.hazmat.primitives import serialization
    raw = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return sha256_bytes(raw)

def generate_worker_keypair(private_path: str | Path, public_path: str | Path) -> str:
    """Generate an Ed25519 keypair. Protect the private key as a worker-only secret."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    private_path, public_path = Path(private_path), Path(public_path)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    public_bytes = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(private_bytes)
    public_path.write_bytes(public_bytes)
    return key_id_for_public_key(private.public_key())

def sign_worker_result(*, report_bytes: bytes, artifact_sha256: Mapping[str, str], private_key_path: str | Path, worker_id: str, worker_image: str) -> dict:
    from cryptography.hazmat.primitives import serialization
    private = serialization.load_pem_private_key(Path(private_key_path).read_bytes(), password=None)
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    if not isinstance(private, Ed25519PrivateKey):
        raise ValueError("worker signing key must be Ed25519")
    public = private.public_key()
    key_id = key_id_for_public_key(public)
    unsigned = {
        "schema_version": 1,
        "worker_id": worker_id,
        "key_id": key_id,
        "worker_image": worker_image,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "nonce": uuid4().hex,
        "report_sha256": sha256_bytes(report_bytes),
        "artifact_sha256": dict(sorted(artifact_sha256.items())),
    }
    signature = private.sign(canonical_json(unsigned))
    return {**unsigned, "signature": base64.b64encode(signature).decode("ascii")}

def verify_worker_result(*, attestation: object, report_bytes: bytes, artifact_sha256: Mapping[str, str], trusted_worker_keys: Mapping[str, bytes | str], expected_image: str, expected_worker_id: str | None = None) -> dict:
    """Verify signature, trusted key identity, image pin, report digest and independently hashed inputs."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.exceptions import InvalidSignature
    fields = {"schema_version", "worker_id", "key_id", "worker_image", "issued_at", "nonce", "report_sha256", "artifact_sha256", "signature"}
    if not isinstance(attestation, dict) or set(attestation) != fields or isinstance(attestation.get("schema_version"), bool) or attestation.get("schema_version") != 1:
        raise ValueError("worker attestation schema is invalid")
    key_id = attestation["key_id"]
    if not isinstance(key_id, str) or key_id not in trusted_worker_keys:
        raise ValueError("worker signing key is not trusted")
    if not isinstance(attestation["worker_id"], str) or not attestation["worker_id"].strip():
        raise ValueError("worker_id must be non-empty")
    if expected_worker_id is not None and attestation["worker_id"] != expected_worker_id:
        raise ValueError("worker identity does not match configured identity")
    if attestation["worker_image"] != expected_image:
        raise ValueError("attested worker image does not match configured immutable image digest")
    if attestation["report_sha256"] != sha256_bytes(report_bytes):
        raise ValueError("attestation report digest mismatch")
    if attestation["artifact_sha256"] != dict(sorted(artifact_sha256.items())):
        raise ValueError("attested input artifact digests do not match controller-observed inputs")
    if not isinstance(attestation["nonce"], str) or len(attestation["nonce"]) != 32:
        raise ValueError("attestation nonce is invalid")
    public_source = trusted_worker_keys[key_id]
    public_bytes = Path(public_source).read_bytes() if isinstance(public_source, str) else public_source
    public = serialization.load_pem_public_key(public_bytes)
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if not isinstance(public, Ed25519PublicKey):
        raise ValueError("trusted worker public key must be Ed25519")
    if key_id_for_public_key(public) != key_id:
        raise ValueError("trusted public key fingerprint does not match key_id")
    unsigned = {k: attestation[k] for k in fields if k != "signature"}
    try:
        public.verify(base64.b64decode(attestation["signature"], validate=True), canonical_json(unsigned))
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise ValueError("worker attestation signature verification failed") from exc
    return {"worker_id": attestation["worker_id"], "key_id": key_id, "worker_image": expected_image,
            "report_sha256": attestation["report_sha256"], "artifact_sha256": dict(artifact_sha256),
            "issued_at": attestation["issued_at"], "nonce": attestation["nonce"]}
