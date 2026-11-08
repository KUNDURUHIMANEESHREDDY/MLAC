"""Cryptographically signed, content-verified dataset manifests.

The manifest records exact dataset file paths, sizes, hashes, and split membership.
The signer key must be managed independently from model and worker processes.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _safe_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("dataset paths must be non-empty POSIX relative paths")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("dataset path must be relative and must not escape the dataset root")
    target = root.joinpath(*path.parts)
    # Resolve and reject symlinks so the signed path cannot be redirected.
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"dataset file is missing, not regular, or a symlink: {relative}")
    resolved = target.resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError("dataset file resolves outside the dataset root")
    return resolved


def _load_private_key(value: bytes | str):
    from cryptography.hazmat.primitives import serialization
    key = serialization.load_pem_private_key(value.encode() if isinstance(value, str) else value, password=None)
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("dataset manifest signer must use Ed25519")
    return key


def _load_public_key(value: bytes | str):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    key = serialization.load_pem_public_key(value.encode() if isinstance(value, str) else value)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("trusted dataset signer key must use Ed25519")
    return key


def _public_key_digest(key) -> str:
    from cryptography.hazmat.primitives import serialization
    raw = key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(raw).hexdigest()


def create_signed_dataset_manifest(
    dataset_root: str | Path,
    *,
    dataset_id: str,
    revision: str,
    splits: Mapping[str, list[str]],
    signer_id: str,
    private_key_pem: bytes | str,
    dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Build and sign a deterministic manifest over explicitly assigned split files."""
    if not all(isinstance(x, str) and x.strip() for x in (dataset_id, revision, signer_id)):
        raise ValueError("dataset_id, revision, and signer_id must be non-empty strings")
    if not isinstance(splits, Mapping) or not splits:
        raise ValueError("splits must be a non-empty mapping")
    if dataset_fingerprint is not None and (not isinstance(dataset_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", dataset_fingerprint)):
        raise ValueError("dataset_fingerprint must be a lowercase SHA-256 digest")
    root = Path(dataset_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("dataset_root must be a directory")
    normalized: dict[str, list[str]] = {}
    seen: set[str] = set()
    for split, paths in sorted(splits.items()):
        if not isinstance(split, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", split):
            raise ValueError("split names must be lowercase identifiers")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"split {split!r} must contain at least one file")
        normalized_paths = sorted(paths)
        for relative in normalized_paths:
            if relative in seen:
                raise ValueError(f"dataset file may belong to only one split: {relative}")
            seen.add(relative)
            _safe_file(root, relative)
        normalized[split] = normalized_paths
    files = []
    for relative in sorted(seen):
        path = _safe_file(root, relative)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": digest})
    from datetime import datetime, timezone
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": normalized,
        "files": files,
    }
    if dataset_fingerprint is not None:
        manifest["dataset_fingerprint"] = dataset_fingerprint
    key = _load_private_key(private_key_pem)
    signature = base64.b64encode(key.sign(canonical_json(manifest))).decode("ascii")
    return {
        "schema_version": 1,
        "manifest": manifest,
        "signature": {
            "algorithm": "Ed25519",
            "signer_id": signer_id,
            "public_key_sha256": _public_key_digest(key.public_key()),
            "value": signature,
        },
    }


def verify_signed_dataset_manifest(
    envelope: dict[str, Any],
    dataset_root: str | Path,
    *,
    trusted_signers: Mapping[str, bytes | str],
) -> dict[str, Any]:
    """Verify signer trust, signature, split membership, and every dataset byte."""
    from cryptography.exceptions import InvalidSignature
    if not isinstance(envelope, dict) or set(envelope) != {"schema_version", "manifest", "signature"}:
        raise ValueError("dataset manifest envelope schema is invalid")
    if envelope["schema_version"] != 1:
        raise ValueError("unsupported dataset manifest envelope version")
    manifest, signature = envelope["manifest"], envelope["signature"]
    if not isinstance(manifest, dict) or set(manifest) not in (
        {"schema_version", "dataset_id", "revision", "created_at", "splits", "files"},
        {"schema_version", "dataset_id", "revision", "created_at", "splits", "files", "dataset_fingerprint"},
    ):
        raise ValueError("dataset manifest schema is invalid")
    if isinstance(manifest["schema_version"], bool) or manifest["schema_version"] != 1:
        raise ValueError("unsupported dataset manifest version")
    for field in ("dataset_id", "revision", "created_at"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise ValueError(f"dataset manifest {field} must be a non-empty string")
    if "dataset_fingerprint" in manifest and (not isinstance(manifest["dataset_fingerprint"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", manifest["dataset_fingerprint"])):
        raise ValueError("dataset manifest fingerprint must be a lowercase SHA-256 digest")
    if not isinstance(signature, dict) or set(signature) != {
        "algorithm", "signer_id", "public_key_sha256", "value"
    } or signature["algorithm"] != "Ed25519":
        raise ValueError("dataset manifest signature schema is invalid")
    signer_id = signature["signer_id"]
    if not isinstance(signer_id, str) or signer_id not in trusted_signers:
        raise PermissionError("dataset manifest signer is not trusted")
    key = _load_public_key(trusted_signers[signer_id])
    if _public_key_digest(key) != signature["public_key_sha256"]:
        raise PermissionError("dataset manifest signer key fingerprint mismatch")
    try:
        key.verify(base64.b64decode(signature["value"], validate=True), canonical_json(manifest))
    except (ValueError, InvalidSignature) as exc:
        raise PermissionError("dataset manifest signature verification failed") from exc

    splits = manifest["splits"]
    files = manifest["files"]
    if not isinstance(splits, dict) or not splits or not isinstance(files, list):
        raise ValueError("dataset manifest splits/files are invalid")
    membership: dict[str, str] = {}
    for split, paths in splits.items():
        if not isinstance(split, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", split):
            raise ValueError("dataset split name is invalid")
        if not isinstance(paths, list) or not paths:
            raise ValueError("dataset split must contain at least one file")
        for relative in paths:
            if not isinstance(relative, str) or not relative:
                raise ValueError("dataset split paths must be non-empty strings")
            if relative in membership:
                raise ValueError("dataset file appears in multiple splits")
            membership[relative] = split
    file_map = {}
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
            raise ValueError("dataset file record schema is invalid")
        relative = record["path"]
        if not isinstance(relative, str) or relative in file_map:
            raise ValueError("dataset file path is invalid or duplicated")
        size, digest = record["size_bytes"], record["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("dataset file size must be a non-negative integer")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("dataset file SHA-256 is invalid")
        file_map[relative] = record
    if set(file_map) != set(membership):
        raise ValueError("dataset split membership does not match file records")

    root = Path(dataset_root).expanduser().resolve(strict=True)
    verified = []
    for relative, record in sorted(file_map.items()):
        path = _safe_file(root, relative)
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if len(data) != record["size_bytes"] or actual != record["sha256"]:
            raise PermissionError(f"dataset file integrity check failed: {relative}")
        verified.append({"path": relative, "split": membership[relative], "sha256": actual,
                         "size_bytes": len(data)})
    return {
        "dataset_id": manifest["dataset_id"],
        "revision": manifest["revision"],
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "signer_id": signer_id,
        "manifest_sha256": hashlib.sha256(canonical_json(manifest)).hexdigest(),
        "verified_files": verified,
    }
