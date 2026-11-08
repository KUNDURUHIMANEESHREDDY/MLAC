"""Verify a downloaded MLAC state backup before an operator restores it."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

_SQLITE_SIGNATURE = b"SQLite format 3\x00"
_POSTGRES_CUSTOM_SIGNATURE = b"PGDMP"
_READ_CHUNK_BYTES = 1024 * 1024


def _regular_file_identity(path: Path) -> tuple[int, int]:
    """Reject symlinks and non-regular backup paths."""
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot inspect backup file: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("backup must be a regular, non-symlink file")
    return metadata.st_dev, metadata.st_ino


def _sha256(path: Path, expected_identity: tuple[int, int]) -> str:
    """Hash the backup in bounded memory and reject path replacement."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as backup:
            metadata = os.fstat(backup.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("backup must be a regular file")
            if (metadata.st_dev, metadata.st_ino) != expected_identity:
                raise ValueError("backup changed while it was being opened")
            for chunk in iter(lambda: backup.read(_READ_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read backup file: {exc}") from exc
    if _regular_file_identity(path) != expected_identity:
        raise ValueError("backup path changed while it was being verified")
    return digest.hexdigest()


def _detect_format(path: Path) -> str:
    """Identify only the two backup formats emitted by MLAC."""
    try:
        with path.open("rb") as backup:
            signature = backup.read(16)
    except OSError as exc:
        raise ValueError(f"cannot read backup signature: {exc}") from exc
    if signature.startswith(_SQLITE_SIGNATURE):
        return "sqlite"
    if signature.startswith(_POSTGRES_CUSTOM_SIGNATURE):
        return "postgres-custom"
    raise ValueError("unsupported backup signature; expected SQLite or PostgreSQL custom format")


def _verify_sqlite(path: Path, expected_identity: tuple[int, int]) -> None:
    """Run SQLite integrity checks without allowing writes or trusted schema code."""
    if _regular_file_identity(path) != expected_identity:
        raise ValueError("backup path changed before SQLite verification")
    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro&immutable=1",
        uri=True,
        timeout=10,
    )
    try:
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA query_only = ON")
        results = connection.execute("PRAGMA integrity_check").fetchall()
        if results != [("ok",)]:
            raise ValueError("SQLite backup integrity_check did not return exactly 'ok'")
        connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
        ).fetchone()
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"SQLite backup is not readable or structurally valid: {exc}") from exc
    finally:
        connection.close()
    if _regular_file_identity(path) != expected_identity:
        raise ValueError("backup path changed during SQLite verification")


def _verify_postgres(path: Path) -> None:
    """Validate a PostgreSQL custom archive's table of contents without restoring it."""
    pg_restore = shutil.which("pg_restore")
    if pg_restore is None:
        raise ValueError("pg_restore is required to inspect a PostgreSQL custom archive")
    try:
        result = subprocess.run(
            [pg_restore, "--list", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"pg_restore could not inspect the backup: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise ValueError(f"pg_restore rejected the backup archive: {detail}")
    entries = [line for line in result.stdout.splitlines() if re.match(r"^\d+;", line)]
    if not entries:
        raise ValueError("PostgreSQL archive contains no readable table-of-contents entries")


def verify_backup(
    backup_path: Path,
    expected_sha256: str,
    backup_format: str = "auto",
) -> dict[str, str]:
    """Verify trusted digest metadata and the backup's format-specific integrity."""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected SHA-256 must contain exactly 64 hexadecimal characters")
    if backup_format not in {"auto", "sqlite", "postgres-custom"}:
        raise ValueError("backup format must be auto, sqlite, or postgres-custom")

    path = Path(backup_path)
    identity = _regular_file_identity(path)
    actual_digest = _sha256(path, identity)
    if not hmac.compare_digest(actual_digest, expected_sha256.lower()):
        raise ValueError("backup SHA-256 does not match trusted metadata")

    detected_format = _detect_format(path)
    if backup_format != "auto" and backup_format != detected_format:
        raise ValueError(
            f"backup format mismatch: expected {backup_format}, detected {detected_format}"
        )

    if detected_format == "sqlite":
        _verify_sqlite(path, identity)
    else:
        _verify_postgres(path)

    return {"format": detected_format, "sha256": actual_digest}


def main(argv: list[str] | None = None) -> int:
    """Verify one downloaded backup and print a non-secret result."""
    parser = argparse.ArgumentParser(
        description="Verify an MLAC SQLite or PostgreSQL custom-format backup."
    )
    parser.add_argument("backup", type=Path, help="downloaded backup file")
    parser.add_argument(
        "--sha256",
        required=True,
        help="trusted SHA-256 value from the backup object's metadata",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "sqlite", "postgres-custom"),
        default="auto",
        help="expected backup format (default: detect from file signature)",
    )
    arguments = parser.parse_args(argv)

    try:
        result = verify_backup(arguments.backup, arguments.sha256, arguments.format)
    except (OSError, ValueError) as exc:
        print(f"Backup verification failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Backup verification passed: "
        f"format={result['format']} sha256={result['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
