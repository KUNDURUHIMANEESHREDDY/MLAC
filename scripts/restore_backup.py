"""Restore an MLAC SQLite backup into a new, isolated file without overwriting data."""

from __future__ import annotations

import argparse
import os
import sqlite3
import stat
import sys
from pathlib import Path
from typing import Any

from scripts.verify_backup import verify_backup


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    """Return the device and inode used to identify the newly created destination."""
    return metadata.st_dev, metadata.st_ino


def restore_sqlite_backup(
    backup_path: Path,
    destination_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Restore a verified SQLite backup to a new file and validate the result.

    The destination must not exist and its parent directory must not be writable
    by group or other users. Existing files are never replaced or truncated.
    PostgreSQL archives are verified by the companion verifier but deliberately
    require a separately administered disposable database for restoration.
    """
    backup = Path(backup_path)
    destination = Path(destination_path)
    verified = verify_backup(backup, expected_sha256)

    if verified["format"] != "sqlite":
        raise ValueError(
            "automatic restore supports SQLite backups only; restore PostgreSQL "
            "archives into a dedicated disposable database using the operator runbook"
        )

    if not destination.name or destination.name in {".", ".."}:
        raise ValueError("destination must name a new file")
    try:
        parent = destination.parent.resolve(strict=True)
        parent_info = parent.stat()
    except OSError as exc:
        raise ValueError(f"destination parent directory is unavailable: {exc}") from exc
    if not stat.S_ISDIR(parent_info.st_mode):
        raise ValueError("destination parent must be a directory")
    if stat.S_IMODE(parent_info.st_mode) & 0o022:
        raise ValueError("destination parent directory must not be group/world writable")

    target = parent / destination.name
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing restore target: {target}") from exc
    except OSError as exc:
        raise ValueError(f"cannot create restore target safely: {exc}") from exc

    created_identity = _identity(os.fstat(descriptor))
    os.close(descriptor)

    try:
        source_uri = backup.resolve(strict=True).as_uri() + "?mode=ro&immutable=1"
        with (
            sqlite3.connect(source_uri, uri=True, timeout=10) as source,
            sqlite3.connect(target, timeout=10) as restored,
        ):
            source.backup(restored)
            restored.execute("PRAGMA trusted_schema = OFF")
            restored.execute("PRAGMA query_only = ON")
            integrity = restored.execute("PRAGMA integrity_check").fetchall()
            if integrity != [("ok",)]:
                raise ValueError("restored SQLite database failed integrity_check")
            table_count = restored.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
            ).fetchone()[0]

        current = target.lstat()
        if not stat.S_ISREG(current.st_mode) or _identity(current) != created_identity:
            raise ValueError("restore target identity changed during restore")
        os.chmod(target, 0o600)
        with target.open("rb") as restored_file:
            os.fsync(restored_file.fileno())

        return {
            "format": "sqlite",
            "sha256": verified["sha256"],
            "destination": str(target),
            "table_count": int(table_count),
        }
    except Exception:
        try:
            current = target.lstat()
            if stat.S_ISREG(current.st_mode) and _identity(current) == created_identity:
                target.unlink()
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    """Restore one verified SQLite backup into a new isolated destination."""
    parser = argparse.ArgumentParser(
        description="Restore an MLAC SQLite backup into a new file without overwriting."
    )
    parser.add_argument("backup", type=Path, help="downloaded backup file")
    parser.add_argument("destination", type=Path, help="new restore destination (must not exist)")
    parser.add_argument(
        "--sha256",
        required=True,
        help="trusted SHA-256 value from the backup object's metadata",
    )
    arguments = parser.parse_args(argv)

    try:
        result = restore_sqlite_backup(
            arguments.backup, arguments.destination, arguments.sha256
        )
    except (OSError, ValueError) as exc:
        print(f"Backup restore failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Isolated SQLite restore passed: "
        f"destination={result['destination']} "
        f"sha256={result['sha256']} tables={result['table_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
