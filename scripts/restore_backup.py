"""Restore an MLAC SQLite backup into a new, isolated file without overwriting data."""

from __future__ import annotations

import argparse
import os
import sqlite3
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts.verify_backup import verify_backup


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    """Return the device and inode used to identify the newly created destination."""
    return metadata.st_dev, metadata.st_ino


def _stable_source_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Capture identity and mutation-sensitive metadata for a source backup."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _copy_backup_snapshot(backup_path: Path, snapshot_path: Path) -> None:
    """Copy one no-follow regular-file handle into the private restore workspace."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("restore requires no-follow file opening support")

    source_flags = os.O_RDONLY | os.O_NOFOLLOW
    snapshot_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        source_flags |= os.O_CLOEXEC
        snapshot_flags |= os.O_CLOEXEC

    try:
        source_descriptor = os.open(backup_path, source_flags)
    except OSError as exc:
        raise ValueError(f"cannot open backup safely: {exc}") from exc

    try:
        source_before = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_before.st_mode):
            raise ValueError("backup must be a regular, non-symlink file")
        source_identity = _stable_source_identity(source_before)
        try:
            path_before = backup_path.lstat()
        except OSError as exc:
            raise ValueError(f"cannot inspect backup path: {exc}") from exc
        if (
            not stat.S_ISREG(path_before.st_mode)
            or _stable_source_identity(path_before) != source_identity
        ):
            raise ValueError("backup path changed before snapshot creation")

        try:
            snapshot_descriptor = os.open(snapshot_path, snapshot_flags, 0o600)
        except OSError as exc:
            raise ValueError(f"cannot create private backup snapshot: {exc}") from exc

        try:
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(snapshot_descriptor, remaining)
                    if written <= 0:
                        raise OSError("short write while creating backup snapshot")
                    remaining = remaining[written:]
            os.fsync(snapshot_descriptor)

            source_after = os.fstat(source_descriptor)
            try:
                path_after = backup_path.lstat()
            except OSError as exc:
                raise ValueError(f"backup path changed while snapshot was copied: {exc}") from exc
            if (
                _stable_source_identity(source_after) != source_identity
                or _stable_source_identity(path_after) != source_identity
            ):
                raise ValueError("backup changed while its stable snapshot was being copied")
        finally:
            os.close(snapshot_descriptor)
    finally:
        os.close(source_descriptor)


def restore_sqlite_backup(
    backup_path: Path,
    destination_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Restore a verified SQLite snapshot to a new file and validate the result.

    The source backup is copied through one no-follow file descriptor into a
    private temporary directory. Digest verification and SQLite restoration use
    that same immutable snapshot, so a later replacement or mutation of the
    downloaded pathname cannot change the bytes being restored.

    The destination must not exist and its parent directory must not be writable
    by group or other users. Existing files are never replaced or truncated.
    PostgreSQL archives are verified but require an operator-run restore into a
    separately provisioned disposable database.
    """
    backup = Path(backup_path)
    destination = Path(destination_path)

    if os.name != "posix" or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("restore requires POSIX directory fsync support for durable creation")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("restore requires no-follow file opening support")
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
    with tempfile.TemporaryDirectory(prefix=".mlac-restore-", dir=parent) as workspace:
        snapshot = Path(workspace) / "backup.snapshot"
        _copy_backup_snapshot(backup, snapshot)

        # Both verification and restore consume this private, stable snapshot.
        verified = verify_backup(snapshot, expected_sha256)
        if verified["format"] != "sqlite":
            raise ValueError(
                "automatic restore supports SQLite backups only; restore PostgreSQL "
                "archives into a dedicated disposable database using the operator runbook"
            )

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            descriptor = os.open(target, flags, 0o600)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing restore target: {target}") from exc
        except OSError as exc:
            raise ValueError(f"cannot create restore target safely: {exc}") from exc

        try:
            created_identity = _identity(os.fstat(descriptor))
        finally:
            os.close(descriptor)

        try:
            source: sqlite3.Connection | None = None
            restored: sqlite3.Connection | None = None
            try:
                source_uri = snapshot.resolve(strict=True).as_uri() + "?mode=ro&immutable=1"
                source = sqlite3.connect(source_uri, uri=True, timeout=10)
                restored = sqlite3.connect(target, timeout=10)
                source.backup(restored)
                restored.execute("PRAGMA trusted_schema = OFF")
                restored.execute("PRAGMA query_only = ON")
                integrity = restored.execute("PRAGMA integrity_check").fetchall()
                if integrity != [("ok",)]:
                    raise ValueError("restored SQLite database failed integrity_check")
                table_count = restored.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
                ).fetchone()[0]
            finally:
                if restored is not None:
                    restored.close()
                if source is not None:
                    source.close()

            current = target.lstat()
            if not stat.S_ISREG(current.st_mode) or _identity(current) != created_identity:
                raise ValueError("restore target identity changed during restore")
            os.chmod(target, 0o600)
            sync_flags = os.O_RDWR | os.O_NOFOLLOW
            if hasattr(os, "O_CLOEXEC"):
                sync_flags |= os.O_CLOEXEC
            sync_descriptor = os.open(target, sync_flags)
            try:
                if _identity(os.fstat(sync_descriptor)) != created_identity:
                    raise ValueError("restore target identity changed before sync")
                os.fsync(sync_descriptor)
            finally:
                os.close(sync_descriptor)

            directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)

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
