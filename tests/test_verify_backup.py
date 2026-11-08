from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.verify_backup import verify_backup


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sqlite_backup(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE state (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO state (value) VALUES (?)", ("preserved",))
    return _sha256(path)


def test_verify_backup_accepts_integral_sqlite_file(tmp_path: Path) -> None:
    backup = tmp_path / "state.sqlite3"
    digest = _sqlite_backup(backup)

    result = verify_backup(backup, digest)

    assert result == {"format": "sqlite", "sha256": digest}


def test_verify_backup_rejects_digest_mismatch_before_parsing(tmp_path: Path) -> None:
    backup = tmp_path / "state.sqlite3"
    _sqlite_backup(backup)

    with pytest.raises(ValueError, match="SHA-256"):
        verify_backup(backup, "0" * 64)


def test_verify_backup_rejects_invalid_digest_syntax(tmp_path: Path) -> None:
    backup = tmp_path / "state.sqlite3"
    _sqlite_backup(backup)

    with pytest.raises(ValueError, match="64 hexadecimal"):
        verify_backup(backup, "not-a-digest")


def test_verify_backup_rejects_symlink(tmp_path: Path) -> None:
    backup = tmp_path / "state.sqlite3"
    _sqlite_backup(backup)
    link = tmp_path / "linked.sqlite3"
    try:
        link.symlink_to(backup)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are unavailable on this platform: {exc}")

    with pytest.raises(ValueError, match="regular, non-symlink"):
        verify_backup(link, _sha256(backup))


def test_verify_backup_rejects_corrupt_sqlite_file(tmp_path: Path) -> None:
    backup = tmp_path / "corrupt.sqlite3"
    backup.write_bytes(b"SQLite format 3\x00not a database")

    with pytest.raises(ValueError, match="SQLite backup"):
        verify_backup(backup, _sha256(backup))


def test_verify_backup_checks_postgres_archive_with_pg_restore(
    tmp_path: Path,
) -> None:
    backup = tmp_path / "state.dump"
    backup.write_bytes(b"PGDMP\x01\x0f\x00fixture")
    digest = _sha256(backup)
    completed = SimpleNamespace(
        returncode=0,
        stdout="; PostgreSQL custom archive\n1; 0 0 TABLE public state owner\n",
        stderr="",
    )

    with (
        patch("scripts.verify_backup.shutil.which", return_value="/usr/bin/pg_restore"),
        patch("scripts.verify_backup.subprocess.run", return_value=completed) as run,
    ):
        result = verify_backup(backup, digest)

    assert result == {"format": "postgres-custom", "sha256": digest}
    run.assert_called_once_with(
        ["/usr/bin/pg_restore", "--list", str(backup)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_verify_backup_rejects_postgres_archive_without_pg_restore(
    tmp_path: Path,
) -> None:
    backup = tmp_path / "state.dump"
    backup.write_bytes(b"PGDMP\x01\x0f\x00fixture")

    with (
        patch("scripts.verify_backup.shutil.which", return_value=None),
        pytest.raises(ValueError, match="pg_restore is required"),
    ):
        verify_backup(backup, _sha256(backup))


def test_verify_backup_rejects_postgres_archive_rejected_by_pg_restore(
    tmp_path: Path,
) -> None:
    backup = tmp_path / "state.dump"
    backup.write_bytes(b"PGDMP\x01\x0f\x00fixture")
    rejected = subprocess.CompletedProcess(
        args=["pg_restore", "--list"],
        returncode=1,
        stdout="",
        stderr="invalid archive",
    )

    with (
        patch("scripts.verify_backup.shutil.which", return_value="/usr/bin/pg_restore"),
        patch("scripts.verify_backup.subprocess.run", return_value=rejected),
        pytest.raises(ValueError, match="pg_restore rejected"),
    ):
        verify_backup(backup, _sha256(backup))


def test_verify_backup_rejects_explicit_format_mismatch(tmp_path: Path) -> None:
    backup = tmp_path / "state.sqlite3"
    digest = _sqlite_backup(backup)

    with pytest.raises(ValueError, match="format mismatch"):
        verify_backup(backup, digest, "postgres-custom")
