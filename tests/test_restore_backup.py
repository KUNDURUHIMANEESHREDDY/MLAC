from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.restore_backup import restore_sqlite_backup


def _backup(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE state (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO state (value) VALUES (?)", ("preserved",))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_restore_creates_private_integrity_checked_copy(tmp_path: Path) -> None:
    backup = tmp_path / "backup.sqlite3"
    digest = _backup(backup)
    destination = tmp_path / "isolated" / "restored.sqlite3"
    destination.parent.mkdir(mode=0o700)

    result = restore_sqlite_backup(backup, destination, digest)

    assert result == {
        "format": "sqlite",
        "sha256": digest,
        "destination": str(destination),
        "table_count": 1,
    }
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    with sqlite3.connect(destination) as restored:
        assert restored.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert restored.execute("SELECT value FROM state").fetchone() == ("preserved",)


def test_restore_refuses_to_overwrite_existing_destination(tmp_path: Path) -> None:
    backup = tmp_path / "backup.sqlite3"
    digest = _backup(backup)
    destination = tmp_path / "existing.sqlite3"
    destination.write_bytes(b"do not replace")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        restore_sqlite_backup(backup, destination, digest)

    assert destination.read_bytes() == b"do not replace"


def test_restore_rejects_digest_mismatch_before_creating_destination(tmp_path: Path) -> None:
    backup = tmp_path / "backup.sqlite3"
    _backup(backup)
    destination = tmp_path / "restored.sqlite3"

    with pytest.raises(ValueError, match="SHA-256"):
        restore_sqlite_backup(backup, destination, "0" * 64)

    assert not destination.exists()


def test_restore_rejects_postgres_archive_without_creating_destination(tmp_path: Path) -> None:
    backup = tmp_path / "backup.dump"
    backup.write_bytes(b"PGDMP fixture")
    destination = tmp_path / "restored.sqlite3"

    with patch(
        "scripts.restore_backup.verify_backup",
        return_value={"format": "postgres-custom", "sha256": "a" * 64},
    ):
        with pytest.raises(ValueError, match="PostgreSQL archives"):
            restore_sqlite_backup(backup, destination, "a" * 64)

    assert not destination.exists()


def test_restore_requires_private_destination_directory(tmp_path: Path) -> None:
    backup = tmp_path / "backup.sqlite3"
    digest = _backup(backup)
    public_directory = tmp_path / "shared"
    public_directory.mkdir(mode=0o777)
    public_directory.chmod(0o777)

    with pytest.raises(ValueError, match="group/world writable"):
        restore_sqlite_backup(backup, public_directory / "restored.sqlite3", digest)

    assert not (public_directory / "restored.sqlite3").exists()


def test_restore_closes_source_if_destination_connection_fails(
    tmp_path: Path,
) -> None:
    backup = tmp_path / "backup.sqlite3"
    digest = _backup(backup)
    destination = tmp_path / "restored.sqlite3"
    real_connect = sqlite3.connect
    opened_source = None

    def fail_destination_connect(database, *args, **kwargs):
        nonlocal opened_source
        if opened_source is None:
            opened_source = real_connect(database, *args, **kwargs)
            return opened_source
        raise sqlite3.OperationalError("simulated destination open failure")

    with (
        patch("scripts.restore_backup.verify_backup", return_value={"format": "sqlite", "sha256": digest}),
        patch("scripts.restore_backup.sqlite3.connect", side_effect=fail_destination_connect),
        pytest.raises(sqlite3.OperationalError, match="destination open failure"),
    ):
        restore_sqlite_backup(backup, destination, digest)

    assert opened_source is not None
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened_source.execute("SELECT 1")
    assert not destination.exists()


def test_restore_syncs_file_and_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backup = tmp_path / "backup.sqlite3"
    digest = _backup(backup)
    destination = tmp_path / "isolated" / "restored.sqlite3"
    destination.parent.mkdir(mode=0o700)
    real_fsync = os.fsync
    synced_modes: list[int] = []

    def record_fsync(descriptor: int) -> None:
        synced_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr("scripts.restore_backup.os.fsync", record_fsync)

    restore_sqlite_backup(backup, destination, digest)

    assert len(synced_modes) == 2
    assert stat.S_ISREG(synced_modes[0])
    assert stat.S_ISDIR(synced_modes[1])
