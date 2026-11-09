"""Disposable PostgreSQL restore-drill integration and target-validation tests."""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import uuid
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest

import scripts.restore_postgres_backup as restore_module
from scripts.restore_postgres_backup import (
    parse_postgres_target,
    restore_postgres_backup,
)


def test_restore_target_requires_dedicated_database_name() -> None:
    """Reject production database names before opening a connection."""
    with pytest.raises(ValueError, match="mlac_restore_drill_"):
        parse_postgres_target(
            "postgresql://mlac:secret@localhost:5432/production?sslmode=disable"
        )


def test_restore_target_requires_tls_for_remote_hosts() -> None:
    """Require TLS whenever the restore target is not loopback."""
    with pytest.raises(ValueError, match="remote restore targets must require TLS"):
        parse_postgres_target(
            "postgresql://mlac:secret@db.example.com:5432/"
            "mlac_restore_drill_remote?sslmode=disable"
        )


def test_restore_target_rejects_unreviewed_libpq_options() -> None:
    """Reject connection options that could redirect or alter the restore."""
    with pytest.raises(ValueError, match="only one sslmode and one connect_timeout"):
        parse_postgres_target(
            "postgresql://mlac:secret@localhost:5432/"
            "mlac_restore_drill_local?sslmode=disable&options=-c%20search_path%3Dpublic"
        )


def _database_url(base_url: str, database: str) -> str:
    """Build a local-only test DSN while preserving the service credentials."""
    parsed = urlsplit(base_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["sslmode"] = "disable"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"/{database}", urlencode(query), "")
    )


@pytest.fixture
def postgres_archive_and_target(tmp_path: Path):
    """Create a source archive and a fresh disposable target in the CI PostgreSQL service."""
    base_url = os.getenv("MLAC_TEST_POSTGRES_URL")
    if not base_url:
        pytest.skip("MLAC_TEST_POSTGRES_URL is not configured")
    if not shutil.which("pg_dump") or not shutil.which("pg_restore"):
        pytest.skip("PostgreSQL client tools are not installed")

    psycopg = pytest.importorskip("psycopg")
    from psycopg import sql

    parsed = urlsplit(base_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["sslmode"] = "disable"
    admin_url = urlunsplit(
        (parsed.scheme, parsed.netloc, "/postgres", urlencode(query), "")
    )

    suffix = uuid.uuid4().hex[:12]
    source_name = f"mlac_restore_drill_source_{suffix}"
    target_name = f"mlac_restore_drill_target_{suffix}"
    source_url = _database_url(base_url, source_name)
    target_url = _database_url(base_url, target_name)
    created: list[str] = []

    try:
        admin = psycopg.connect(admin_url)
        try:
            admin.autocommit = True
            with admin.cursor() as cursor:
                for name in (source_name, target_name):
                    cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
                    created.append(name)
        finally:
            admin.close()

        source = parse_postgres_target(source_url)
        connection = psycopg.connect(**source.connection_kwargs())
        try:
            with connection, connection.cursor() as cursor:
                cursor.execute(
                    "CREATE TABLE restore_probe (id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
                )
                cursor.execute(
                    "INSERT INTO restore_probe (id, value) VALUES (%s, %s)",
                    (1, "preserved"),
                )
        finally:
            connection.close()

        backup = tmp_path / "state.dump"
        subprocess.run(
            [
                shutil.which("pg_dump"),
                "--format=custom",
                "--no-owner",
                "--no-acl",
                "--file",
                str(backup),
            ],
            env=source.environment(),
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        digest = hashlib.sha256(backup.read_bytes()).hexdigest()
        yield backup, digest, target_url, target_name
    finally:
        if created:
            cleanup = psycopg.connect(admin_url)
            try:
                cleanup.autocommit = True
                with cleanup.cursor() as cursor:
                    for name in reversed(created):
                        cursor.execute(
                            sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name))
                        )
            finally:
                cleanup.close()


def test_restore_target_pins_one_server_address_for_both_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin one resolved IP for psycopg inspection and pg_restore execution."""
    for name in ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE", "PGOPTIONS"):
        monkeypatch.delenv(name, raising=False)

    def resolve_target(host: str, port: int, **kwargs):
        """Return a deterministic test address without making a network request."""
        assert host == "db.example.com"
        assert port == 5432
        assert kwargs.get("type") == socket.SOCK_STREAM
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("192.0.2.10", port))
        ]

    monkeypatch.setattr(restore_module.socket, "getaddrinfo", resolve_target)
    target = parse_postgres_target(
        "postgresql://mlac:secret@db.example.com:5432/"
        "mlac_restore_drill_pinned?sslmode=require"
    )

    assert target.hostaddr == "192.0.2.10"
    connection = target.connection_kwargs()
    environment = target.environment()
    assert connection["hostaddr"] == target.hostaddr
    assert connection["options"] == ""
    assert "service" not in connection
    assert environment["PGHOSTADDR"] == target.hostaddr
    assert environment["PGHOST"] == target.host


@pytest.mark.parametrize(
    "name",
    ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE", "PGOPTIONS", "pghostaddr"),
)
def test_restore_target_rejects_libpq_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Fail closed rather than let ambient libpq settings redirect inspection."""
    target = parse_postgres_target(
        "postgresql://mlac:secret@127.0.0.1:5432/"
        "mlac_restore_drill_environment?sslmode=disable"
    )
    monkeypatch.setenv(name, "attacker-controlled-override")
    with pytest.raises(ValueError, match="libpq environment overrides to be unset"):
        target.connection_kwargs()


def test_postgres_backup_restores_into_empty_disposable_database(
    postgres_archive_and_target,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore a verified archive and assert its data and safe invocation."""
    backup, digest, target_url, target_name = postgres_archive_and_target
    psycopg = pytest.importorskip("psycopg")
    target = parse_postgres_target(target_url)
    real_run = subprocess.run
    restore_commands: list[list[str]] = []

    def capture_restore_command(*args, **kwargs):
        """Capture only the actual restore invocation for argv assertions."""
        command = args[0]
        if (
            isinstance(command, list)
            and command
            and Path(command[0]).name == "pg_restore"
            and "--list" not in command
        ):
            # verify_backup uses pg_restore --list for archive inspection; capture
            # only the separate invocation that executes the restore transaction.
            assert kwargs.get("shell", False) is False
            restore_commands.append(command)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(restore_module.subprocess, "run", capture_restore_command)
    result = restore_postgres_backup(backup, target_url, digest)

    assert len(restore_commands) == 1
    restore_command = restore_commands[0]
    assert "--dbname" in restore_command
    assert restore_command[restore_command.index("--dbname") + 1] == target.database
    fixed_flags = {
        "--exit-on-error",
        "--no-owner",
        "--no-acl",
        "--single-transaction",
        "--no-password",
        "--dbname",
    }
    assert fixed_flags <= set(restore_command[1:-1])
    connection_args = [arg for arg in restore_command[1:-1] if arg not in fixed_flags]
    # The target database is the only non-flag argument before the archive path.
    # Credentials must remain in the sanitized PG* environment, never argv.
    assert connection_args == [target.database]
    assert target.password not in connection_args
    assert result["format"] == "postgres-custom"
    assert result["sha256"] == digest
    assert result["database"] == target_name
    assert result["table_count"] == 1

    target = parse_postgres_target(target_url)
    connection = psycopg.connect(**target.connection_kwargs())
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT value FROM restore_probe WHERE id = %s", (1,))
            assert cursor.fetchone() == ("preserved",)
    finally:
        connection.close()


def test_postgres_restore_refuses_nonempty_target_without_changing_it(
    postgres_archive_and_target,
) -> None:
    """Refuse a populated target and preserve its existing row."""
    backup, digest, target_url, _target_name = postgres_archive_and_target
    psycopg = pytest.importorskip("psycopg")
    target = parse_postgres_target(target_url)

    connection = psycopg.connect(**target.connection_kwargs())
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute("CREATE TABLE operator_owned (value TEXT NOT NULL)")
            cursor.execute(
                "INSERT INTO operator_owned (value) VALUES (%s)",
                ("keep-this-row",),
            )
    finally:
        connection.close()

    with pytest.raises(ValueError, match="target is not empty"):
        restore_postgres_backup(backup, target_url, digest)

    connection = psycopg.connect(**target.connection_kwargs())
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT value FROM operator_owned")
            assert cursor.fetchone() == ("keep-this-row",)
    finally:
        connection.close()


def test_postgres_restore_refuses_target_with_large_objects_without_changing_it(
    postgres_archive_and_target,
) -> None:
    """Refuse a target containing only a PostgreSQL large object."""
    backup, digest, target_url, _target_name = postgres_archive_and_target
    psycopg = pytest.importorskip("psycopg")
    target = parse_postgres_target(target_url)

    connection = psycopg.connect(**target.connection_kwargs())
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute("SELECT lo_create(0)")
            row = cursor.fetchone()
            assert row is not None
            large_object_oid = row[0]
    finally:
        connection.close()

    with pytest.raises(ValueError, match="target is not empty"):
        restore_postgres_backup(backup, target_url, digest)

    connection = psycopg.connect(**target.connection_kwargs())
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_largeobject_metadata WHERE oid = %s)",
                (large_object_oid,),
            )
            assert cursor.fetchone() == (True,)
    finally:
        connection.close()


def test_postgres_restore_rejects_digest_mismatch_before_connecting(
    postgres_archive_and_target,
) -> None:
    """Reject a backup whose bytes do not match trusted metadata."""
    backup, _digest, target_url, _target_name = postgres_archive_and_target
    with pytest.raises(ValueError, match="SHA-256"):
        restore_postgres_backup(backup, target_url, "0" * 64)


def test_postgres_restore_uses_the_exact_verified_snapshot(
    postgres_archive_and_target,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore the private verified snapshot despite source-path mutation."""
    backup, digest, target_url, target_name = postgres_archive_and_target
    real_verify = restore_module.verify_backup

    def verify_snapshot_then_mutate_original(
        snapshot_path: Path,
        expected_sha256: str,
        backup_format: str = "auto",
    ):
        """Verify the private snapshot before mutating the caller's source path."""
        snapshot = Path(snapshot_path)
        assert snapshot != backup
        result = real_verify(
            snapshot,
            expected_sha256,
            backup_format=backup_format,
        )
        backup.write_bytes(b"replacement after verification")
        return result

    monkeypatch.setattr(
        restore_module,
        "verify_backup",
        verify_snapshot_then_mutate_original,
    )
    result = restore_module.restore_postgres_backup(backup, target_url, digest)

    assert result["database"] == target_name
    assert result["sha256"] == digest
    target = restore_module.parse_postgres_target(target_url)
    psycopg = pytest.importorskip("psycopg")
    connection = psycopg.connect(**target.connection_kwargs())
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT value FROM restore_probe WHERE id = %s", (1,))
            assert cursor.fetchone() == ("preserved",)
    finally:
        connection.close()
