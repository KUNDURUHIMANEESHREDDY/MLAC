"""Disposable PostgreSQL restore-drill integration and target-validation tests."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest

from scripts.restore_postgres_backup import parse_postgres_target, restore_postgres_backup


def test_restore_target_requires_dedicated_database_name() -> None:
    with pytest.raises(ValueError, match="mlac_restore_drill_"):
        parse_postgres_target(
            "postgresql://mlac:secret@localhost:5432/production?sslmode=disable"
        )


def test_restore_target_requires_tls_for_remote_hosts() -> None:
    with pytest.raises(ValueError, match="remote restore targets must require TLS"):
        parse_postgres_target(
            "postgresql://mlac:secret@db.example.com:5432/"
            "mlac_restore_drill_remote?sslmode=disable"
        )


def test_restore_target_rejects_unreviewed_libpq_options() -> None:
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

    psycopg2 = pytest.importorskip("psycopg2")
    from psycopg2 import sql

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

    admin = psycopg2.connect(admin_url)
    try:
        admin.autocommit = True
        with admin.cursor() as cursor:
            for name in (source_name, target_name):
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
                created.append(name)
    finally:
        admin.close()

    try:
        source = parse_postgres_target(source_url)
        connection = psycopg2.connect(**source.connection_kwargs())
        try:
            with connection:
                with connection.cursor() as cursor:
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
        cleanup = psycopg2.connect(admin_url)
        try:
            cleanup.autocommit = True
            with cleanup.cursor() as cursor:
                for name in reversed(created):
                    cursor.execute(
                        sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name))
                    )
        finally:
            cleanup.close()


def test_postgres_backup_restores_into_empty_disposable_database(
    postgres_archive_and_target,
) -> None:
    backup, digest, target_url, target_name = postgres_archive_and_target
    psycopg2 = pytest.importorskip("psycopg2")

    result = restore_postgres_backup(backup, target_url, digest)

    assert result["format"] == "postgres-custom"
    assert result["sha256"] == digest
    assert result["database"] == target_name
    assert result["table_count"] == 1

    target = parse_postgres_target(target_url)
    connection = psycopg2.connect(**target.connection_kwargs())
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT value FROM restore_probe WHERE id = %s", (1,))
            assert cursor.fetchone() == ("preserved",)
    finally:
        connection.close()


def test_postgres_restore_refuses_nonempty_target_without_changing_it(
    postgres_archive_and_target,
) -> None:
    backup, digest, target_url, _target_name = postgres_archive_and_target
    psycopg2 = pytest.importorskip("psycopg2")
    target = parse_postgres_target(target_url)

    connection = psycopg2.connect(**target.connection_kwargs())
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE TABLE operator_owned (value TEXT NOT NULL)")
                cursor.execute(
                    "INSERT INTO operator_owned (value) VALUES (%s)",
                    ("keep-this-row",),
                )
    finally:
        connection.close()

    with pytest.raises(ValueError, match="target is not empty"):
        restore_postgres_backup(backup, target_url, digest)

    connection = psycopg2.connect(**target.connection_kwargs())
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT value FROM operator_owned")
            assert cursor.fetchone() == ("keep-this-row",)
    finally:
        connection.close()


def test_postgres_restore_rejects_digest_mismatch_before_connecting(
    postgres_archive_and_target,
) -> None:
    backup, _digest, target_url, _target_name = postgres_archive_and_target
    with pytest.raises(ValueError, match="SHA-256"):
        restore_postgres_backup(backup, target_url, "0" * 64)
