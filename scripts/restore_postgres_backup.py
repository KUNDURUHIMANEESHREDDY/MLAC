"""Restore a PostgreSQL custom-format backup into a new, empty drill database.

The command never creates or drops a database and never uses pg_restore's
--clean or --create options. It requires an explicitly named disposable target,
checks that the target contains no user objects, and runs the restore in one
transaction so a failed restore cannot leave a partially populated database.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, unquote, urlparse

from scripts.restore_backup import _copy_backup_snapshot
from scripts.verify_backup import verify_backup

_DATABASE_NAME = re.compile(r"^mlac_restore_drill_[a-z0-9_]{1,44}$")
_SSL_MODES = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
_ALLOWED_QUERY_KEYS = {"sslmode", "connect_timeout"}


@dataclass(frozen=True)
class PostgresTarget:
    """Validated connection settings for an explicitly disposable database."""

    host: str
    port: int
    username: str
    password: str
    database: str
    sslmode: str
    connect_timeout: int

    def connection_kwargs(self) -> dict[str, object]:
        """Return explicit libpq keyword arguments without exposing a DSN."""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.username,
            "password": self.password,
            "dbname": self.database,
            "sslmode": self.sslmode,
            "connect_timeout": self.connect_timeout,
            "application_name": "mlac-postgres-restore-drill",
        }

    def environment(self) -> dict[str, str]:
        """Build a sanitized libpq environment for pg_restore."""
        environment = os.environ.copy()
        for name in (
            "PGSERVICE",
            "PGSERVICEFILE",
            "PGOPTIONS",
            "PGHOSTADDR",
            "PGHOST",
            "PGPORT",
            "PGUSER",
            "PGPASSWORD",
            "PGDATABASE",
            "PGSSLMODE",
            "PGCONNECT_TIMEOUT",
            "PGAPPNAME",
        ):
            environment.pop(name, None)
        environment.update(
            {
                "PGHOST": self.host,
                "PGPORT": str(self.port),
                "PGUSER": self.username,
                "PGPASSWORD": self.password,
                "PGDATABASE": self.database,
                "PGSSLMODE": self.sslmode,
                "PGCONNECT_TIMEOUT": str(self.connect_timeout),
                "PGAPPNAME": "mlac-postgres-restore-drill",
            }
        )
        return environment


def parse_postgres_target(database_url: str) -> PostgresTarget:
    """Validate a PostgreSQL URL and require the dedicated restore-drill prefix."""
    if not isinstance(database_url, str) or not database_url.strip():
        raise ValueError("MLAC_RESTORE_DATABASE_URL must be configured")

    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ValueError("restore target must be a PostgreSQL URL with a host")
    if parsed.fragment or parsed.params or parsed.path.count("/") != 1:
        raise ValueError("restore target URL contains an unsupported component")

    database = unquote(parsed.path.lstrip("/"))
    if not _DATABASE_NAME.fullmatch(database):
        raise ValueError(
            "restore target database name must start with mlac_restore_drill_"
        )

    if not parsed.username or parsed.password is None:
        raise ValueError("restore target URL must include a username and password")

    query = parse_qs(parsed.query, keep_blank_values=True)
    if set(query) - _ALLOWED_QUERY_KEYS or any(len(values) != 1 for values in query.values()):
        raise ValueError(
            "restore target URL may contain only one sslmode and one connect_timeout"
        )

    sslmode = query.get("sslmode", ["require"])[0]
    if sslmode not in _SSL_MODES:
        raise ValueError("restore target sslmode is invalid")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname.casefold() == "localhost"
    if sslmode in {"disable", "allow", "prefer"} and not loopback:
        raise ValueError("remote restore targets must require TLS")

    try:
        port = parsed.port or 5432
        connect_timeout = int(query.get("connect_timeout", ["10"])[0])
    except (ValueError, TypeError) as exc:
        raise ValueError("restore target port or connect_timeout is invalid") from exc
    if not 1 <= port <= 65535 or not 1 <= connect_timeout <= 30:
        raise ValueError("restore target port or connect_timeout is outside its allowed range")

    return PostgresTarget(
        host=parsed.hostname,
        port=port,
        username=unquote(parsed.username),
        password=unquote(parsed.password),
        database=database,
        sslmode=sslmode,
        connect_timeout=connect_timeout,
    )


def _target_is_empty(target: PostgresTarget) -> bool:
    """Refuse any target containing user schemas, relations, functions, or types."""
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is required; install the MLAC production extra"
        ) from exc

    connection = None
    try:
        connection = psycopg.connect(**target.connection_kwargs())
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    EXISTS (
                        SELECT 1
                        FROM pg_catalog.pg_class AS relation
                        JOIN pg_catalog.pg_namespace AS namespace
                          ON namespace.oid = relation.relnamespace
                        WHERE namespace.nspname !~ '^pg_' AND namespace.nspname <> 'information_schema'
                          AND relation.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
                    )
                    OR EXISTS (
                        SELECT 1
                        FROM pg_catalog.pg_proc AS proc
                        JOIN pg_catalog.pg_namespace AS namespace
                          ON namespace.oid = proc.pronamespace
                        WHERE namespace.nspname !~ '^pg_' AND namespace.nspname <> 'information_schema'
                    )
                    OR EXISTS (
                        SELECT 1
                        FROM pg_catalog.pg_type AS typ
                        JOIN pg_catalog.pg_namespace AS namespace
                          ON namespace.oid = typ.typnamespace
                        WHERE namespace.nspname !~ '^pg_' AND namespace.nspname <> 'information_schema'
                    )
                    OR EXISTS (
                        SELECT 1
                        FROM pg_catalog.pg_namespace
                        WHERE nspname !~ '^pg_' AND nspname NOT IN ('information_schema', 'public')
                    )
                """
            )
            row = cursor.fetchone()
            return row is not None and not bool(row[0])
    except Exception as exc:
        if exc.__class__.__module__.startswith("psycopg"):
            raise RuntimeError(
                f"cannot inspect the disposable PostgreSQL target ({type(exc).__name__})"
            ) from exc
        raise
    finally:
        if connection is not None:
            connection.close()


def _count_restored_tables(target: PostgresTarget) -> int:
    """Count user tables after restore without returning row contents."""
    import psycopg

    connection = None
    try:
        connection = psycopg.connect(**target.connection_kwargs())
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            row = cursor.fetchone()
            if row is None or row[0] != target.database:
                raise RuntimeError("connected database identity changed during restore")
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM pg_catalog.pg_class AS relation
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname !~ '^pg_' AND namespace.nspname <> 'information_schema'
                  AND relation.relkind IN ('r', 'p')
                """
            )
            count = cursor.fetchone()
            return int(count[0]) if count else 0
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        if exc.__class__.__module__.startswith("psycopg"):
            raise RuntimeError(
                f"could not verify the restored PostgreSQL database ({type(exc).__name__})"
            ) from exc
        raise
    finally:
        if connection is not None:
            connection.close()


def restore_postgres_backup(
    backup_path: Path,
    database_url: str,
    expected_sha256: str,
) -> dict[str, object]:
    """Restore one verified custom archive into a new empty drill database.

    The target name must begin with `mlac_restore_drill_`. The command never
    drops or creates a database and never asks pg_restore to clean existing
    objects. A single transaction makes archive-restore failures atomic.
    """
    backup = Path(backup_path)
    target = parse_postgres_target(database_url)

    pg_restore = shutil.which("pg_restore")
    if pg_restore is None:
        raise RuntimeError("pg_restore is required for PostgreSQL restore drills")

    # Hash verification and restore must consume the same private snapshot. A
    # caller replacing the downloaded pathname after verification must not be
    # able to change the bytes pg_restore receives.
    with TemporaryDirectory(prefix="mlac-postgres-restore-") as workspace:
        snapshot = Path(workspace) / "backup.snapshot"
        _copy_backup_snapshot(backup, snapshot)
        verified = verify_backup(
            snapshot,
            expected_sha256,
            backup_format="postgres-custom",
        )

        if not _target_is_empty(target):
            raise ValueError(
                "restore target is not empty; refusing to modify an existing database"
            )

        try:
            result = subprocess.run(
                [
                    pg_restore,
                    "--exit-on-error",
                    "--no-owner",
                    "--no-acl",
                    "--single-transaction",
                    "--no-password",
                    # Supplying --dbname is required for pg_restore to execute
                    # the archive. Without it pg_restore has no output target and
                    # exits with: "one of -d/--dbname and -f/--file must be specified".
                    # The database name is safe to expose; credentials remain in
                    # the sanitized PG* environment and never appear in argv.
                    "--dbname",
                    target.database,
                    str(snapshot),
                ],
                env=target.environment(),
                capture_output=True,
                text=True,
                check=False,
                timeout=900,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("pg_restore exceeded the 15-minute restore limit") from exc
        except OSError as exc:
            raise RuntimeError("pg_restore could not be started") from exc

        if result.returncode != 0:
            # Restore output can contain schema names and database values. Keep it
            # out of routine logs; the operator can inspect protected runner logs.
            raise RuntimeError(
                f"pg_restore failed with exit code {result.returncode}; "
                "restore output was suppressed to avoid logging database contents"
            )

        table_count = _count_restored_tables(target)
        return {
            "format": "postgres-custom",
            "sha256": verified["sha256"],
            "database": target.database,
            "table_count": table_count,
        }


def main(argv: list[str] | None = None) -> int:
    """Restore one PostgreSQL archive using an operator-supplied disposable DSN."""
    parser = argparse.ArgumentParser(
        description=(
            "Restore a PostgreSQL custom archive into an empty "
            "mlac_restore_drill_* database."
        )
    )
    parser.add_argument("backup", type=Path, help="downloaded PostgreSQL custom archive")
    parser.add_argument(
        "--sha256",
        required=True,
        help="trusted SHA-256 value from the backup object's metadata",
    )
    arguments = parser.parse_args(argv)
    database_url = os.getenv("MLAC_RESTORE_DATABASE_URL", "")

    try:
        result = restore_postgres_backup(
            arguments.backup,
            database_url,
            arguments.sha256,
        )
    except (OSError, RuntimeError, ValueError, TypeError, ImportError) as exc:
        detail = str(exc).replace(database_url, "[database URL redacted]") if database_url else str(exc)
        print(f"PostgreSQL restore drill failed: {detail}", file=sys.stderr)
        return 1

    print(
        "Isolated PostgreSQL restore passed: "
        f"database={result['database']} "
        f"sha256={result['sha256']} "
        f"tables={result['table_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
