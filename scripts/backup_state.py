"""Create integrity-checked PostgreSQL or SQLite backups in S3 Object Lock."""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


def _required_env(name: str) -> str:
    """Read a required environment variable without exposing its value."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value


def _postgres_environment(database_url: str) -> dict[str, str]:
    """Translate a PostgreSQL URL into libpq environment variables."""
    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ValueError("MLAC_API_DATABASE_URL must be a valid PostgreSQL URL")
    database = unquote(parsed.path.lstrip("/"))
    if not database or not parsed.username:
        raise ValueError("PostgreSQL URL must include a database and username")
    query = parse_qs(parsed.query)
    environment = os.environ.copy()
    environment.update(
        {
            "PGHOST": parsed.hostname,
            "PGPORT": str(parsed.port or 5432),
            "PGUSER": unquote(parsed.username),
            "PGPASSWORD": unquote(parsed.password or ""),
            "PGDATABASE": database,
            "PGSSLMODE": query.get("sslmode", ["require"])[0],
            "PGCONNECT_TIMEOUT": "10",
        }
    )
    return environment


def _create_database_backup(destination: Path) -> str:
    """Create a transactionally consistent backup and return its format."""
    database_url = os.getenv("MLAC_API_DATABASE_URL")
    sqlite_path = os.getenv("MLAC_API_STATE_DB")
    if bool(database_url) == bool(sqlite_path):
        raise RuntimeError(
            "Configure exactly one of MLAC_API_DATABASE_URL or MLAC_API_STATE_DB"
        )

    if database_url:
        pg_dump = shutil.which("pg_dump")
        pg_restore = shutil.which("pg_restore")
        if not pg_dump or not pg_restore:
            raise RuntimeError("PostgreSQL client tools pg_dump and pg_restore are required")
        environment = _postgres_environment(database_url)
        subprocess.run(
            [
                pg_dump,
                "--format=custom",
                "--no-owner",
                "--no-acl",
                "--file",
                str(destination),
            ],
            env=environment,
            check=True,
            timeout=900,
        )
        subprocess.run(
            [pg_restore, "--list", str(destination)],
            env=environment,
            check=True,
            stdout=subprocess.DEVNULL,
            timeout=60,
        )
        return "postgres-custom"

    database = Path(sqlite_path or "")
    if not database.is_file() or database.is_symlink():
        raise RuntimeError("configured SQLite state database is not a regular file")
    with (
        sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as source,
        sqlite3.connect(destination) as target,
    ):
        source.backup(target)
        result = target.execute("PRAGMA integrity_check").fetchone()
        if result is None or result[0] != "ok":
            raise RuntimeError("SQLite backup integrity check failed")
    return "sqlite"


def main() -> None:
    """Back up the configured database and retain it in COMPLIANCE mode."""
    import boto3

    bucket = _required_env("MLAC_BACKUP_S3_BUCKET")
    retention_days = int(os.getenv("MLAC_BACKUP_RETENTION_DAYS", "90"))
    if retention_days < 1:
        raise ValueError("MLAC_BACKUP_RETENTION_DAYS must be at least 1")

    client = boto3.client("s3")
    versioning = client.get_bucket_versioning(Bucket=bucket)
    if versioning.get("Status") != "Enabled":
        raise RuntimeError("backup bucket versioning must be enabled")
    lock = client.get_object_lock_configuration(Bucket=bucket).get(
        "ObjectLockConfiguration", {}
    )
    if lock.get("ObjectLockEnabled") != "Enabled":
        raise RuntimeError("backup bucket Object Lock must be enabled")

    with tempfile.TemporaryDirectory(prefix="mlac-backup-") as directory:
        suffix = ".dump" if os.getenv("MLAC_API_DATABASE_URL") else ".sqlite3"
        backup_path = Path(directory) / f"api-state{suffix}"
        backup_format = _create_database_backup(backup_path)
        os.chmod(backup_path, 0o600)

        digest_hash = hashlib.sha256()
        with backup_path.open("rb") as backup_file:
            for chunk in iter(lambda: backup_file.read(1024 * 1024), b""):
                digest_hash.update(chunk)
        digest = digest_hash.hexdigest()

        created = datetime.now(timezone.utc)
        retain_until = (created + timedelta(days=retention_days)).replace(microsecond=0)
        key = (
            f"mlac/api-state/{created:%Y/%m/%d}/"
            f"api-state-{created:%Y%m%dT%H%M%SZ}-{digest[:16]}{suffix}"
        )
        extra = {
            "ContentType": "application/octet-stream",
            "Metadata": {"sha256": digest, "format": backup_format},
            "ObjectLockMode": "COMPLIANCE",
            "ObjectLockRetainUntilDate": retain_until,
            "ServerSideEncryption": (
                "aws:kms" if os.getenv("MLAC_BACKUP_KMS_KEY_ID") else "AES256"
            ),
        }
        if os.getenv("MLAC_BACKUP_KMS_KEY_ID"):
            extra["SSEKMSKeyId"] = os.environ["MLAC_BACKUP_KMS_KEY_ID"]

        client.upload_file(str(backup_path), bucket, key, ExtraArgs=extra)
        uploaded = client.head_object(Bucket=bucket, Key=key)
        if uploaded.get("ObjectLockMode") != "COMPLIANCE":
            raise RuntimeError("uploaded backup is not protected by COMPLIANCE retention")
        if uploaded.get("Metadata", {}).get("sha256") != digest:
            raise RuntimeError("uploaded backup SHA-256 metadata does not match")
        actual_retention = uploaded.get("ObjectLockRetainUntilDate")
        if actual_retention is None or actual_retention < retain_until:
            raise RuntimeError("uploaded backup retention is shorter than requested")
        print(
            f"Backup uploaded: s3://{bucket}/{key} "
            f"sha256={digest} format={backup_format} "
            f"retain_until={retain_until.isoformat()}"
        )


if __name__ == "__main__":
    main()
