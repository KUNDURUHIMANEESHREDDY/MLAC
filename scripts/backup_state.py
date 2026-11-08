"""Create and upload an integrity-checked SQLite backup to S3 Object Lock."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value


def main() -> None:
    """Back up the live state database and retain the object in COMPLIANCE mode."""
    import boto3

    database = Path(_required_env("MLAC_API_STATE_DB"))
    bucket = _required_env("MLAC_BACKUP_S3_BUCKET")
    retention_days = int(os.getenv("MLAC_BACKUP_RETENTION_DAYS", "90"))
    if retention_days < 1:
        raise ValueError("MLAC_BACKUP_RETENTION_DAYS must be at least 1")
    if not database.is_file() or database.is_symlink():
        raise RuntimeError("configured state database is not a regular file")

    client = boto3.client("s3")
    versioning = client.get_bucket_versioning(Bucket=bucket)
    if versioning.get("Status") != "Enabled":
        raise RuntimeError("backup bucket versioning must be enabled")
    lock = client.get_object_lock_configuration(Bucket=bucket).get("ObjectLockConfiguration", {})
    if lock.get("ObjectLockEnabled") != "Enabled":
        raise RuntimeError("backup bucket Object Lock must be enabled")

    with tempfile.TemporaryDirectory(prefix="mlac-backup-") as directory:
        backup_path = Path(directory) / "api-state.sqlite3"
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as source:
            with sqlite3.connect(backup_path) as destination:
                source.backup(destination)
                result = destination.execute("PRAGMA integrity_check").fetchone()
                if result is None or result[0] != "ok":
                    raise RuntimeError("SQLite backup integrity check failed")
        os.chmod(backup_path, 0o600)
        digest = hashlib.sha256(backup_path.read_bytes()).hexdigest()
        created = datetime.now(timezone.utc)
        retain_until = created + timedelta(days=retention_days)
        key = f"mlac/api-state/{created:%Y/%m/%d}/api-state-{created:%Y%m%dT%H%M%SZ}-{digest[:16]}.sqlite3"
        extra = {
            "ContentType": "application/vnd.sqlite3",
            "Metadata": {"sha256": digest},
            "ObjectLockMode": "COMPLIANCE",
            "ObjectLockRetainUntilDate": retain_until,
            "ServerSideEncryption": "aws:kms" if os.getenv("MLAC_BACKUP_KMS_KEY_ID") else "AES256",
        }
        if os.getenv("MLAC_BACKUP_KMS_KEY_ID"):
            extra["SSEKMSKeyId"] = os.environ["MLAC_BACKUP_KMS_KEY_ID"]
        client.upload_file(str(backup_path), bucket, key, ExtraArgs=extra)
        print(f"Backup uploaded: s3://{bucket}/{key} sha256={digest} retain_until={retain_until.isoformat()}")


if __name__ == "__main__":
    main()
