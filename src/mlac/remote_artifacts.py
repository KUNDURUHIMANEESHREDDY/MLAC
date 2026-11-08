"""S3-compatible content-addressed storage with per-object WORM retention."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

@dataclass(frozen=True)
class RemoteArtifact:
    sha256: str
    size_bytes: int
    bucket: str
    key: str
    version_id: str
    retention_until: str
    object_lock_mode: str

class S3ObjectLockArtifactStore:
    """S3 Object Lock COMPLIANCE backend. No delete operation is exposed."""
    def __init__(self, *, bucket: str, retention_days: int, client: Any | None = None,
                 prefix: str = "mlac/sha256", clock: Any | None = None):
        if not bucket or not bucket.strip():
            raise ValueError("bucket must be non-empty")
        if isinstance(retention_days, bool) or retention_days < 1:
            raise ValueError("retention_days must be a positive integer")
        if not prefix or prefix.startswith("/") or ".." in prefix.split("/"):
            raise ValueError("prefix must be a safe relative object prefix")
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("install the remote-storage extra to use S3") from exc
            client = boto3.client("s3")
        self.bucket, self.retention_days, self.client = bucket, retention_days, client
        self.prefix, self.clock = prefix.rstrip("/"), clock or (lambda: datetime.now(timezone.utc))
        if self.client.get_bucket_versioning(Bucket=bucket).get("Status") != "Enabled":
            raise RuntimeError("artifact bucket versioning must be enabled")
        lock = self.client.get_object_lock_configuration(Bucket=bucket)
        if lock.get("ObjectLockConfiguration", {}).get("ObjectLockEnabled") != "Enabled":
            raise RuntimeError("artifact bucket must have Object Lock enabled")

    @staticmethod
    def _validate_digest(digest: str):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("sha256 must be a 64-character lowercase hex digest")

    def _key(self, digest: str) -> str:
        self._validate_digest(digest)
        return f"{self.prefix}/{digest[:2]}/{digest}"

    def put_bytes(self, data: bytes, *, media_type: str = "application/octet-stream") -> RemoteArtifact:
        if not isinstance(data, bytes):
            raise TypeError("artifact data must be bytes")
        digest, key = hashlib.sha256(data).hexdigest(), None
        key = self._key(digest)
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        retain_until = now.astimezone(timezone.utc) + timedelta(days=self.retention_days)
        try:
            response = self.client.put_object(
                Bucket=self.bucket, Key=key, Body=data, ContentType=media_type,
                Metadata={"sha256": digest}, IfNoneMatch="*", ObjectLockMode="COMPLIANCE",
                ObjectLockRetainUntilDate=retain_until)
            version_id = response.get("VersionId")
        except Exception as exc:
            status = getattr(exc, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status != 412:
                raise
            if self.get_bytes(digest) != data:
                raise RuntimeError("content-address collision or corrupted remote object") from exc
            version_id = self.client.head_object(Bucket=self.bucket, Key=key).get("VersionId")
        if not version_id:
            raise RuntimeError("object store did not return a version ID")
        head = self.client.head_object(Bucket=self.bucket, Key=key, VersionId=version_id)
        if head.get("Metadata", {}).get("sha256") != digest:
            raise RuntimeError("remote object metadata digest mismatch")
        if head.get("ObjectLockMode") != "COMPLIANCE":
            raise RuntimeError("remote object is not protected by COMPLIANCE retention")
        actual = head.get("ObjectLockRetainUntilDate")
        if actual is None or actual < retain_until:
            raise RuntimeError("remote object retention is shorter than requested")
        return RemoteArtifact(digest, len(data), self.bucket, key, version_id,
                              actual.astimezone(timezone.utc).isoformat(), "COMPLIANCE")

    def get_bytes(self, digest: str, *, version_id: str | None = None) -> bytes:
        args = {"Bucket": self.bucket, "Key": self._key(digest)}
        if version_id is not None:
            args["VersionId"] = version_id
        data = self.client.get_object(**args)["Body"].read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("downloaded artifact failed SHA-256 verification")
        return data

    def verify(self, artifact: RemoteArtifact) -> bool:
        if artifact.bucket != self.bucket or artifact.key != self._key(artifact.sha256):
            return False
        data = self.get_bytes(artifact.sha256, version_id=artifact.version_id)
        head = self.client.head_object(Bucket=self.bucket, Key=artifact.key, VersionId=artifact.version_id)
        return (len(data) == artifact.size_bytes and head.get("ObjectLockMode") == "COMPLIANCE"
                and head.get("Metadata", {}).get("sha256") == artifact.sha256)
