import hashlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from io import BytesIO

import pytest

from mlac.remote_artifacts import S3ObjectLockArtifactStore


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.get_calls = []

    def get_bucket_versioning(self, **kwargs):
        return {"Status": "Enabled"}

    def get_object_lock_configuration(self, **kwargs):
        return {"ObjectLockConfiguration": {"ObjectLockEnabled": "Enabled"}}

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        if key in self.objects:
            error = RuntimeError("precondition failed")
            error.response = {"ResponseMetadata": {"HTTPStatusCode": 412}}
            raise error
        version_id = f"version-{len(self.objects) + 1}"
        self.objects[key] = {
            "Body": kwargs["Body"],
            "Metadata": kwargs["Metadata"],
            "Mode": kwargs["ObjectLockMode"],
            "Until": kwargs["ObjectLockRetainUntilDate"],
            "VersionId": version_id,
        }
        return {"VersionId": version_id}

    def head_object(self, **kwargs):
        obj = self.objects[kwargs["Key"]]
        if kwargs.get("VersionId") not in (None, obj["VersionId"]):
            raise KeyError("unknown version")
        return {
            "VersionId": obj["VersionId"],
            "Metadata": obj["Metadata"],
            "ObjectLockMode": obj["Mode"],
            "ObjectLockRetainUntilDate": obj["Until"],
        }

    def get_object(self, **kwargs):
        self.get_calls.append(dict(kwargs))
        obj = self.objects[kwargs["Key"]]
        if kwargs.get("VersionId") not in (None, obj["VersionId"]):
            raise KeyError("unknown version")
        return {"Body": BytesIO(obj["Body"])}


def test_put_get_and_verify():
    client = FakeS3()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = S3ObjectLockArtifactStore(
        bucket="worm", retention_days=30, client=client, clock=lambda: now)
    item = store.put_bytes(b"immutable")
    assert item.sha256 == hashlib.sha256(b"immutable").hexdigest()
    assert item.object_lock_mode == "COMPLIANCE"
    assert store.get_bytes(item.sha256, version_id=item.version_id) == b"immutable"
    assert store.verify(item)


def test_existing_content_is_deduplicated_and_reads_pinned_version():
    client = FakeS3()
    store = S3ObjectLockArtifactStore(bucket="worm", retention_days=30, client=client)
    first = store.put_bytes(b"same")
    second = store.put_bytes(b"same")
    assert first.sha256 == second.sha256
    assert second.version_id == first.version_id
    assert client.get_calls[-1]["VersionId"] == second.version_id


def test_verify_rejects_changed_retention_metadata():
    store = S3ObjectLockArtifactStore(bucket="worm", retention_days=30, client=FakeS3())
    item = store.put_bytes(b"immutable")
    changed = replace(
        item,
        retention_until=(datetime.fromisoformat(item.retention_until) + timedelta(days=1)).isoformat(),
    )
    assert not store.verify(changed)


def test_versioning_is_required():
    class NoVersioning(FakeS3):
        def get_bucket_versioning(self, **kwargs):
            return {}

    with pytest.raises(RuntimeError, match="versioning"):
        S3ObjectLockArtifactStore(bucket="bad", retention_days=7, client=NoVersioning())
