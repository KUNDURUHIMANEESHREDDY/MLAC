import hashlib
from datetime import datetime, timezone
from io import BytesIO
import pytest
from mlac.remote_artifacts import S3ObjectLockArtifactStore

class FakeS3:
    def __init__(self): self.objects = {}
    def get_bucket_versioning(self, **kwargs): return {"Status":"Enabled"}
    def get_object_lock_configuration(self, **kwargs): return {"ObjectLockConfiguration":{"ObjectLockEnabled":"Enabled"}}
    def put_object(self, **kwargs):
        key=kwargs["Key"]
        if key in self.objects:
            error=RuntimeError("precondition failed"); error.response={"ResponseMetadata":{"HTTPStatusCode":412}}; raise error
        self.objects[key]={"Body":kwargs["Body"],"Metadata":kwargs["Metadata"],"Mode":kwargs["ObjectLockMode"],"Until":kwargs["ObjectLockRetainUntilDate"],"VersionId":"version-1"}
        return {"VersionId":"version-1"}
    def head_object(self, **kwargs):
        obj=self.objects[kwargs["Key"]]
        return {"VersionId":obj["VersionId"],"Metadata":obj["Metadata"],"ObjectLockMode":obj["Mode"],"ObjectLockRetainUntilDate":obj["Until"]}
    def get_object(self, **kwargs): return {"Body":BytesIO(self.objects[kwargs["Key"]]["Body"])}

def test_put_get_and_verify():
    client=FakeS3(); now=datetime(2026,1,1,tzinfo=timezone.utc)
    store=S3ObjectLockArtifactStore(bucket="worm",retention_days=30,client=client,clock=lambda:now)
    item=store.put_bytes(b"immutable")
    assert item.sha256==hashlib.sha256(b"immutable").hexdigest()
    assert item.object_lock_mode=="COMPLIANCE"
    assert store.get_bytes(item.sha256,version_id=item.version_id)==b"immutable"
    assert store.verify(item)

def test_existing_content_is_deduplicated():
    store=S3ObjectLockArtifactStore(bucket="worm",retention_days=30,client=FakeS3())
    assert store.put_bytes(b"same").sha256==store.put_bytes(b"same").sha256

def test_versioning_is_required():
    class NoVersioning(FakeS3):
        def get_bucket_versioning(self,**kwargs): return {}
    with pytest.raises(RuntimeError,match="versioning"):
        S3ObjectLockArtifactStore(bucket="bad",retention_days=7,client=NoVersioning())
