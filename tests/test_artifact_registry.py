from mlac.artifact_registry import ArtifactRegistry
import pytest

def test_content_addressed_artifacts_are_deduplicated_and_verified(tmp_path):
    registry=ArtifactRegistry(tmp_path/"store")
    first=registry.put_bytes(b"same bytes")
    second=registry.put_bytes(b"same bytes")
    assert first["sha256"]==second["sha256"]
    assert registry.get(first["sha256"])["data"]==b"same bytes"

def test_registry_links_artifacts_to_evaluation(tmp_path):
    registry=ArtifactRegistry(tmp_path/"store")
    registry.link_evaluation(evaluation_id="e1",model_id="m1",candidate_id="c1",
        artifacts={"report.json":b'{"ok":true}',"checkpoint":b"weights"})
    links=registry.list_evaluation("e1")
    assert {item["role"] for item in links}=={"report.json","checkpoint"}
    assert len({item["sha256"] for item in links})==2

def test_registry_rejects_empty_evaluation_link(tmp_path):
    with pytest.raises(ValueError):
        ArtifactRegistry(tmp_path/"store").link_evaluation(
            evaluation_id="",model_id="m",candidate_id="c",artifacts={"x":b"x"})

def test_registry_rejects_replayed_attestation_nonce(tmp_path):
    registry=ArtifactRegistry(tmp_path/"store")
    fields={"nonce":"one-time-nonce","key_id":"key","report_sha256":"digest","evaluation_id":"e1"}
    registry.record_attestation(**fields)
    with pytest.raises(ValueError,match="already been used"):
        registry.record_attestation(**{**fields,"evaluation_id":"e2"})
