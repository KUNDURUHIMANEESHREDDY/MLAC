import base64
import json
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from mlac.dataset_manifest import (
    canonical_json, create_signed_dataset_manifest, verify_signed_dataset_manifest,
)

def pem_pair():
    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    public = key.public_key().public_bytes(serialization.Encoding.PEM,
                                           serialization.PublicFormat.SubjectPublicKeyInfo)
    return private, public

def test_sign_and_verify_dataset_files(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"x":1}\n', encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text('{"x":2}\n', encoding="utf-8")
    private, public = pem_pair()
    envelope = create_signed_dataset_manifest(
        tmp_path, dataset_id="tiny-corpus", revision="v1",
        splits={"train": ["train.jsonl"], "validation": ["eval.jsonl"]},
        signer_id="dataset-release", private_key_pem=private,
    )
    result = verify_signed_dataset_manifest(
        envelope, tmp_path, trusted_signers={"dataset-release": public}
    )
    assert result["dataset_id"] == "tiny-corpus"
    assert len(result["verified_files"]) == 2

def test_rejects_modified_dataset_bytes(tmp_path):
    (tmp_path / "train.jsonl").write_text("original", encoding="utf-8")
    private, public = pem_pair()
    envelope = create_signed_dataset_manifest(
        tmp_path, dataset_id="d", revision="1", splits={"train": ["train.jsonl"]},
        signer_id="signer", private_key_pem=private,
    )
    (tmp_path / "train.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(PermissionError, match="integrity"):
        verify_signed_dataset_manifest(envelope, tmp_path, trusted_signers={"signer": public})

def test_rejects_untrusted_signer(tmp_path):
    (tmp_path / "data.jsonl").write_text("x", encoding="utf-8")
    private, public = pem_pair()
    envelope = create_signed_dataset_manifest(
        tmp_path, dataset_id="d", revision="1", splits={"train": ["data.jsonl"]},
        signer_id="signer", private_key_pem=private,
    )
    with pytest.raises(PermissionError, match="not trusted"):
        verify_signed_dataset_manifest(envelope, tmp_path, trusted_signers={})

def test_rejects_path_traversal(tmp_path):
    private, _ = pem_pair()
    with pytest.raises(ValueError, match="relative"):
        create_signed_dataset_manifest(
            tmp_path, dataset_id="d", revision="1", splits={"train": ["../secret"]},
            signer_id="signer", private_key_pem=private,
        )

def test_rejects_duplicate_split_membership(tmp_path):
    (tmp_path / "data.jsonl").write_text("x", encoding="utf-8")
    private, _ = pem_pair()
    with pytest.raises(ValueError, match="only one split"):
        create_signed_dataset_manifest(
            tmp_path, dataset_id="d", revision="1",
            splits={"train": ["data.jsonl"], "validation": ["data.jsonl"]},
            signer_id="signer", private_key_pem=private,
        )


def test_signed_materialized_dataset_fingerprint_round_trip(tmp_path):
    (tmp_path / "train.jsonl").write_text("source rows", encoding="utf-8")
    private, public = pem_pair()
    fingerprint = "a" * 64
    envelope = create_signed_dataset_manifest(
        tmp_path, dataset_id="materialized", revision="r1",
        splits={"train": ["train.jsonl"]}, signer_id="signer",
        private_key_pem=private, dataset_fingerprint=fingerprint,
    )
    result = verify_signed_dataset_manifest(envelope, tmp_path, trusted_signers={"signer": public})
    assert result["dataset_fingerprint"] == fingerprint
