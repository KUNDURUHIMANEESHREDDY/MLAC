# Signed Dataset Manifests

MLAC can bind an evaluation to exact dataset bytes and split membership using
an Ed25519-signed manifest. The signer is an operator-controlled trust root;
training workers and model-generated code must not possess the private key.

## Create a manifest

Install the cryptography dependency (included in the `dev` and `attestation`
extras), then create a manifest from a dataset directory:

```python
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from mlac.dataset_manifest import create_signed_dataset_manifest

private_key = Path("/secure/dataset-signing-key.pem").read_bytes()
envelope = create_signed_dataset_manifest(
    "./dataset",
    dataset_id="reasoning-eval",
    revision="2026-11-08",
    splits={
        "train": ["train.jsonl"],
        "validation": ["validation.jsonl"],
        "test": ["test.jsonl"],
    },
    signer_id="dataset-release-prod",
    private_key_pem=private_key,
)
Path("dataset-manifest.json").write_text(
    __import__("json").dumps(envelope, indent=2), encoding="utf-8"
)
```

## Verify before evaluation

```python
import json
from pathlib import Path
from mlac.dataset_manifest import verify_signed_dataset_manifest

envelope = json.loads(Path("dataset-manifest.json").read_text(encoding="utf-8"))
trusted_public_key = Path("/etc/mlac/trust/dataset-release-prod.pem").read_bytes()
verified = verify_signed_dataset_manifest(
    envelope,
    "./dataset",
    trusted_signers={"dataset-release-prod": trusted_public_key},
)
print(verified["manifest_sha256"])
```

Verification checks the signer identity and Ed25519 signature, rejects
untrusted keys and duplicate split membership, prevents path traversal and
symlink escapes, and recomputes the SHA-256 digest and byte length of every
listed file. For in-memory Training Engine data, the operator must additionally
sign the exact materialized tensor fingerprint (`TrainingDataset.digest`):

```python
envelope = create_signed_dataset_manifest(
    dataset_root,
    dataset_id="training-corpus",
    revision=training_dataset.revision,
    splits={"train": ["train.jsonl"], "validation": ["validation.jsonl"], "test": ["test.jsonl"]},
    signer_id="dataset-release-prod",
    private_key_pem=private_key,
    dataset_fingerprint=training_dataset.digest,
)
```

The Training Engine rejects unsigned bindings and any mismatch between the
signed fingerprint and the tensors actually supplied to training. Evaluation
manifests must include the exact dataset/input filenames referenced by the
worker's trusted `manifest.json`.

## Trust and limitations

- Keep signing keys outside the dataset directory, worker image, candidate
  bundle, and model-controlled environment. Rotate keys through an operator
  trust-store update.
- A valid signature establishes that a trusted signer approved the manifest;
  it does not establish dataset licensing, correctness, representativeness,
  contamination status, or quality.
- `CandidateEvaluationPipeline` requires a signed manifest in the trusted
  evaluation directory and verifies it with its operator-configured
  `trusted_dataset_signers` before starting the worker. The verified dataset
  provenance is included in the pipeline result and experiment record.
- The Training Engine requires a `SignedDatasetBinding`, verifies its
  signature, file hashes, revision, split declarations, and signed materialized
  tensor fingerprint before starting a run. It records the manifest digest
  and signer ID in training evidence.
- The controller pipeline requires a signed manifest and checks that all
  evaluation dataset/input files referenced by the worker manifest are listed.
  The worker attestation additionally binds every file in both input mounts.
- File signatures do not prove dataset licensing, label correctness, absence
  of contamination, or correctness of the source-to-tensor preprocessing
  implementation. The operator remains responsible for those controls.
- The manifest records file-level hashes and split membership. It does not
  normalize JSONL, CSV, tokenizer, prompt, or preprocessing semantics.
