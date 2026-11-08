# MLAC 0.9.0 release verification

This checklist separates implementation from verified and deployed status.

## Release scope

- Mandatory Ed25519 dataset manifest verification before candidate worker execution.
- Worker-manifest dataset/input path coverage against signed file records.
- Signed tensor fingerprint binding for Training Engine runs.
- Dataset provenance carried into evaluation results and experiment ledger.
- Recursive controller-model evaluation workflow with append-only SQLite history.
- Database triggers preventing update/delete of controller evolution records.
- Existing worker signature, digest-pinned sandbox, quality gates, candidate governance, and immutable recovery boundaries remain in force.

## Automated verification

- Install `.[dev,torch,api]`.
- Run Ruff on API, CLI, training, pipeline, dataset manifest, and self-evolution modules.
- Require a successful GitHub Actions run on the exact reviewed PR head; local test results do not satisfy release acceptance.
- Build the wheel and verify `mlac version` reports `0.9.0`.

## Deployment acceptance

- Run the fixed worker image against an operator-owned dataset with a production dataset-signing key and independently verify the signature.
- Run end-to-end candidate evaluation using the actual target model and held-out dataset.
- Verify worker image digest, runtime identity authority, remote object-lock storage, and backup/restore in the target environment.
- The `mlac serve` CLI rejects non-loopback binds. Any other launcher exposing `mlac.api:app` must enforce network isolation until durable tenant-scoped storage, distributed rate limiting, a TLS gateway, and deployed security acceptance are complete.
- Hosted API model weights cannot be self-modified. Recursive model evolution requires an operator-designated local/open-weight controller model and trusted evaluation bundle.

No release is considered production-deployed until these checks pass in the target environment.
