# MLAC 0.9.0 release verification

This checklist separates implementation from verified and deployed status.

## Release scope in this branch

- Mandatory Ed25519 dataset manifest verification before candidate worker execution.
- Worker-manifest dataset/input path coverage against signed file records.
- Signed tensor fingerprint binding for Training Engine runs.
- Dataset provenance carried into evaluation results and the experiment ledger.
- Recursive controller-model evaluation workflow with append-only SQLite history.
- Database triggers preventing update/delete of controller evolution records.
- Existing worker signature, digest-pinned sandbox, quality gates, candidate governance, and immutable recovery boundaries remain in force.

## Required automated verification

- Install `.[dev,torch,api]`.
- Run Ruff on API, CLI, training, pipeline, and dataset manifest modules.
- Run the complete pytest suite.
- Build the package and verify `mlac version` reports `0.9.0`.

## Deployment acceptance still required

- Run the fixed worker image against an operator-owned dataset with a production dataset-signing key and verify its signature independently.
- Run a complete end-to-end candidate evaluation using the actual target model and held-out dataset.
- Verify worker image digest, runtime identity authority, remote object-lock storage, and backup/restore in the target environment.
- The FastAPI service remains loopback-only because its registry and rate limiting are process-local. Durable tenant-scoped storage, distributed rate limiting, TLS gateway, and deployed security acceptance are release blockers for public network exposure.
- A hosted API model's provider-managed weights cannot be self-modified. Recursive model evolution requires an operator-designated local/open-weight controller model and a trusted evaluation bundle.

No release is considered production-deployed until these checks pass in the target environment.
