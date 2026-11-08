# Milestone: Remote Immutable Artifacts and Verifiable Runtime Identity

This milestone extends Phase 7's local content-addressed registry and worker
result signatures with a separately operated object-lock service and runtime
attestation authority.

## Remote artifact storage

S3ObjectLockArtifactStore stores objects at a SHA-256-derived key. It requires
bucket versioning and Object Lock, conditionally creates each object, applies
per-object COMPLIANCE retention, requires a version ID, checks retention
metadata, and re-hashes every downloaded object. It deliberately exposes no
delete operation.

Create a dedicated bucket with Object Lock enabled at bucket creation, enable
versioning, and deny routine principals permission to delete object versions
or alter retention. Use separate administrator credentials, TLS, server-side
encryption (prefer a customer-managed KMS key), access logging, and tested
cross-region replication. Credentials must come from short-lived workload
identity, not source code.

Install boto3 with: pip install -e ".[remote-storage]"

Example:

    from mlac.remote_artifacts import S3ObjectLockArtifactStore
    remote = S3ObjectLockArtifactStore(bucket="mlac-production-artifacts", retention_days=365)
    record = remote.put_bytes(checkpoint_bytes, media_type="application/x-pytorch")
    assert remote.verify(record)

The local ArtifactRegistry remains the local cache/index. Persist remote bucket,
key, version ID, digest and retention metadata with each evaluation. Never treat
an S3 ETag as a content hash.

## Independent runtime identity

The prior worker-result signature proves possession of a worker private key;
it does not prove which binary or host used that key. RuntimeIdentityVerifier
accepts a statement signed by a separately trusted runtime authority. The
statement binds worker ID, SPIFFE workload ID, exact OCI image digest,
attestation mechanism, measured-evidence SHA-256, issuance/expiry and nonce.
MLAC verifies the authority signature, expected worker/image/trust domain,
freshness and one-time nonce using a durable SQLite ledger.

The runtime authority must independently validate platform evidence before
signing. Suitable designs include SPIRE node attestation with workload/image
selectors, or a TPM/TEE-backed verifier (for example TPM 2.0/IMA or a TEE
quote). A SPIFFE ID alone is workload identity, not proof of image measurement.
A service that merely repeats worker-supplied claims is not independent
attestation. Do not reuse the worker-result signing key as the runtime authority.

This verifier checks the authority-issued assertion; it does not itself
validate TPM quotes, Nitro documents, IMA measurements, or SPIRE node
attestation. Those checks belong in the independently administered authority.

## Required deployment trust chain

1. Build and sign the worker image in a protected release pipeline.
2. Pin its OCI digest in controller configuration.
3. Run workers on dedicated hosts or a hardened orchestrator.
4. Measure the running workload using an independently administered runtime
   attestor and issue the signed identity statement.
5. Deliver that statement to the controller over an authenticated channel.
6. Verify runtime identity before accepting the worker's signed result.
7. Upload reports, attestations, inputs and checkpoints to object-lock storage.
8. Independently audit object version, retention, bytes and both signatures.

## Limits and rollout

Object Lock retention is intentionally difficult to shorten; test it in a
non-production bucket. It protects object versions, not availability, KMS keys,
account access or all metadata. Provider compatibility must be tested.

These modules are reusable primitives, not yet a complete production rollout.
The remote artifact adapter and runtime verifier still need wiring into the
evaluation pipeline and a deployed independent attestation authority. Until
that integration is complete, do not claim end-to-end remote attestation.
