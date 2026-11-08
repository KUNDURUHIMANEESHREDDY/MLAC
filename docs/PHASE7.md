# Phase 7: Signed Worker Attestations and Durable Artifact Registry

Phase 7 adds cryptographic provenance to the Phase 6 worker report and preserves the exact evidence inputs and outputs in a content-addressed registry.

## Signed result envelope

The fixed worker writes a schema-version-1 envelope containing:
- `report`: the Phase 6 evaluation payload, serialized as canonical JSON for signing.
- `attestation`: Ed25519 signature metadata with worker_id, key_id, configured worker image digest, issued_at, nonce, report SHA-256, input artifact SHA-256 map, and signature.

The worker signs the canonical attestation fields with an operator-provisioned private key mounted read-only at `/identity/worker_ed25519_private.pem`. The key is not included in the worker image, candidate mount, or trusted evaluation mount.

## Controller verification order

1. Hash every direct regular file in candidate and trusted input directories before worker execution.
2. Run the fixed worker command with separate candidate, trusted, and identity mounts.
3. Reject timeout, non-zero exit, missing report, oversized report, or malformed envelope.
4. Re-hash both input directories and reject changes during execution.
5. Verify the Ed25519 signature against controller-configured public keys; require the expected worker ID and exact configured immutable image digest; compare report and input digests.
6. Persist the attestation nonce in SQLite with a unique constraint; reject replay.
7. Apply Phase 6 report validation and evaluation policy only after cryptographic verification.
8. Store the full report, canonical report payload, attestation, candidate checkpoint, and trusted input files in the artifact registry; associate each blob with evaluation/model/candidate identifiers.
9. Persist the scorecard and governance result in the existing experiment ledger.

## Key provisioning

Use `mlac.attestation.generate_worker_keypair(private_path, public_path)` in a protected operator environment. Configure the returned key_id and public PEM bytes in `trusted_worker_keys`. Provision the private PEM in a dedicated host directory mounted read-only by Docker. The worker runs as UID/GID 65532; provision host ownership/permissions so only that identity can read the private key. Never commit the private key, bake it into the image, pass it in environment variables, or expose it to candidate-controlled paths.

Rotate keys by trusting the new public key before deploying the new private key, then retire the old key after the worker transition. A key_id is the SHA-256 fingerprint of the DER SubjectPublicKeyInfo public key.

## Durable artifact registry

`ArtifactRegistry` stores bytes by SHA-256 under `objects/sha256/<prefix>/<suffix>` and maintains SQLite metadata and evaluation links. Repeated content reuses the same blob. Reads recompute and verify the digest. Attestation nonces are recorded in a unique table to prevent replay across controller restarts when the same registry database is retained.

The registry captures the candidate checkpoint and trusted evaluation files in addition to the signed report and attestation. These files may contain proprietary model and evaluation data; protect the registry directory and its backups.

## Trust boundaries and limitations

- A valid signature proves that the configured private key signed the attestation. It does not prove that the evaluation result is truthful, the key was uncompromised, or the worker host is secure.
- The worker image digest is supplied to the worker by the controller's fixed command and included in the signature. This is not independent runtime measurement or remote attestation.
- A compromised worker image with access to the private key can sign false results. Review, scan, release, and pin the worker image independently.
- Docker daemon, host kernel, and worker host remain trusted computing base components.
- The local SQLite/filesystem registry is durable across process restarts but is not WORM, remotely replicated, encrypted, or independently administered. Production should use object-lock/WORM storage, encryption, access controls, backup/restore validation, and a separately operated trust root.
- Candidate source code is still not executed by this fixed worker. Only supported tensor checkpoints are evaluated.
