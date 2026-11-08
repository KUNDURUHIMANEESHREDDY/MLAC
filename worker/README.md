# MLAC trusted worker image

Build from the repository root:

    docker build -f worker/Dockerfile -t mlac-worker:local .

Resolve the built image to an immutable digest and configure that digest in
CandidateEvaluationPipeline and DockerSandboxRunner. Do not configure a mutable
tag in production. The image contains only the minimum MLAC modules needed to
instantiate TransformerBlock; it excludes controller governance, policy, API,
and recovery modules.

## Separate candidate and trusted evaluation inputs

The controller mounts two different host directories:

- candidate directory -> /workspace (read-only), containing candidate.pt only
- operator-managed evaluation directory -> /trusted (read-only), containing manifest.json, baseline.pt, input.pt, and target.pt

Never place the trusted evaluation directory under candidate-controlled storage.
The pipeline requires both directories separately and DockerSandboxRunner rejects
using the same directory for both mounts. The candidate cannot replace the
baseline, input, target, architecture, or worker command.

Example trusted manifest:

    {
      "schema_version": 1,
      "architecture": {
        "d_model": 64,
        "num_heads": 4,
        "mlp_ratio": 4,
        "dropout": 0.0
      },
      "baseline_state_dict": "baseline.pt",
      "input_tensor": "input.pt",
      "target_tensor": "target.pt",
      "iterations": 20,
      "warmup": 5
    }

The candidate directory contains only candidate.pt. All artifact filenames
must be simple names in the relevant directory; symlinks and path traversal
are rejected. Each artifact is limited to 512 MiB. Tensor/checkpoint loading
uses torch.load(weights_only=True), and state dictionaries must contain only
named tensors. The worker constructs the supported MLAC TransformerBlock from
the operator-controlled manifest and loads both state dictionaries strictly.

## Quality metric

This initial worker supports a fixed regression task. It computes negative
mean-squared error against the operator-controlled target.pt for both baseline
and candidate; higher (less negative) is better. This is not a universal
LLM/task-quality measure. For language modeling or other tasks, add a
separately reviewed worker harness and task-specific trusted dataset/metric.

Generated Python source is not imported or executed by this worker. Candidate
code execution needs a separately reviewed harness and remains confined to
the container boundary. The worker image itself is trusted code and must be
built, reviewed, scanned, and pinned by digest in operator-controlled config.

## Worker identity and signed evidence (Phase 7)

The controller mounts an operator-provisioned identity directory read-only at
`/identity`. It must contain `worker_ed25519_private.pem`; the private key is
not baked into the image, passed through environment variables, or placed in
candidate/trusted evaluation directories. The worker runs as UID/GID 65532, so
provision the file so that this UID can read it while other host users cannot
(for example, owner UID 65532 and mode 0400 on a dedicated worker host).

Generate a keypair in a protected operator environment:

    from mlac.attestation import generate_worker_keypair
    key_id = generate_worker_keypair(
        "./worker-identity/worker_ed25519_private.pem",
        "./trusted-workers/worker_ed25519_public.pem",
    )

Register the public key bytes under that exact key_id in the controller's
`trusted_worker_keys` mapping. Keep the private key out of Git, image layers,
logs, CI artifacts, and the candidate/trusted input directories. Rotate keys
by adding the new public key to controller trust configuration, deploying the
new private key to the worker identity mount, then retiring the old key.

The worker signs a canonical JSON report digest plus the SHA-256 digest of
every direct regular file in both input mounts. The attestation includes
worker_id, key_id, the configured immutable image digest, timestamp, and a
random nonce. The controller independently hashes inputs before and after
execution, verifies the signature and trusted key, checks the expected worker
and image, then rejects previously used nonces in the durable artifact registry.

This is a cryptographic identity assertion by the worker key, not remote
attestation of the host, kernel, Docker daemon, or actual image measurement.
The image digest is supplied by the controller's fixed command and signed by
the worker; it is not an independent measurement. A compromised worker image
with access to the private key can produce signed false evidence. Build,
review, scan, and pin the worker image in a trusted release process.

The local registry stores blobs under SHA-256-derived paths and metadata and
evaluation links in SQLite. It detects content mismatch on read. It is not a
WORM service, remote object store, encrypted backup system, or access-control
boundary. Production deployments should use an independently administered
object-lock/WORM backend, encryption, backup/restore testing, and restricted
registry write permissions. The registry stores candidate and trusted input
files as well as the report and attestation; protect it accordingly.


## Host-side output collection

The writable /output mount is controlled by the worker process. The controller
therefore does not use a check-then-read pathname walk. It opens the output root
once, traverses relative to pinned directory descriptors, opens children with
O_NOFOLLOW, verifies the opened inode/type, and enforces the aggregate byte
limit while reading. Symlinks, special files, and entries changed during an
open race are skipped; a growing file cannot exceed the configured byte limit.

The runner checks for POSIX descriptor-relative openat and O_NOFOLLOW support
before starting Docker. On hosts without those primitives (including native
Windows Python), candidate execution fails closed rather than using a
race-prone pathname fallback. Use a Linux host or VM until native Windows
reparse-point-safe collection is implemented.
