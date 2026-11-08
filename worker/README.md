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

## Host-side output collection

The writable /output mount is controlled by the worker process. The controller
therefore does not use a check-then-read pathname walk. It opens the output root
once, traverses relative to pinned directory descriptors, opens children with
O_NOFOLLOW, verifies the opened inode/type, and enforces the aggregate byte
limit while reading. Symlinks, special files, and entries changed during an
open race are skipped; a growing file cannot exceed the configured byte limit.

This host-side collector requires POSIX descriptor-relative openat and
O_NOFOLLOW support. On hosts without those primitives (including native
Windows Python), candidate execution fails closed rather than falling back to
a race-prone pathname read. Use a Linux host/VM for the Docker candidate runner
until native Windows reparse-point-safe collection is implemented.
