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
