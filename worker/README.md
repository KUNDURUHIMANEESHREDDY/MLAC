# MLAC trusted worker image

Build from the repository root:

    docker build -f worker/Dockerfile -t mlac-worker:local .

Resolve the built image to an immutable digest and configure that digest in
CandidateEvaluationPipeline and DockerSandboxRunner. Do not configure a mutable
tag in production. The Dockerfile copies only the minimum MLAC modules needed
to instantiate the supported TransformerBlock; it does not include the
controller's governance, policy, API, or recovery modules.

## Bundle schema

A candidate bundle is a read-only directory containing these files:

- manifest.json
- baseline.pt — tensor-only state_dict
- candidate.pt — tensor-only state_dict
- input.pt — floating-point input tensor [batch, sequence, d_model]
- target.pt — floating-point regression target with the same shape

Example manifest:

    {
      "schema_version": 1,
      "architecture": {
        "d_model": 64,
        "num_heads": 4,
        "mlp_ratio": 4,
        "dropout": 0.0
      },
      "baseline_state_dict": "baseline.pt",
      "candidate_state_dict": "candidate.pt",
      "input_tensor": "input.pt",
      "target_tensor": "target.pt",
      "iterations": 20,
      "warmup": 5
    }

All filenames must be simple names in the bundle root; symlinks and path
traversal are rejected. Each artifact is limited to 512 MiB. Tensor/checkpoint
loading uses torch.load(weights_only=True), and state dictionaries must contain
only named tensors. The worker constructs the supported MLAC TransformerBlock
from the manifest and loads both state dictionaries strictly.

## Quality metric

This initial worker supports a fixed regression task. It computes negative
mean-squared error against target.pt for both baseline and candidate; higher
(less negative) is better. This is an explicit baseline quality metric, not a
universal LLM/task-quality measure. For language modeling or other tasks, add
a separately reviewed worker harness and task-specific dataset/metric rather
than trusting metrics emitted by candidate code.

Generated Python source is not imported or executed by this worker. Candidate
code execution needs a separately reviewed harness and remains confined to the
container boundary. The worker image itself is trusted code and must be built,
reviewed, scanned, and pinned by digest in operator-controlled configuration.
