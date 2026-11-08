# Phase 6: Controller-Orchestrated Candidate Evaluation

Phase 6 connects the Phase 5 Docker runner to evaluation policy, governance, and the SQLite experiment ledger through CandidateEvaluationPipeline. The repository includes an initial trusted worker image for MLAC's supported TransformerBlock and a fixed regression benchmark.

## Controlled workflow

1. The controller accepts a candidate bundle and a separate operator-managed trusted evaluation directory.
2. The controller chooses a fixed worker entrypoint inside the digest-pinned image. The candidate cannot supply the command.
3. Docker mounts candidate.pt at /workspace (read-only) and the trusted manifest, baseline checkpoint, input, and target at /trusted (read-only). The directories must be distinct.
4. The trusted worker image loads baseline and candidate tensor-only state dictionaries and writes /output/report.json.
5. The controller validates the versioned report schema, field types, finite values, shapes, and task-quality metrics.
6. Controller-owned EvaluationPolicy gates finite outputs, shape compatibility, latency regression, parameter growth, and task-quality non-regression.
7. Only a passing evaluation is submitted to GovernancePolicy for candidate-only authorization. This path cannot authorize production mutation or deployment.
8. The complete scorecard and governance outcome are persisted in ExperimentStore.

## Build the worker image

From the repository root:

    docker build -f worker/Dockerfile -t mlac-worker:local .

Resolve the image to its immutable repository digest, then configure that digest in both DockerSandboxRunner and CandidateEvaluationPipeline. Do not use a mutable tag in production. See worker/README.md for the supported bundle format and trusted data separation.

## Worker report contract (schema version 1)

The fixed worker executable must produce exactly these top-level keys:

- schema_version: integer 1
- baseline and candidate: each contains parameter_count (integer), latency_ms_per_iteration (finite number), iterations (positive integer), output_shape (integer array), finite_output (boolean)
- quality_baseline and quality_candidate: non-empty maps of metric names to finite numbers; both must include EvaluationPolicy.quality_metric (default: score)

The controller rejects unknown/missing fields, malformed types, NaN/infinity, missing quality metrics, invalid JSON, and oversized reports. The worker executable path is controller configuration and is not read from candidate files.

## Supported worker bundle and task metric

The candidate directory contains candidate.pt only. A separate operator-managed trusted directory contains manifest.json, baseline.pt, input.pt, and target.pt. The manifest defines architecture, baseline checkpoint, dataset tensors, and benchmark iteration counts. Candidate-controlled storage cannot replace trusted baseline or evaluation data.

The worker loads checkpoints with torch.load(weights_only=True), requires tensor-only state dictionaries, rejects symlinks/path traversal, and uses strict state-dict loading. The initial task metric is negative mean-squared error against the trusted target tensor; higher (less negative) is better. This is a concrete regression benchmark, not a universal language-model quality metric. A language-model or other task requires its own separately reviewed trusted worker harness and dataset/metric.

Generated Python source is not imported or executed by this initial worker. Supporting generated source requires a separately reviewed harness and remains confined to the container boundary; do not let candidate-provided scripts or metric values define the trusted evaluation.

## Worker image trust

A digest pin gives image immutability, not trustworthiness. Build the worker image from reviewed source in a trusted release pipeline, scan it, pin its digest in operator-controlled configuration, and keep the evaluation harness and dependencies in that image. The harness measures baseline and candidate using the same operator-controlled input/target artifacts and does not accept candidate-supplied metric values as ground truth. Do not inject signing keys, credentials, Docker socket, controller state, or policy files.

## Governance semantics

- A passing evaluation is evidence, not deployment permission.
- Candidate governance checks the operation and parameter-growth budget.
- A governance denial is persisted with the evaluation evidence.
- Production authorization remains a separate Phase 4 flow requiring external policy digest and scoped signed operator approval.
- No candidate output is automatically copied into the trusted repository, policy store, recovery path, or production deployment.

## Limitations

The worker report is trustworthy only to the extent that the pinned worker image and runtime are trusted. The Docker daemon, host kernel, and worker host remain part of the trusted computing base. The pipeline does not yet provide a remote worker fleet, durable artifact store, dataset registry, signed report attestation, or one-time production approval replay protection.
