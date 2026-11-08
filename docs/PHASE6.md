# Phase 6: Controller-Orchestrated Candidate Evaluation

Phase 6 connects the Phase 5 Docker runner to evaluation policy, governance, and the SQLite experiment ledger through CandidateEvaluationPipeline.

## Controlled workflow

1. The controller accepts a candidate bundle directory and stable model/candidate identifiers.
2. The controller chooses a fixed worker entrypoint inside the digest-pinned image. The candidate cannot supply the command.
3. Docker runs that worker with candidate input mounted read-only, no network, no controller source/policy/recovery mounts, and a separate output directory.
4. The trusted worker image evaluates baseline and candidate and writes /output/report.json.
5. The controller validates the versioned report schema, field types, finite values, shapes, and task-quality metrics.
6. Controller-owned EvaluationPolicy gates finite outputs, shape compatibility, latency regression, parameter growth, and task-quality non-regression.
7. Only a passing evaluation is submitted to GovernancePolicy for candidate-only authorization. This path cannot authorize production mutation or deployment.
8. The complete scorecard and governance outcome are persisted in ExperimentStore.

## Worker report contract (schema version 1)

The fixed trusted worker executable must produce exactly these top-level keys:

- schema_version: integer 1
- baseline and candidate: each contains parameter_count (integer), latency_ms_per_iteration (finite number), iterations (positive integer), output_shape (integer array), finite_output (boolean)
- quality_baseline and quality_candidate: non-empty maps of metric names to finite numbers; both must include EvaluationPolicy.quality_metric (default: score)

The controller rejects unknown/missing fields, malformed types, NaN/infinity, missing quality metrics, invalid JSON, and oversized reports. The worker executable path is controller configuration and is not read from candidate files.

## Worker image trust

A digest pin gives image immutability, not trustworthiness. Build the worker image from reviewed source in a trusted release pipeline, scan it, pin its digest in operator-controlled configuration, and keep the evaluation harness and dependencies in that image. The harness must measure baseline and candidate using the same fixed inputs/dataset manifest and must not accept candidate-supplied metric values as ground truth. Dataset and baseline artifacts should be explicit bundle inputs, not ambient host mounts. Do not inject signing keys, credentials, Docker socket, controller state, or policy files.

## Governance semantics

- A passing evaluation is evidence, not deployment permission.
- Candidate governance checks the operation and parameter-growth budget.
- A governance denial is persisted with the evaluation evidence.
- Production authorization remains a separate Phase 4 flow requiring external policy digest and scoped signed operator approval.
- No candidate output is automatically copied into the trusted repository, policy store, recovery path, or production deployment.

## Limitations

The worker report is trustworthy only to the extent that the pinned worker image and runtime are trusted. The Docker daemon, host kernel, and worker host remain part of the trusted computing base. The pipeline does not yet provide a remote worker fleet, durable artifact store, dataset registry, signed report attestation, or one-time production approval replay protection.
