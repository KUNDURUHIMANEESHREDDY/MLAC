# MLAC architecture

MLAC separates model metadata, supported transformations, evaluation evidence, experiment history, governance, and candidate execution.

- **IR:** versioned, framework-neutral component and tensor metadata.
- **Inspector:** reads an already-instantiated, trusted PyTorch module; does not load checkpoints or run a forward pass.
- **Registry:** in-memory model metadata store.
- **Mutation:** explicit dropout-configuration update on an IR snapshot, not a live graph or weight rewrite.
- **Architecture engineering:** MLAC's own PyTorch TransformerBlock supports head pruning with retained-weight migration.
- **Benchmark and evaluation:** baseline/candidate measurements, finite-output and shape checks, latency/parameter budgets, and mandatory task-quality gates.
- **Experiment ledger:** SQLite stores immutable evaluation scorecards and workflow outcomes, not model weights or executable checkpoints.
- **Evolution:** records hypotheses as proposals; it does not independently execute generated code or promote deployments.
- **Safety kernel:** integrity-checks trusted guardrails, permits only monotonic tightening, and requires externally pinned policy digests and scoped operator approval for production authorization.
- **Sandbox runner:** digest-pinned Docker execution with read-only candidate input, separate output, disabled network, dropped capabilities, non-root identity, and resource/time controls.
- **Candidate pipeline:** connects the fixed worker entrypoint to strict report validation, controller-owned evaluation gates, candidate-only governance, and durable scorecard recording. Candidate code cannot choose the worker command or access controller trust roots.

Keep the unauthenticated development API on localhost. Before network exposure, add authentication, authorization, durable registry state, rate limits, artifact isolation, and independently enforced sandboxing.

The Docker daemon, host kernel, worker image, and worker host remain in the sandbox trust boundary. Use a dedicated worker VM/host and stronger runtime isolation for hostile workloads. Trusted authorization, recovery, governance, and deployment trust roots must remain outside autonomous self-modification.
