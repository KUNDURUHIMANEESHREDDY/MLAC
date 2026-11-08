# MLAC architecture

MLAC separates model metadata, supported transformations, evaluation evidence, experiment history, governance, and candidate execution.

- **IR:** versioned, framework-neutral component and tensor metadata.
- **Inspector:** reads an already-instantiated, trusted PyTorch module; does not load checkpoints or run a forward pass.
- **Registry:** in-memory model metadata store.
- **Mutation:** explicit dropout-configuration update on an IR snapshot. This is not a live graph or weight rewrite.
- **Architecture engineering:** the first executable target is MLAC's own PyTorch TransformerBlock; attention-head pruning creates a candidate and migrates retained weights.
- **Benchmark:** fixed-input local measurement of latency, parameter count, output shape, and finite outputs.
- **Evaluation:** compares baseline and candidate with explicit performance and task-quality gates. Passing creates evidence and possible candidate eligibility only; it does not deploy or promote.
- **Experiment ledger:** SQLite stores immutable evaluation scorecards and identifiers, not model weights or executable checkpoints.
- **Evolution:** records hypotheses as proposals; it does not execute generated code, train candidates, or promote deployments.
- **Governance:** candidate proposals are allowed by default; production mutations remain default-deny.
- **Safety kernel:** integrity-checks trusted guardrails, permits only monotonic tightening, and requires externally pinned policy digests and scoped operator approval for production authorization.
- **Sandbox runner:** Phase 5 adds an explicit Docker-backed worker boundary. Candidate bundles are read-only, output is separate and ephemeral, networking is disabled, and CPU/memory/PID/time/output limits are enforced. No candidate code executes in the controller process.

Install the API extra and run mlac serve. Keep the unauthenticated development API on localhost. Before network exposure, add authentication, authorization, durable registry state, rate limits, artifact isolation, and independently enforced sandboxing.

The Docker daemon, host kernel, and worker image remain in the sandbox trust boundary. Use a dedicated worker host/VM and stronger runtime isolation for hostile workloads. The trusted authorization, recovery, and governance boundaries must remain outside autonomous self-modification.
