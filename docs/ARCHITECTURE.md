# MLAC architecture

MLAC separates model metadata, supported transformations, evaluation evidence, experiment history, and governance.

- **IR:** versioned, framework-neutral component and tensor metadata.
- **Inspector:** reads an already-instantiated, trusted PyTorch module; does not load checkpoints or run a forward pass.
- **Registry:** in-memory model metadata store.
- **Mutation:** explicit dropout-configuration update on an IR snapshot. This is not a live graph or weight rewrite.
- **Architecture engineering:** the first executable target is MLAC's own PyTorch `TransformerBlock`; attention-head pruning creates a candidate and migrates retained weights.
- **Benchmark:** fixed-input local measurement of latency, parameter count, output shape, and finite outputs.
- **Evaluation:** compares baseline and candidate with explicit performance and task-quality gates. Passing creates evidence and possible candidate eligibility only; it does not deploy or promote.
- **Experiment ledger:** SQLite stores immutable evaluation scorecards and identifiers, not model weights or executable checkpoints.
- **Evolution:** records hypotheses as proposals; it does not execute generated code, train candidates, or promote deployments.
- **Governance:** candidate proposals are allowed by default; production mutations remain default-deny.

Install the API extra and run `mlac serve`. Keep the unauthenticated development API on localhost. Before network exposure, add authentication, authorization, durable registry state, rate limits, artifact isolation, and an independently enforced sandbox.

PostgreSQL, object storage, MLflow, Temporal, distributed training, model-provider adapters, sandboxed execution, and hardware schedulers remain roadmap items. The trusted authorization, recovery, and governance boundaries must remain outside autonomous self-modification.

- **Safety kernel:** integrity-checks the trusted guardrail policy, permits only monotonic tightening, and requires externally signed, scoped, unexpired approval for production operations when production is explicitly enabled. This is an in-process application control, not a sandbox; production deployments still need external trust roots, replay protection, KMS/HSM-backed keys, and isolated execution.
