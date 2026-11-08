# MLAC architecture

MLAC separates model metadata, supported transformations, evaluation evidence, experiment history, governance, and candidate execution.

- **IR:** versioned, framework-neutral component and tensor metadata.
- **Inspector:** reads an already-instantiated, trusted PyTorch module; does not load checkpoints or run a forward pass.
- **Registry:** in-memory model metadata store.
- **Mutation:** explicit dropout-configuration update on an IR snapshot, not a live graph or weight rewrite.
- **Architecture engineering:** MLAC's own PyTorch TransformerBlock supports head pruning with retained-weight migration.
- **Benchmark and evaluation:** baseline/candidate measurements, finite-output and shape checks, latency/parameter budgets, and mandatory task-quality gates.
- **Experiment ledger:** SQLite stores immutable evaluation scorecards and workflow outcomes.
- **Artifact registry:** content-addressed SHA-256 blobs with SQLite evaluation associations, integrity checks, and one-time worker-attestation nonce records.
- **Worker identity:** Ed25519 result signatures bind canonical report bytes, configured worker identity/image claim, and controller-recomputed input artifact hashes. Public keys are pinned in controller trust configuration; private keys are mounted read-only only into the fixed worker.
- **Evolution:** records hypotheses as proposals; it does not independently execute generated code or promote deployments.
- **Safety kernel:** integrity-checks trusted guardrails, permits only monotonic tightening, and requires externally pinned policy digests and scoped operator approval for production authorization.
- **Sandbox runner:** digest-pinned Docker execution with read-only candidate input, separate output, disabled network, dropped capabilities, non-root identity, and resource/time controls.
- **Candidate pipeline:** connects the fixed worker entrypoint to signature and input-digest verification, strict report validation, controller-owned evaluation gates, candidate-only governance, and durable artifact/scorecard recording. Candidate code cannot choose the worker command or access controller trust roots.

The optional FastAPI surface validates OIDC JWTs against a configured JWKS endpoint, issuer, and audience. It resolves every non-health request to an issuer/subject principal and scopes model and proposal reads/writes to that principal. Request bodies are capped at 1 MiB and a process-local per-principal limiter is applied. The registry remains in-memory and non-durable; rate limits do not coordinate across workers. The CLI therefore rejects all non-loopback binds even when OIDC is configured. Durable tenant-scoped storage, transactional migrations, distributed rate limiting, TLS gateway deployment, artifact isolation, and deployed acceptance remain required before network exposure.

The Docker daemon, host kernel, worker image, and worker host remain in the sandbox trust boundary. Use a dedicated worker VM/host and stronger runtime isolation for hostile workloads. Trusted authorization, recovery, governance, and deployment trust roots must remain outside autonomous self-modification.
