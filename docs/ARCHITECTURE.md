# MLAC architecture

MLAC separates model metadata, supported transformations, evaluation evidence, experiment history, governance, and candidate execution.

- **IR:** versioned, framework-neutral component and tensor metadata.
- **Inspector:** reads an already-instantiated, trusted PyTorch module; does not load checkpoints or run a forward pass.
- **Registry:** in-memory model metadata store.
- **Mutation:** explicit dropout-configuration update on an IR snapshot, not a live graph or weight rewrite.
- **Architecture engineering:** the transactional Modification Engine supports isolated attention pruning/resizing, MLP hidden-width and activation edits, dimension-checked layer-stack changes, and top-k MoE conversion. The source model is not mutated.
- **Benchmark and evaluation:** baseline/candidate measurements, finite-output and shape checks, latency/parameter budgets, and mandatory task-quality gates.
- **Experiment ledger:** SQLite stores immutable evaluation scorecards and workflow outcomes.
- **Artifact registry:** content-addressed SHA-256 blobs with SQLite evaluation associations, integrity checks, and one-time worker-attestation nonce records.
- **Worker identity:** Ed25519 result signatures bind canonical report bytes, configured worker identity/image claim, and controller-recomputed input artifact hashes. Public keys are pinned in controller trust configuration; private keys are mounted read-only only into the fixed worker.
- **Master Intelligence:** normalizes objectives, diagnoses architecture metadata and ranks validated proposal-only experiment plans. Its optional OpenAI Responses backend returns schema-constrained suggestions that remain untrusted and allowlisted.
- **Intelligence workflow:** connects locally validated structural plans to the Modification Engine, emits a candidate checkpoint plus an operation manifest, and sends both to the fixed worker. The worker independently reconstructs the candidate from the trusted baseline before strict state loading. The LLM does not execute code, control the worker, or authorize promotion.
- **Evolution:** proposal-only evolution remains available. The recursive controller workflow can run the Master Intelligence plan-to-evaluation path against an operator-designated local/open-weight controller model and writes an append-only SQLite history of proposals and candidate outcomes. It never rewrites controller code, governance, authorization, or recovery, and never promotes a candidate automatically.
- **Safety kernel:** integrity-checks trusted guardrails, permits only monotonic tightening, and requires externally pinned policy digests and scoped operator approval for production authorization.
- **Sandbox runner:** digest-pinned Docker execution with read-only candidate input, separate output, disabled network, dropped capabilities, non-root identity, and resource/time controls.
- **Candidate pipeline:** connects the fixed worker entrypoint to signature and input-digest verification, strict report validation, controller-owned evaluation gates, candidate-only governance, and durable artifact/scorecard recording. Candidate code cannot choose the worker command or access controller trust roots.

The optional FastAPI surface validates OIDC JWTs against a configured JWKS endpoint, issuer, and audience. It resolves every non-health request to an issuer/subject principal and scopes model and proposal reads/writes to that principal. Request bodies are capped at 1 MiB; a process-local per-peer limiter bounds unauthenticated requests before JWKS resolution, and a separate process-local per-principal limiter applies after authentication. JWKS requests have a five-second timeout, and PyJWT is constrained to a patched release line. The registry remains in-memory and non-durable; neither limiter coordinates across workers. The CLI rejects non-loopback binds and disables Uvicorn proxy-header trust. Durable tenant-scoped storage, transactional migrations, distributed rate limiting, backup/restore, TLS gateway deployment, artifact isolation, and deployed acceptance remain required before network exposure.

The Docker daemon, host kernel, worker image, and worker host remain in the sandbox trust boundary. Use a dedicated worker VM/host and stronger runtime isolation for hostile workloads. Trusted authorization, recovery, governance, and deployment trust roots must remain outside autonomous self-modification.


The extended PyTorch inspector adds best-effort FX graph connectivity and shape metadata, attention configuration evidence, explicit-dimension KV-cache estimates, and opt-in forward-memory measurements. See `docs/ARCHITECTURE_INSPECTOR.md` for supported fields and limitations.
