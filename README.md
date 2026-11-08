# MLAC — Master LLM Architecture Controller

MLAC is an open-source research and engineering platform for inspecting machine-learning models, representing structure in a framework-neutral intermediate representation (MLAC-IR), and running governed architecture experiments.

> **Status: Phase 7 engineering preview.** MLAC includes a supported PyTorch transformer target, candidate evaluation and scorecards, a safety kernel, a Docker-backed execution boundary, and a controller-owned sandbox/evaluation/governance workflow. It does not autonomously deploy models.

## Implemented

- Framework-neutral architecture, component, and tensor representation.
- Master Intelligence Engine with an OpenAI Responses API provider (schema-constrained proposals), plus a governed plan-to-candidate evaluation workflow for the supported TransformerBlock.
- In-memory model registry and read-only inspector for trusted, instantiated PyTorch modules.
- Versioned dropout-configuration mutation on IR snapshots.
- Proposal-only evolution controller and default-deny production mutation policy.
- Executable TransformerBlock with fused QKV attention and attention-head pruning with weight migration.
- Local microbenchmark and baseline-versus-candidate evaluation with task-quality and performance gates.
- SQLite-backed append-only experiment scorecard ledger.
- Safety kernel with policy integrity checks, monotonic tightening, and externally signed production approvals.
- Docker-backed candidate runner with no network, read-only candidate input, separate trusted evaluation mount, isolated output, dropped capabilities, non-root identity, and resource/time limits.
- Controller-owned candidate pipeline: fixed trusted worker, signed evidence verification, evaluation gates, candidate-only governance, and persisted outcome.
- Ed25519 worker-result attestations bound to report and input-artifact SHA-256 digests, with durable nonce replay rejection.
- Content-addressed artifact registry with SHA-256 verification and SQLite evaluation associations.
- Optional S3 Object Lock COMPLIANCE artifact backend with versioned, content-addressed uploads.
- Independent runtime-identity verifier for authority-signed SPIFFE/image/measurement statements, with durable nonce replay protection.
- Evaluation-pipeline hooks for mandatory runtime evidence and remote artifact archival.
- Optional API authentication with externally issued OIDC JWTs, JWKS signature verification, issuer/audience validation, and issuer/subject-scoped model and proposal access; bounded request bodies and pre-/post-authentication rate limits keep the API loopback-only until durable multi-tenant storage and deployment acceptance are complete.

## Quick start

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev,torch]"
    pytest
    mlac version

Optional API: install `pip install -e ".[api]"` and configure `MLAC_API_JWKS_URL`, `MLAC_API_ISSUER`, and `MLAC_API_AUDIENCE` from a trusted OIDC provider before running `mlac serve`. Every non-health route requires a signed bearer JWT with issuer, audience, subject, issued-at, and expiry claims. The API verifies signing keys through the configured JWKS endpoint and scopes models/proposals to the issuer plus subject. Missing identity configuration fails closed with HTTP 503.

## Master Intelligence planning

The proposal-only planning engine works without a model-provider API:

    from mlac.master_intelligence import MasterIntelligenceEngine
    report = MasterIntelligenceEngine().plan(
        model_id="baseline-001", architecture=architecture,
        objective="Reduce inference latency by 30%",
    )
    print(report.to_dict())

Configure `MLAC_OPENAI_API_KEY` (or `OPENAI_API_KEY`) and optionally `MLAC_OPENAI_MODEL` (default `gpt-6-astra`) to activate live OpenAI reasoning. `GET /intelligence/provider` reports provider configuration; `POST /intelligence/plan` accepts `model_id`, `architecture`, `objective`, and optional `constraints`. Without a key, the planner safely falls back to deterministic rules.

To execute proposals through the existing candidate pipeline, construct `IntelligenceExperimentWorkflow` with an initialized `CandidateEvaluationPipeline` and call `run(..., trusted_evaluation_dir=...)`. The initial trusted candidate builder supports only the worker's `TransformerBlock`: baseline profiling and attention-head pruning. It loads operator-controlled tensor state dictionaries, writes a candidate checkpoint, and sends it through the digest-pinned worker, signed-report verification, evaluation gates, candidate governance, artifact registry, and ledger. See `docs/MASTER_INTELLIGENCE.md` for setup and limits. The LLM never executes code, controls the worker, or authorizes a candidate.

## Candidate evaluation workflow

Build the initial worker image from the repository root:

    docker build -f worker/Dockerfile -t mlac-worker:local .

Resolve its immutable image digest and configure that digest in DockerSandboxRunner and CandidateEvaluationPipeline. The candidate directory contains candidate.pt only. A separate operator-managed trusted evaluation directory contains manifest.json, baseline.pt, input.pt, and target.pt. See worker/README.md for the manifest format.

    from mlac.pipeline import CandidateEvaluationPipeline
    from mlac.sandbox import DockerSandboxRunner
    from pathlib import Path
    from mlac.experiments import ExperimentStore
    from mlac.artifact_registry import ArtifactRegistry
    from mlac.attestation import key_id_for_public_key
    from cryptography.hazmat.primitives import serialization

    image = "registry.example/mlac-worker@sha256:<64-hex-digest>"
    worker_identity_dir = Path("./worker-identity")
    trusted_public_key = Path("./trusted-workers/worker-ed25519-public.pem").read_bytes()
    key_id = key_id_for_public_key(serialization.load_pem_public_key(trusted_public_key))
    pipeline = CandidateEvaluationPipeline(
        runner=DockerSandboxRunner(image=image),
        image=image,
        store=ExperimentStore("mlac-experiments.sqlite3"),
        trusted_worker_keys={key_id: trusted_public_key},
        worker_identity_dir=worker_identity_dir,
        artifact_registry=ArtifactRegistry("./mlac-artifacts"),
    )
    result = pipeline.run(
        candidate_dir="./candidate-bundle",
        trusted_evaluation_dir="./operator-controlled-evaluation",
        model_id="baseline-model-id",
        candidate_id="candidate-001",
    )
    print(result.status, result.evaluation.evaluation_id)

The worker identity directory contains worker_ed25519_private.pem and is mounted read-only at /identity. Protect the private key so only the worker UID (65532 in the container) can read it; do not commit it or bake it into the image. Register the corresponding public key in controller trust configuration. The worker signs the canonical report and hashes of every direct input artifact. The controller verifies the signature, worker key, configured image digest, report hash, and independently re-hashed inputs before evaluation. The SQLite artifact registry stores immutable SHA-256-addressed blobs and links them to the evaluation; a persistent nonce table rejects replayed attestations. See docs/PHASE7.md for worker-key provisioning and docs/MILESTONE_REMOTE_TRUST.md for remote WORM storage and runtime identity deployment. These are opt-in primitives; production use requires an independently operated attestation authority and an orchestrator that supplies evidence outside the worker container.

The pipeline can authorize only a candidate operation. It does not deploy or promote to production. The initial worker does not execute generated Python source; that requires a separately reviewed harness. See worker/README.md, docs/PHASE5.md, docs/PHASE6.md, and docs/PHASE7.md.

## Safety and limitations

- Production mutations are disabled by default and require external policy pinning plus scoped operator approval.
- Evaluation evidence is not deployment authorization.
- Candidate input and trusted evaluation artifacts are mounted read-only and separately; no controller, policy, recovery, credential, or home-directory mounts are provided.
- Worker reports are trusted only to the extent that the reviewed, digest-pinned worker image and runtime are trusted.
- Docker is not a complete defense against a compromised host, kernel, daemon, or runtime. Use a dedicated worker VM/host and stronger isolation for hostile code.
- The optional API requires OIDC JWT authentication and issuer/subject ownership checks for every non-health route. Request bodies are capped at 1 MiB; pre-authentication requests are limited per peer before JWKS resolution, and authenticated principals have a separate process-local limit. The CLI disables proxy-header trust and rejects non-loopback binds.
- The API registry and both rate limiters remain process-local and non-durable; they do not coordinate across workers. Durable tenant-scoped storage, transactional migrations, distributed rate limiting, backup/restore, TLS gateway deployment, and deployed acceptance remain release blockers.
- Trusted authorization, recovery, governance, and deployment trust roots are not self-modifying.

See docs/ARCHITECTURE.md and docs/PHASE4.md through docs/PHASE7.md.

## Roadmap

1. Remote WORM/object-lock artifact backend, encrypted replication, and signed dataset manifests.
2. External worker orchestration and independently measured runtime identity.
3. More supported architecture transformations and reproducible task evaluation.
4. Provider-independent master intelligence and experiment planning.
5. Recursive controller evaluation with immutable recovery and authorization boundaries.

## License

Apache-2.0. See LICENSE.
