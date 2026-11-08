# MLAC — Master LLM Architecture Controller

MLAC is an open-source research and engineering platform for inspecting machine-learning models, representing structure in a framework-neutral intermediate representation (MLAC-IR), and running governed architecture experiments.

> **Status: Phase 6 engineering preview.** MLAC includes a supported PyTorch transformer target, candidate evaluation and scorecards, a safety kernel, a Docker-backed execution boundary, and a controller-owned sandbox/evaluation/governance workflow. It does not autonomously deploy models.

## Implemented

- Framework-neutral architecture, component, and tensor representation.
- In-memory model registry and read-only inspector for trusted, instantiated PyTorch modules.
- Versioned dropout-configuration mutation on IR snapshots.
- Proposal-only evolution controller and default-deny production mutation policy.
- Executable TransformerBlock with fused QKV attention and attention-head pruning with weight migration.
- Local microbenchmark and baseline-versus-candidate evaluation with task-quality and performance gates.
- SQLite-backed append-only experiment scorecard ledger.
- Safety kernel with policy integrity checks, monotonic tightening, and externally signed production approvals.
- Docker-backed candidate runner with no network, read-only candidate input, isolated output, dropped capabilities, non-root identity, and resource/time limits.
- Controller-owned candidate pipeline: fixed trusted worker, validated report, evaluation gates, candidate-only governance, and persisted outcome.

## Quick start

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev,torch]"
    pytest
    mlac version

Optional API: pip install -e ".[api]" then mlac serve. Keep the unauthenticated API on localhost.

## Candidate evaluation workflow

The worker image must be trusted and pinned by immutable digest. Its fixed /opt/mlac-worker/evaluate executable evaluates baseline and candidate artifacts from the read-only bundle and writes /output/report.json using schema version 1.

    from mlac.pipeline import CandidateEvaluationPipeline
    from mlac.sandbox import DockerSandboxRunner
    from mlac.experiments import ExperimentStore

    image = "registry.example/mlac-worker@sha256:<64-hex-digest>"
    pipeline = CandidateEvaluationPipeline(
        runner=DockerSandboxRunner(image=image),
        image=image,
        store=ExperimentStore("mlac-experiments.sqlite3"),
    )
    result = pipeline.run(
        candidate_dir="./candidate-bundle",
        model_id="baseline-model-id",
        candidate_id="candidate-001",
    )
    print(result.status, result.evaluation.evaluation_id)

The pipeline can authorize only a candidate operation. It does not deploy or promote to production. See docs/PHASE5.md and docs/PHASE6.md.

## Safety and limitations

- Production mutations are disabled by default and require external policy pinning plus scoped operator approval.
- Evaluation evidence is not deployment authorization.
- Candidate input is read-only; no controller, policy, recovery, credential, or home-directory mounts are provided.
- Worker reports are trusted only to the extent that the reviewed, digest-pinned worker image and runtime are trusted.
- Docker is not a complete defense against a compromised host, kernel, daemon, or runtime. Use a dedicated worker VM/host and stronger isolation for hostile code.
- The optional API is unauthenticated; keep it on localhost.
- Trusted authorization, recovery, governance, and deployment trust roots are not self-modifying.

See docs/ARCHITECTURE.md, docs/PHASE4.md, docs/PHASE5.md, and docs/PHASE6.md.

## Roadmap

1. Durable artifact store and immutable dataset manifests.
2. Signed worker attestations and external worker orchestration.
3. More supported architecture transformations and reproducible task evaluation.
4. Provider-independent master intelligence and experiment planning.
5. Recursive controller evaluation with immutable recovery and authorization boundaries.

## License

Apache-2.0. See LICENSE.
