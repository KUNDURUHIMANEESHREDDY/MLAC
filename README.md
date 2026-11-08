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
- Docker-backed candidate runner with no network, read-only candidate input, separate trusted evaluation mount, isolated output, dropped capabilities, non-root identity, and resource/time limits.
- Controller-owned candidate pipeline: fixed trusted worker, validated report, evaluation gates, candidate-only governance, and persisted outcome.

## Quick start

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev,torch]"
    pytest
    mlac version

Optional API: pip install -e ".[api]" then mlac serve. Keep the unauthenticated API on localhost.

## Candidate evaluation workflow

Build the initial worker image from the repository root:

    docker build -f worker/Dockerfile -t mlac-worker:local .

Resolve its immutable image digest and configure that digest in DockerSandboxRunner and CandidateEvaluationPipeline. The candidate directory contains candidate.pt only. A separate operator-managed trusted evaluation directory contains manifest.json, baseline.pt, input.pt, and target.pt. See worker/README.md for the manifest format.

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
        trusted_evaluation_dir="./operator-controlled-evaluation",
        model_id="baseline-model-id",
        candidate_id="candidate-001",
    )
    print(result.status, result.evaluation.evaluation_id)

The pipeline can authorize only a candidate operation. It does not deploy or promote to production. The initial worker does not execute generated Python source; that requires a separately reviewed harness. See worker/README.md, docs/PHASE5.md, and docs/PHASE6.md.

## Safety and limitations

- Production mutations are disabled by default and require external policy pinning plus scoped operator approval.
- Evaluation evidence is not deployment authorization.
- Candidate input and trusted evaluation artifacts are mounted read-only and separately; no controller, policy, recovery, credential, or home-directory mounts are provided.
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
