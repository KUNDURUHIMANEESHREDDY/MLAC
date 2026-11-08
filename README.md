# MLAC — Master LLM Architecture Controller

MLAC is an open-source research and engineering platform for inspecting machine-learning models, representing structure in a framework-neutral intermediate representation (MLAC-IR), and running governed architecture experiments.

> **Status: Phase 5 engineering preview.** MLAC includes a supported PyTorch transformer target, candidate evaluation and scorecards, a safety kernel, and a Docker-backed isolated execution primitive. It does not autonomously train, promote, or deploy models.

## Implemented

- Framework-neutral architecture, component, and tensor representation.
- In-memory model registry and read-only inspector for trusted, instantiated PyTorch modules.
- Versioned dropout-configuration mutation on IR snapshots (not live model weights).
- Proposal-only evolution controller and default-deny production mutation policy.
- Executable TransformerBlock with fused QKV attention and attention-head pruning with weight migration.
- Local microbenchmark and baseline-versus-candidate evaluation with task-quality and performance gates.
- SQLite-backed append-only experiment scorecard ledger.
- Safety kernel with policy integrity checks, monotonic tightening, and externally signed production approvals.
- Docker-backed candidate runner with no network, read-only candidate input, isolated output, dropped capabilities, non-root identity, and resource/time/output limits.
- GitHub Actions test workflow.

## Quick start

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev,torch]"
    pytest
    mlac version

Optional API: pip install -e ".[api]" then mlac serve. API docs: http://127.0.0.1:8000/docs.

## Run an isolated candidate

The runner requires Docker and a worker image pinned by immutable digest. It never falls back to in-process execution.

    from mlac.sandbox import DockerSandboxRunner, SandboxLimits

    runner = DockerSandboxRunner(
        image="registry.example/mlac-worker@sha256:<64-hex-digest>",
        limits=SandboxLimits(timeout_seconds=90, memory="4g", cpus=2),
    )
    result = runner.run(
        candidate_dir="./candidate-bundle",
        command=["python", "-I", "/workspace/evaluate_candidate.py"],
    )

See docs/PHASE5.md before running untrusted workloads.

## Safety and limitations

- Production mutations are disabled by default. Candidate evaluation is evidence, not deployment authorization.
- Only trusted, already-instantiated PyTorch modules are inspected; no untrusted checkpoint loading.
- Head pruning supports MLAC's own TransformerBlock, not arbitrary model graph surgery.
- The optional API is unauthenticated; keep it on localhost.
- The sandbox is a container execution boundary, not a complete defense against a compromised host, kernel, Docker daemon, or container runtime. Use a dedicated non-root worker VM/host and stronger isolation for hostile code.
- Candidate code cannot write to its mounted input; no controller, policy, recovery, credential, or home-directory mounts are provided. Output is ephemeral and is not automatically promoted.
- The trusted authorization, recovery, governance, and deployment trust roots are not self-modifying.

See docs/ARCHITECTURE.md, docs/PHASE4.md, and docs/PHASE5.md.

## Roadmap

1. Durable registry, artifact references, and audit log.
2. More supported architecture transformations and weight migration.
3. Reproducible dataset manifests, robust evaluation statistics, and external worker orchestration.
4. Provider-independent master intelligence and experiment planning.
5. Recursive controller evaluation with immutable recovery and authorization boundaries.
6. Production hardening and deployment.

## License

Apache-2.0. See LICENSE.
