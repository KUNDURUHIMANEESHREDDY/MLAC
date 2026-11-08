# MLAC — Master LLM Architecture Controller

MLAC is an open-source research and engineering platform for inspecting machine-learning models, representing structure in a framework-neutral intermediate representation (MLAC-IR), and managing governed architecture-change proposals.

> **Status: Phase 2 engineering preview.** The foundation MVP now includes an executable, explicitly supported PyTorch transformer block, attention-head pruning with weight migration, a local microbenchmark, and CI coverage. This is not arbitrary model graph surgery and does not autonomously train or promote candidates.

## Implemented
- Framework-neutral architecture, component, and tensor representation.
- In-memory model registry.
- Optional read-only PyTorch module inspector.
- Versioned dropout-configuration mutation on IR snapshots (not live model weights).
- Proposal-only evolution controller.
- Default-deny production mutation policy.
- Unit tests for core behavior.
- Executable `TransformerBlock` with fused QKV attention.
- Attention-head pruning that creates an isolated candidate and migrates retained Q/K/V and output-projection weights.
- Local candidate microbenchmark for parameter count, latency, output shape, and finite outputs.
- GitHub Actions test workflow.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,torch]"
pytest
mlac version
```

Optional API: `pip install -e ".[api]"` then `mlac serve`. API docs: http://127.0.0.1:8000/docs.

## Inspect a model
```python
from examples.tiny_model import TinyMLP
from mlac.inspector import inspect_torch_model
snapshot = inspect_torch_model(TinyMLP(), name="tiny-mlp")
print(snapshot.to_dict())
```

## Run an architecture experiment
\n```bash\npip install -e ".[torch,dev]"\npython examples/architecture_search.py\n```\n\nSee [Phase 2 engineering notes](docs/PHASE2.md) for the supported transformation and benchmark limitations.\n\n## Safety and limitations\n- Production mutations are disabled by default.
- Only inspect trusted, already-instantiated PyTorch modules; no untrusted checkpoint loading.
- Head pruning currently supports only MLAC's own `TransformerBlock`; it is not a generic Hugging Face/PyTorch model surgery adapter.
- Head pruning structurally removes heads and migrates weights, but does not guarantee preserved task quality; task-specific evaluation is required.
- The benchmark is a local microbenchmark, not a statistically robust quality/performance evaluation suite.
- Candidate proposals require independent evaluation before promotion.
- Registry and proposal state are in memory and reset on restart.
- The optional API is unauthenticated; keep it on localhost.
- Training, GPU scheduling, sandbox execution, autonomous code generation, and production deployment are not implemented.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Roadmap
1. Durable registry, artifacts, and audit log.
2. Executable graph transformations and weight migration for supported model families.
3. Isolated training and reproducible benchmark evaluation.
4. Provider-independent master intelligence and experiment planning.
5. Recursive controller evaluation with immutable recovery and authorization boundaries.
6. Production hardening and deployment.

## License
Apache-2.0. See [LICENSE](LICENSE).
