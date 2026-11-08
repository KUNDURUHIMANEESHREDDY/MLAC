# MLAC — Master LLM Architecture Controller

MLAC is an open-source research and engineering platform for inspecting machine-learning models, representing structure in a framework-neutral intermediate representation (MLAC-IR), and managing governed architecture-change proposals.

> **Status: early MVP (v0.3.0).** This release includes MLAC-IR, an in-memory registry, optional PyTorch inspection, a small allow-listed IR metadata mutation, proposal-based evolution, governance defaults, and core tests. It does not autonomously redesign arbitrary LLMs, train models, rewrite live weights, or deploy candidates.

## Implemented
- Framework-neutral architecture, component, and tensor representation.
- In-memory model registry.
- Optional read-only PyTorch module inspector.
- Versioned dropout-configuration mutation on IR snapshots (not live model weights).
- Proposal-only evolution controller.
- Default-deny production mutation policy.
- Unit tests for core behavior.

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

## Safety and limitations
- Production mutations are disabled by default.
- Only inspect trusted, already-instantiated PyTorch modules; no untrusted checkpoint loading.
- IR mutation is not a live neural-network topology or weight transformation.
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
