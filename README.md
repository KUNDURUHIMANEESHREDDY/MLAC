# MLAC — Master LLM Architecture Controller

MLAC is an open-source research and engineering platform for inspecting machine-learning models, representing structure in a framework-neutral intermediate representation (MLAC-IR), and running governed architecture experiments.

> **Status: Phase 3 engineering preview.** MLAC includes a narrow executable PyTorch transformer target, attention-head pruning with weight migration, candidate scorecards, a SQLite experiment ledger, and CI coverage. It does not perform arbitrary model graph surgery, autonomously train models, or deploy candidates.

## Implemented

- Framework-neutral architecture, component, and tensor representation.
- In-memory model registry and read-only inspector for trusted, instantiated PyTorch modules.
- Versioned dropout-configuration mutation on IR snapshots (not live model weights).
- Proposal-only evolution controller and default-deny production mutation policy.
- Executable `TransformerBlock` with fused QKV attention.
- Attention-head pruning that creates an isolated candidate and migrates retained Q/K/V and output-projection weights.
- Local microbenchmark for parameter count, latency, output shape, and finite outputs.
- Baseline-versus-candidate evaluation with output validity, shape, latency, parameter-growth, and task-quality gates.
- SQLite-backed append-only experiment scorecard ledger.
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

```bash
python examples/architecture_search.py
```

See [Phase 2 engineering notes](docs/PHASE2.md) for supported structural pruning and [Phase 3 evaluation notes](docs/PHASE3.md) for candidate scorecards and the experiment ledger.

## Evaluate and record a candidate

```python
import torch
from mlac.evaluation import evaluate_candidate
from mlac.experiments import ExperimentStore
from mlac.transformer import TransformerBlock, prune_attention_heads

baseline = TransformerBlock(d_model=32, num_heads=4).eval()
candidate = prune_attention_heads(baseline, [1]).module.eval()
sample = torch.randn(2, 16, 32)

# Supply a real task-specific validation metric for your model and dataset.
def quality(model, batch):
    return {"score": float(task_validation_score(model, batch))}

result = evaluate_candidate(baseline, candidate, sample, quality_evaluator=quality)
ExperimentStore("experiments.sqlite3").record(
    result, model_id="baseline-v1", candidate_id="pruned-v1"
)
```

## Safety and limitations

- Production mutations are disabled by default.
- Candidate evaluation is evidence, not deployment authorization. MLAC never promotes or deploys candidates.
- Quality evaluation is required by default; performance-only results are exploratory and cannot be promotion-eligible.
- Only inspect trusted, already-instantiated PyTorch modules; no untrusted checkpoint loading.
- Head pruning currently supports only MLAC's own `TransformerBlock`; it is not a generic Hugging Face/PyTorch model surgery adapter.
- The benchmark is a local microbenchmark, not a statistically robust quality or serving-performance evaluation suite.
- Registry and proposal state are in memory. The experiment ledger is SQLite and stores scorecards, not model artifacts.
- The optional API is unauthenticated; keep it on localhost.
- Training, GPU scheduling, sandbox execution, autonomous code generation, and production deployment are not implemented.
- The trusted authorization, recovery, and governance boundaries are not self-modifying.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Roadmap

1. Durable registry, artifact references, and audit log.
2. More supported architecture transformations and weight migration.
3. Reproducible dataset manifests, robust evaluation statistics, resource metrics, and isolated candidate execution.
4. Provider-independent master intelligence and experiment planning.
5. Recursive controller evaluation with immutable recovery and authorization boundaries.
6. Production hardening and deployment.

## License

Apache-2.0. See [LICENSE](LICENSE).
