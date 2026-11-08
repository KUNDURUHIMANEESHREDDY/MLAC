# Phase 3: Evaluation Laboratory and Experiment Ledger

MLAC now compares an explicitly supported baseline module with an isolated candidate and records the evidence. Evaluation does not deploy, promote, or mutate either model.

## Candidate scorecard

`evaluate_candidate(baseline, candidate, sample, ...)` benchmarks both modules with the same input, warmup count, and iteration count. It checks:

- Finite outputs from both modules.
- Matching output shapes.
- Candidate latency regression within policy (default maximum: 10%).
- Parameter growth within policy (default maximum: 10%).
- A caller-supplied task-quality metric that does not regress (required by default).

The quality evaluator returns a mapping of metric names to finite numeric values. The configured primary metric defaults to `score`; higher is better. The evaluator runs in evaluation mode under `torch.inference_mode()`, and each module's original training mode is restored afterward.

A performance-only run can be requested with `EvaluationPolicy(require_quality=False)`. It is marked exploratory and is never promotion-eligible. Even a passing quality-aware result is only marked `promotion_eligible`; MLAC does not deploy it.

## Durable experiment ledger

`ExperimentStore` uses SQLite from the Python standard library. It records immutable JSON scorecards, timestamps, model/candidate identifiers, and evaluation status. It does not store model weights, checkpoints, or executable artifacts. The default database path is `mlac-experiments.sqlite3`; choose a controlled path in production.

## Example

```python
import torch
from mlac.evaluation import EvaluationPolicy, evaluate_candidate
from mlac.experiments import ExperimentStore
from mlac.transformer import TransformerBlock, prune_attention_heads

baseline = TransformerBlock(d_model=32, num_heads=4).eval()
candidate = prune_attention_heads(baseline, [1]).module.eval()
sample = torch.randn(2, 16, 32)

# Replace this illustrative callback with a real, task-specific validation set.
def quality(model, batch):
    return {"score": float(task_validation_accuracy(model, batch))}

result = evaluate_candidate(
    baseline, candidate, sample,
    quality_evaluator=quality,
    policy=EvaluationPolicy(quality_metric="score"),
)
store = ExperimentStore("experiments.sqlite3")
store.record(result, model_id="baseline-v1", candidate_id="pruned-v1")
print(result.to_dict())
```

## Limitations and next work

- The latency measurement remains a local microbenchmark, not a statistically rigorous serving benchmark.
- The caller is responsible for representative validation data and trustworthy task metrics.
- Candidate execution is not yet isolated in a sandbox.
- SQLite is a local ledger, not a multi-worker experiment service.
- No training, checkpoint transformation, automatic promotion, or governance self-modification is implemented.

Next: add reproducible dataset/evaluation manifests, repeated measurements with confidence intervals, resource/memory metrics, artifact references, and sandboxed candidate execution.
