import sqlite3

import pytest

torch = pytest.importorskip("torch")
from mlac.evaluation import EvaluationPolicy, evaluate_candidate
from mlac.experiments import ExperimentStore
from mlac.transformer import TransformerBlock, prune_attention_heads


def test_evaluation_requires_task_quality_before_promotion():
    torch.manual_seed(11)
    baseline = TransformerBlock(d_model=16, num_heads=4, mlp_ratio=2).eval()
    candidate = prune_attention_heads(baseline, [0]).module.eval()
    sample = torch.randn(2, 5, 16)
    result = evaluate_candidate(
        baseline, candidate, sample,
        quality_evaluator=lambda model, x: {"score": 0.8},
        policy=EvaluationPolicy(max_latency_regression_ratio=1000),
        iterations=2, warmup=1,
    )
    assert result.checks["quality_evaluated"]
    assert result.promotion_eligible
    assert result.passed
    assert result.to_dict()["evaluation_id"] == result.evaluation_id


def test_performance_only_evaluation_is_never_promotion_eligible():
    model = TransformerBlock(d_model=16, num_heads=4).eval()
    result = evaluate_candidate(
        model, model, torch.randn(1, 3, 16),
        policy=EvaluationPolicy(require_quality=False, max_latency_regression_ratio=1000),
        iterations=1, warmup=0,
    )
    assert result.passed
    assert not result.promotion_eligible
    assert any("exploratory" in reason for reason in result.reasons)


def test_quality_regression_fails_evaluation():
    model = TransformerBlock(d_model=16, num_heads=4).eval()
    result = evaluate_candidate(
        model, model, torch.randn(1, 3, 16),
        quality_evaluator=lambda module, sample: {"score": 0.2},
        policy=EvaluationPolicy(minimum_quality_delta=0.01), iterations=1, warmup=0,
    )
    assert not result.passed
    assert not result.promotion_eligible
    assert not result.checks["quality_non_regression"]


def test_quality_metric_is_required_and_finite():
    model = TransformerBlock(d_model=8, num_heads=2)
    with pytest.raises(ValueError, match="configured metric"):
        evaluate_candidate(model, model, torch.randn(1, 2, 8),
                           quality_evaluator=lambda m, x: {"other": 1.0},
                           iterations=1, warmup=0)
    with pytest.raises(ValueError, match="finite"):
        evaluate_candidate(model, model, torch.randn(1, 2, 8),
                           quality_evaluator=lambda m, x: {"score": float("nan")},
                           iterations=1, warmup=0)


def test_experiment_store_roundtrip_and_filter(tmp_path):
    model = TransformerBlock(d_model=8, num_heads=2).eval()
    result = evaluate_candidate(
        model, model, torch.randn(1, 2, 8),
        quality_evaluator=lambda m, x: {"score": 1.0},
        policy=EvaluationPolicy(max_latency_regression_ratio=1000), iterations=1, warmup=0,
    )
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    stored = store.record(result, model_id="base", candidate_id="candidate-1")
    assert stored["result"]["evaluation_id"] == result.evaluation_id
    assert store.get(result.evaluation_id)["status"] == "eligible"
    assert len(store.list(model_id="base")) == 1
    assert store.list(model_id="other") == []
    with pytest.raises(KeyError):
        store.get("missing")
    with pytest.raises(ValueError):
        store.list(limit=0)


def test_experiment_store_rejects_duplicate_ids(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    payload = {"evaluation_id": "same", "created_at": "now", "passed": True}
    store.record(payload)
    with pytest.raises(sqlite3.IntegrityError):
        store.record(payload)


def test_quality_evaluator_runs_in_eval_mode_and_restores_training_mode():
    model = TransformerBlock(d_model=8, num_heads=2)
    model.train()
    observed_modes = []

    def evaluator(module, sample):
        observed_modes.append(module.training)
        return {"score": 0.5}

    evaluate_candidate(model, model, torch.randn(1, 2, 8),
                       quality_evaluator=evaluator,
                       policy=EvaluationPolicy(max_latency_regression_ratio=1000),
                       iterations=1, warmup=0)
    assert observed_modes == [False, False]
    assert model.training


def test_task_quality_is_required_by_default():
    model = TransformerBlock(d_model=8, num_heads=2).eval()
    result = evaluate_candidate(
        model, model, torch.randn(1, 2, 8), iterations=1, warmup=0,
        policy=EvaluationPolicy(max_latency_regression_ratio=1000),
    )
    assert not result.passed
    assert not result.promotion_eligible
    assert not result.checks["quality_evaluated"]
