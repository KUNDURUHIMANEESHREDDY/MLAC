import pytest
from mlac.ir import Architecture, Component, TensorSpec
from mlac.master_intelligence import MasterIntelligenceEngine, CallableReasoningBackend


def sample_architecture():
    return Architecture(name="tiny-transformer", family="transformer", framework="pytorch",
        dimensions={"parameter_count": 100_000, "module_count": 3},
        components=(Component("encoder.attn", "MultiheadAttention", {"num_heads": 4}),
                    Component("encoder.dropout", "Dropout", {"p": 0.1})),
        tensors=(TensorSpec("encoder.weight", (128, 128), "float32"),))


def test_plan_objective_diagnosis_and_attention_experiment():
    report = MasterIntelligenceEngine().plan(model_id="base-1", architecture=sample_architecture(),
                                              objective="Reduce inference latency by 30%")
    assert report.objective.primary_metric == "latency"
    assert report.objective.target_value == 30
    assert report.objective.direction == "minimize"
    assert any(d.code == "attention_compute_candidate" for d in report.diagnoses)
    assert {p.operation for p in report.plans} == {"profile_model", "prune_attention_heads"}
    pruning = next(p for p in report.plans if p.operation == "prune_attention_heads")
    assert pruning.parameters == {"module_name": "encoder.attn", "heads_to_prune": [3]}
    assert "task_quality" in pruning.required_measurements


def test_profile_only_when_no_structural_evidence():
    report = MasterIntelligenceEngine().plan(model_id="opaque-1",
        architecture=Architecture(name="opaque", family="custom"), objective="Improve quality")
    assert len(report.plans) == 1
    assert report.plans[0].operation == "profile_model"


def test_rejects_unknown_constraints_and_invalid_objective():
    engine = MasterIntelligenceEngine()
    with pytest.raises(ValueError, match="unsupported constraint"):
        engine.plan(model_id="m", architecture=sample_architecture(),
                    objective="lower latency", constraints={"allow_deploy": 1})
    with pytest.raises(ValueError, match="non-empty"):
        engine.plan(model_id="m", architecture=sample_architecture(), objective=" ")


def test_llm_backend_suggestions_are_allowlist_validated():
    def provider(**kwargs):
        return [{"operation": "shell_command", "parameters": {},
                 "hypothesis": "run arbitrary code", "rationale": "because",
                 "expected_effect": {"latency": "lower"}, "falsification_criteria": ["not faster"]}]
    engine = MasterIntelligenceEngine(backend=CallableReasoningBackend(provider))
    with pytest.raises(ValueError, match="unsupported operation"):
        engine.plan(model_id="m", architecture=sample_architecture(), objective="reduce latency")


def test_llm_backend_can_add_validated_plan():
    def provider(**kwargs):
        return [{"operation": "prune_attention_heads",
                 "parameters": {"module_name": "encoder.attn", "heads_to_prune": [0]},
                 "hypothesis": "Prune a different attention head",
                 "rationale": "The provider identified a distinct head ablation to evaluate.",
                 "expected_effect": {"latency": "decrease_expected"},
                 "falsification_criteria": ["No measured latency improvement"],
                 "risk": "low", "estimated_value": 0.3, "feasibility": 0.9}]
    report = MasterIntelligenceEngine(backend=CallableReasoningBackend(provider)).plan(
        model_id="m", architecture=sample_architecture(), objective="reduce latency")
    assert report.planner == "rules+llm"
    assert any(p.operation == "prune_attention_heads" and p.parameters["heads_to_prune"] == [0] for p in report.plans)


def test_rejects_invalid_pruning_indices_from_backend():
    def provider(**kwargs):
        return [{"operation": "prune_attention_heads",
                 "parameters": {"module_name": "encoder.attn", "heads_to_prune": [-1]},
                 "hypothesis": "prune", "rationale": "test", "expected_effect": {},
                 "falsification_criteria": ["quality regression"]}]
    with pytest.raises(ValueError, match="non-negative"):
        MasterIntelligenceEngine(backend=CallableReasoningBackend(provider)).plan(
            model_id="m", architecture=sample_architecture(), objective="reduce latency")
