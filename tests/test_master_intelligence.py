import pytest
from mlac.ir import Architecture, Component, TensorSpec
from mlac.master_intelligence import MasterIntelligenceEngine, CallableReasoningBackend


def sample_architecture():
    return Architecture(name="tiny-transformer", family="transformer", framework="pytorch",
        dimensions={"parameter_count": 100_000, "module_count": 3},
        components=(Component("encoder.attn", "MultiheadAttention", {"num_heads": 4}),
                    Component("encoder.mlp", "FeedForward", {"hidden_size": 128}),
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


def test_rejects_non_finite_objective_constraints():
    engine = MasterIntelligenceEngine()
    with pytest.raises(ValueError, match="finite number"):
        engine.plan(
            model_id="m",
            architecture=sample_architecture(),
            objective="reduce latency",
            constraints={"max_latency_regression_ratio": float("nan")},
        )


def test_profile_plan_parameters_are_bounded_and_allowlisted():
    def provider(**kwargs):
        return [{
            "operation": "profile_model",
            "parameters": {"measurement_iterations": 1000000},
            "hypothesis": "Profile carefully",
            "rationale": "A bounded profile is required.",
            "expected_effect": {"latency": "measured"},
            "falsification_criteria": ["Measurements are unstable"],
            "risk": "low",
            "estimated_value": 0.5,
            "feasibility": 0.9,
        }]
    with pytest.raises(ValueError, match="measurement_iterations"):
        MasterIntelligenceEngine(backend=CallableReasoningBackend(provider)).plan(
            model_id="m", architecture=sample_architecture(), objective="reduce latency"
        )


def test_decision_ranking_penalizes_risk_after_value_and_feasibility():
    def provider(**kwargs):
        proposals = []
        for head, risk in ((0, "high"), (1, "low")):
            proposals.append({
                "operation": "prune_attention_heads",
                "parameters": {"module_name": "encoder.attn", "heads_to_prune": [head]},
                "hypothesis": f"Prune head {head}",
                "rationale": "Compare isolated head ablations.",
                "expected_effect": {"latency": "measure", "quality": "measure"},
                "falsification_criteria": ["Quality regresses"],
                "risk": risk,
                "estimated_value": 0.8,
                "feasibility": 0.8,
            })
        return proposals

    report = MasterIntelligenceEngine(
        backend=CallableReasoningBackend(provider), max_plans=5
    ).plan(model_id="m", architecture=sample_architecture(), objective="reduce latency")
    low_risk = next(i for i, plan in enumerate(report.plans)
                    if plan.operation == "prune_attention_heads"
                    and plan.parameters["heads_to_prune"] == [1])
    high_risk = next(i for i, plan in enumerate(report.plans)
                     if plan.operation == "prune_attention_heads"
                     and plan.parameters["heads_to_prune"] == [0])
    assert low_risk < high_risk
    assert "Top-ranked proposal:" in report.reasoning_summary


def test_structural_modification_proposals_are_bounded_and_allowlisted():
    proposals = [
        ("resize_attention_heads", {"num_heads": 2}),
        ("resize_mlp_hidden", {"hidden_size": 96}),
        ("set_mlp_activation", {"activation": "silu"}),
        ("convert_mlp_to_moe", {"num_experts": 4, "top_k": 2, "expert_noise_std": 0.001}),
    ]
    for operation, parameters in proposals:
        def provider(**kwargs):
            return [{
                "operation": operation,
                "parameters": parameters,
                "hypothesis": "Evaluate a bounded architecture edit.",
                "rationale": "The architecture metadata exposes the required target.",
                "expected_effect": {"quality": "measure", "latency": "measure"},
                "falsification_criteria": ["Quality or performance violates policy"],
                "risk": "medium", "estimated_value": 0.5, "feasibility": 0.8,
            }]
        report = MasterIntelligenceEngine(
            backend=CallableReasoningBackend(provider), max_plans=10
        ).plan(
            model_id="m", architecture=sample_architecture(), objective="reduce latency"
        )
        assert any(plan.operation == operation and plan.parameters == parameters
                   for plan in report.plans)


def test_structural_modification_proposals_reject_invalid_values():
    invalid = [
        ("resize_attention_heads", {"num_heads": 0}),
        ("resize_mlp_hidden", {"hidden_size": 100000}),
        ("set_mlp_activation", {"activation": "custom"}),
        ("convert_mlp_to_moe", {"num_experts": 2, "top_k": 3}),
    ]
    for operation, parameters in invalid:
        def provider(**kwargs):
            return [{
                "operation": operation,
                "parameters": parameters,
                "hypothesis": "Try an invalid architecture edit.",
                "rationale": "This must be rejected locally.",
                "expected_effect": {"quality": "measure"},
                "falsification_criteria": ["Validation rejects this proposal"],
                "risk": "low", "estimated_value": 0.5, "feasibility": 0.8,
            }]
        with pytest.raises(ValueError):
            MasterIntelligenceEngine(
                backend=CallableReasoningBackend(provider), max_plans=10
            ).plan(
                model_id="m", architecture=sample_architecture(), objective="reduce latency"
            )
