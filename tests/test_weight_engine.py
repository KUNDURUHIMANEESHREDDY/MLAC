import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mlac.weight_engine import (
    LoRALinear,
    WeightOptimizationPlan,
    WeightOptimizationStep,
    distillation_loss,
    measure_sparsity,
    migrate_checkpoint,
    optimize_weights,
)


def plan(*steps):
    return WeightOptimizationPlan(tuple(steps))


def test_unstructured_pruning_is_transactional_and_measurable():
    model = nn.Sequential(nn.Linear(8, 8, bias=False))
    original = model[0].weight.detach().clone()
    result = optimize_weights(model, plan(
        WeightOptimizationStep("prune_unstructured", "0.weight", {"amount": 0.25}),
    ))
    assert result.sparsity_after.zero_parameters == 16
    assert result.sparsity_after.sparsity == pytest.approx(0.25)
    assert torch.equal(model[0].weight, original)
    assert result.records[0].metrics["pruned_parameters"] == 16


def test_structured_output_pruning_zeros_weight_rows_and_bias():
    model = nn.Sequential(nn.Linear(5, 4))
    result = optimize_weights(model, plan(
        WeightOptimizationStep("prune_structured", "0.weight", {
            "dimension": "output", "indices": [1, 3],
        }),
    ))
    assert torch.count_nonzero(result.model[0].weight[[1, 3]]) == 0
    assert torch.count_nonzero(result.model[0].bias[[1, 3]]) == 0
    assert result.records[0].metrics["pruned_indices"] == (1, 3)


def test_fake_int8_quantization_reports_error_and_keeps_float_storage():
    model = nn.Linear(4, 3)
    result = optimize_weights(model, plan(
        WeightOptimizationStep("fake_quantize_int8", "weight"),
    ))
    assert result.model.weight.dtype == torch.float32
    assert result.records[0].metrics["scheme"] == "symmetric_per_tensor_int8_fake_quant"
    assert result.records[0].metrics["max_absolute_error"] >= 0
    assert not torch.equal(result.model.weight, model.weight)


def test_lora_injection_forward_merge_and_unmerge():
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(5, 4)).eval()
    inputs = torch.randn(2, 3, 5)
    injected = optimize_weights(model, plan(
        WeightOptimizationStep("inject_lora", "0", {"rank": 2, "alpha": 4}),
    )).model
    assert isinstance(injected[0], LoRALinear)
    with torch.no_grad():
        injected[0].lora_B.normal_(std=0.1)
    unmerged_output = injected(inputs)
    merged = optimize_weights(injected, plan(
        WeightOptimizationStep("merge_lora", "0"),
    )).model
    assert merged[0].merged
    assert torch.allclose(merged(inputs), unmerged_output, atol=1e-6, rtol=1e-5)
    unmerged = optimize_weights(merged, plan(
        WeightOptimizationStep("unmerge_lora", "0"),
    )).model
    assert not unmerged[0].merged
    assert torch.allclose(unmerged(inputs), unmerged_output, atol=1e-6, rtol=1e-5)


def test_distillation_loss_is_differentiable_and_teacher_is_detached():
    student = torch.randn(3, 7, requires_grad=True)
    teacher = torch.randn(3, 7, requires_grad=True)
    labels = torch.tensor([1, 2, 3])
    losses = distillation_loss(student, teacher, labels=labels, temperature=2.5, alpha=0.7)
    losses["loss"].backward()
    assert student.grad is not None
    assert teacher.grad is None
    assert losses["loss"].ndim == 0


def test_distillation_rejects_mismatched_logits():
    with pytest.raises(ValueError, match="identical shapes"):
        distillation_loss(torch.randn(2, 3), torch.randn(2, 4))


def test_checkpoint_migration_requires_exact_tensor_contracts():
    source = {"old.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    target = {
        "new.weight": torch.zeros(2, 3),
        "new.bias": torch.zeros(2),
    }
    report = migrate_checkpoint(source, target, {"new.weight": "old.weight"})
    assert torch.equal(report.state_dict["new.weight"], source["old.weight"])
    assert report.missing_targets == ("new.bias",)
    with pytest.raises(ValueError, match="unmapped target"):
        migrate_checkpoint(source, target, {"new.weight": "old.weight"}, strict=True)


def test_checkpoint_migration_rejects_shape_and_dtype_mismatch():
    source = {"old.weight": torch.zeros(3, 2, dtype=torch.float32)}
    target = {"new.weight": torch.zeros(2, 3, dtype=torch.float32)}
    with pytest.raises(ValueError, match="contract mismatch"):
        migrate_checkpoint(source, target, {"new.weight": "old.weight"})


def test_failed_plan_does_not_expose_partially_modified_candidate():
    model = nn.Sequential(nn.Linear(4, 4))
    before = model[0].weight.detach().clone()
    with pytest.raises(ValueError, match="unsupported weight optimization"):
        optimize_weights(model, plan(
            WeightOptimizationStep("prune_unstructured", "0.weight", {"amount": 0.2}),
            WeightOptimizationStep("run_python", "0"),
        ))
    assert torch.equal(model[0].weight, before)


def test_sparsity_measurement_counts_parameter_zeros_only():
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.zero_()
    metrics = measure_sparsity(model)
    assert metrics.zero_parameters == model.weight.numel()
    assert metrics.total_parameters == model.weight.numel() + model.bias.numel()
