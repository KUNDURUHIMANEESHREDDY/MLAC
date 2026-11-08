import pytest

torch = pytest.importorskip("torch")

from mlac.ir import (
    Architecture, ParameterMapping, TensorSpec, WeightTransformationRequirement,
)
from mlac.weight_transforms import apply_weight_transformations


def _split_architecture(target_shape=(2, 2)):
    return Architecture(
        name="split-qkv",
        family="transformer",
        tensors=(
            TensorSpec("qkv", (6, 2), "fp32"),
            TensorSpec("q", target_shape, "fp32"),
            TensorSpec("k", (2, 2), "fp32"),
            TensorSpec("v", (2, 2), "fp32"),
        ),
        parameter_mappings=(
            ParameterMapping("fused.qkv", "qkv", shape=(6, 2)),
            ParameterMapping("q", "q", shape=target_shape),
            ParameterMapping("k", "k", shape=(2, 2)),
            ParameterMapping("v", "v", shape=(2, 2)),
        ),
        weight_transformations=(
            WeightTransformationRequirement(
                "split_qkv", ("fused.qkv",), ("q", "k", "v"),
                {"axis": 0, "sections": [2, 2, 2]},
            ),
        ),
    )


def test_split_qkv_executes_atomically_and_preserves_original():
    architecture = _split_architecture()
    source = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    original = {"fused.qkv": source}
    result = apply_weight_transformations(architecture, original)
    assert torch.equal(original["fused.qkv"], source)
    assert torch.equal(result.parameters["q"], source[:2])
    assert torch.equal(result.parameters["k"], source[2:4])
    assert torch.equal(result.parameters["v"], source[4:])
    assert result.records[0].operation == "split_qkv"
    assert "fused.qkv" in result.unchanged_parameters


def test_shape_contract_failure_does_not_mutate_input():
    architecture = _split_architecture(target_shape=(3, 2))
    source = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    original = {"fused.qkv": source}
    with pytest.raises(ValueError, match="shape"):
        apply_weight_transformations(architecture, original)
    assert set(original) == {"fused.qkv"}
    assert torch.equal(original["fused.qkv"], source)


def test_missing_source_fails_closed():
    with pytest.raises(KeyError, match="missing transformation sources"):
        apply_weight_transformations(_split_architecture(), {})
