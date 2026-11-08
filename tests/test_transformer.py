import pytest

torch = pytest.importorskip("torch")
from mlac.transformer import TransformerBlock, prune_attention_heads
from mlac.benchmark import benchmark_module


def test_head_pruning_migrates_weights_and_preserves_shape():
    torch.manual_seed(7)
    model = TransformerBlock(d_model=32, num_heads=4, mlp_ratio=2).eval()
    x = torch.randn(2, 9, 32)
    result = prune_attention_heads(model, [1])
    assert result.module.attention.num_heads == 3
    assert result.retained_heads == (0, 2, 3)
    assert result.parameter_reduction > 0
    assert result.parameters_after < result.parameters_before
    assert result.module(x).shape == model(x).shape
    assert model.attention.num_heads == 4


def test_pruning_rejects_invalid_head_sets():
    model = TransformerBlock(d_model=16, num_heads=4)
    with pytest.raises(ValueError):
        prune_attention_heads(model, [0, 1, 2, 3])
    with pytest.raises(ValueError):
        prune_attention_heads(model, [4])
    with pytest.raises(ValueError):
        prune_attention_heads(model, [])


def test_benchmark_reports_valid_output_and_restores_mode():
    model = TransformerBlock(d_model=16, num_heads=4)
    model.train()
    result = benchmark_module(model, torch.randn(2, 5, 16), iterations=2, warmup=1)
    assert result.finite_output
    assert result.output_shape == (2, 5, 16)
    assert result.parameter_count > 0
    assert model.training
