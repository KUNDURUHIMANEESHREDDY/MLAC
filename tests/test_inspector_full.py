import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mlac.inspector import (
    estimate_kv_cache_bytes,
    inspect_torch_model,
    inspect_torch_model_full,
)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.num_key_value_heads = 2
        self.head_dim = 8
        self.q_proj = nn.Linear(32, 32)
        self.k_proj = nn.Linear(32, 16)
        self.v_proj = nn.Linear(32, 16)

    def forward(self, x):
        return self.q_proj(x) + self.k_proj(x).repeat_interleave(2, dim=-1) + self.v_proj(x).repeat_interleave(2, dim=-1)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = TinyAttention()
        self.norm = nn.LayerNorm(32)

    def forward(self, x):
        return self.norm(self.attention(x))


def test_basic_inspector_remains_compatible_and_reports_parameter_count():
    model = TinyModel()
    architecture = inspect_torch_model(model)
    assert architecture.framework == "pytorch"
    assert architecture.dimensions["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert any(component.name == "attention" for component in architecture.components)


def test_full_inspection_traces_graph_and_attention_with_shapes():
    model = TinyModel().eval()
    inputs = (torch.randn(2, 5, 32),)
    report = inspect_torch_model_full(model, example_inputs=inputs)
    assert report.trace_error is None
    assert report.graph_nodes
    assert report.graph_edges
    assert any(node.output_shape == (2, 5, 32) for node in report.graph_nodes)
    attention = next(item for item in report.attention if item.module_name == "attention")
    assert attention.num_heads == 4
    assert attention.num_kv_heads == 2
    assert attention.head_dim == 8
    assert attention.grouped_query_attention is True


def test_kv_cache_estimate_uses_explicit_dimensions_and_dtype():
    result = estimate_kv_cache_bytes(
        batch_size=2, sequence_length=1024, num_layers=24,
        num_kv_heads=8, head_dim=128, dtype=torch.bfloat16,
    )
    assert result.bytes_per_element == 2
    assert result.total_bytes == 2 * 2 * 1024 * 24 * 8 * 128 * 2


def test_kv_cache_estimate_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="positive integer"):
        estimate_kv_cache_bytes(batch_size=0, sequence_length=10, num_layers=2,
                                num_kv_heads=2, head_dim=8)


def test_memory_profile_reports_parameter_buffer_input_and_activation_bytes():
    model = TinyModel().train()
    model.attention.eval()
    inputs = (torch.randn(2, 5, 32),)
    report = inspect_torch_model_full(model, example_inputs=inputs, profile_memory=True)
    assert report.memory is not None
    assert report.memory.parameter_bytes == sum(p.numel() * p.element_size() for p in model.parameters())
    assert report.memory.input_bytes == inputs[0].numel() * inputs[0].element_size()
    assert report.memory.activation_output_bytes > 0
    assert model.training is True
    assert model.attention.training is False


def test_full_inspector_rejects_non_tuple_example_inputs():
    with pytest.raises(TypeError, match="tuple"):
        inspect_torch_model_full(TinyModel(), example_inputs=torch.randn(2, 5, 32))
