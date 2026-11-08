import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mlac.modification import (
    Modification,
    ModificationPlan,
    MixtureOfExperts,
    apply_modifications,
    modify_model,
)
from mlac.transformer import TransformerBlock


def test_resize_mlp_hidden_and_activation_preserve_external_shape():
    torch.manual_seed(3)
    model = TransformerBlock(d_model=16, num_heads=4, mlp_ratio=2).eval()
    sample = torch.randn(2, 7, 16)
    before = model(sample)
    result = modify_model(
        model,
        Modification("resize_mlp_hidden", {"hidden_size": 48}),
        Modification("set_mlp_activation", {"activation": "silu"}),
    )
    assert result.module.mlp[0].out_features == 48
    assert isinstance(result.module.mlp[1], nn.SiLU)
    assert result.module(sample).shape == before.shape
    assert model.mlp[0].out_features == 32
    assert result.parameter_delta > 0


def test_attention_head_pruning_and_expansion_migrate_common_weights():
    model = TransformerBlock(d_model=16, num_heads=4, mlp_ratio=2).eval()
    sample = torch.randn(2, 5, 16)
    pruned = modify_model(model, Modification("prune_attention_heads", {"heads": [1]}))
    assert pruned.module.attention.num_heads == 3
    assert pruned.module(sample).shape == sample.shape
    expanded = modify_model(
        pruned.module, Modification("resize_attention_heads", {"num_heads": 5})
    )
    assert expanded.module.attention.num_heads == 5
    assert expanded.module(sample).shape == sample.shape
    old_attention = pruned.module.attention
    new_attention = expanded.module.attention
    for section in range(3):
        old_start = section * old_attention.inner_dim
        new_start = section * new_attention.inner_dim
        assert torch.equal(
            new_attention.qkv.weight[new_start:new_start + old_attention.inner_dim],
            old_attention.qkv.weight[old_start:old_start + old_attention.inner_dim],
        )
    assert model.attention.num_heads == 4


def test_moe_conversion_is_topk_routed_and_differentiable():
    torch.manual_seed(5)
    model = TransformerBlock(d_model=16, num_heads=4, mlp_ratio=2)
    result = modify_model(
        model,
        Modification("convert_mlp_to_moe", {
            "num_experts": 3, "top_k": 2, "expert_noise_std": 1e-3,
        }),
    )
    assert isinstance(result.module.mlp, MixtureOfExperts)
    assert result.module.mlp.num_experts == 3
    assert result.module.mlp.top_k == 2
    output = result.module(torch.randn(2, 6, 16))
    assert output.shape == (2, 6, 16)
    output.square().mean().backward()
    assert result.module.mlp.router.weight.grad is not None
    assert model.mlp.__class__ is nn.Sequential


def test_transformer_stack_can_add_and_remove_layers():
    stack = nn.Sequential(
        TransformerBlock(d_model=16, num_heads=4),
        TransformerBlock(d_model=16, num_heads=4),
    ).eval()
    sample = torch.randn(2, 4, 16)
    result = modify_model(
        stack,
        Modification("add_transformer_block", {"index": 1}),
        Modification("remove_layer", {"index": 1}),
    )
    assert len(result.module) == 2
    assert result.module(sample).shape == sample.shape
    assert not result.module.training
    assert all(not layer.training for layer in result.module)
    assert len(stack) == 2


def test_linear_layer_can_be_inserted_and_removed_when_dimensions_match():
    stack = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
    sample = torch.randn(3, 4)
    result = modify_model(
        stack,
        Modification("add_linear_layer", {
            "index": 2, "in_features": 8, "out_features": 8,
        }),
        Modification("remove_layer", {"index": 2}),
    )
    assert len(result.module) == 3
    assert result.module(sample).shape == (3, 2)


def test_incompatible_linear_layer_removal_and_insertions_fail_closed():
    stack = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    with pytest.raises(ValueError, match="endpoint Linear"):
        modify_model(stack, Modification("remove_layer", {"index": 0}))
    with pytest.raises(ValueError, match="preceding Linear output"):
        modify_model(
            stack,
            Modification("add_linear_layer", {
                "index": 1, "in_features": 7, "out_features": 8,
            }),
        )


def test_modification_plan_is_atomic_and_does_not_mutate_source():
    model = TransformerBlock(d_model=16, num_heads=4, mlp_ratio=2)
    original_hidden = model.mlp[0].out_features
    plan = ModificationPlan((
        Modification("resize_mlp_hidden", {"hidden_size": 64}),
        Modification("execute_arbitrary_code", {}),
    ))
    with pytest.raises(ValueError, match="unsupported architecture modification"):
        apply_modifications(model, plan)
    assert model.mlp[0].out_features == original_hidden


def test_moe_rejects_invalid_router_configuration():
    expert = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
    with pytest.raises(ValueError, match="at least 2"):
        MixtureOfExperts(expert, d_model=4, num_experts=1)
    with pytest.raises(ValueError, match="top_k"):
        MixtureOfExperts(expert, d_model=4, num_experts=3, top_k=4)
