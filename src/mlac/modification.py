"""Transactional, allow-listed PyTorch architecture modifications.

Only explicitly supported module families and operations are accepted. The
source module is never mutated: a deep-copied candidate is transformed and
returned only after all operations and structural checks succeed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Architecture modification requires 'pip install mlac[torch]'") from exc

from mlac.transformer import PrunableSelfAttention, TransformerBlock, prune_attention_heads


_SUPPORTED_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


@dataclass(frozen=True)
class Modification:
    """One allow-listed operation and its JSON-like configuration."""

    operation: str
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModificationPlan:
    """Ordered operations applied atomically to an isolated model copy."""

    operations: tuple[Modification, ...]
    name: str = "architecture-modification"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("modification plan name must be non-empty")
        if not self.operations:
            raise ValueError("modification plan must contain at least one operation")


@dataclass(frozen=True)
class ModificationRecord:
    operation: str
    details: Mapping[str, Any]
    parameters_before: int
    parameters_after: int


@dataclass(frozen=True)
class ModificationResult:
    module: nn.Module
    plan_name: str
    parameters_before: int
    parameters_after: int
    records: tuple[ModificationRecord, ...]

    @property
    def parameter_delta(self) -> int:
        return self.parameters_after - self.parameters_before


class MixtureOfExperts(nn.Module):
    """Token-wise top-k sparse router over independent feed-forward experts.

    Input and output retain the final hidden dimension. Expert copies are
    lightly perturbed by default to break symmetry; conversion therefore
    requires task evaluation and usually fine-tuning before promotion.
    """

    def __init__(
        self,
        expert: nn.Module,
        *,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        expert_noise_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_experts < 2:
            raise ValueError("d_model must be positive and num_experts must be at least 2")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if not isinstance(expert_noise_std, (int, float)) or not torch.isfinite(torch.tensor(float(expert_noise_std))) or expert_noise_std < 0:
            raise ValueError("expert_noise_std must be a finite non-negative number")
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router = nn.Linear(self.d_model, self.num_experts)
        self.experts = nn.ModuleList(deepcopy(expert) for _ in range(self.num_experts))
        if expert_noise_std:
            with torch.no_grad():
                for index, expert_copy in enumerate(self.experts):
                    if index == 0:
                        continue
                    generator = torch.Generator(device=next(expert_copy.parameters()).device)
                    generator.manual_seed(index)
                    for parameter in expert_copy.parameters():
                        if parameter.ndim > 1:
                            noise = torch.randn(
                                parameter.shape, dtype=parameter.dtype,
                                device=parameter.device, generator=generator,
                            )
                            parameter.add_(noise * expert_noise_std)

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.d_model:
            raise ValueError(
                f"MoE expected final dimension {self.d_model}, received {value.shape[-1]}"
            )
        original_shape = value.shape
        flat = value.reshape(-1, self.d_model)
        logits = self.router(flat)
        selected_logits, selected_experts = torch.topk(logits, self.top_k, dim=-1)
        selected_weights = torch.softmax(selected_logits, dim=-1)
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            positions = (selected_experts == expert_index).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            token_indices = positions[:, 0]
            slots = positions[:, 1]
            expert_input = flat.index_select(0, token_indices)
            expert_output = expert(expert_input)
            weights = selected_weights[token_indices, slots].unsqueeze(-1)
            output.index_add_(0, token_indices, expert_output * weights)
        return output.reshape(original_shape)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _linear_neighbors(layers: Sequence[nn.Module], index: int) -> tuple[nn.Linear | None, nn.Linear | None]:
    previous = next(
        (layer for layer in reversed(layers[:index]) if isinstance(layer, nn.Linear)),
        None,
    )
    following = next(
        (layer for layer in layers[index:] if isinstance(layer, nn.Linear)),
        None,
    )
    return previous, following


def _resize_mlp_hidden(block: TransformerBlock, hidden_size: int) -> dict[str, Any]:
    if type(hidden_size) is not int or hidden_size < 1:
        raise ValueError("hidden_size must be a positive integer")
    if not isinstance(block.mlp, nn.Sequential) or len(block.mlp) < 2:
        raise TypeError("TransformerBlock MLP must be a Sequential with two Linear layers")
    old_in = block.mlp[0]
    old_out = block.mlp[-1]
    if not isinstance(old_in, nn.Linear) or not isinstance(old_out, nn.Linear):
        raise TypeError("TransformerBlock MLP must start and end with Linear layers")
    old_hidden = old_in.out_features
    if old_hidden == hidden_size:
        return {"old_hidden_size": old_hidden, "new_hidden_size": hidden_size, "changed": False}

    new_in = nn.Linear(old_in.in_features, hidden_size, bias=old_in.bias is not None)
    new_out = nn.Linear(hidden_size, old_out.out_features, bias=old_out.bias is not None)
    new_in = new_in.to(device=old_in.weight.device, dtype=old_in.weight.dtype)
    new_out = new_out.to(device=old_out.weight.device, dtype=old_out.weight.dtype)
    overlap = min(old_hidden, hidden_size)
    with torch.no_grad():
        new_in.weight[:overlap].copy_(old_in.weight[:overlap])
        if old_in.bias is not None:
            new_in.bias[:overlap].copy_(old_in.bias[:overlap])
        new_out.weight[:, :overlap].copy_(old_out.weight[:, :overlap])
        if old_out.bias is not None:
            new_out.bias.copy_(old_out.bias)
    layers = list(block.mlp.children())
    layers[0] = new_in
    layers[-1] = new_out
    block.mlp = nn.Sequential(*layers)
    block.mlp_ratio = max(1, hidden_size // block.d_model)
    return {"old_hidden_size": old_hidden, "new_hidden_size": hidden_size, "copied_channels": overlap}


def _resize_attention_heads(block: TransformerBlock, num_heads: int) -> dict[str, Any]:
    if type(num_heads) is not int or num_heads < 1:
        raise ValueError("num_heads must be a positive integer")
    old = block.attention
    if not isinstance(old, PrunableSelfAttention):
        raise TypeError("TransformerBlock attention must be PrunableSelfAttention")
    if old.num_heads == num_heads:
        return {"old_num_heads": old.num_heads, "new_num_heads": num_heads, "changed": False}
    new = PrunableSelfAttention(
        old.d_model, num_heads, bias=old.qkv.bias is not None, head_dim=old.head_dim
    ).to(device=old.qkv.weight.device, dtype=old.qkv.weight.dtype)
    new.train(block.training)
    retained_count = min(old.num_heads, num_heads)
    indices = torch.arange(retained_count * old.head_dim, device=old.qkv.weight.device)
    retained_width = retained_count * old.head_dim
    with torch.no_grad():
        for section in range(3):
            old_start = section * old.inner_dim
            new_start = section * new.inner_dim
            new.qkv.weight[new_start : new_start + retained_width].copy_(
                old.qkv.weight[old_start : old_start + retained_width]
            )
            if old.qkv.bias is not None:
                new.qkv.bias[new_start : new_start + retained_width].copy_(
                    old.qkv.bias[old_start : old_start + retained_width]
                )
        new.out_proj.weight[:, : retained_count * old.head_dim].copy_(
            old.out_proj.weight.index_select(1, indices)
        )
        if old.out_proj.bias is not None:
            new.out_proj.bias.copy_(old.out_proj.bias)
    block.attention = new
    return {
        "old_num_heads": old.num_heads,
        "new_num_heads": num_heads,
        "retained_heads": retained_count,
        "head_dim": old.head_dim,
    }


def _add_transformer_block(module: nn.Sequential, config: Mapping[str, Any]) -> dict[str, Any]:
    index = int(config.get("index", len(module)))
    if not 0 <= index <= len(module):
        raise ValueError("insert index is outside the layer stack")
    if any(not isinstance(layer, TransformerBlock) for layer in module):
        raise TypeError("add_transformer_block requires a Sequential of TransformerBlock layers")
    reference = module[0] if len(module) else None
    d_model = int(config.get("d_model", reference.d_model if reference else 0))
    num_heads = int(config.get("num_heads", reference.attention.num_heads if reference else 0))
    mlp_ratio = int(config.get("mlp_ratio", reference.mlp_ratio if reference else 4))
    if d_model < 1 or num_heads < 1:
        raise ValueError("d_model and num_heads are required for an empty layer stack")
    if any(layer.d_model != d_model for layer in module):
        raise ValueError("all transformer blocks in a stack must share d_model")
    block = TransformerBlock(
        d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
        dropout=float(config.get("dropout", 0.0)),
    )
    if reference is not None:
        parameter = next(reference.parameters())
        block = block.to(device=parameter.device, dtype=parameter.dtype)
    block.train(module.training)
    was_training = module.training
    layers = list(module.children())
    layers.insert(index, block)
    module.__init__(*layers)
    module.train(was_training)
    return {"index": index, "d_model": d_model, "num_heads": num_heads, "mlp_ratio": mlp_ratio}


def _remove_layer(module: nn.Sequential, config: Mapping[str, Any]) -> dict[str, Any]:
    index = int(config["index"])
    layers = list(module.children())
    if not 0 <= index < len(layers):
        raise ValueError("remove index is outside the layer stack")
    if len(layers) == 1:
        raise ValueError("cannot remove the only layer from a Sequential")
    removed = layers[index]
    if isinstance(removed, nn.Linear):
        previous, following = _linear_neighbors(layers, index)
        if previous is not None and following is not None and previous.out_features != following.in_features:
            raise ValueError(
                "removing this Linear breaks adjacent feature dimensions; resize or replace neighbors explicitly"
            )
        if previous is None or following is None:
            raise ValueError("removing an endpoint Linear requires an explicit replacement architecture")
    if isinstance(removed, TransformerBlock):
        if any(not isinstance(layer, TransformerBlock) for layer in layers):
            raise TypeError("TransformerBlock removal requires a homogeneous transformer stack")
    was_training = module.training
    layers.pop(index)
    module.__init__(*layers)
    module.train(was_training)
    return {"index": index, "removed_type": type(removed).__name__}


def _apply_one(module: nn.Module, item: Modification) -> dict[str, Any]:
    operation, config = item.operation, item.config
    if operation == "prune_attention_heads":
        if not isinstance(module, TransformerBlock):
            raise TypeError("prune_attention_heads requires a TransformerBlock root")
        result = prune_attention_heads(module, config["heads"])
        module.__dict__.update(result.module.__dict__)
        return {"removed_heads": result.removed_heads, "retained_heads": result.retained_heads}
    if operation == "resize_attention_heads":
        if not isinstance(module, TransformerBlock):
            raise TypeError("resize_attention_heads requires a TransformerBlock root")
        return _resize_attention_heads(module, config["num_heads"])
    if operation == "resize_mlp_hidden":
        if not isinstance(module, TransformerBlock):
            raise TypeError("resize_mlp_hidden requires a TransformerBlock root")
        return _resize_mlp_hidden(module, config["hidden_size"])
    if operation == "set_mlp_activation":
        if not isinstance(module, TransformerBlock):
            raise TypeError("set_mlp_activation requires a TransformerBlock root")
        activation = str(config["activation"]).lower()
        activation_type = _SUPPORTED_ACTIVATIONS.get(activation)
        if activation_type is None:
            raise ValueError(f"unsupported MLP activation: {activation!r}")
        if not isinstance(module.mlp, nn.Sequential) or len(module.mlp) < 2:
            raise TypeError("TransformerBlock MLP does not have the supported Sequential structure")
        module.mlp[1] = activation_type()
        return {"activation": activation}
    if operation == "convert_mlp_to_moe":
        if not isinstance(module, TransformerBlock):
            raise TypeError("convert_mlp_to_moe requires a TransformerBlock root")
        if isinstance(module.mlp, MixtureOfExperts):
            raise ValueError("MLP is already a MixtureOfExperts module")
        experts = int(config["num_experts"])
        top_k = int(config.get("top_k", min(2, experts)))
        noise = float(config.get("expert_noise_std", 1e-3))
        parameter = next(module.mlp.parameters())
        moe = MixtureOfExperts(
            module.mlp, d_model=module.d_model, num_experts=experts,
            top_k=top_k, expert_noise_std=noise,
        ).to(device=parameter.device, dtype=parameter.dtype)
        moe.train(module.mlp.training)
        module.mlp = moe
        return {"num_experts": experts, "top_k": top_k, "expert_noise_std": noise}
    if operation == "add_transformer_block":
        if not isinstance(module, nn.Sequential):
            raise TypeError("add_transformer_block requires nn.Sequential root")
        return _add_transformer_block(module, config)
    if operation == "remove_layer":
        if not isinstance(module, nn.Sequential):
            raise TypeError("remove_layer requires nn.Sequential root")
        return _remove_layer(module, config)
    if operation == "add_linear_layer":
        if not isinstance(module, nn.Sequential):
            raise TypeError("add_linear_layer requires nn.Sequential root")
        layers = list(module.children())
        index = int(config.get("index", len(layers)))
        if not 0 <= index <= len(layers):
            raise ValueError("insert index is outside the layer stack")
        in_features, out_features = int(config["in_features"]), int(config["out_features"])
        if in_features < 1 or out_features < 1:
            raise ValueError("linear dimensions must be positive")
        previous, following = _linear_neighbors(layers, index)
        if previous is not None and previous.out_features != in_features:
            raise ValueError("new Linear input does not match preceding Linear output")
        if following is not None and following.in_features != out_features:
            raise ValueError("new Linear output does not match following Linear input")
        layer = nn.Linear(in_features, out_features, bias=bool(config.get("bias", True)))
        if layers:
            parameter = next(module.parameters(), None)
            if parameter is not None:
                layer = layer.to(device=parameter.device, dtype=parameter.dtype)
        was_training = module.training
        layers.insert(index, layer)
        module.__init__(*layers)
        module.train(was_training)
        return {"index": index, "in_features": in_features, "out_features": out_features}
    raise ValueError(f"unsupported architecture modification: {operation!r}")


def apply_modifications(module: nn.Module, plan: ModificationPlan) -> ModificationResult:
    """Apply an ordered plan atomically to an isolated model copy.

    Supported roots are TransformerBlock and nn.Sequential layer stacks.
    The caller's model is never mutated, including when a later operation fails.
    """
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    if not isinstance(plan, ModificationPlan):
        raise TypeError("plan must be a ModificationPlan")
    candidate = deepcopy(module)
    before = _parameter_count(module)
    records: list[ModificationRecord] = []
    for item in plan.operations:
        if not isinstance(item, Modification):
            raise TypeError("plan operations must be Modification instances")
        operation_before = _parameter_count(candidate)
        details = _apply_one(candidate, item)
        operation_after = _parameter_count(candidate)
        records.append(ModificationRecord(
            item.operation, details, operation_before, operation_after
        ))
    after = _parameter_count(candidate)
    return ModificationResult(candidate, plan.name, before, after, tuple(records))


def modify_model(module: nn.Module, *operations: Modification) -> ModificationResult:
    """Convenience API for an ordered set of allow-listed modifications."""
    return apply_modifications(module, ModificationPlan(tuple(operations)))
