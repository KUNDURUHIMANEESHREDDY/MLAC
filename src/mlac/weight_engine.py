"""Unified, transactional weight optimization for supported PyTorch modules.

The engine deliberately uses explicit module paths and allow-listed operations.
It never mutates the caller's model. Quantization is fake/simulated INT8 unless
an operator separately exports and validates a backend-specific quantized model.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("Weight optimization requires 'pip install mlac[torch]'") from exc


@dataclass(frozen=True)
class WeightOptimizationStep:
    operation: str
    target: str
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WeightOptimizationPlan:
    steps: tuple[WeightOptimizationStep, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("weight optimization plan must contain at least one step")
        if any(not isinstance(step, WeightOptimizationStep) for step in self.steps):
            raise TypeError("steps must contain WeightOptimizationStep values")


@dataclass(frozen=True)
class SparsityMetrics:
    zero_parameters: int
    total_parameters: int
    sparsity: float


@dataclass(frozen=True)
class WeightOptimizationRecord:
    operation: str
    target: str
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class WeightOptimizationResult:
    model: nn.Module
    records: tuple[WeightOptimizationRecord, ...]
    sparsity_before: SparsityMetrics
    sparsity_after: SparsityMetrics
    parameters_before: int
    parameters_after: int


@dataclass(frozen=True)
class CheckpointMigrationReport:
    state_dict: dict[str, Tensor]
    copied: tuple[tuple[str, str], ...]
    skipped: tuple[str, ...]
    missing_targets: tuple[str, ...]


def measure_sparsity(model: nn.Module) -> SparsityMetrics:
    """Measure exact zeros across parameters (not buffers or optimizer state)."""
    total = zero = 0
    for parameter in model.parameters():
        total += parameter.numel()
        zero += int(torch.count_nonzero(parameter.detach() == 0).item())
    return SparsityMetrics(zero, total, (zero / total) if total else 0.0)


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _resolve_parent(model: nn.Module, path: str) -> tuple[nn.Module, str, nn.Module]:
    if not path or path.startswith(".") or path.endswith("."):
        raise ValueError("target must be a non-empty module path")
    parts = path.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        if part not in parent._modules:
            raise KeyError(f"unknown module path {path!r}")
        parent = parent._modules[part]
    name = parts[-1]
    child = parent._modules.get(name)
    if child is None:
        raise KeyError(f"unknown module path {path!r}")
    return parent, name, child


def _resolve_parameter(model: nn.Module, path: str) -> tuple[nn.Parameter, nn.Module, str]:
    if not path:
        raise ValueError("parameter target must be non-empty")
    if "." in path:
        module_path, parameter_name = path.rsplit(".", 1)
        module = model.get_submodule(module_path)
    else:
        module, parameter_name = model, path
    parameter = getattr(module, parameter_name, None)
    if not isinstance(parameter, nn.Parameter):
        raise KeyError(f"{path!r} is not a registered parameter")
    return parameter, module, parameter_name


class LoRALinear(nn.Module):
    """Linear layer with an explicit low-rank residual adapter."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank < 1 or rank > min(base.in_features, base.out_features):
            raise ValueError("LoRA rank must be within [1, min(in_features, out_features)]")
        if alpha <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA alpha must be positive and dropout must be in [0, 1)")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                                dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                               dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.merged = False

    def delta_weight(self) -> Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self) -> None:
        if not self.merged:
            with torch.no_grad():
                self.base.weight.add_(self.delta_weight().to(self.base.weight))
            self.merged = True

    def unmerge(self) -> None:
        if self.merged:
            with torch.no_grad():
                self.base.weight.sub_(self.delta_weight().to(self.base.weight))
            self.merged = False

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.base(inputs)
        if self.merged:
            return output
        adapter = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return output + adapter * self.scaling


def _apply_step(model: nn.Module, step: WeightOptimizationStep) -> WeightOptimizationRecord:
    op, target, config = step.operation, step.target, dict(step.config)
    if op == "prune_unstructured":
        parameter, _, _ = _resolve_parameter(model, target)
        amount = float(config.get("amount", 0.1))
        if not 0.0 < amount < 1.0:
            raise ValueError("pruning amount must be strictly between 0 and 1")
        flat = parameter.detach().abs().flatten()
        count = max(1, min(flat.numel() - 1, int(flat.numel() * amount)))
        threshold = torch.kthvalue(flat, count).values
        mask = parameter.detach().abs() > threshold
        # Stable tie handling: fill any remaining quota by deterministic flat index.
        if int((~mask).sum().item()) > count:
            keep = torch.ones(flat.numel(), dtype=torch.bool, device=flat.device)
            indices = torch.argsort(flat, stable=True)[:count]
            keep[indices] = False
            mask = keep.view_as(parameter)
        with torch.no_grad():
            parameter.mul_(mask)
        return WeightOptimizationRecord(op, target, {
            "requested_fraction": amount, "pruned_parameters": int((~mask).sum().item()),
        })

    if op == "prune_structured":
        parameter, module, parameter_name = _resolve_parameter(model, target)
        if not isinstance(module, nn.Linear) or parameter_name != "weight":
            raise TypeError("structured pruning currently supports nn.Linear.weight only")
        dimension = str(config.get("dimension", "output"))
        indices = tuple(sorted(set(int(value) for value in config.get("indices", ()))))
        axis = 0 if dimension == "output" else 1 if dimension == "input" else -1
        if axis < 0:
            raise ValueError("structured pruning dimension must be 'input' or 'output'")
        limit = parameter.shape[axis]
        if not indices or any(index < 0 or index >= limit for index in indices) or len(indices) >= limit:
            raise ValueError("structured pruning indices must be valid and leave at least one channel")
        with torch.no_grad():
            parameter.index_fill_(axis, torch.tensor(indices, device=parameter.device), 0)
            if axis == 0 and module.bias is not None:
                module.bias.index_fill_(0, torch.tensor(indices, device=module.bias.device), 0)
        return WeightOptimizationRecord(op, target, {
            "dimension": dimension, "pruned_indices": indices,
        })

    if op == "fake_quantize_int8":
        parameter, _, _ = _resolve_parameter(model, target)
        if not parameter.is_floating_point():
            raise TypeError("fake INT8 quantization requires a floating-point parameter")
        original = parameter.detach().clone()
        max_abs = float(original.abs().max().item()) if original.numel() else 0.0
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        quantized = torch.clamp(torch.round(original / scale), -127, 127)
        restored = quantized * scale
        with torch.no_grad():
            parameter.copy_(restored)
        return WeightOptimizationRecord(op, target, {
            "scheme": "symmetric_per_tensor_int8_fake_quant",
            "scale": scale,
            "estimated_int8_bytes": original.numel(),
            "stored_parameter_dtype": str(parameter.dtype),
            "max_absolute_error": float((original - restored).abs().max().item()) if original.numel() else 0.0,
        })

    if op == "inject_lora":
        parent, name, module = _resolve_parent(model, target)
        if not isinstance(module, nn.Linear):
            raise TypeError("LoRA injection currently supports nn.Linear targets only")
        if isinstance(module, LoRALinear):
            raise ValueError("target already has a LoRA adapter")
        rank = int(config.get("rank", 4))
        alpha = float(config.get("alpha", rank))
        dropout = float(config.get("dropout", 0.0))
        parent._modules[name] = LoRALinear(module, rank, alpha, dropout)
        return WeightOptimizationRecord(op, target, {
            "rank": rank, "alpha": alpha, "trainable_adapter_parameters": rank * (module.in_features + module.out_features),
        })

    if op in {"merge_lora", "unmerge_lora"}:
        module = model.get_submodule(target)
        if not isinstance(module, LoRALinear):
            raise TypeError(f"{op} requires a LoRALinear target")
        module.merge() if op == "merge_lora" else module.unmerge()
        return WeightOptimizationRecord(op, target, {"merged": module.merged})

    raise ValueError(f"unsupported weight optimization operation: {op!r}")


def optimize_weights(model: nn.Module, plan: WeightOptimizationPlan) -> WeightOptimizationResult:
    """Apply an allow-listed plan to a deep copy; return only a fully valid candidate."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(plan, WeightOptimizationPlan):
        raise TypeError("plan must be a WeightOptimizationPlan")
    candidate = deepcopy(model)
    before = measure_sparsity(model)
    parameters_before = _parameter_count(model)
    records = tuple(_apply_step(candidate, step) for step in plan.steps)
    after = measure_sparsity(candidate)
    return WeightOptimizationResult(
        candidate, records, before, after, parameters_before, _parameter_count(candidate),
    )


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    labels: Tensor | None = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> dict[str, Tensor]:
    """Compute temperature-scaled teacher KL plus optional supervised CE.

    Teacher logits are detached. This is a loss primitive, not a trainer or a
    complete dataset/optimizer/checkpoint workflow.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.ndim < 2:
        raise ValueError("logits must include batch and class dimensions")
    if temperature <= 0 or not 0.0 <= alpha <= 1.0:
        raise ValueError("temperature must be positive and alpha must be in [0, 1]")
    t = float(temperature)
    soft_targets = F.softmax(teacher_logits.detach() / t, dim=-1)
    kl = F.kl_div(F.log_softmax(student_logits / t, dim=-1), soft_targets,
                  reduction="batchmean") * (t * t)
    if labels is None:
        total = kl
        ce = student_logits.new_zeros(())
    else:
        ce = F.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]),
                             labels.reshape(-1))
        total = alpha * kl + (1.0 - alpha) * ce
    return {"loss": total, "distillation_loss": kl, "supervised_loss": ce}


def migrate_checkpoint(
    source_state: Mapping[str, Tensor],
    target_state: Mapping[str, Tensor],
    mapping: Mapping[str, str],
    *,
    strict: bool = True,
) -> CheckpointMigrationReport:
    """Copy explicitly mapped tensors into a target-shaped checkpoint template.

    Mapping direction is target-key -> source-key. No implicit reshape, cast,
    transpose, or fuzzy name matching is performed.
    """
    result = {name: tensor.detach().clone() for name, tensor in target_state.items()}
    copied: list[tuple[str, str]] = []
    skipped: list[str] = []
    for target_name, source_name in mapping.items():
        if target_name not in target_state:
            if strict:
                raise KeyError(f"unknown target checkpoint key {target_name!r}")
            skipped.append(f"{target_name}: unknown target")
            continue
        if source_name not in source_state:
            if strict:
                raise KeyError(f"missing source checkpoint key {source_name!r}")
            skipped.append(f"{target_name}: missing source {source_name}")
            continue
        source, target = source_state[source_name], target_state[target_name]
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("checkpoint values must be torch.Tensor instances")
        if source.shape != target.shape or source.dtype != target.dtype:
            if strict:
                raise ValueError(
                    f"checkpoint contract mismatch for {target_name!r}: "
                    f"source {tuple(source.shape)}/{source.dtype}, target {tuple(target.shape)}/{target.dtype}"
                )
            skipped.append(f"{target_name}: shape/dtype mismatch")
            continue
        result[target_name] = source.detach().clone()
        copied.append((source_name, target_name))
    mapped_targets = {target for _, target in copied}
    missing = tuple(sorted(set(target_state) - mapped_targets))
    if strict and missing:
        raise ValueError(f"unmapped target checkpoint keys: {', '.join(missing)}")
    return CheckpointMigrationReport(result, tuple(copied), tuple(skipped), missing)
