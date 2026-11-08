"""Transactional execution of declared MLAC-IR weight transformations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError("MLAC-IR weight transformations require 'pip install mlac[torch]'") from exc

from mlac.ir import Architecture, WeightTransformationRequirement


@dataclass(frozen=True)
class TransformationRecord:
    operation: str
    sources: tuple[str, ...]
    targets: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class WeightTransformationReport:
    parameters: dict[str, Tensor]
    records: tuple[TransformationRecord, ...]
    unchanged_parameters: tuple[str, ...]


def _execute(
    requirement: WeightTransformationRequirement,
    parameters: Mapping[str, Tensor],
) -> tuple[Tensor, ...]:
    operation, config = requirement.operation, requirement.config
    sources = [parameters[name] for name in requirement.source_parameters]

    if operation in {"identity", "copy"}:
        if len(sources) != len(requirement.target_parameters):
            raise ValueError("identity/copy requires one source per target")
        return tuple(value.clone() for value in sources)
    if operation in {"transpose", "t"}:
        if len(sources) != 1:
            raise ValueError("transpose requires exactly one source")
        dims = config.get("dims")
        value = sources[0]
        return (value.transpose(*dims) if dims is not None else value.transpose(-2, -1),)
    if operation == "permute":
        if len(sources) != 1:
            raise ValueError("permute requires exactly one source")
        return (sources[0].permute(*tuple(config["dims"])).contiguous(),)
    if operation == "reshape":
        if len(sources) != 1:
            raise ValueError("reshape requires exactly one source")
        return (sources[0].reshape(tuple(config["shape"])),)
    if operation in {"split", "split_qkv"}:
        if len(sources) != 1:
            raise ValueError("split requires exactly one source")
        axis = int(config.get("axis", 0))
        if "sections" in config:
            sections = tuple(int(value) for value in config["sections"])
            if sum(sections) != sources[0].shape[axis]:
                raise ValueError("split sections must sum to the source dimension")
            return tuple(torch.split(sources[0], sections, dim=axis))
        if "split_size" in config:
            return tuple(torch.split(sources[0], int(config["split_size"]), dim=axis))
        raise ValueError("split requires config.sections or config.split_size")
    if operation in {"concat", "concatenate"}:
        if len(sources) < 2:
            raise ValueError("concatenate requires at least two sources")
        return (torch.cat(sources, dim=int(config.get("axis", 0))),)
    if operation == "slice":
        if len(sources) != 1:
            raise ValueError("slice requires exactly one source")
        axis = int(config.get("axis", 0))
        start, end = int(config["start"]), int(config["end"])
        slices = [slice(None)] * sources[0].ndim
        slices[axis] = slice(start, end)
        return (sources[0][tuple(slices)].clone(),)
    if operation == "squeeze":
        if len(sources) != 1:
            raise ValueError("squeeze requires exactly one source")
        dim = config.get("dim")
        return (sources[0].squeeze() if dim is None else sources[0].squeeze(int(dim)),)
    if operation == "unsqueeze":
        if len(sources) != 1:
            raise ValueError("unsqueeze requires exactly one source")
        return (sources[0].unsqueeze(int(config["dim"])),)
    raise ValueError(f"unsupported weight transformation operation: {operation!r}")


def _validate_parameter_shape(name: str, value: Tensor,
                             mappings: Mapping[str, Any], tensor_specs: Mapping[str, Any]) -> None:
    mapping = mappings.get(name)
    if mapping is None:
        raise ValueError(f"parameter {name!r} has no IR mapping")
    spec = tensor_specs.get(mapping.tensor_name)
    if spec is None:
        raise ValueError(f"parameter {name!r} references undeclared tensor {mapping.tensor_name!r}")
    if len(spec.shape) != value.ndim:
        raise ValueError(
            f"parameter {name!r} rank {value.ndim} does not match declared rank {len(spec.shape)}"
        )
    for actual, expected in zip(value.shape, spec.shape):
        if isinstance(expected, int) and actual != expected:
            raise ValueError(
                f"parameter {name!r} shape {tuple(value.shape)} does not match declared shape {spec.shape}"
            )
    if mapping.shape and tuple(mapping.shape) != tuple(value.shape):
        raise ValueError(f"parameter {name!r} violates its parameter mapping shape contract")


def apply_weight_transformations(
    architecture: Architecture,
    parameters: Mapping[str, Tensor],
    *,
    strict: bool = True,
) -> WeightTransformationReport:
    """Apply declared transforms atomically to a copy of a named parameter map.

    Inputs are never mutated. The original mapping is unchanged if any
    transformation, source check, or target-shape validation fails.
    """
    architecture.assert_valid()
    working = dict(parameters)
    original_names = set(parameters)
    tensor_specs = {item.name: item for item in architecture.tensors}
    mappings = {item.parameter_name: item for item in architecture.parameter_mappings}
    records: list[TransformationRecord] = []

    for requirement in architecture.weight_transformations:
        missing = [name for name in requirement.source_parameters if name not in working]
        if missing:
            if strict:
                raise KeyError(f"missing transformation sources: {', '.join(missing)}")
            continue
        for source_name in requirement.source_parameters:
            source = working[source_name]
            if not isinstance(source, torch.Tensor):
                raise TypeError(f"transformation source {source_name!r} is not a torch.Tensor")
            _validate_parameter_shape(source_name, source, mappings, tensor_specs)
        outputs = _execute(requirement, working)
        if len(outputs) != len(requirement.target_parameters):
            raise ValueError(
                f"{requirement.operation!r} produced {len(outputs)} tensors for "
                f"{len(requirement.target_parameters)} targets"
            )
        for target_name, output in zip(requirement.target_parameters, outputs):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"transformation target {target_name!r} is not a torch.Tensor")
            _validate_parameter_shape(target_name, output, mappings, tensor_specs)
            working[target_name] = output.clone()

        records.append(TransformationRecord(
            requirement.operation,
            requirement.source_parameters,
            requirement.target_parameters,
            tuple(tuple(int(dim) for dim in item.shape) for item in outputs),
        ))

    return WeightTransformationReport(
        parameters=working,
        records=tuple(records),
        unchanged_parameters=tuple(sorted(original_names - {
            target for record in records for target in record.targets
        })),
    )
