"""Versioned, framework-neutral MLAC intermediate representation.

The IR describes architecture structure and the contracts needed to safely map,
initialize, transform, and execute model parameters. It deliberately does not
import a deep-learning framework.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any


IR_SCHEMA_VERSION = 1
_SUPPORTED_SEVERITIES = {"error", "warning"}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int | str, ...]
    dtype: str = "unknown"
    layout: str = "contiguous"
    trainable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tensor name must be non-empty")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError("tensor dtype must be non-empty")
        if not isinstance(self.layout, str) or not self.layout.strip():
            raise ValueError("tensor layout must be non-empty")
        if any(type(dim) is int and dim < 0 for dim in self.shape):
            raise ValueError("tensor dimensions must be non-negative")
        if any(type(dim) is not int and (not isinstance(dim, str) or not dim.strip()) for dim in self.shape):
            raise TypeError("tensor dimensions must be integers or non-empty symbolic strings")


@dataclass(frozen=True)
class Component:
    name: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.kind.strip():
            raise ValueError("component name and kind must be non-empty")


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    operation: str
    component_name: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id.strip() or not self.operation.strip():
            raise ValueError("graph node id and operation must be non-empty")


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    tensor_name: str | None = None
    source_port: str = "output"
    target_port: str = "input"

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise ValueError("graph edge endpoints must be non-empty")
        if not self.source_port.strip() or not self.target_port.strip():
            raise ValueError("graph edge ports must be non-empty")


@dataclass(frozen=True)
class ParameterMapping:
    """Maps a framework/checkpoint parameter key to an IR tensor."""

    parameter_name: str
    tensor_name: str
    component_name: str | None = None
    checkpoint_key: str | None = None
    shape: tuple[int | str, ...] = ()
    transform: str = "identity"

    def __post_init__(self) -> None:
        if not self.parameter_name.strip() or not self.tensor_name.strip():
            raise ValueError("parameter and tensor names must be non-empty")
        if any(type(dim) is int and dim < 0 for dim in self.shape):
            raise ValueError("parameter mapping dimensions must be non-negative")
        if any(type(dim) is not int and (not isinstance(dim, str) or not dim.strip()) for dim in self.shape):
            raise TypeError("parameter mapping dimensions must be integers or symbolic strings")


@dataclass(frozen=True)
class InitializationStrategy:
    target: str
    method: str
    config: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.target.strip() or not self.method.strip():
            raise ValueError("initialization target and method must be non-empty")


@dataclass(frozen=True)
class CompatibilityConstraint:
    target: str
    operator: str
    value: Any
    severity: str = "error"
    message: str = ""

    def __post_init__(self) -> None:
        if not self.target.strip() or not self.operator.strip():
            raise ValueError("constraint target and operator must be non-empty")
        if self.severity not in _SUPPORTED_SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(_SUPPORTED_SEVERITIES)}")


@dataclass(frozen=True)
class RuntimeRequirements:
    backend: str | None = None
    required_features: tuple[str, ...] = ()
    required_dtypes: tuple[str, ...] = ()
    minimum_memory_bytes: int | None = None
    maximum_sequence_length: int | None = None

    def __post_init__(self) -> None:
        if self.backend is not None and not self.backend.strip():
            raise ValueError("backend must be non-empty when specified")
        if self.minimum_memory_bytes is not None and self.minimum_memory_bytes < 0:
            raise ValueError("minimum_memory_bytes must be non-negative")
        if self.maximum_sequence_length is not None and self.maximum_sequence_length < 1:
            raise ValueError("maximum_sequence_length must be positive")


@dataclass(frozen=True)
class RuntimeCapabilities:
    backend: str
    features: tuple[str, ...] = ()
    supported_dtypes: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    available_memory_bytes: int | None = None
    maximum_sequence_length: int | None = None

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ValueError("runtime backend must be non-empty")
        if self.available_memory_bytes is not None and self.available_memory_bytes < 0:
            raise ValueError("available_memory_bytes must be non-negative")
        if self.maximum_sequence_length is not None and self.maximum_sequence_length < 1:
            raise ValueError("runtime maximum_sequence_length must be positive")


@dataclass(frozen=True)
class WeightTransformationRequirement:
    operation: str
    source_parameters: tuple[str, ...]
    target_parameters: tuple[str, ...]
    config: dict[str, Any] = field(default_factory=dict)
    preserves_function: bool = False

    def __post_init__(self) -> None:
        if not self.operation.strip():
            raise ValueError("weight transformation operation must be non-empty")
        if not self.source_parameters or not self.target_parameters:
            raise ValueError("weight transformations require source and target parameters")


@dataclass(frozen=True)
class Architecture:
    name: str
    family: str
    version: int = 1
    framework: str = "unknown"
    dimensions: dict[str, int | float | str] = field(default_factory=dict)
    components: tuple[Component, ...] = ()
    tensors: tuple[TensorSpec, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    graph_nodes: tuple[GraphNode, ...] = ()
    graph_edges: tuple[GraphEdge, ...] = ()
    parameter_mappings: tuple[ParameterMapping, ...] = ()
    initialization: tuple[InitializationStrategy, ...] = ()
    compatibility_constraints: tuple[CompatibilityConstraint, ...] = ()
    runtime_requirements: RuntimeRequirements = field(default_factory=RuntimeRequirements)
    weight_transformations: tuple[WeightTransformationRequirement, ...] = ()
    schema_version: int = IR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.family.strip():
            raise ValueError("Architecture name and family must be non-empty")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        if self.schema_version != IR_SCHEMA_VERSION:
            raise ValueError(f"unsupported MLAC-IR schema version: {self.schema_version}")

    def validate(self) -> tuple[str, ...]:
        """Return structural/reference errors without mutating the architecture."""
        errors: list[str] = []
        component_names = [item.name for item in self.components]
        tensor_names = [item.name for item in self.tensors]
        node_ids = [item.node_id for item in self.graph_nodes]
        parameter_names = [item.parameter_name for item in self.parameter_mappings]

        for label, values in (
            ("component", component_names), ("tensor", tensor_names),
            ("graph node", node_ids), ("parameter mapping", parameter_names),
        ):
            duplicates = sorted({value for value in values if values.count(value) > 1})
            if duplicates:
                errors.append(f"duplicate {label} identifiers: {', '.join(duplicates)}")

        component_set, tensor_set, node_set = set(component_names), set(tensor_names), set(node_ids)
        indegree = {node_id: 0 for node_id in node_set}
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_set}
        for edge in self.graph_edges:
            if edge.source in node_set and edge.target in node_set:
                adjacency[edge.source].append(edge.target)
                indegree[edge.target] += 1
        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            current = ready.pop()
            visited += 1
            for target in adjacency[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(node_set):
            errors.append("graph contains a cycle; MLAC-IR v1 requires an acyclic dataflow graph")
        for node in self.graph_nodes:
            if node.component_name and node.component_name not in component_set:
                errors.append(f"node {node.node_id!r} references unknown component {node.component_name!r}")
        for edge in self.graph_edges:
            if edge.source not in node_set:
                errors.append(f"edge references unknown source node {edge.source!r}")
            if edge.target not in node_set:
                errors.append(f"edge references unknown target node {edge.target!r}")
            if edge.tensor_name and self.tensors and edge.tensor_name not in tensor_set:
                errors.append(f"edge references unknown tensor {edge.tensor_name!r}")
        for mapping in self.parameter_mappings:
            if self.tensors and mapping.tensor_name not in tensor_set:
                errors.append(f"parameter {mapping.parameter_name!r} maps to unknown tensor {mapping.tensor_name!r}")
            if mapping.component_name and mapping.component_name not in component_set:
                errors.append(f"parameter {mapping.parameter_name!r} references unknown component {mapping.component_name!r}")
            if mapping.shape and mapping.tensor_name in tensor_set:
                tensor = next(item for item in self.tensors if item.name == mapping.tensor_name)
                if tuple(mapping.shape) != tuple(tensor.shape):
                    errors.append(f"parameter {mapping.parameter_name!r} shape does not match tensor {mapping.tensor_name!r}")
        init_targets = [item.target for item in self.initialization]
        if len(init_targets) != len(set(init_targets)):
            errors.append("initialization targets must be unique")
        for item in self.initialization:
            if self.tensors and item.target not in tensor_set:
                errors.append(f"initialization references unknown tensor {item.target!r}")
        mapped_parameters = set(parameter_names)
        for item in self.weight_transformations:
            if mapped_parameters:
                missing = (set(item.source_parameters) | set(item.target_parameters)) - mapped_parameters
                if missing:
                    errors.append(f"weight transformation references unmapped parameters: {', '.join(sorted(missing))}")
        runtime_targets = {
            "backend", "features", "supported_dtypes", "available_memory_bytes",
            "maximum_sequence_length",
        }
        for constraint in self.compatibility_constraints:
            if constraint.operator not in {"eq", "ne", "gte", "lte", "in", "contains"}:
                errors.append(f"unsupported compatibility operator {constraint.operator!r}")
            if constraint.target not in runtime_targets:
                errors.append(f"unsupported runtime compatibility target {constraint.target!r}")
        return tuple(errors)

    def assert_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("invalid MLAC-IR: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Architecture":
        """Load current IR and legacy v1 payloads that contain only core fields."""
        return cls(
            name=data["name"],
            family=data["family"],
            version=int(data.get("version", 1)),
            framework=data.get("framework", "unknown"),
            dimensions=dict(data.get("dimensions", {})),
            components=tuple(Component(**item) for item in data.get("components", ())),
            tensors=tuple(TensorSpec(**{**item, "shape": tuple(item["shape"])}) for item in data.get("tensors", ())),
            metadata=dict(data.get("metadata", {})),
            graph_nodes=tuple(GraphNode(**item) for item in data.get("graph_nodes", ())),
            graph_edges=tuple(GraphEdge(**item) for item in data.get("graph_edges", ())),
            parameter_mappings=tuple(
                ParameterMapping(**{**item, "shape": tuple(item.get("shape", ()))})
                for item in data.get("parameter_mappings", ())
            ),
            initialization=tuple(InitializationStrategy(**item) for item in data.get("initialization", ())),
            compatibility_constraints=tuple(CompatibilityConstraint(**item) for item in data.get("compatibility_constraints", ())),
            runtime_requirements=RuntimeRequirements(**data.get("runtime_requirements", {})),
            weight_transformations=tuple(
                WeightTransformationRequirement(**{
                    **item,
                    "source_parameters": tuple(item["source_parameters"]),
                    "target_parameters": tuple(item["target_parameters"]),
                })
                for item in data.get("weight_transformations", ())
            ),
            schema_version=int(data.get("schema_version", IR_SCHEMA_VERSION)),
        )

    def fingerprint(self) -> str:
        """Stable SHA-256 digest of the serialized architecture description."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_compatibility(
    architecture: Architecture, runtime: RuntimeCapabilities
) -> tuple[str, ...]:
    """Compare declared runtime requirements and constraints with a runtime."""
    issues: list[str] = list(architecture.validate())
    requirements = architecture.runtime_requirements
    if requirements.backend and requirements.backend != runtime.backend:
        issues.append(f"runtime backend {runtime.backend!r} does not satisfy {requirements.backend!r}")
    missing_features = set(requirements.required_features) - set(runtime.features)
    if missing_features:
        issues.append(f"runtime is missing features: {', '.join(sorted(missing_features))}")
    required_dtypes = set(requirements.required_dtypes)
    required_dtypes.update(
        tensor.dtype for tensor in architecture.tensors if tensor.dtype != "unknown"
    )
    unsupported = required_dtypes - set(runtime.supported_dtypes)
    if unsupported:
        issues.append(f"runtime does not support dtypes: {', '.join(sorted(unsupported))}")
    if requirements.minimum_memory_bytes is not None:
        if runtime.available_memory_bytes is None or runtime.available_memory_bytes < requirements.minimum_memory_bytes:
            issues.append("runtime has insufficient or unknown available memory")
    if requirements.maximum_sequence_length is not None:
        if runtime.maximum_sequence_length is None or runtime.maximum_sequence_length < requirements.maximum_sequence_length:
            issues.append("runtime maximum sequence length is below the architecture requirement")

    values: dict[str, Any] = {
        "backend": runtime.backend,
        "features": set(runtime.features),
        "supported_dtypes": set(runtime.supported_dtypes),
        "available_memory_bytes": runtime.available_memory_bytes,
        "maximum_sequence_length": runtime.maximum_sequence_length,
    }
    for constraint in architecture.compatibility_constraints:
        if constraint.target not in values or constraint.operator not in {"eq", "ne", "gte", "lte", "in", "contains"}:
            continue  # Invalid declarations were already reported by structural validation.
        actual = values[constraint.target]
        expected = constraint.value
        try:
            if constraint.operator == "eq":
                passed = actual == expected
            elif constraint.operator == "ne":
                passed = actual != expected
            elif constraint.operator == "gte":
                passed = actual is not None and actual >= expected
            elif constraint.operator == "lte":
                passed = actual is not None and actual <= expected
            elif constraint.operator == "in":
                passed = actual in expected
            else:
                passed = actual is not None and expected in actual
        except (TypeError, ValueError):
            passed = False
        if not passed:
            message = constraint.message or (
                f"runtime constraint failed: {constraint.target} {constraint.operator} {expected!r}"
            )
            issues.append(message if constraint.severity == "error" else f"warning: {message}")
    return tuple(issues)
