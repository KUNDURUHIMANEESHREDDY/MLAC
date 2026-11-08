"""Compile validated MLAC-IR DAGs into executable PyTorch modules.

The compiler is registry-driven: any acyclic graph composed of registered
operators can be compiled. Dynamic control flow and unknown operators fail
closed; adapters can extend the operator registry without editing the IR.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised in minimal installs
    raise ImportError("MLAC-IR compilation requires 'pip install mlac[torch]'") from exc

from mlac.ir import Architecture, GraphNode, ParameterMapping


OperatorFactory = Callable[[Mapping[str, Any]], nn.Module]


class _Functional(nn.Module):
    def __init__(self, function: Callable[..., Any]):
        super().__init__()
        self.function = function

    def forward(self, *values: Tensor) -> Any:
        return self.function(*values)


class _Reshape(nn.Module):
    def __init__(self, shape: Sequence[int | str]):
        super().__init__()
        if any(not isinstance(dim, int) or dim == 0 for dim in shape):
            raise ValueError("compiled reshape dimensions must be non-zero integers")
        self.shape = tuple(int(dim) for dim in shape)

    def forward(self, value: Tensor) -> Tensor:
        return value.reshape(self.shape)


class _Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0, self.dim1 = int(dim0), int(dim1)

    def forward(self, value: Tensor) -> Tensor:
        return value.transpose(self.dim0, self.dim1)


def _builtin_factories() -> dict[str, OperatorFactory]:
    return {
        "identity": lambda a: nn.Identity(),
        "linear": lambda a: nn.Linear(
            int(a["in_features"]), int(a["out_features"]), bias=bool(a.get("bias", True))
        ),
        "relu": lambda a: nn.ReLU(inplace=False),
        "gelu": lambda a: nn.GELU(approximate=str(a.get("approximate", "none"))),
        "silu": lambda a: nn.SiLU(),
        "tanh": lambda a: nn.Tanh(),
        "sigmoid": lambda a: nn.Sigmoid(),
        "softmax": lambda a: _Functional(lambda x: torch.softmax(x, dim=int(a.get("dim", -1)))),
        "layer_norm": lambda a: nn.LayerNorm(
            tuple(a["normalized_shape"]) if isinstance(a["normalized_shape"], list)
            else a["normalized_shape"],
            eps=float(a.get("eps", 1e-5)),
            elementwise_affine=bool(a.get("elementwise_affine", True)),
        ),
        "dropout": lambda a: nn.Dropout(float(a.get("p", 0.5))),
        "embedding": lambda a: nn.Embedding(
            int(a["num_embeddings"]), int(a["embedding_dim"]),
            padding_idx=a.get("padding_idx"),
        ),
        "add": lambda a: _Functional(lambda x, y: x + y),
        "multiply": lambda a: _Functional(lambda x, y: x * y),
        "matmul": lambda a: _Functional(lambda x, y: torch.matmul(x, y)),
        "concat": lambda a: _Functional(lambda *xs: torch.cat(xs, dim=int(a.get("dim", -1)))),
        "flatten": lambda a: nn.Flatten(
            start_dim=int(a.get("start_dim", 1)), end_dim=int(a.get("end_dim", -1))
        ),
        "reshape": lambda a: _Reshape(a["shape"]),
        "transpose": lambda a: _Transpose(a["dim0"], a["dim1"]),
        "permute": lambda a: _Functional(lambda x: x.permute(*tuple(a["dims"]))),
    }


@dataclass(frozen=True)
class CheckpointLoadReport:
    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


class CompiledArchitecture(nn.Module):
    """Executable DAG compiled from an Architecture declaration."""

    def __init__(
        self,
        architecture: Architecture,
        ordered_nodes: Sequence[GraphNode],
        modules: Sequence[nn.Module],
        incoming: Mapping[str, Sequence[str]],
        input_nodes: Sequence[str],
        output_nodes: Sequence[str],
        parameter_keys: Mapping[str, str],
    ):
        super().__init__()
        self.architecture = architecture
        self.ordered_node_ids = tuple(node.node_id for node in ordered_nodes)
        self.nodes = nn.ModuleList(modules)
        self.incoming = {key: tuple(value) for key, value in incoming.items()}
        self.input_nodes = tuple(input_nodes)
        self.output_nodes = tuple(output_nodes)
        self.parameter_keys = dict(parameter_keys)
        self._module_index = {
            node_id: index for index, node_id in enumerate(self.ordered_node_ids)
        }

    def forward(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        if len(inputs) != len(self.input_nodes):
            raise ValueError(
                f"expected {len(self.input_nodes)} graph inputs, received {len(inputs)}"
            )
        values: dict[str, Tensor] = dict(zip(self.input_nodes, inputs))
        for node_id in self.ordered_node_ids:
            args = [values[source] for source in self.incoming[node_id]]
            values[node_id] = self.nodes[self._module_index[node_id]](*args)
        outputs = tuple(values[node_id] for node_id in self.output_nodes)
        return outputs[0] if len(outputs) == 1 else outputs

    def load_checkpoint(
        self, checkpoint: Mapping[str, Tensor], *, strict: bool = True
    ) -> CheckpointLoadReport:
        """Load checkpoint tensors through declared IR parameter mappings.

        GraphNode.attributes['parameter_bindings'] maps a local module parameter
        (for example 'weight') to ParameterMapping.parameter_name.
        """
        state = dict(self.named_parameters())
        loaded: list[str] = []
        missing: list[str] = []
        unexpected = sorted(set(checkpoint) - {
            mapping.checkpoint_key or mapping.parameter_name
            for mapping in self.architecture.parameter_mappings
        })
        pending: list[tuple[str, Tensor]] = []
        for mapping in self.architecture.parameter_mappings:
            source_key = mapping.checkpoint_key or mapping.parameter_name
            target_key = self.parameter_keys.get(mapping.parameter_name)
            if source_key not in checkpoint or target_key is None or target_key not in state:
                missing.append(source_key)
                continue
            source, target = checkpoint[source_key], state[target_key]
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"checkpoint tensor {source_key!r} shape {tuple(source.shape)} "
                    f"does not match {target_key!r} shape {tuple(target.shape)}"
                )
            if mapping.shape and tuple(source.shape) != tuple(mapping.shape):
                raise ValueError(f"checkpoint tensor {source_key!r} violates its IR shape contract")
            pending.append((target_key, source))
            loaded.append(source_key)
        if strict and (missing or unexpected):
            raise KeyError(
                f"checkpoint mapping mismatch; missing={sorted(missing)}, "
                f"unexpected={unexpected}"
            )
        with torch.no_grad():
            for target_key, source in pending:
                state[target_key].copy_(source.to(
                    device=state[target_key].device, dtype=state[target_key].dtype
                ))
        return CheckpointLoadReport(tuple(loaded), tuple(sorted(missing)), tuple(unexpected))


def _topological_order(architecture: Architecture) -> tuple[list[GraphNode], dict[str, list[str]]]:
    nodes = {node.node_id: node for node in architecture.graph_nodes}
    order_index = {node.node_id: i for i, node in enumerate(architecture.graph_nodes)}
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in nodes}
    for edge in architecture.graph_edges:
        incoming[edge.target].append(edge.source)
        outgoing[edge.source].append(edge.target)
        indegree[edge.target] += 1
    ready = deque(sorted(
        (node_id for node_id, degree in indegree.items() if degree == 0),
        key=order_index.__getitem__,
    ))
    ordered: list[GraphNode] = []
    while ready:
        node_id = ready.popleft()
        ordered.append(nodes[node_id])
        for target in outgoing[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(ordered) != len(nodes):
        raise ValueError("MLAC-IR compiler does not support cyclic graphs")
    for node_id, sources in incoming.items():
        sources.sort(key=order_index.__getitem__)
    return ordered, dict(incoming)


def compile_architecture(
    architecture: Architecture,
    *,
    custom_operators: Mapping[str, OperatorFactory] | None = None,
) -> CompiledArchitecture:
    """Compile an acyclic graph from registered operators into a PyTorch module."""
    architecture.assert_valid()
    if not architecture.graph_nodes:
        raise ValueError("architecture has no graph nodes")
    factories = _builtin_factories()
    if custom_operators:
        for name, factory in custom_operators.items():
            if not name or not callable(factory):
                raise ValueError("custom operators require a non-empty name and callable factory")
            if name in {"input"}:
                raise ValueError("'input' is reserved by the MLAC-IR compiler")
            factories[name] = factory

    ordered, incoming = _topological_order(architecture)
    input_nodes = sorted(
        (node for node in architecture.graph_nodes if node.operation == "input"),
        key=lambda node: (int(node.attributes.get("input_index", 0)), node.node_id),
    )
    if not input_nodes:
        raise ValueError("compiled architecture requires at least one input node")
    if len({int(node.attributes.get("input_index", i)) for i, node in enumerate(input_nodes)}) != len(input_nodes):
        raise ValueError("input node indices must be unique")
    for node in input_nodes:
        if incoming.get(node.node_id):
            raise ValueError(f"input node {node.node_id!r} cannot have incoming edges")

    modules: list[nn.Module] = []
    compiled_nodes: list[GraphNode] = []
    module_index: dict[str, int] = {}
    for node in ordered:
        if node.operation == "input":
            continue
        factory = factories.get(node.operation)
        if factory is None:
            raise ValueError(f"unregistered MLAC-IR operator: {node.operation!r}")
        if not incoming.get(node.node_id):
            raise ValueError(f"non-input node {node.node_id!r} has no incoming edges")
        module = factory(node.attributes)
        if not isinstance(module, nn.Module):
            raise TypeError(f"operator factory {node.operation!r} did not return torch.nn.Module")
        module_index[node.node_id] = len(modules)
        modules.append(module)
        compiled_nodes.append(node)

    outputs = architecture.metadata.get("outputs")
    if outputs is None:
        targets = {edge.target for edge in architecture.graph_edges}
        output_nodes = [node.node_id for node in architecture.graph_nodes if node.node_id not in targets]
    else:
        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise ValueError("metadata.outputs must be a non-empty list of node ids")
        output_nodes = list(outputs)
        unknown = set(output_nodes) - {node.node_id for node in architecture.graph_nodes}
        if unknown:
            raise ValueError(f"unknown graph outputs: {', '.join(sorted(unknown))}")
    if not output_nodes:
        raise ValueError("compiled architecture has no graph outputs")

    parameter_keys: dict[str, str] = {}
    for node in compiled_nodes:
        bindings = node.attributes.get("parameter_bindings", {})
        if not isinstance(bindings, dict):
            raise TypeError(f"node {node.node_id!r} parameter_bindings must be a mapping")
        for local_name, parameter_name in bindings.items():
            if not isinstance(local_name, str) or not isinstance(parameter_name, str):
                raise TypeError("parameter bindings must map strings to strings")
            if parameter_name in parameter_keys:
                raise ValueError(f"duplicate parameter binding: {parameter_name!r}")
            parameter_keys[parameter_name] = f"nodes.{module_index[node.node_id]}.{local_name}"

    declared = {mapping.parameter_name for mapping in architecture.parameter_mappings}
    dangling = set(parameter_keys) - declared if declared else set()
    if dangling:
        raise ValueError(f"parameter bindings have no IR mapping: {', '.join(sorted(dangling))}")

    return CompiledArchitecture(
        architecture, compiled_nodes, modules, incoming,
        [node.node_id for node in input_nodes], output_nodes, parameter_keys,
    )
