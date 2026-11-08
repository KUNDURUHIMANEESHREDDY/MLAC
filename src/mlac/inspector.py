"""Read-only PyTorch architecture, graph, attention, KV-cache and memory inspection.

The basic inspector only reads module metadata. Full graph tracing is opt-in and
uses torch.fx on a trusted, already-instantiated module. Shape propagation and
forward-memory profiling execute a caller-supplied example input and therefore
must only be used with trusted models and inputs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from mlac.ir import Architecture, Component, TensorSpec


@dataclass(frozen=True)
class GraphNode:
    name: str
    operation: str
    target: str
    module_type: str | None = None
    inputs: tuple[str, ...] = ()
    output_shape: tuple[int | str, ...] | None = None
    output_dtype: str | None = None
    parameter_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str


@dataclass(frozen=True)
class AttentionAnalysis:
    module_name: str
    module_type: str
    num_heads: int | None
    num_kv_heads: int | None
    head_dim: int | None
    hidden_size: int | None
    grouped_query_attention: bool
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class KVCacheEstimate:
    batch_size: int
    sequence_length: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    bytes_per_element: int
    total_bytes: int
    formula: str = "2 * batch_size * sequence_length * num_layers * num_kv_heads * head_dim * bytes_per_element"

    @property
    def total_mib(self) -> float:
        return self.total_bytes / (1024 ** 2)


@dataclass(frozen=True)
class MemoryProfile:
    device: str
    parameter_bytes: int
    buffer_bytes: int
    input_bytes: int | None = None
    activation_output_bytes: int | None = None
    cuda_allocated_before_bytes: int | None = None
    cuda_allocated_after_bytes: int | None = None
    cuda_peak_allocated_bytes: int | None = None
    measurement_note: str = "Parameter and buffer sizes are exact tensor storage sizes; activation output bytes sum module outputs and may include reused tensors."


@dataclass(frozen=True)
class ArchitectureInspection:
    architecture: Architecture
    graph_nodes: tuple[GraphNode, ...] = ()
    graph_edges: tuple[GraphEdge, ...] = ()
    attention: tuple[AttentionAnalysis, ...] = ()
    memory: MemoryProfile | None = None
    kv_cache: KVCacheEstimate | None = None
    trace_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _dtype_bytes(dtype: Any) -> tuple[str, int]:
    name = str(dtype).replace("torch.", "")
    widths = {"float64": 8, "int64": 8, "complex128": 16,
              "float32": 4, "int32": 4, "complex64": 8,
              "float16": 2, "bfloat16": 2, "int16": 2,
              "uint16": 2, "int8": 1, "uint8": 1, "bool": 1}
    if name not in widths:
        raise ValueError(f"unsupported cache dtype: {name}")
    return name, widths[name]


def estimate_kv_cache_bytes(*, batch_size: int, sequence_length: int,
                            num_layers: int, num_kv_heads: int,
                            head_dim: int, dtype: Any = "float16") -> KVCacheEstimate:
    """Estimate decoder KV storage; excludes allocator, metadata and fragmentation."""
    for name, value in (("batch_size", batch_size), ("sequence_length", sequence_length),
                        ("num_layers", num_layers), ("num_kv_heads", num_kv_heads),
                        ("head_dim", head_dim)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    dtype_name, width = _dtype_bytes(dtype)
    total = 2 * batch_size * sequence_length * num_layers * num_kv_heads * head_dim * width
    return KVCacheEstimate(batch_size, sequence_length, num_layers, num_kv_heads,
                           head_dim, dtype_name, width, total)


def inspect_torch_model(model, *, name="pytorch-model") -> Architecture:
    """Return basic architecture metadata without executing the model."""
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("Install PyTorch with: pip install 'mlac[torch]'") from exc
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    counts, components, tensors = Counter(), [], []
    total = trainable = 0
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        kind = type(module).__name__
        counts[kind] += 1
        config = {}
        for attr in ("in_features", "out_features", "num_features", "num_heads",
                     "num_key_value_heads", "num_kv_heads", "head_dim", "hidden_size",
                     "intermediate_size", "embed_dim", "d_model", "p", "kernel_size", "stride"):
            value = getattr(module, attr, None)
            if isinstance(value, (int, float, str, tuple)):
                config[attr] = value
        components.append(Component(module_name, kind, config))
    for param_name, param in model.named_parameters():
        total += param.numel()
        trainable += param.numel() if param.requires_grad else 0
        tensors.append(TensorSpec(param_name, tuple(int(x) for x in param.shape), str(param.dtype)))
    return Architecture(name, "pytorch_module", framework="pytorch",
        dimensions={"parameter_count": total, "trainable_parameter_count": trainable,
                    "module_count": len(components)},
        components=tuple(components), tensors=tuple(tensors),
        metadata={"module_types": dict(counts)})


def _attention_analysis(model: Any) -> tuple[AttentionAnalysis, ...]:
    found = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        kind = type(module).__name__
        attrs = {}
        for key in ("num_heads", "num_attention_heads", "num_key_value_heads",
                    "num_kv_heads", "head_dim", "hidden_size", "embed_dim", "d_model"):
            value = getattr(module, key, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                attrs[key] = value
        looks_like_attention = "attention" in kind.lower() or "attn" in kind.lower() or bool(
            set(attrs) & {"num_heads", "num_attention_heads", "num_key_value_heads", "num_kv_heads"})
        if not looks_like_attention:
            continue
        heads = attrs.get("num_heads", attrs.get("num_attention_heads"))
        kv_heads = attrs.get("num_key_value_heads", attrs.get("num_kv_heads", heads))
        head_dim = attrs.get("head_dim")
        hidden = attrs.get("hidden_size", attrs.get("embed_dim", attrs.get("d_model")))
        if head_dim is None and hidden is not None and heads and hidden % heads == 0:
            head_dim = hidden // heads
        evidence = tuple(sorted(attrs))
        found.append(AttentionAnalysis(module_name, kind, heads, kv_heads, head_dim,
                                       hidden, bool(heads and kv_heads and kv_heads < heads), evidence))
    return tuple(found)


def _shape_dtype(node: Any) -> tuple[tuple[int | str, ...] | None, str | None]:
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return None, None
    shape = getattr(meta, "shape", None)
    dtype = getattr(meta, "dtype", None)
    if shape is None:
        return None, str(dtype) if dtype is not None else None
    return tuple(int(x) if isinstance(x, int) else str(x) for x in shape), str(dtype) if dtype is not None else None


def _profile_memory(model: Any, example_inputs: tuple[Any, ...] | None) -> MemoryProfile:
    import torch
    parameters = sum(_tensor_bytes(p) for p in model.parameters())
    buffers = sum(_tensor_bytes(b) for b in model.buffers())
    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        device = "cpu"
    if example_inputs is None:
        return MemoryProfile(device, parameters, buffers)

    input_bytes = 0
    def count_inputs(value: Any) -> None:
        nonlocal input_bytes
        if isinstance(value, torch.Tensor):
            input_bytes += _tensor_bytes(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                count_inputs(item)
        elif isinstance(value, dict):
            for item in value.values():
                count_inputs(item)
    count_inputs(example_inputs)

    activation_bytes = 0
    handles = []
    def output_bytes(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return _tensor_bytes(value)
        if isinstance(value, (tuple, list)):
            return sum(output_bytes(item) for item in value)
        if isinstance(value, dict):
            return sum(output_bytes(item) for item in value.values())
        return 0
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        nonlocal activation_bytes
        activation_bytes += output_bytes(output)
    for _name, module in model.named_modules():
        if module is not model:
            handles.append(module.register_forward_hook(hook))

    training_states = [(module, module.training) for module in model.modules()]
    cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    before = after = peak = None
    if cuda:
        torch.cuda.synchronize()
        before = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    try:
        model.eval()
        with torch.inference_mode():
            model(*example_inputs)
        if cuda:
            torch.cuda.synchronize()
            after = int(torch.cuda.memory_allocated(device))
            peak = int(torch.cuda.max_memory_allocated(device))
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in training_states:
            module.training = was_training
    return MemoryProfile(device, parameters, buffers, input_bytes, activation_bytes,
                         before, after, peak)


def inspect_torch_model_full(
    model: Any,
    *,
    name: str = "pytorch-model",
    example_inputs: tuple[Any, ...] | None = None,
    trace_graph: bool = True,
    profile_memory: bool = False,
    kv_cache: Mapping[str, Any] | None = None,
) -> ArchitectureInspection:
    """Inspect a trusted PyTorch model, optionally tracing and profiling a forward pass.

    `example_inputs` enables FX shape propagation and is required for measured
    forward-memory profiling. `kv_cache` supplies explicit decoder dimensions
    for an analytical KV-cache estimate; no generic cache is inferred from
    module names.
    """
    import torch
    from torch import nn
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    if example_inputs is not None and not isinstance(example_inputs, tuple):
        raise TypeError("example_inputs must be a tuple of positional model inputs")
    architecture = inspect_torch_model(model, name=name)
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    trace_error = None
    if trace_graph:
        try:
            import torch.fx as fx
            traced = fx.symbolic_trace(model)
            if example_inputs is not None:
                from torch.fx.passes.shape_prop import ShapeProp
                ShapeProp(traced).propagate(*example_inputs)
            module_map = dict(traced.named_modules())
            for node in traced.graph.nodes:
                target = str(node.target)
                module_type = None
                parameter_names: tuple[str, ...] = ()
                if node.op == "call_module":
                    module = module_map.get(target)
                    if module is not None:
                        module_type = type(module).__name__
                        parameter_names = tuple(
                            f"{target}.{param_name}" if param_name else target
                            for param_name, _param in module.named_parameters(recurse=True)
                        )
                shape, dtype = _shape_dtype(node)
                inputs = tuple(source.name for source in node.all_input_nodes)
                nodes.append(GraphNode(node.name, node.op, target, module_type,
                                       inputs, shape, dtype, parameter_names))
                edges.extend(GraphEdge(source.name, node.name) for source in node.all_input_nodes)
        except Exception as exc:
            trace_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            nodes.clear()
            edges.clear()
    memory = _profile_memory(model, example_inputs) if profile_memory else None
    cache = None
    if kv_cache is not None:
        allowed = {"batch_size", "sequence_length", "num_layers", "num_kv_heads", "head_dim", "dtype"}
        if set(kv_cache) - allowed:
            raise ValueError("kv_cache contains unsupported fields")
        required = allowed - {"dtype"}
        if not required.issubset(kv_cache):
            raise ValueError("kv_cache requires batch_size, sequence_length, num_layers, num_kv_heads, and head_dim")
        cache = estimate_kv_cache_bytes(**dict(kv_cache))
    return ArchitectureInspection(architecture, tuple(nodes), tuple(edges),
                                  _attention_analysis(model), memory, cache, trace_error)
