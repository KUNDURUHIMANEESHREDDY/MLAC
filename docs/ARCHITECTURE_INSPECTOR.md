# Architecture Inspector v1

## Capabilities

The legacy `inspect_torch_model` API remains available and performs metadata-only inspection without running the model. It returns MLAC-IR architecture metadata, module configuration, parameter shapes, parameter counts, and trainable parameter counts.

The extended `inspect_torch_model_full` API adds:

- **Computational graph:** optional `torch.fx.symbolic_trace` node and edge extraction, module targets, input connectivity, parameter-name mappings, and output shape/dtype metadata when example inputs are supplied.
- **Attention analysis:** discovers common attention configuration attributes including query-head count, KV-head count, head dimension, hidden size, and grouped-query attention indicators. The evidence fields identify which attributes were observed.
- **KV-cache sizing:** analytical estimate using explicit batch size, sequence length, layer count, KV-head count, head dimension, and dtype. It accounts for keys and values, but excludes allocator overhead, cache metadata, fragmentation, paging, and implementation-specific layouts.
- **Memory profile:** exact parameter and buffer storage bytes; optional input tensor bytes and a sum of module output tensor bytes; CUDA allocated-before, allocated-after, and peak-allocated measurements when the model is on CUDA.

## Usage

```python
import torch
from mlac.inspector import inspect_torch_model_full

model = load_trusted_model().eval()
example_inputs = (torch.zeros(1, 128, 2048),)

report = inspect_torch_model_full(
    model,
    name="candidate-model",
    example_inputs=example_inputs,
    trace_graph=True,
    profile_memory=True,
    kv_cache={
        "batch_size": 1,
        "sequence_length": 4096,
        "num_layers": 24,
        "num_kv_heads": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
    },
)
print(report.to_dict())
```

## Safety and interpretation

- Only inspect models that are already instantiated and trusted. This module does not load checkpoints or execute arbitrary generated code.
- Graph tracing is best-effort. Models with unsupported Python control flow, dynamic behavior, or custom operators may fail symbolic tracing; the report preserves a bounded `trace_error` rather than claiming a complete graph.
- Supplying `example_inputs` for shape propagation executes a forward pass through the traced graph.
- `profile_memory=True` executes the original model once under inference mode. It restores each module's individual training/evaluation flag and removes hooks even if the forward pass fails.
- `activation_output_bytes` is the sum of observed module outputs, not a peak-live-activation measurement. Shared/reused tensors can be counted more than once.
- CUDA peak allocation is reported only when the model's first parameter is on a CUDA device and CUDA is available. CPU allocator peak usage is not claimed.
- KV-cache size is a formula-based estimate from operator-supplied dimensions. It is not a measurement of a running model's actual cache implementation.
- Attention discovery is heuristic and reports only attributes found on modules; it does not infer hidden internals from opaque modules.

This is an initial inspection layer, not yet a universal compiler-grade graph extractor or runtime KV-cache profiler.
