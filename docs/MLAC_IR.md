# MLAC-IR v1

MLAC-IR is the framework-neutral architecture contract used by the controller,
inspector, mutation engine, weight tooling, and runtime adapters. It describes
what a model is independently of the framework that created it.

The Python implementation is in `src/mlac/ir.py`. It has no PyTorch or
Transformers dependency. The machine-readable JSON Schema is published at
`schemas/mlac-ir-v1.schema.json` (JSON Schema Draft 2020-12).

## Schema coverage

| Area | IR types / fields | Purpose |
|---|---|---|
| Identity and versioning | `Architecture.name`, `family`, `version`, `schema_version` | Distinguish architecture revisions from the IR schema |
| Dimensions and components | `dimensions`, `Component` | Record model dimensions and configurable building blocks |
| Tensor contract | `TensorSpec` | Shape (including symbolic dimensions), dtype, layout, trainability |
| Graph connectivity | `GraphNode`, `GraphEdge` | Represent operations and directed tensor flow |
| Checkpoint mapping | `ParameterMapping` | Map framework parameter/checkpoint keys to IR tensors |
| Initialization | `InitializationStrategy` | Describe initialization method, parameters, and optional seed |
| Compatibility | `CompatibilityConstraint` | Express runtime predicates with error or warning severity |
| Runtime contract | `RuntimeRequirements`, `RuntimeCapabilities` | Compare backend, features, devices, dtypes, memory and context needs against an execution target |
| Weight migration | `WeightTransformationRequirement` | Declare source/target parameters and required transformations |

The IR remains framework-neutral, while optional PyTorch adapters make a
validated subset executable. The compiler turns acyclic dataflow graphs into
PyTorch modules, and the weight-transformation executor applies declared
parameter migrations with shape checks and transactional semantics.

## Example

This first example shows a model-family architecture description. Its
`attention` operation is a semantic label, not one of the compiler's built-in
operators; compile it only after registering a compatible attention operator.

```python
from mlac.ir import (
    Architecture, Component, GraphEdge, GraphNode, ParameterMapping,
    RuntimeRequirements, TensorSpec, validate_compatibility,
    RuntimeCapabilities,
)

architecture = Architecture(
    name="candidate-transformer",
    family="transformer",
    version=1,
    framework="pytorch",
    dimensions={
        "hidden_size": 2048,
        "layers": 24,
        "attention_heads": 16,
        "kv_heads": 8,
        "intermediate_size": 8192,
    },
    components=(
        Component("attention", "grouped_query_attention", {"heads": 16, "kv_heads": 8}),
        Component("normalization", "rms_norm"),
        Component("activation", "swiglu"),
        Component("position", "rope"),
    ),
    tensors=(TensorSpec("layers.0.q_proj.weight", (2048, 2048), "bf16"),),
    graph_nodes=(
        GraphNode("input", "embedding"),
        GraphNode("layer0.attention", "attention", "attention"),
    ),
    graph_edges=(GraphEdge("input", "layer0.attention", "layers.0.q_proj.weight"),),
    parameter_mappings=(
        ParameterMapping(
            "layers.0.attention.q_proj.weight",
            "layers.0.q_proj.weight",
            component_name="attention",
            checkpoint_key="model.layers.0.self_attn.q_proj.weight",
            shape=(2048, 2048),
        ),
    ),
    runtime_requirements=RuntimeRequirements(
        backend="pytorch",
        required_features=("bf16",),
        required_devices=("cuda",),
        required_dtypes=("bf16",),
    ),
)

architecture.assert_valid()
runtime = RuntimeCapabilities(
    backend="pytorch",
    features=("bf16",),
    supported_dtypes=("bf16", "fp32"),
    devices=("cuda",),
)
assert validate_compatibility(architecture, runtime) == ()
```

## Validation and compatibility

- `Architecture.validate()` checks unique identifiers and references between
  graph nodes, components, tensors, parameter mappings, initialization rules,
  and weight-transformation requirements. It returns a tuple of errors.
- `Architecture.assert_valid()` raises `ValueError` when structural errors
  exist.
- `validate_compatibility(architecture, runtime)` checks the declared
  backend, required runtime features and devices, supported dtypes, memory floor,
  sequence length, and explicit compatibility predicates. It returns a tuple of issues.
  Warning-severity predicates are prefixed with `warning:`; callers must not
  treat warnings as hard failures.
- A missing runtime measurement does not count as satisfying a minimum memory
  or maximum sequence-length requirement.

## Graph compiler

Install the optional dependency with `pip install mlac[torch]`. The compiler
entry point is `mlac.compiler.compile_architecture(architecture)`; it validates
the IR and compiles a directed acyclic graph into a `torch.nn.Module`.

Built-in operators:

| Operator | Node attributes |
|---|---|
| `input` | Optional `input_index`; graph input placeholder |
| `identity`, `relu`, `gelu`, `silu`, `tanh`, `sigmoid` | None |
| `linear` | `in_features`, `out_features`, optional `bias` |
| `layer_norm` | `normalized_shape`, optional `eps`, `elementwise_affine` |
| `dropout` | Optional `p` |
| `embedding` | `num_embeddings`, `embedding_dim`, optional `padding_idx` |
| `softmax` | Optional `dim` |
| `add`, `multiply`, `matmul` | Two incoming edges |
| `concat` | Two or more incoming edges; optional `dim` |
| `flatten` | Optional `start_dim`, `end_dim` |
| `reshape` | `shape` |
| `transpose` | `dim0`, `dim1` |
| `permute` | `dims` |

A custom operator can be registered per compile call by passing a mapping of
operator names to factories that accept node attributes and return
`torch.nn.Module` instances. This is the extension point for model-family
specific operations. Unknown operators fail closed. Cycles, dynamic control
flow, and multi-output node ports are not supported by the v1 PyTorch compiler;
the compiler supports arbitrary *acyclic compositions of registered operators*,
not every possible computation graph.

Declared `InitializationStrategy` entries are applied during compilation to
parameters mapped to the target tensor. Built-in methods include zeros, ones,
constant, normal, uniform, Xavier uniform/normal, Kaiming uniform/normal, and
orthogonal initialization. Strategy-specific seeded draws are isolated from
the caller's RNG state; normal framework module construction still follows
PyTorch's standard initialization behavior.

A node can bind module parameters to canonical IR parameter names through
`attributes.parameter_bindings`. For example,
`{"weight": "encoder.proj.weight", "bias": "encoder.proj.bias"}`.
Each canonical name must have a `ParameterMapping`. Checkpoint loading then
uses `checkpoint_key` (or the canonical name when absent), verifies tensor
shapes, and copies tensors into the compiled module. Strict loading rejects
missing and unexpected checkpoint keys.

Executable compiler example:

```python
from mlac.ir import Architecture, GraphEdge, GraphNode, ParameterMapping, TensorSpec
from mlac.compiler import compile_architecture

executable = Architecture(
    name="tiny-projection",
    family="mlp",
    graph_nodes=(
        GraphNode("input", "input"),
        GraphNode("projection", "linear", attributes={
            "in_features": 4,
            "out_features": 2,
            "parameter_bindings": {
                "weight": "projection.weight",
                "bias": "projection.bias",
            },
        }),
    ),
    graph_edges=(GraphEdge("input", "projection"),),
    tensors=(
        TensorSpec("projection.weight", (2, 4), "fp32"),
        TensorSpec("projection.bias", (2,), "fp32"),
    ),
    parameter_mappings=(
        ParameterMapping("projection.weight", "projection.weight", shape=(2, 4)),
        ParameterMapping("projection.bias", "projection.bias", shape=(2,)),
    ),
)
compiled = compile_architecture(executable)
output = compiled(input_tensor)  # input_tensor shape: [batch, 4]
report = compiled.load_checkpoint(checkpoint_state_dict, strict=True)
```

## Weight transformation execution

Install `mlac[torch]` and call
`mlac.weight_transforms.apply_weight_transformations(architecture, parameters)`.
The executor supports:

- `identity` / `copy`
- `transpose` / `t`
- `permute`
- `reshape`
- `cast` (explicit dtype conversion, for example `{"dtype": "bf16"}`)
- `split` / `split_qkv` (explicit `sections` or `split_size`, plus optional `axis`)
- `concat` / `concatenate`
- `slice` (`axis`, `start`, `end`)
- `squeeze` and `unsqueeze`

Each transformation reads named source parameters and writes named target
parameters. Target names must resolve through `ParameterMapping` to declared
`TensorSpec` entries. Result rank, concrete dimensions, and supported dtype
labels are checked against the IR. The compiler also casts module parameters to
their declared supported PyTorch dtypes before applying initialization. The executor stages changes in a new mapping, clones generated tensors,
and returns a report; it does not mutate the caller's input mapping. If any
step fails, no partial result is returned. Operations execute in declaration
order, so a later step may consume a target created by an earlier step.

The executor transforms tensors supplied by the caller; it does not fetch
checkpoints, write artifact storage, train models, or promote a candidate.
Those responsibilities remain with the checkpoint, training, and governance
services. The compiler and transformation executor are optional PyTorch
adapters; the core IR module itself remains framework-independent.

## Serialization and versioning

`Architecture.to_dict()` serializes the complete IR. `Architecture.from_dict()`
accepts the original core payload (name, family, dimensions, components,
tensors, metadata) and fills the expanded fields with defaults. This preserves
compatibility with the repository's original IR objects.

`version` is the architecture revision. `schema_version` identifies the
serialized MLAC-IR schema and currently must equal `IR_SCHEMA_VERSION`.
Unknown schema versions fail closed rather than being silently interpreted.
`fingerprint()` returns a deterministic SHA-256 digest of the serialized
architecture description.

## Invariants and limits

1. Tensor dimensions are non-negative integers or symbolic strings.
2. Graph edges must reference declared nodes. If tensor declarations are
   present, an edge's optional tensor reference must resolve.
3. Parameter mappings resolve to declared tensors/components when those
   declarations are present; an explicit mapping shape must match its tensor.
4. Initialization targets are unique and resolve to declared tensors when
   tensor declarations are present.
5. If parameter mappings exist, every parameter referenced by a weight
   transformation must be mapped.
6. Compatibility predicates currently support `eq`, `ne`, `gte`, `lte`,
   `in`, and `contains`.

MLAC-IR v1 does not yet define a universal operator semantics catalogue,
control-flow/loop semantics, sparse or sharded tensor layouts, quantization
metadata beyond dtype labels, or optimizer/training state. The PyTorch compiler
is an optional adapter with a finite built-in operator set; other frameworks
and operators require explicit adapters. Unsupported features must fail closed.
Graph validation checks references and rejects cycles for the v1 dataflow
contract; it does not prove numerical equivalence. Parameter mappings are
declarations until a checkpoint is loaded and verified against the compiled
module's parameter shapes. The compiler's supported operator set is explicit
and extensible, not a universal implementation of every framework operator.
