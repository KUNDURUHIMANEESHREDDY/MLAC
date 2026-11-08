# MLAC-IR v1

MLAC-IR is the framework-neutral architecture contract used by the controller,
inspector, mutation engine, weight tooling, and runtime adapters. It describes
what a model is independently of the framework that created it.

The Python implementation is in `src/mlac/ir.py`. It has no PyTorch or
Transformers dependency.

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
| Runtime contract | `RuntimeRequirements`, `RuntimeCapabilities` | Compare declared needs against an available execution target |
| Weight migration | `WeightTransformationRequirement` | Declare source/target parameters and required transformations |

The schema is intentionally descriptive. It does not execute graph nodes,
initialize weights, load checkpoints, or perform tensor transformations.

## Example

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
        allowed_dtypes=("bf16",),
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
  backend, required runtime features, supported dtypes, memory floor, sequence
  length, and explicit compatibility predicates. It returns a tuple of issues.
  Warning-severity predicates are prefixed with `warning:`; callers must not
  treat warnings as hard failures.
- A missing runtime measurement does not count as satisfying a minimum memory
  or maximum sequence-length requirement.

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
metadata beyond dtype labels, optimizer/training state, or a framework
compiler. Adapters must declare unsupported features rather than guessing.
Graph validation currently checks references, not acyclicity or numerical
equivalence. Parameter mappings are declarations, not proof that checkpoint
keys or tensor contents are correct.
