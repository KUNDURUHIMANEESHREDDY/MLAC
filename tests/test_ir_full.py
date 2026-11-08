from mlac.ir import (
    Architecture,
    CompatibilityConstraint,
    Component,
    GraphEdge,
    GraphNode,
    InitializationStrategy,
    ParameterMapping,
    RuntimeCapabilities,
    RuntimeRequirements,
    TensorSpec,
    WeightTransformationRequirement,
    validate_compatibility,
)


def test_legacy_ir_payload_remains_compatible():
    architecture = Architecture.from_dict({
        "name": "tiny",
        "family": "mlp",
        "components": [{"name": "block", "kind": "mlp", "config": {}}],
        "tensors": [{"name": "weight", "shape": [4, 4], "dtype": "fp32"}],
    })
    assert architecture.version == 1
    assert architecture.schema_version == 1
    assert architecture.tensors[0].shape == (4, 4)
    assert architecture.validate() == ()


def test_full_ir_round_trip_and_fingerprint_are_stable():
    architecture = Architecture(
        name="tiny-transformer",
        family="transformer",
        framework="pytorch",
        dimensions={"hidden_size": 32, "layers": 2},
        components=(Component("attn", "grouped_query_attention", {"heads": 4, "kv_heads": 2}),),
        tensors=(TensorSpec("qkv.weight", (96, 32), "bf16"),),
        graph_nodes=(
            GraphNode("embed", "embedding"),
            GraphNode("attn", "attention", component_name="attn"),
        ),
        graph_edges=(GraphEdge("embed", "attn", "qkv.weight"),),
        parameter_mappings=(
            ParameterMapping("blocks.0.attn.qkv.weight", "qkv.weight", "attn", "model.layers.0.qkv.weight", (96, 32)),
        ),
        initialization=(InitializationStrategy("qkv.weight", "xavier_uniform", {"gain": 1.0}, seed=7),),
        runtime_requirements=RuntimeRequirements(
            backend="pytorch", required_features=("bf16",), required_dtypes=("bf16",),
            minimum_memory_bytes=1024, maximum_sequence_length=2048,
        ),
        weight_transformations=(
            WeightTransformationRequirement(
                "split_qkv", ("blocks.0.attn.qkv.weight",),
                ("blocks.0.attn.q.weight", "blocks.0.attn.k.weight"),
                {"split": [64, 16, 16]},
            ),
        ),
    )
    restored = Architecture.from_dict(architecture.to_dict())
    assert restored == architecture
    assert restored.fingerprint() == architecture.fingerprint()
    assert len(restored.graph_edges) == 1
    assert restored.parameter_mappings[0].checkpoint_key == "model.layers.0.qkv.weight"


def test_structural_validation_catches_broken_references():
    architecture = Architecture(
        name="broken",
        family="transformer",
        components=(Component("attention", "attention"),),
        tensors=(TensorSpec("q", (8, 8)),),
        graph_nodes=(GraphNode("n1", "attention", "missing-component"),),
        graph_edges=(GraphEdge("n1", "missing-node", "unknown-tensor"),),
        parameter_mappings=(ParameterMapping("weight", "missing-tensor"),),
        initialization=(InitializationStrategy("missing-tensor", "normal"),),
    )
    errors = architecture.validate()
    assert any("unknown component" in item for item in errors)
    assert any("unknown target node" in item for item in errors)
    assert any("unknown tensor" in item for item in errors)


def test_runtime_compatibility_checks_backend_features_dtype_and_memory():
    architecture = Architecture(
        name="accelerated",
        family="transformer",
        runtime_requirements=RuntimeRequirements(
            backend="pytorch",
            required_features=("bf16", "flash_attention"),
            required_devices=("cuda",),
            required_dtypes=("bf16",),
            minimum_memory_bytes=4096,
            maximum_sequence_length=4096,
        ),
    )
    runtime = RuntimeCapabilities(
        backend="pytorch",
        features=("bf16",),
        supported_dtypes=("fp32",),
        available_memory_bytes=2048,
        maximum_sequence_length=2048,
    )
    issues = validate_compatibility(architecture, runtime)
    assert any("missing features" in item for item in issues)
    assert any("missing devices" in item for item in issues)
    assert any("does not support dtypes" in item for item in issues)
    assert any("insufficient" in item for item in issues)
    assert any("sequence length" in item for item in issues)


def test_compatibility_constraints_support_warnings():
    architecture = Architecture(
        name="cpu-model",
        family="mlp",
        compatibility_constraints=(
            CompatibilityConstraint("backend", "eq", "cuda", message="CUDA preferred", severity="warning"),
        ),
    )
    issues = validate_compatibility(
        architecture,
        RuntimeCapabilities(backend="cpu"),
    )
    assert issues == ("warning: CUDA preferred",)


def test_invalid_schema_and_invalid_tensor_shapes_are_rejected():
    import pytest

    with pytest.raises(ValueError, match="unsupported MLAC-IR"):
        Architecture(name="x", family="mlp", schema_version=99)
    with pytest.raises(ValueError, match="non-negative"):
        TensorSpec("bad", (4, -1))



def test_public_package_exposes_ir_graph_types_and_lazy_adapters():
    from mlac import IRGraphEdge, IRGraphNode, apply_weight_transformations, compile_architecture

    assert IRGraphNode("input", "input").operation == "input"
    assert IRGraphEdge("input", "output").source == "input"
    assert callable(compile_architecture)
    assert callable(apply_weight_transformations)


def test_ir_rejects_cycles_and_invalid_runtime_capabilities():
    import pytest

    cyclic = Architecture(
        name="cyclic",
        family="graph",
        graph_nodes=(GraphNode("a", "input"), GraphNode("b", "identity")),
        graph_edges=(GraphEdge("a", "b"), GraphEdge("b", "a")),
    )
    assert any("graph contains a cycle" in error for error in cyclic.validate())
    with pytest.raises(ValueError, match="runtime backend"):
        RuntimeCapabilities(backend="")
