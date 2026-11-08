import pytest

torch = pytest.importorskip("torch")

from mlac.compiler import compile_architecture
from mlac.ir import Architecture, GraphEdge, GraphNode, ParameterMapping, TensorSpec


def test_compile_linear_relu_graph_and_load_checkpoint():
    architecture = Architecture(
        name="linear-relu",
        family="mlp",
        graph_nodes=(
            GraphNode("input", "input"),
            GraphNode("linear", "linear", attributes={
                "in_features": 3, "out_features": 2,
                "parameter_bindings": {"weight": "fc.weight", "bias": "fc.bias"},
            }),
            GraphNode("relu", "relu"),
        ),
        graph_edges=(GraphEdge("input", "linear"), GraphEdge("linear", "relu")),
        tensors=(
            TensorSpec("fc.weight", (2, 3), "fp32"),
            TensorSpec("fc.bias", (2,), "fp32"),
        ),
        parameter_mappings=(
            ParameterMapping("fc.weight", "fc.weight", shape=(2, 3), checkpoint_key="layer.weight"),
            ParameterMapping("fc.bias", "fc.bias", shape=(2,), checkpoint_key="layer.bias"),
        ),
    )
    model = compile_architecture(architecture)
    report = model.load_checkpoint({
        "layer.weight": torch.tensor([[1., 0., 0.], [0., 1., 0.]]),
        "layer.bias": torch.tensor([-1., 1.]),
    })
    assert report.loaded == ("layer.weight", "layer.bias")
    result = model(torch.tensor([[2., 3., 4.]]))
    assert torch.equal(result, torch.tensor([[1., 4.]]))


def test_compiler_supports_multi_input_ops_and_multiple_outputs():
    architecture = Architecture(
        name="two-input-add",
        family="graph",
        graph_nodes=(
            GraphNode("left", "input", attributes={"input_index": 0}),
            GraphNode("right", "input", attributes={"input_index": 1}),
            GraphNode("sum", "add"),
            GraphNode("scaled", "multiply"),
        ),
        graph_edges=(
            GraphEdge("left", "sum", target_port="left"),
            GraphEdge("right", "sum", target_port="right"),
            GraphEdge("sum", "scaled", target_port="left"),
            GraphEdge("right", "scaled", target_port="right"),
        ),
        metadata={"outputs": ["sum", "scaled"]},
    )
    model = compile_architecture(architecture)
    left, right = torch.tensor([1., 2.]), torch.tensor([3., 4.])
    first, second = model(left, right)
    assert torch.equal(first, torch.tensor([4., 6.]))
    assert torch.equal(second, torch.tensor([12., 24.]))


def test_compiler_accepts_custom_registered_operators():
    from torch import nn

    class Double(nn.Module):
        def forward(self, value):
            return value * 2

    architecture = Architecture(
        name="custom",
        family="graph",
        graph_nodes=(GraphNode("input", "input"), GraphNode("double", "double")),
        graph_edges=(GraphEdge("input", "double"),),
    )
    model = compile_architecture(
        architecture, custom_operators={"double": lambda attrs: Double()}
    )
    x = torch.tensor([2.])
    assert torch.equal(model(x), torch.tensor([4.]))


def test_compiler_rejects_cycles_and_unknown_operators():
    cyclic = Architecture(
        name="cycle", family="graph",
        graph_nodes=(GraphNode("a", "input"), GraphNode("b", "identity")),
        graph_edges=(GraphEdge("a", "b"), GraphEdge("b", "a")),
    )
    with pytest.raises(ValueError, match="cycle"):
        compile_architecture(cyclic)

    unknown = Architecture(
        name="unknown", family="graph",
        graph_nodes=(GraphNode("input", "input"), GraphNode("mystery", "not_registered")),
        graph_edges=(GraphEdge("input", "mystery"),),
    )
    with pytest.raises(ValueError, match="unregistered"):
        compile_architecture(unknown)
