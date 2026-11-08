"""Constraint-aware, deterministic architecture candidate generation.

MLAC Architecture Designer v1 generates executable feed-forward PyTorch graph
specifications in framework-neutral MLAC-IR. It does not claim to search
arbitrary transformer topology or predict task quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable

from mlac.ir import (
    Architecture,
    Component,
    GraphEdge,
    GraphNode,
    InitializationStrategy,
    ParameterMapping,
    TensorSpec,
)


_SUPPORTED_ACTIVATIONS = {"relu", "gelu", "silu", "tanh"}


@dataclass(frozen=True)
class DesignConstraints:
    input_size: int
    output_size: int
    hidden_sizes: tuple[int, ...] = (64, 128, 256)
    min_layers: int = 1
    max_layers: int = 3
    allowed_activations: tuple[str, ...] = ("gelu", "relu")
    max_parameters: int | None = None
    target_parameters: int | None = None
    bias: bool = True
    dtype: str = "fp32"
    candidate_limit: int = 100
    search_evaluation_limit: int = 10000

    def __post_init__(self) -> None:
        if type(self.input_size) is not int or self.input_size < 1:
            raise ValueError("input_size must be a positive integer")
        if type(self.output_size) is not int or self.output_size < 1:
            raise ValueError("output_size must be a positive integer")
        if not self.hidden_sizes or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integers")
        if len(set(self.hidden_sizes)) != len(self.hidden_sizes):
            raise ValueError("hidden_sizes must be unique")
        if type(self.min_layers) is not int or type(self.max_layers) is not int:
            raise ValueError("layer bounds must be integers")
        if self.min_layers < 1 or self.max_layers < self.min_layers:
            raise ValueError("layer bounds must satisfy 1 <= min_layers <= max_layers")
        if self.max_layers > 8:
            raise ValueError("max_layers cannot exceed 8 in designer v1")
        if not self.allowed_activations or any(
            name not in _SUPPORTED_ACTIVATIONS for name in self.allowed_activations
        ):
            raise ValueError(
                f"allowed_activations must be selected from {sorted(_SUPPORTED_ACTIVATIONS)}"
            )
        if self.max_parameters is not None and (
            type(self.max_parameters) is not int or self.max_parameters < 1
        ):
            raise ValueError("max_parameters must be a positive integer")
        if self.target_parameters is not None and (
            type(self.target_parameters) is not int or self.target_parameters < 1
        ):
            raise ValueError("target_parameters must be a positive integer")
        if type(self.candidate_limit) is not int or not 1 <= self.candidate_limit <= 10000:
            raise ValueError("candidate_limit must be an integer between 1 and 10000")
        if type(self.search_evaluation_limit) is not int or not 1 <= self.search_evaluation_limit <= 100000:
            raise ValueError("search_evaluation_limit must be an integer between 1 and 100000")
        if not isinstance(self.dtype, str) or self.dtype.lower() not in {
            "fp16", "bf16", "fp32", "fp64"
        }:
            raise ValueError("dtype must be one of fp16, bf16, fp32, fp64")


@dataclass(frozen=True)
class DesignCandidate:
    architecture: Architecture
    hidden_widths: tuple[int, ...]
    activation: str
    parameter_count: int
    estimated_multiply_accumulates_per_example: int
    score: float
    rationale: str


class ArchitectureDesigner:
    """Generate, validate, and rank a bounded population of MLP designs."""

    def generate(self, constraints: DesignConstraints) -> tuple[DesignCandidate, ...]:
        if not isinstance(constraints, DesignConstraints):
            raise TypeError("constraints must be a DesignConstraints instance")
        search_size = len(constraints.allowed_activations) * sum(
            len(constraints.hidden_sizes) ** depth
            for depth in range(constraints.min_layers, constraints.max_layers + 1)
        )
        if search_size > constraints.search_evaluation_limit:
            raise ValueError(
                f"design search contains {search_size} candidates, exceeding "
                f"search_evaluation_limit={constraints.search_evaluation_limit}"
            )
        candidates: list[DesignCandidate] = []
        for depth in range(constraints.min_layers, constraints.max_layers + 1):
            for widths in product(constraints.hidden_sizes, repeat=depth):
                for activation in constraints.allowed_activations:
                    candidate = self._build(constraints, tuple(widths), activation)
                    if (
                        constraints.max_parameters is not None
                        and candidate.parameter_count > constraints.max_parameters
                    ):
                        continue
                    candidates.append(candidate)
        candidates.sort(
            key=lambda item: (
                item.score,
                -item.parameter_count,
                -len(item.hidden_widths),
                item.hidden_widths,
                item.activation,
            ),
            reverse=True,
        )
        return tuple(candidates[:constraints.candidate_limit])

    def design(self, constraints: DesignConstraints) -> DesignCandidate:
        candidates = self.generate(constraints)
        if not candidates:
            raise ValueError("no architecture satisfies the supplied design constraints")
        return candidates[0]

    @staticmethod
    def _build(
        constraints: DesignConstraints,
        hidden_widths: tuple[int, ...],
        activation: str,
    ) -> DesignCandidate:
        widths = (constraints.input_size, *hidden_widths, constraints.output_size)
        nodes = [GraphNode("input", "input", attributes={"input_index": 0})]
        edges: list[GraphEdge] = []
        components: list[Component] = []
        tensors: list[TensorSpec] = []
        mappings: list[ParameterMapping] = []
        initializers: list[InitializationStrategy] = []
        parameter_count = 0
        macs = 0
        previous = "input"

        for layer_index, (in_features, out_features) in enumerate(zip(widths, widths[1:])):
            linear_id = f"linear_{layer_index}"
            weight_name = f"layers.{layer_index}.weight"
            bias_name = f"layers.{layer_index}.bias"
            bindings = {"weight": weight_name}
            nodes.append(GraphNode(
                linear_id,
                "linear",
                component_name=f"dense_{layer_index}",
                attributes={
                    "in_features": in_features,
                    "out_features": out_features,
                    "bias": constraints.bias,
                    "parameter_bindings": bindings,
                },
            ))
            edges.append(GraphEdge(previous, linear_id))
            components.append(Component(
                f"dense_{layer_index}", "linear",
                {"in_features": in_features, "out_features": out_features, "bias": constraints.bias},
            ))
            weight_shape = (out_features, in_features)
            tensors.append(TensorSpec(weight_name, weight_shape, constraints.dtype))
            mappings.append(ParameterMapping(
                weight_name, weight_name, f"dense_{layer_index}", shape=weight_shape
            ))
            initializers.append(InitializationStrategy(weight_name, "xavier_uniform"))
            parameter_count += in_features * out_features
            macs += in_features * out_features
            if constraints.bias:
                bindings["bias"] = bias_name
                bias_shape = (out_features,)
                tensors.append(TensorSpec(bias_name, bias_shape, constraints.dtype))
                mappings.append(ParameterMapping(
                    bias_name, bias_name, f"dense_{layer_index}", shape=bias_shape
                ))
                initializers.append(InitializationStrategy(bias_name, "zeros"))
                parameter_count += out_features

            if layer_index < len(widths) - 2:
                activation_id = f"activation_{layer_index}"
                nodes.append(GraphNode(activation_id, activation))
                edges.append(GraphEdge(linear_id, activation_id))
                components.append(Component(f"activation_{layer_index}", activation))
                previous = activation_id
            else:
                previous = linear_id

        architecture = Architecture(
            name=f"mlp-{constraints.input_size}-{'-'.join(map(str, hidden_widths))}-{constraints.output_size}-{activation}",
            family="mlp",
            framework="pytorch",
            dimensions={
                "input_size": constraints.input_size,
                "hidden_sizes": list(hidden_widths),
                "output_size": constraints.output_size,
                "layers": len(widths) - 1,
            },
            components=tuple(components),
            tensors=tuple(tensors),
            graph_nodes=tuple(nodes),
            graph_edges=tuple(edges),
            parameter_mappings=tuple(mappings),
            initialization=tuple(initializers),
            metadata={
                "outputs": [previous],
                "designer": "mlac.architecture_designer.v1",
                "activation": activation,
                "bias": constraints.bias,
            },
        )
        architecture.assert_valid()
        target = constraints.target_parameters
        if target is not None:
            score = math.exp(-abs(math.log((parameter_count + 1) / (target + 1))))
            rationale = (
                f"{parameter_count:,} parameters; target is {target:,}; "
                "score rewards a close parameter-budget match."
            )
        else:
            budget = constraints.max_parameters
            score = (
                parameter_count / budget if budget is not None
                else 1.0 / (1.0 + parameter_count / 1_000_000)
            )
            rationale = (
                f"{parameter_count:,} parameters and {macs:,} multiply-accumulates "
                "per example; score favors efficient designs within the supplied budget."
            )
        return DesignCandidate(
            architecture=architecture,
            hidden_widths=hidden_widths,
            activation=activation,
            parameter_count=parameter_count,
            estimated_multiply_accumulates_per_example=macs,
            score=score,
            rationale=rationale,
        )
