"""Master Intelligence Engine: objective analysis, diagnosis, and governed experiment planning.

Proposal-only: never executes model code, changes weights, or authorizes promotion.
"""
from __future__ import annotations
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4
from mlac.ir import Architecture


@dataclass(frozen=True)
class OptimizationObjective:
    statement: str
    primary_metric: str
    direction: str
    target_value: float | None = None
    target_unit: str | None = None
    constraints: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.statement.strip():
            raise ValueError("objective statement must be non-empty")
        if self.primary_metric not in {"latency", "parameters", "memory", "quality", "throughput"}:
            raise ValueError("unsupported primary metric")
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be minimize or maximize")
        if self.target_value is not None and (
            isinstance(self.target_value, bool)
            or not math.isfinite(self.target_value)
            or self.target_value <= 0
        ):
            raise ValueError("target_value must be finite and positive")


@dataclass(frozen=True)
class Diagnosis:
    code: str
    severity: str
    finding: str
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5


@dataclass(frozen=True)
class ExperimentHypothesis:
    statement: str
    rationale: str
    expected_effect: dict[str, str]
    falsification_criteria: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentPlan:
    plan_id: str
    objective: OptimizationObjective
    hypothesis: ExperimentHypothesis
    operation: str
    parameters: dict[str, Any]
    required_measurements: tuple[str, ...]
    risk: str
    estimated_value: float
    feasibility: float
    status: str = "proposed"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class IntelligenceReport:
    request_id: str
    model_id: str
    architecture_name: str
    objective: OptimizationObjective
    diagnoses: tuple[Diagnosis, ...]
    plans: tuple[ExperimentPlan, ...]
    reasoning_summary: str
    planner: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReasoningBackend(Protocol):
    """Optional LLM backend returning JSON-compatible plan proposals."""
    def propose(self, *, objective: dict[str, Any], architecture: dict[str, Any],
                diagnoses: list[dict[str, Any]], constraints: dict[str, Any]) -> list[dict[str, Any]]: ...


class CallableReasoningBackend:
    """Adapter for a user-supplied function or provider SDK wrapper."""
    def __init__(self, fn: Callable[..., list[dict[str, Any]]]):
        self.fn = fn

    def propose(self, **kwargs: Any) -> list[dict[str, Any]]:
        result = self.fn(**kwargs)
        if not isinstance(result, list):
            raise ValueError("reasoning backend must return a list of plan proposals")
        return result


class MasterIntelligenceEngine:
    """Turns model goals into ranked, constrained, auditable experiment proposals."""

    ALLOWED_OPERATIONS = {"profile_model", "prune_attention_heads", "resize_attention_heads", "resize_mlp_hidden", "set_mlp_activation", "convert_mlp_to_moe"}
    MEASUREMENTS = ("task_quality", "latency", "parameter_count", "peak_memory", "finite_outputs")

    def __init__(self, *, backend: ReasoningBackend | None = None,
                 max_plans: int = 5, max_parameter_growth_ratio: float = 0.10):
        if not 1 <= max_plans <= 20:
            raise ValueError("max_plans must be between 1 and 20")
        if (
            isinstance(max_parameter_growth_ratio, bool)
            or not math.isfinite(max_parameter_growth_ratio)
            or max_parameter_growth_ratio < 0
        ):
            raise ValueError("max_parameter_growth_ratio must be finite and >= 0")
        self.backend = backend
        self.max_plans = max_plans
        self.max_parameter_growth_ratio = max_parameter_growth_ratio

    @staticmethod
    def _architecture(value: Architecture | Mapping[str, Any]) -> Architecture:
        if isinstance(value, Architecture):
            return value
        if isinstance(value, Mapping):
            return Architecture.from_dict(dict(value))
        raise TypeError("architecture must be an Architecture or architecture mapping")

    @staticmethod
    def parse_objective(statement: str, *, constraints: Mapping[str, float] | None = None) -> OptimizationObjective:
        text = statement.strip()
        if not text:
            raise ValueError("objective statement must be non-empty")
        lowered = text.lower()
        if any(word in lowered for word in ("accuracy", "quality", "loss", "perplexity", "reasoning")):
            metric, direction = "quality", "maximize"
        elif any(word in lowered for word in ("throughput", "tokens per second", "requests per second")):
            metric, direction = "throughput", "maximize"
        elif any(word in lowered for word in ("memory", "vram", "ram")):
            metric, direction = "memory", "minimize"
        elif any(word in lowered for word in ("parameter", "model size", "smaller", "size")):
            metric, direction = "parameters", "minimize"
        else:
            metric, direction = "latency", "minimize"
        match = re.search(r"(?:by|to|under|below|less than|at most)\s*(\d+(?:\.\d+)?)\s*(%|ms|milliseconds|gb|mb)?", lowered)
        target = float(match.group(1)) if match else None
        unit = match.group(2) if match else None
        normalized = {}
        for key, val in (constraints or {}).items():
            if key not in {"max_latency_regression_ratio", "max_parameter_growth_ratio",
                           "minimum_quality_delta", "max_memory_growth_ratio"}:
                raise ValueError(f"unsupported constraint: {key}")
            if isinstance(val, bool):
                raise ValueError(f"{key} must be a finite number")
            number = float(val)
            if not math.isfinite(number):
                raise ValueError(f"{key} must be a finite number")
            if number < 0 and key != "minimum_quality_delta":
                raise ValueError(f"{key} must be >= 0")
            normalized[key] = number
        return OptimizationObjective(text, metric, direction, target, unit, normalized)

    def diagnose(self, architecture: Architecture, objective: OptimizationObjective) -> list[Diagnosis]:
        components = list(architecture.components)
        attention = [c for c in components if "attention" in c.kind.lower() or "attn" in c.name.lower()]
        dropout = [c for c in components if "dropout" in c.kind.lower()]
        diagnoses = []
        if objective.primary_metric in {"latency", "throughput", "memory"}:
            if attention:
                diagnoses.append(Diagnosis("attention_compute_candidate", "medium",
                    "Attention modules are plausible compute or memory optimization targets.",
                    {"attention_modules": [c.name for c in attention],
                     "head_counts": [c.config.get("num_heads") for c in attention
                                     if isinstance(c.config.get("num_heads"), int)]}, 0.72))
            else:
                diagnoses.append(Diagnosis("attention_topology_unknown", "low",
                    "No explicitly named attention module was identified; profile before structural changes.",
                    {"module_count": len(components)}, 0.55))
        if objective.primary_metric in {"parameters", "memory"} and "parameter_count" in architecture.dimensions:
            diagnoses.append(Diagnosis("parameter_footprint_known", "info",
                "The architecture reports a parameter count for size and memory comparisons.",
                {"parameter_count": architecture.dimensions["parameter_count"]}, 0.95))
        if dropout:
            diagnoses.append(Diagnosis("dropout_present", "info",
                "Dropout modules are present; their configuration is not assumed to improve inference.",
                {"dropout_modules": [c.name for c in dropout]}, 0.9))
        if not diagnoses:
            diagnoses.append(Diagnosis("insufficient_architecture_evidence", "low",
                "Architecture metadata is insufficient to justify a structural edit; establish a baseline first.",
                {"module_count": len(components), "known_dimensions": sorted(architecture.dimensions)}, 0.9))
        return diagnoses

    def _rule_plans(self, architecture: Architecture, objective: OptimizationObjective) -> list[dict[str, Any]]:
        plans = [{
            "operation": "profile_model",
            "parameters": {"warmup_iterations": 5, "measurement_iterations": 30},
            "hypothesis": "A reproducible profile will identify the dominant cost before structural mutation.",
            "rationale": "Profiling is the lowest-risk first experiment and establishes comparable baseline evidence.",
            "expected_effect": {"latency": "measured_baseline", "quality": "unchanged"},
            "falsification_criteria": ["Repeated measurements have unstable variance",
                                        "The profile identifies no actionable hotspots"],
            "risk": "low", "estimated_value": 0.55, "feasibility": 0.99,
        }]
        if objective.primary_metric in {"latency", "memory", "parameters", "throughput"}:
            for component in architecture.components:
                heads = component.config.get("num_heads")
                if ("attention" in component.kind.lower() or "attn" in component.name.lower()) and isinstance(heads, int) and heads > 1:
                    plans.append({
                        "operation": "prune_attention_heads",
                        "parameters": {"module_name": component.name, "heads_to_prune": [heads - 1]},
                        "hypothesis": f"Removing one head from {component.name} may reduce attention compute and parameter footprint.",
                        "rationale": "The inspected architecture exposes an attention module and integer head count.",
                        "expected_effect": {"latency": "decrease_expected", "parameters": "decrease_expected",
                                            "quality": "must_be_measured"},
                        "falsification_criteria": ["Task quality falls below the configured non-regression threshold",
                                                   "Latency or memory does not improve beyond measurement noise"],
                        "risk": "medium", "estimated_value": 0.78, "feasibility": 0.72,
                    })
        return plans

    def _validate_proposal(
        self,
        raw: Mapping[str, Any],
        objective: OptimizationObjective,
        architecture: Architecture,
        index: int,
    ) -> ExperimentPlan:
        if not isinstance(raw, Mapping):
            raise ValueError(f"proposal {index}: proposal must be an object")
        operation = raw.get("operation")
        if operation not in self.ALLOWED_OPERATIONS:
            raise ValueError(f"proposal {index}: unsupported operation {operation!r}")
        supplied_params = raw.get("parameters", {})
        if not isinstance(supplied_params, Mapping):
            raise ValueError(f"proposal {index}: parameters must be an object")
        params = {key: value for key, value in supplied_params.items() if value is not None}

        if operation == "prune_attention_heads":
            if set(params) != {"module_name", "heads_to_prune"}:
                raise ValueError(f"proposal {index}: pruning parameters must contain only module_name and heads_to_prune")
            name, heads = params.get("module_name"), params.get("heads_to_prune")
            if not isinstance(name, str) or not name.strip() or not isinstance(heads, list) or not heads:
                raise ValueError(f"proposal {index}: pruning requires module_name and non-empty heads_to_prune")
            if any(isinstance(head, bool) or not isinstance(head, int) or head < 0 for head in heads):
                raise ValueError(f"proposal {index}: head indices must be non-negative integers")
            component = next((item for item in architecture.components if item.name == name), None)
            if component is None or not (
                "attention" in component.kind.lower() or "attn" in component.name.lower()
            ):
                raise ValueError(f"proposal {index}: pruning target is not an inspected attention component")
            head_count = component.config.get("num_heads")
            if (
                not isinstance(head_count, int)
                or isinstance(head_count, bool)
                or any(head >= head_count for head in heads)
                or len(set(heads)) != len(heads)
                or len(set(heads)) >= head_count
            ):
                raise ValueError(
                    f"proposal {index}: pruned head indices must be unique, in range, and leave at least one head"
                )
        elif operation == "resize_attention_heads":
            if set(params) != {"num_heads"}:
                raise ValueError(f"proposal {index}: resize_attention_heads requires only num_heads")
            count = params["num_heads"]
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 128:
                raise ValueError(f"proposal {index}: num_heads must be an integer from 1 to 128")
            if not any(
                "attention" in item.kind.lower() or "attn" in item.name.lower()
                for item in architecture.components
            ):
                raise ValueError(f"proposal {index}: no inspected attention component is available")
            params = {"num_heads": count}
        elif operation == "resize_mlp_hidden":
            if set(params) != {"hidden_size"}:
                raise ValueError(f"proposal {index}: resize_mlp_hidden requires only hidden_size")
            width = params["hidden_size"]
            if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 65536:
                raise ValueError(f"proposal {index}: hidden_size must be an integer from 1 to 65536")
            if not any(
                "mlp" in item.name.lower() or "feedforward" in item.name.lower()
                for item in architecture.components
            ):
                raise ValueError(f"proposal {index}: no inspected MLP component is available")
            params = {"hidden_size": width}
        elif operation == "set_mlp_activation":
            if set(params) != {"activation"}:
                raise ValueError(f"proposal {index}: set_mlp_activation requires only activation")
            activation = params["activation"]
            if activation not in {"gelu", "relu", "silu", "tanh"}:
                raise ValueError(f"proposal {index}: unsupported MLP activation")
            if not any(
                "mlp" in item.name.lower() or "feedforward" in item.name.lower()
                for item in architecture.components
            ):
                raise ValueError(f"proposal {index}: no inspected MLP component is available")
            params = {"activation": activation}
        elif operation == "convert_mlp_to_moe":
            allowed = {"num_experts", "top_k", "expert_noise_std"}
            if set(params) - allowed or not {"num_experts", "top_k"}.issubset(params):
                raise ValueError(f"proposal {index}: convert_mlp_to_moe requires num_experts and top_k")
            experts, top_k = params["num_experts"], params["top_k"]
            noise = params.get("expert_noise_std", 1e-3)
            if isinstance(experts, bool) or not isinstance(experts, int) or not 2 <= experts <= 32:
                raise ValueError(f"proposal {index}: num_experts must be an integer from 2 to 32")
            if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= experts:
                raise ValueError(f"proposal {index}: top_k must be an integer from 1 to num_experts")
            if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not math.isfinite(noise) or not 0 <= noise <= 0.05:
                raise ValueError(f"proposal {index}: expert_noise_std must be between 0 and 0.05")
            if not any(
                "mlp" in item.name.lower() or "feedforward" in item.name.lower()
                for item in architecture.components
            ):
                raise ValueError(f"proposal {index}: no inspected MLP component is available")
            params = {"num_experts": experts, "top_k": top_k, "expert_noise_std": float(noise)}
        else:
            allowed = {"warmup_iterations", "measurement_iterations"}
            if set(params) - allowed:
                raise ValueError(f"proposal {index}: profile_model contains unsupported parameters")
            warmup = params.get("warmup_iterations", 5)
            iterations = params.get("measurement_iterations", 30)
            if (
                isinstance(warmup, bool)
                or not isinstance(warmup, int)
                or not 0 <= warmup <= 1000
            ):
                raise ValueError(f"proposal {index}: warmup_iterations must be an integer from 0 to 1000")
            if (
                isinstance(iterations, bool)
                or not isinstance(iterations, int)
                or not 1 <= iterations <= 1000
            ):
                raise ValueError(f"proposal {index}: measurement_iterations must be an integer from 1 to 1000")
            params = {"warmup_iterations": warmup, "measurement_iterations": iterations}

