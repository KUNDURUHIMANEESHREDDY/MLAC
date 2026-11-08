"""Master Intelligence Engine: objective analysis, diagnosis, and governed experiment planning.

Proposal-only: never executes model code, changes weights, or authorizes promotion.
"""
from __future__ import annotations
import json
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
        if self.target_value is not None and self.target_value <= 0:
            raise ValueError("target_value must be positive")


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

    ALLOWED_OPERATIONS = {"profile_model", "prune_attention_heads", "reduce_dropout"}
    MEASUREMENTS = ("task_quality", "latency", "parameter_count", "peak_memory", "finite_outputs")

    def __init__(self, *, backend: ReasoningBackend | None = None,
                 max_plans: int = 5, max_parameter_growth_ratio: float = 0.10):
        if not 1 <= max_plans <= 20:
            raise ValueError("max_plans must be between 1 and 20")
        if max_parameter_growth_ratio < 0:
            raise ValueError("max_parameter_growth_ratio must be >= 0")
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
            number = float(val)
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

    def _validate_proposal(self, raw: Mapping[str, Any], objective: OptimizationObjective, architecture: Architecture, index: int) -> ExperimentPlan:
        operation = raw.get("operation")
        if operation not in self.ALLOWED_OPERATIONS:
            raise ValueError(f"proposal {index}: unsupported operation {operation!r}")
        params = raw.get("parameters", {})
        if not isinstance(params, Mapping):
            raise ValueError(f"proposal {index}: parameters must be an object")
        if operation == "prune_attention_heads":
            name, heads = params.get("module_name"), params.get("heads_to_prune")
            if not isinstance(name, str) or not name.strip() or not isinstance(heads, list) or not heads:
                raise ValueError(f"proposal {index}: pruning requires module_name and non-empty heads_to_prune")
            if any(isinstance(h, bool) or not isinstance(h, int) or h < 0 for h in heads):
                raise ValueError(f"proposal {index}: head indices must be non-negative integers")
            component = next((c for c in architecture.components if c.name == name), None)
            if component is None or not ("attention" in component.kind.lower() or "attn" in component.name.lower()):
                raise ValueError(f"proposal {index}: pruning target is not an inspected attention component")
            head_count = component.config.get("num_heads")
            if not isinstance(head_count, int) or any(h >= head_count for h in heads) or len(set(heads)) != len(heads) or len(set(heads)) >= head_count:
                raise ValueError(f"proposal {index}: pruned head indices must be unique, in range, and leave at least one head")
        if operation == "reduce_dropout":
            name, probability = params.get("module_name"), params.get("p")
            component = next((c for c in architecture.components if c.name == name), None)
            if component is None or "dropout" not in component.kind.lower():
                raise ValueError(f"proposal {index}: dropout target is not an inspected dropout component")
            if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                raise ValueError(f"proposal {index}: dropout p must be between 0 and 1")
        for key in ("hypothesis", "rationale"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise ValueError(f"proposal {index}: {key} must be non-empty")
        expected, criteria = raw.get("expected_effect", {}), raw.get("falsification_criteria", [])
        if not isinstance(expected, Mapping) or not isinstance(criteria, list) or not criteria:
            raise ValueError(f"proposal {index}: expected_effect and falsification criteria are required")
        risk = raw.get("risk", "medium")
        if risk not in {"low", "medium", "high"}:
            raise ValueError(f"proposal {index}: risk must be low, medium, or high")
        value, feasibility = float(raw.get("estimated_value", 0.5)), float(raw.get("feasibility", 0.5))
        if not 0 <= value <= 1 or not 0 <= feasibility <= 1:
            raise ValueError(f"proposal {index}: value and feasibility must be between 0 and 1")
        hypothesis = ExperimentHypothesis(raw["hypothesis"], raw["rationale"],
            {str(k): str(v) for k, v in expected.items()}, tuple(str(x) for x in criteria))
        return ExperimentPlan(str(uuid4()), objective, hypothesis, operation, dict(params),
            self.MEASUREMENTS, risk, value, feasibility)

    def plan(self, *, model_id: str, architecture: Architecture | Mapping[str, Any],
             objective: str, constraints: Mapping[str, float] | None = None) -> IntelligenceReport:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        arch = self._architecture(architecture)
        goal = self.parse_objective(objective, constraints=constraints)
        diagnoses = self.diagnose(arch, goal)
        raw_plans = self._rule_plans(arch, goal)
        planner = "rules"
        if self.backend is not None:
            proposals = self.backend.propose(objective=asdict(goal), architecture=arch.to_dict(),
                diagnoses=[asdict(d) for d in diagnoses],
                constraints={"allowed_operations": sorted(self.ALLOWED_OPERATIONS),
                             "max_parameter_growth_ratio": self.max_parameter_growth_ratio})
            if not isinstance(proposals, list):
                raise ValueError("reasoning backend must return a list")
            raw_plans.extend(dict(p) for p in proposals)
            planner = "rules+llm"
        plans, seen = [], set()
        for index, raw in enumerate(raw_plans):
            plan = self._validate_proposal(raw, goal, arch, index)
            fingerprint = json.dumps([plan.operation, plan.parameters], sort_keys=True, default=str)
            if fingerprint not in seen:
                plans.append(plan)
                seen.add(fingerprint)
        plans.sort(key=lambda p: (p.estimated_value * p.feasibility,
                                  {"low": 2, "medium": 1, "high": 0}[p.risk]), reverse=True)
        plans = plans[:self.max_plans]
        summary = (f"Analyzed {len(arch.components)} components and {len(arch.tensors)} tensors. "
                   f"Found {len(diagnoses)} diagnostic signals and ranked {len(plans)} experiment plans. "
                   "Plans are proposals; execution, evaluation, and authorization remain separate.")
        return IntelligenceReport(str(uuid4()), model_id, arch.name, goal,
            tuple(diagnoses), tuple(plans), summary, planner)
