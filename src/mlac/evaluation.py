"""Baseline-versus-candidate evaluation with explicit promotion eligibility gates.

Evaluation is evidence, not authorization: this module never deploys, promotes,
or mutates either model. The caller supplies a task-specific evaluator because
MLAC cannot infer task quality from latency or parameter counts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import Callable, Mapping
from uuid import uuid4

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised in no-torch installs
    raise ImportError("Candidate evaluation requires 'pip install mlac[torch]'") from exc

from mlac.benchmark import BenchmarkResult, benchmark_module

QualityEvaluator = Callable[[nn.Module, Tensor], Mapping[str, float]]


@dataclass(frozen=True)
class EvaluationPolicy:
    """Hard limits for a candidate comparison.

    Quality evaluation is mandatory by default. Set ``require_quality=False``
    only for explicitly exploratory performance-only experiments; such results
    remain ineligible for promotion.
    """

    max_latency_regression_ratio: float = 0.10
    max_parameter_growth_ratio: float = 0.10
    quality_metric: str = "score"
    minimum_quality_delta: float = 0.0
    require_quality: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.max_latency_regression_ratio) or self.max_latency_regression_ratio < 0:
            raise ValueError("max_latency_regression_ratio must be finite and >= 0")
        if not isfinite(self.max_parameter_growth_ratio) or self.max_parameter_growth_ratio < 0:
            raise ValueError("max_parameter_growth_ratio must be finite and >= 0")
        if not self.quality_metric.strip():
            raise ValueError("quality_metric must be non-empty")
        if not isfinite(self.minimum_quality_delta):
            raise ValueError("minimum_quality_delta must be finite")


@dataclass(frozen=True)
class EvaluationResult:
    evaluation_id: str
    created_at: str
    baseline: BenchmarkResult
    candidate: BenchmarkResult
    quality_baseline: dict[str, float] = field(default_factory=dict)
    quality_candidate: dict[str, float] = field(default_factory=dict)
    quality_delta: float | None = None
    latency_regression_ratio: float = 0.0
    parameter_growth_ratio: float = 0.0
    checks: dict[str, bool] = field(default_factory=dict)
    passed: bool = False
    promotion_eligible: bool = False
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def _quality_values(evaluator: QualityEvaluator, module: nn.Module,
                    sample: Tensor) -> dict[str, float]:
    raw = evaluator(module, sample)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("quality evaluator must return a non-empty metric mapping")
    values: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("quality metric names must be non-empty strings")
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"quality metric {key!r} must be finite")
        values[key] = number
    return values


def evaluate_candidate(
    baseline: nn.Module,
    candidate: nn.Module,
    sample: Tensor,
    *,
    quality_evaluator: QualityEvaluator | None = None,
    policy: EvaluationPolicy | None = None,
    iterations: int = 20,
    warmup: int = 5,
) -> EvaluationResult:
    """Evaluate two modules under identical inputs and return a non-deploying scorecard."""
    policy = policy or EvaluationPolicy()
    baseline_benchmark = benchmark_module(baseline, sample, iterations=iterations, warmup=warmup)
    candidate_benchmark = benchmark_module(candidate, sample, iterations=iterations, warmup=warmup)

    latency_delta = (
        candidate_benchmark.latency_ms_per_iteration - baseline_benchmark.latency_ms_per_iteration
    ) / max(baseline_benchmark.latency_ms_per_iteration, 1e-12)
    parameter_delta = (
        candidate_benchmark.parameter_count - baseline_benchmark.parameter_count
    ) / max(baseline_benchmark.parameter_count, 1)

    quality_baseline: dict[str, float] = {}
    quality_candidate: dict[str, float] = {}
    quality_delta: float | None = None
    quality_ok = False
    reasons: list[str] = []
    if quality_evaluator is not None:
        baseline_training, candidate_training = baseline.training, candidate.training
        baseline.eval()
        candidate.eval()
        try:
            with torch.inference_mode():
                quality_baseline = _quality_values(quality_evaluator, baseline, sample)
                quality_candidate = _quality_values(quality_evaluator, candidate, sample)
        finally:
            baseline.train(baseline_training)
            candidate.train(candidate_training)
        metric = policy.quality_metric
        if metric not in quality_baseline or metric not in quality_candidate:
            raise ValueError(f"quality evaluator must return configured metric {metric!r} for both models")
        quality_delta = quality_candidate[metric] - quality_baseline[metric]
        quality_ok = quality_delta >= policy.minimum_quality_delta
        if not quality_ok:
            reasons.append(
                f"quality metric {metric!r} delta {quality_delta:.6g} is below "
                f"required {policy.minimum_quality_delta:.6g}"
            )
    elif policy.require_quality:
        reasons.append("task-specific quality evaluation is required but was not supplied")

    finite_ok = baseline_benchmark.finite_output and candidate_benchmark.finite_output
    shape_ok = baseline_benchmark.output_shape == candidate_benchmark.output_shape
    latency_ok = latency_delta <= policy.max_latency_regression_ratio
    parameters_ok = parameter_delta <= policy.max_parameter_growth_ratio
    if not finite_ok:
        reasons.append("one or both models produced non-finite output")
    if not shape_ok:
        reasons.append("candidate output shape differs from baseline")
    if not latency_ok:
        reasons.append("candidate latency regression exceeds policy")
    if not parameters_ok:
        reasons.append("candidate parameter growth exceeds policy")

    checks = {
        "finite_outputs": finite_ok,
        "same_output_shape": shape_ok,
        "latency_within_budget": latency_ok,
        "parameter_growth_within_budget": parameters_ok,
        "quality_non_regression": quality_ok if quality_evaluator is not None else not policy.require_quality,
        "quality_evaluated": quality_evaluator is not None,
    }
    passed = all(checks[k] for k in (
        "finite_outputs", "same_output_shape", "latency_within_budget",
        "parameter_growth_within_budget", "quality_non_regression",
    ))
    promotion_eligible = passed and quality_evaluator is not None
    if passed and not promotion_eligible:
        reasons.append("performance-only result is exploratory and is not promotion-eligible")

    return EvaluationResult(
        evaluation_id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
        baseline=baseline_benchmark, candidate=candidate_benchmark,
        quality_baseline=quality_baseline, quality_candidate=quality_candidate,
        quality_delta=quality_delta, latency_regression_ratio=latency_delta,
        parameter_growth_ratio=parameter_delta, checks=checks, passed=passed,
        promotion_eligible=promotion_eligible, reasons=tuple(reasons),
    )
