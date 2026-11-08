"""Controller-owned sandbox -> evaluation -> governance -> ledger workflow."""
from __future__ import annotations
import json, math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4
from mlac.benchmark import BenchmarkResult
from mlac.evaluation import EvaluationPolicy, EvaluationResult
from mlac.experiments import ExperimentStore
from mlac.governance import GovernancePolicy

class SandboxRunner(Protocol):
    def run(self, *, candidate_dir: str | Path, command: list[str],
            trusted_dir: str | Path | None = None) -> Any: ...

@dataclass(frozen=True)
class PipelineResult:
    evaluation: EvaluationResult
    model_id: str
    candidate_id: str
    operation: str
    governance_authorized: bool
    governance_reason: str
    sandbox_exit_code: int
    sandbox_timed_out: bool
    sandbox_image: str
    ledger_record: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        if self.sandbox_timed_out: return "execution_timeout"
        if self.sandbox_exit_code != 0: return "execution_failed"
        if not self.evaluation.passed: return "evaluation_failed"
        if not self.evaluation.promotion_eligible: return "evaluation_exploratory"
        if not self.governance_authorized: return "governance_denied"
        return "candidate_authorized"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.evaluation.to_dict(), "model_id": self.model_id,
            "candidate_id": self.candidate_id, "operation": self.operation,
            "workflow_status": self.status,
            "governance": {"authorized": self.governance_authorized, "reason": self.governance_reason},
            "sandbox": {"exit_code": self.sandbox_exit_code, "timed_out": self.sandbox_timed_out,
                        "image": self.sandbox_image},
        }

def _benchmark(value: Any, label: str) -> BenchmarkResult:
    required = {"parameter_count","latency_ms_per_iteration","iterations","output_shape","finite_output"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label} benchmark fields must be exactly {sorted(required)}")
    count, iters, latency, shape, finite = (value[k] for k in
        ("parameter_count","iterations","latency_ms_per_iteration","output_shape","finite_output"))
    if isinstance(count,bool) or not isinstance(count,int) or count < 0:
        raise ValueError(f"{label}.parameter_count must be a non-negative integer")
    if isinstance(iters,bool) or not isinstance(iters,int) or iters < 1:
        raise ValueError(f"{label}.iterations must be a positive integer")
    if isinstance(latency,bool) or not isinstance(latency,(int,float)) or not math.isfinite(latency) or latency < 0:
        raise ValueError(f"{label}.latency_ms_per_iteration must be finite and >= 0")
    if not isinstance(shape,list) or any(isinstance(d,bool) or not isinstance(d,int) or d < 0 for d in shape):
        raise ValueError(f"{label}.output_shape must be a list of non-negative integers")
    if not isinstance(finite,bool): raise ValueError(f"{label}.finite_output must be boolean")
    return BenchmarkResult(count,float(latency),iters,tuple(shape),finite)

def _quality(value: Any, label: str) -> dict[str,float]:
    if not isinstance(value,dict) or not value: raise ValueError(f"{label} must be a non-empty metric object")
    result={}
    for key, raw in value.items():
        if not isinstance(key,str) or not key.strip(): raise ValueError("quality metric names must be non-empty")
        if isinstance(raw,bool) or not isinstance(raw,(int,float)) or not math.isfinite(raw):
            raise ValueError(f"quality metric {key!r} must be finite numeric")
        result[key]=float(raw)
    return result

def evaluate_worker_report(report: dict[str,Any], *, policy: EvaluationPolicy|None=None) -> EvaluationResult:
    policy=policy or EvaluationPolicy()
    expected={"schema_version","baseline","candidate","quality_baseline","quality_candidate"}
    if (not isinstance(report,dict) or isinstance(report.get("schema_version"),bool)
            or report.get("schema_version") != 1):
        raise ValueError("worker report schema_version must be integer 1")
    if set(report) != expected: raise ValueError(f"worker report fields must be exactly {sorted(expected)}")
    base=_benchmark(report["baseline"],"baseline")
    cand=_benchmark(report["candidate"],"candidate")
    qb=_quality(report["quality_baseline"],"quality_baseline")
    qc=_quality(report["quality_candidate"],"quality_candidate")
    metric=policy.quality_metric
    if metric not in qb or metric not in qc: raise ValueError(f"worker report must include quality metric {metric!r}")
    ld=(cand.latency_ms_per_iteration-base.latency_ms_per_iteration)/max(base.latency_ms_per_iteration,1e-12)
    pd=(cand.parameter_count-base.parameter_count)/max(base.parameter_count,1)
    qd=qc[metric]-qb[metric]
    checks={
        "finite_outputs":base.finite_output and cand.finite_output,
        "same_output_shape":base.output_shape==cand.output_shape,
        "latency_within_budget":ld<=policy.max_latency_regression_ratio,
        "parameter_growth_within_budget":pd<=policy.max_parameter_growth_ratio,
        "quality_non_regression":qd>=policy.minimum_quality_delta,
        "quality_evaluated":True,
    }
    reasons=[]
    for key,message in [
        ("finite_outputs","one or both models produced non-finite output"),
        ("same_output_shape","candidate output shape differs from baseline"),
        ("latency_within_budget","candidate latency regression exceeds policy"),
        ("parameter_growth_within_budget","candidate parameter growth exceeds policy"),
        ("quality_non_regression",f"quality metric {metric!r} delta {qd:.6g} is below required {policy.minimum_quality_delta:.6g}")]:
        if not checks[key]: reasons.append(message)
    passed=all(checks[k] for k in ("finite_outputs","same_output_shape","latency_within_budget","parameter_growth_within_budget","quality_non_regression"))
    return EvaluationResult(
        evaluation_id=str(uuid4()),created_at=datetime.now(timezone.utc).isoformat(),
        baseline=base,candidate=cand,quality_baseline=qb,quality_candidate=qc,quality_delta=qd,
        latency_regression_ratio=ld,parameter_growth_ratio=pd,checks=checks,passed=passed,
        promotion_eligible=passed,reasons=tuple(reasons))

class CandidateEvaluationPipeline:
    """Fixed worker -> strict report validation -> controller gates -> candidate governance -> ledger."""
    def __init__(self, *, runner: SandboxRunner, image: str, store: ExperimentStore,
                 governance: GovernancePolicy|None=None, evaluation_policy: EvaluationPolicy|None=None,
                 worker_entrypoint: str="/opt/mlac-worker/evaluate", max_report_bytes: int=1024*1024):
        if not worker_entrypoint.startswith("/") or "\x00" in worker_entrypoint:
            raise ValueError("worker_entrypoint must be an absolute trusted image path")
        if max_report_bytes < 1: raise ValueError("max_report_bytes must be positive")
        self.runner,self.image,self.store=runner,image,store
        self.governance=governance or GovernancePolicy()
        self.evaluation_policy=evaluation_policy or EvaluationPolicy()
        self.worker_entrypoint,self.max_report_bytes=worker_entrypoint,max_report_bytes

    def run(self, *, candidate_dir: str|Path, trusted_evaluation_dir: str|Path,
            model_id: str, candidate_id: str,
            operation: str="register-evaluated-candidate") -> PipelineResult:
        if not model_id.strip() or not candidate_id.strip() or not operation.strip():
            raise ValueError("model_id, candidate_id and operation must be non-empty")
        execution=self.runner.run(candidate_dir=candidate_dir,trusted_dir=trusted_evaluation_dir,command=[
            self.worker_entrypoint,"--candidate","/workspace/candidate.pt",
            "--trusted","/trusted","--report","/output/report.json"])
        if execution.timed_out: raise RuntimeError("Candidate worker exceeded its wall-clock limit")
        if execution.exit_code != 0:
            raise RuntimeError(f"Candidate worker failed with exit code {execution.exit_code}: {execution.stderr[:2000]}")
        report_bytes=execution.output_files.get("report.json")
        if not isinstance(report_bytes,bytes): raise RuntimeError("Trusted worker did not produce /output/report.json")
        if len(report_bytes)>self.max_report_bytes: raise RuntimeError("Worker report exceeds configured report size limit")
        try: report=json.loads(report_bytes.decode("utf-8"))
        except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError("Worker report is not valid UTF-8 JSON") from exc
        evaluation=evaluate_worker_report(report,policy=self.evaluation_policy)
        authorized=False
        reason="Evaluation did not pass; candidate authorization was not requested"
        if evaluation.promotion_eligible:
            try:
                self.governance.authorize(target="candidate",operation=operation,
                    parameter_growth_ratio=max(0.0,evaluation.parameter_growth_ratio))
                authorized=True
                reason="Candidate operation authorized by controller governance"
            except (PermissionError,ValueError) as exc: reason=str(exc)
        result=PipelineResult(evaluation,model_id,candidate_id,operation,authorized,reason,
            execution.exit_code,execution.timed_out,self.image)
        ledger=self.store.record(result,model_id=model_id,candidate_id=candidate_id)
        return replace(result,ledger_record=ledger)
