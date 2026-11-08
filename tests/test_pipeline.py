"""Controller-owned sandbox -> evaluation -> governance -> ledger workflow."""
from __future__ import annotations
import json, math, re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4
from mlac.benchmark import BenchmarkResult
from mlac.evaluation import EvaluationPolicy, EvaluationResult
from mlac.experiments import ExperimentStore
from mlac.governance import GovernancePolicy
from mlac.attestation import canonical_json, directory_digests, verify_worker_result
from mlac.artifact_registry import ArtifactRegistry
from mlac.runtime_identity import RuntimeIdentityVerifier

class SandboxRunner(Protocol):
    def run(self, *, candidate_dir: str | Path, command: list[str],
            trusted_dir: str | Path | None = None,
            identity_dir: str | Path | None = None) -> Any: ...

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
    attestation: dict[str, Any] | None = None
    artifacts: tuple[dict[str, Any], ...] = ()
    runtime_identity: dict[str, Any] | None = None

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
            "attestation": self.attestation,
            "artifacts": list(self.artifacts),
            "runtime_identity": self.runtime_identity,
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
                 trusted_worker_keys: dict[str, bytes | str], worker_identity_dir: str | Path,
                 worker_id: str = "mlac-evaluator", artifact_registry: ArtifactRegistry | None = None,
                 governance: GovernancePolicy|None=None, evaluation_policy: EvaluationPolicy|None=None,
                 worker_entrypoint: str="/opt/mlac-worker/evaluate", max_report_bytes: int=1024*1024,
                 runtime_identity_verifier: RuntimeIdentityVerifier | None = None,
                 require_runtime_identity: bool = False, remote_artifact_store: Any | None = None):
        if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
            raise ValueError("pipeline image must be pinned by an immutable sha256 digest")
        if getattr(runner, "image", image) != image:
            raise ValueError("pipeline image must match the sandbox runner image")
        if not worker_entrypoint.startswith("/") or "\x00" in worker_entrypoint:
            raise ValueError("worker_entrypoint must be an absolute trusted image path")
        if max_report_bytes < 1: raise ValueError("max_report_bytes must be positive")
        if not trusted_worker_keys: raise ValueError("at least one trusted worker public key is required")
        if not isinstance(worker_id, str) or not worker_id.strip(): raise ValueError("worker_id must be non-empty")
        self.runner,self.image,self.store=runner,image,store
        self.trusted_worker_keys=dict(trusted_worker_keys)
        self.worker_identity_dir=Path(worker_identity_dir).expanduser().resolve(strict=True)
        if not self.worker_identity_dir.is_dir(): raise ValueError("worker_identity_dir must be an existing directory")
        private_key=self.worker_identity_dir / "worker_ed25519_private.pem"
        if not private_key.is_file() or private_key.is_symlink():
            raise ValueError("worker_identity_dir must contain a regular worker_ed25519_private.pem")
        if require_runtime_identity and runtime_identity_verifier is None:
            raise ValueError("runtime_identity_verifier is required when runtime identity is mandatory")
        self.runtime_identity_verifier=runtime_identity_verifier
        self.require_runtime_identity=require_runtime_identity
        self.remote_artifact_store=remote_artifact_store
        self.worker_id=worker_id
        self.artifact_registry=artifact_registry or ArtifactRegistry()
        self.governance=governance or GovernancePolicy()
        self.evaluation_policy=evaluation_policy or EvaluationPolicy()
        self.worker_entrypoint,self.max_report_bytes=worker_entrypoint,max_report_bytes

    def run(self, *, candidate_dir: str|Path, trusted_evaluation_dir: str|Path,
            model_id: str, candidate_id: str,
            operation: str="register-evaluated-candidate") -> PipelineResult:
        if not model_id.strip() or not candidate_id.strip() or not operation.strip():
            raise ValueError("model_id, candidate_id and operation must be non-empty")
        candidate_path=Path(candidate_dir).expanduser().resolve(strict=True)
        trusted_path=Path(trusted_evaluation_dir).expanduser().resolve(strict=True)
        identity_path=self.worker_identity_dir.resolve(strict=True)
        mounts=(candidate_path,trusted_path,identity_path)
        for index,left in enumerate(mounts):
            for right in mounts[index+1:]:
                if left==right or left in right.parents or right in left.parents:
                    raise ValueError("candidate, trusted evaluation, and worker identity directories must be distinct and non-overlapping")
        before_digests = directory_digests(candidate_path, trusted_path)
        execution=self.runner.run(candidate_dir=candidate_path,trusted_dir=trusted_path,
            identity_dir=identity_path,command=[
            self.worker_entrypoint,"--candidate","/workspace/candidate.pt",
            "--trusted","/trusted","--report","/output/report.json",
            "--identity-key","/identity/worker_ed25519_private.pem",
            "--worker-id",self.worker_id,"--worker-image",self.image])
        if execution.timed_out: raise RuntimeError("Candidate worker exceeded its wall-clock limit")
        runtime_identity = None
        if self.require_runtime_identity:
            evidence = getattr(execution, "runtime_identity_evidence", None)
            if not isinstance(evidence, dict):
                raise PermissionError("independent runtime identity evidence is required but was not supplied")
            runtime_identity = self.runtime_identity_verifier.verify(
                evidence, expected_worker_id=self.worker_id, expected_image_digest=self.image)
        if execution.exit_code != 0:
            raise RuntimeError(f"Candidate worker failed with exit code {execution.exit_code}: {execution.stderr[:2000]}")
        report_bytes=execution.output_files.get("report.json")
        if not isinstance(report_bytes,bytes): raise RuntimeError("Trusted worker did not produce /output/report.json")
        if len(report_bytes)>self.max_report_bytes: raise RuntimeError("Worker report exceeds configured report size limit")
        try: envelope=json.loads(report_bytes.decode("utf-8"))
        except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError("Worker report is not valid UTF-8 JSON") from exc
        if not isinstance(envelope,dict) or set(envelope)!={"schema_version","report","attestation"} or isinstance(envelope.get("schema_version"),bool) or envelope.get("schema_version")!=1:
            raise ValueError("worker result envelope schema is invalid")
        report=envelope["report"]
        report_bytes_canonical=canonical_json(report)
        after_digests=directory_digests(candidate_path,trusted_path)
        if before_digests != after_digests:
            raise ValueError("candidate or trusted evaluation inputs changed during worker execution")
        verified=verify_worker_result(attestation=envelope["attestation"],report_bytes=report_bytes_canonical,
            artifact_sha256=after_digests,trusted_worker_keys=self.trusted_worker_keys,expected_image=self.image,
            expected_worker_id=self.worker_id)
        evaluation=evaluate_worker_report(report,policy=self.evaluation_policy)
        self.artifact_registry.record_attestation(nonce=verified["nonce"],key_id=verified["key_id"],
            report_sha256=verified["report_sha256"],evaluation_id=evaluation.evaluation_id)
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
        artifact_bytes={"worker-report.json":report_bytes,
            "worker-report-payload.json":report_bytes_canonical,
            "worker-attestation.json":canonical_json(envelope["attestation"])}
        for role, directory in (("candidate",candidate_path),("trusted",trusted_path)):
            for path in sorted(directory.iterdir(),key=lambda item:item.name):
                artifact_bytes[f"{role}/{path.name}"]=path.read_bytes()
        linked=self.artifact_registry.link_evaluation(evaluation_id=evaluation.evaluation_id,
            model_id=model_id,candidate_id=candidate_id,artifacts=artifact_bytes)
        artifact_records=[]
        for item in linked:
            record={k:v for k,v in item.items() if k!="object_path"}
            if self.remote_artifact_store is not None:
                remote=self.remote_artifact_store.put_bytes(artifact_bytes[item["role"]])
                record["remote"]={"bucket":remote.bucket,"key":remote.key,"version_id":remote.version_id,
                    "sha256":remote.sha256,"retention_until":remote.retention_until,
                    "object_lock_mode":remote.object_lock_mode}
            artifact_records.append(record)
        result=replace(result,attestation=verified,artifacts=tuple(artifact_records),
            runtime_identity=runtime_identity)
        ledger=self.store.record(result,model_id=model_id,candidate_id=candidate_id)
        return replace(result,ledger_record=ledger)



def test_pipeline_accepts_runtime_identity_and_rejects_replay(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from mlac.runtime_identity import RuntimeIdentityVerifier

    pipe, runner, candidate, trusted, _ = make_pipe(tmp_path)
    authority = Ed25519PrivateKey.generate()
    public = authority.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    now = datetime.now(timezone.utc)
    statement = {
        "authority_id": "runtime-authority",
        "worker_id": pipe.worker_id,
        "spiffe_id": f"spiffe://mlac.example/workers/{pipe.worker_id}",
        "image_digest": pipe.image,
        "attestation_type": "tpm2-ima",
        "measurement_sha256": hashlib.sha256(b"measured-runtime").hexdigest(),
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=2)).isoformat(),
        "nonce": "runtimeproofnonce_1234567890",
    }
    runner.runtime_identity_evidence = {
        "schema_version": 1,
        "statement": statement,
        "signature": base64.b64encode(authority.sign(canonical_json(statement))).decode(),
    }
    pipe.runtime_identity_verifier = RuntimeIdentityVerifier(
        trusted_authorities={"runtime-authority": public},
        nonce_db=tmp_path / "runtime-nonces.sqlite3",
        trust_domain="mlac.example",
        clock=lambda: now,
    )
    pipe.require_runtime_identity = True

    result = pipe.run(
        candidate_dir=candidate,
        trusted_evaluation_dir=trusted,
        model_id="base",
        candidate_id="runtime-attested",
    )
    assert result.status == "candidate_authorized"
    assert result.runtime_identity["statement_sha256"]

    with pytest.raises(PermissionError, match="already been used"):
        pipe.run(
            candidate_dir=candidate,
            trusted_evaluation_dir=trusted,
            model_id="base",
            candidate_id="runtime-attested-replay",
        )
