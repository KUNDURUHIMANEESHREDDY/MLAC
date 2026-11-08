import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from mlac.attestation import canonical_json, directory_digests, generate_worker_keypair, sign_worker_result
from mlac.artifact_registry import ArtifactRegistry
from mlac.evaluation import EvaluationPolicy
from mlac.experiments import ExperimentStore
from mlac.governance import GovernancePolicy
from mlac.pipeline import CandidateEvaluationPipeline, evaluate_worker_report

def bm(params, latency=1.0):
    return {"parameter_count":params,"latency_ms_per_iteration":latency,"iterations":10,
            "output_shape":[2,4],"finite_output":True}

def report(params=105, latency=1.02, quality=.9):
    return {"schema_version":1,"baseline":bm(100),"candidate":bm(params,latency),
            "quality_baseline":{"score":.9},"quality_candidate":{"score":quality}}

class FakeRunner:
    def __init__(self,payload):
        self.payload=payload
        self.command=None
        self.candidate_dir=None
        self.trusted_dir=None
        self.identity_dir=None
        self.tamper=False
    def run(self,*,candidate_dir,command,trusted_dir=None,identity_dir=None):
        self.candidate_dir,self.command,self.trusted_dir,self.identity_dir=candidate_dir,command,trusted_dir,identity_dir
        def arg(name): return command[command.index(name)+1]
        report_bytes=canonical_json(self.payload)
        attestation=sign_worker_result(report_bytes=report_bytes,
            artifact_sha256=directory_digests(candidate_dir,trusted_dir),
            private_key_path=Path(identity_dir)/"worker_ed25519_private.pem",worker_id=arg("--worker-id"),worker_image=arg("--worker-image"))
        if self.tamper: attestation["worker_image"]="attacker@sha256:"+"b"*64
        envelope={"schema_version":1,"report":self.payload,"attestation":attestation}
        return SimpleNamespace(exit_code=0,timed_out=False,stderr="",
            output_files={"report.json":canonical_json(envelope)})

def dirs(tmp_path):
    candidate=tmp_path/"candidate"
    trusted=tmp_path/"trusted"
    identity=tmp_path/"identity"
    candidate.mkdir(); trusted.mkdir(); identity.mkdir()
    (candidate/"candidate.pt").write_bytes(b"candidate checkpoint placeholder")
    (trusted/"manifest.json").write_text("operator-controlled")
    key_id=generate_worker_keypair(identity/"worker_ed25519_private.pem",tmp_path/"worker-public.pem")
    public=(tmp_path/"worker-public.pem").read_bytes()
    return candidate,trusted,identity,key_id,public

def make_pipe(tmp_path, payload=None, *, policy=None, governance=None, tamper=False):
    candidate,trusted,identity,key_id,public=dirs(tmp_path)
    runner=FakeRunner(payload or report())
    runner.tamper=tamper
    registry=ArtifactRegistry(tmp_path/"artifacts")
    pipe=CandidateEvaluationPipeline(runner=runner,image="registry.example/worker@sha256:"+"a"*64,
        store=ExperimentStore(tmp_path/"ledger.sqlite"),trusted_worker_keys={key_id:public},
        worker_identity_dir=identity,artifact_registry=registry,evaluation_policy=policy,governance=governance)
    return pipe,runner,candidate,trusted,registry

def test_report_is_validated_and_scored():
    result=evaluate_worker_report(report())
    assert result.passed and result.promotion_eligible
    assert result.parameter_growth_ratio==pytest.approx(.05)

def test_report_rejects_unknown_schema_and_nonfinite_metrics():
    with pytest.raises(ValueError,match="schema_version"): evaluate_worker_report({"schema_version":9})
    bad=report()
    bad["candidate"]["latency_ms_per_iteration"]=float("nan")
    with pytest.raises(ValueError,match="finite"): evaluate_worker_report(bad)

def test_pipeline_verifies_worker_signature_and_records_artifacts(tmp_path):
    pipe,runner,candidate,trusted,registry=make_pipe(tmp_path)
    result=pipe.run(candidate_dir=candidate,trusted_evaluation_dir=trusted,model_id="base",candidate_id="c1")
    assert result.status=="candidate_authorized" and result.governance_authorized
    assert runner.command==["/opt/mlac-worker/evaluate","--candidate","/workspace/candidate.pt",
        "--trusted","/trusted","--report","/output/report.json","--identity-key",
        "/identity/worker_ed25519_private.pem","--worker-id","mlac-evaluator","--worker-image",pipe.image]
    assert runner.candidate_dir==candidate and runner.trusted_dir==trusted and runner.identity_dir==pipe.worker_identity_dir
    assert result.ledger_record["status"]=="eligible"
    assert pipe.store.get(result.evaluation.evaluation_id)["result"]["workflow_status"]=="candidate_authorized"
    records=registry.list_evaluation(result.evaluation.evaluation_id)
    assert {r["role"] for r in records} >= {"worker-report.json","worker-attestation.json","candidate/candidate.pt","trusted/manifest.json"}
    for item in records: assert registry.get(item["sha256"])["sha256"]==item["sha256"]
    assert result.attestation["worker_id"]=="mlac-evaluator"

def test_pipeline_rejects_wrong_worker_image_attestation(tmp_path):
    pipe,runner,candidate,trusted,_=make_pipe(tmp_path,tamper=True)
    with pytest.raises(ValueError,match="image"):
        pipe.run(candidate_dir=candidate,trusted_evaluation_dir=trusted,model_id="base",candidate_id="tampered")

def test_pipeline_records_governance_denial(tmp_path):
    pipe,runner,candidate,trusted,_=make_pipe(tmp_path,report(params=120),
        policy=EvaluationPolicy(max_parameter_growth_ratio=.25),
        governance=GovernancePolicy(max_parameter_growth_ratio=.1))
    result=pipe.run(candidate_dir=candidate,trusted_evaluation_dir=trusted,model_id="base",candidate_id="large")
    assert result.evaluation.passed and result.status=="governance_denied"
    assert not result.governance_authorized and result.ledger_record["status"]=="governance_denied"

def test_pipeline_does_not_authorize_quality_regression(tmp_path):
    pipe,runner,candidate,trusted,_=make_pipe(tmp_path,report(quality=.5))
    result=pipe.run(candidate_dir=candidate,trusted_evaluation_dir=trusted,model_id="base",candidate_id="bad")
    assert result.status=="evaluation_failed" and not result.governance_authorized

def test_pipeline_worker_entrypoint_is_trusted_configuration(tmp_path):
    candidate,trusted,identity,key_id,public=dirs(tmp_path)
    with pytest.raises(ValueError,match="absolute trusted image path"):
        CandidateEvaluationPipeline(runner=FakeRunner(report()),image="worker@sha256:"+"a"*64,
            store=ExperimentStore(tmp_path/"ledger.sqlite"),trusted_worker_keys={key_id:public},
            worker_identity_dir=identity,worker_entrypoint="candidate.py")

def test_pipeline_allows_parameter_reduction_under_growth_budget(tmp_path):
    pipe,runner,candidate,trusted,_=make_pipe(tmp_path,report(params=80))
    result=pipe.run(candidate_dir=candidate,trusted_evaluation_dir=trusted,model_id="base",candidate_id="smaller")
    assert result.status=="candidate_authorized" and result.evaluation.parameter_growth_ratio<0

def test_report_rejects_boolean_schema_version():
    bad=report(); bad["schema_version"]=True
    with pytest.raises(ValueError,match="integer 1"): evaluate_worker_report(bad)

def test_controller_detects_changed_inputs_during_execution(tmp_path):
    class MutatingRunner(FakeRunner):
        def run(self,**kwargs):
            result=super().run(**kwargs)
            (kwargs["candidate_dir"]/"candidate.pt").write_bytes(b"changed after signature")
            return result
    pipe,_,candidate,trusted,_=make_pipe(tmp_path)
    pipe.runner=MutatingRunner(report())
    with pytest.raises(ValueError,match="changed during"):
        pipe.run(candidate_dir=candidate,trusted_evaluation_dir=trusted,model_id="base",candidate_id="race")

def test_pipeline_rejects_replayed_worker_attestation(tmp_path):
    pipe,_,candidate,trusted,_=make_pipe(tmp_path)
    class CachedRunner(FakeRunner):
        def __init__(self):
            super().__init__(report())
            self.cached=None
        def run(self,**kwargs):
            if self.cached is None:
                self.cached=super().run(**kwargs)
            return self.cached
    pipe.runner=CachedRunner()
    pipe.run(candidate_dir=candidate,trusted_evaluation_dir=trusted,model_id="base",candidate_id="first")
    with pytest.raises(ValueError,match="already been used"):
        pipe.run(candidate_dir=candidate,trusted_evaluation_dir=trusted,model_id="base",candidate_id="replay")
