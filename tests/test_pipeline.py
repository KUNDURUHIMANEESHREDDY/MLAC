import json
from types import SimpleNamespace
import pytest
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
    def run(self,*,candidate_dir,command):
        self.candidate_dir,self.command=candidate_dir,command
        return SimpleNamespace(exit_code=0,timed_out=False,stderr="",
            output_files={"report.json":json.dumps(self.payload).encode()})

def test_report_is_validated_and_scored():
    result=evaluate_worker_report(report())
    assert result.passed and result.promotion_eligible
    assert result.parameter_growth_ratio==pytest.approx(.05)

def test_report_rejects_unknown_schema_and_nonfinite_metrics():
    with pytest.raises(ValueError,match="schema_version"): evaluate_worker_report({"schema_version":9})
    bad=report()
    bad["candidate"]["latency_ms_per_iteration"]=float("nan")
    with pytest.raises(ValueError,match="finite"): evaluate_worker_report(bad)

def test_pipeline_runs_fixed_worker_governs_and_records(tmp_path):
    runner=FakeRunner(report())
    store=ExperimentStore(tmp_path/"ledger.sqlite")
    pipe=CandidateEvaluationPipeline(runner=runner,image="worker@sha256:"+"a"*64,store=store)
    result=pipe.run(candidate_dir=tmp_path,model_id="base",candidate_id="c1")
    assert result.status=="candidate_authorized" and result.governance_authorized
    assert runner.command==["/opt/mlac-worker/evaluate","--input","/workspace","--report","/output/report.json"]
    assert result.ledger_record["status"]=="eligible"
    assert store.get(result.evaluation.evaluation_id)["result"]["workflow_status"]=="candidate_authorized"

def test_pipeline_records_governance_denial(tmp_path):
    runner=FakeRunner(report(params=120))
    pipe=CandidateEvaluationPipeline(runner=runner,image="worker@sha256:"+"a"*64,
        store=ExperimentStore(tmp_path/"ledger.sqlite"),
        evaluation_policy=EvaluationPolicy(max_parameter_growth_ratio=.25),
        governance=GovernancePolicy(max_parameter_growth_ratio=.1))
    result=pipe.run(candidate_dir=tmp_path,model_id="base",candidate_id="large")
    assert result.evaluation.passed and result.status=="governance_denied"
    assert not result.governance_authorized
    assert result.ledger_record["status"]=="governance_denied"

def test_pipeline_does_not_authorize_quality_regression(tmp_path):
    pipe=CandidateEvaluationPipeline(runner=FakeRunner(report(quality=.5)),
        image="worker@sha256:"+"a"*64,store=ExperimentStore(tmp_path/"ledger.sqlite"))
    result=pipe.run(candidate_dir=tmp_path,model_id="base",candidate_id="bad")
    assert result.status=="evaluation_failed" and not result.governance_authorized

def test_pipeline_worker_entrypoint_is_trusted_configuration(tmp_path):
    with pytest.raises(ValueError,match="absolute trusted image path"):
        CandidateEvaluationPipeline(runner=FakeRunner(report()),image="worker@sha256:"+"a"*64,
            store=ExperimentStore(tmp_path/"ledger.sqlite"),worker_entrypoint="candidate.py")

def test_pipeline_allows_parameter_reduction_under_growth_budget(tmp_path):
    runner=FakeRunner(report(params=80))
    pipe=CandidateEvaluationPipeline(runner=runner,image="worker@sha256:"+"a"*64,
        store=ExperimentStore(tmp_path/"ledger.sqlite"))
    result=pipe.run(candidate_dir=tmp_path,model_id="base",candidate_id="smaller")
    assert result.status=="candidate_authorized"
    assert result.evaluation.parameter_growth_ratio < 0

def test_report_rejects_boolean_schema_version():
    bad=report()
    bad["schema_version"]=True
    with pytest.raises(ValueError,match="integer 1"):
        evaluate_worker_report(bad)
