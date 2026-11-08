import sqlite3
import pytest

from mlac.self_evolution import (
    ControllerEvolutionLedger,
    ControllerEvolutionResult,
    ControllerSelfEvolutionWorkflow,
)


def test_controller_evolution_ledger_is_durable_and_append_only(tmp_path):
    ledger = ControllerEvolutionLedger(tmp_path / "evolution.sqlite3")
    result = ControllerEvolutionResult(
        run_id="run-1",
        model_id="mlac-controller-local",
        objective="Improve held-out planning accuracy",
        architecture_sha256="a" * 64,
        proposal_count=2,
        experiment_results=({"plan_id": "p1", "pipeline_result": {"workflow_status": "candidate_authorized"}},),
        candidate_authorized_ids=("p1",),
        promotion_performed=False,
        created_at="2026-11-08T12:00:00+00:00",
    )
    saved = ledger.append(result)
    assert saved["candidate_authorized_ids"] == ["p1"]
    assert saved["promotion_performed"] is False
    assert ledger.list(model_id="mlac-controller-local")[0]["run_id"] == "run-1"
    with pytest.raises(sqlite3.IntegrityError):
        ledger.append(result)


def test_self_evolution_workflow_uses_trusted_experiment_workflow_and_never_promotes(tmp_path):
    class Report:
        plans = [object(), object()]

    class FakeWorkflow:
        def run(self, **kwargs):
            self.kwargs = kwargs
            return type("WorkflowResult", (), {
                "report": Report(),
                "experiments": (
                    {"plan_id": "p1", "pipeline_result": {"workflow_status": "candidate_authorized"}},
                    {"plan_id": "p2", "pipeline_result": {"workflow_status": "governance_denied"}},
                ),
            })()

    workflow = FakeWorkflow()
    ledger = ControllerEvolutionLedger(tmp_path / "history.sqlite3")
    service = ControllerSelfEvolutionWorkflow(intelligence_workflow=workflow, ledger=ledger)
    result = service.run(
        controller_model_id="mlac-controller-local",
        architecture={"layers": 4, "width": 64},
        objective="Improve planning quality without increasing latency",
        trusted_evaluation_dir=tmp_path,
        max_experiments=2,
    )
    assert workflow.kwargs["model_id"] == "mlac-controller-local"
    assert result.candidate_authorized_ids == ("p1",)
    assert result.promotion_performed is False
    assert ledger.get(result.run_id)["architecture_sha256"] == result.architecture_sha256


def test_self_evolution_rejects_invalid_inputs(tmp_path):
    service = ControllerSelfEvolutionWorkflow(
        intelligence_workflow=type("Fake", (), {"run": lambda self, **kwargs: None})(),
        ledger=ControllerEvolutionLedger(tmp_path / "history.sqlite3"),
    )
    with pytest.raises(ValueError, match="controller_model_id"):
        service.run(controller_model_id="", architecture={}, objective="x",
                    trusted_evaluation_dir=tmp_path)
