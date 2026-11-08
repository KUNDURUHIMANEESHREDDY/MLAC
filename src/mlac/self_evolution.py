"""Durable recursive controller-evaluation cycles with immutable safety boundaries.

This workflow can evaluate a controller's own open-weight model when the
operator supplies its architecture and trusted evaluation bundle. It records
proposals and candidate outcomes but never rewrites controller code, policy,
authorization, recovery, or deploys a candidate.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class ControllerEvolutionResult:
    run_id: str
    model_id: str
    objective: str
    architecture_sha256: str
    proposal_count: int
    experiment_results: tuple[dict[str, Any], ...]
    candidate_authorized_ids: tuple[str, ...]
    promotion_performed: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "objective": self.objective,
            "architecture_sha256": self.architecture_sha256,
            "proposal_count": self.proposal_count,
            "experiment_results": list(self.experiment_results),
            "candidate_authorized_ids": list(self.candidate_authorized_ids),
            "promotion_performed": self.promotion_performed,
            "created_at": self.created_at,
        }


class ControllerEvolutionLedger:
    """Append-only SQLite history for recursive controller improvement cycles."""

    def __init__(self, database: str | Path = "mlac-controller-evolution.sqlite3"):
        self.database = str(database)
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS controller_evolution_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    architecture_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )"""
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_controller_evolution_created "
                "ON controller_evolution_runs(created_at)"
            )

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def append(self, result: ControllerEvolutionResult) -> dict[str, Any]:
        if not isinstance(result, ControllerEvolutionResult):
            raise TypeError("result must be a ControllerEvolutionResult")
        payload = _canonical(result.to_dict()).decode("utf-8")
        with self._connect() as db:
            db.execute(
                "INSERT INTO controller_evolution_runs "
                "(run_id,created_at,model_id,architecture_sha256,payload_json) VALUES (?,?,?,?,?)",
                (result.run_id, result.created_at, result.model_id,
                 result.architecture_sha256, payload),
            )
        return self.get(result.run_id)

    def get(self, run_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload_json FROM controller_evolution_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown controller evolution run: {run_id}")
        return json.loads(row["payload_json"])

    def list(self, *, limit: int = 100, model_id: str | None = None) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        query = "SELECT run_id FROM controller_evolution_runs"
        params: list[Any] = []
        if model_id is not None:
            query += " WHERE model_id=?"
            params.append(model_id)
        query += " ORDER BY created_at DESC,run_id LIMIT ?"
        params.append(limit)
        with self._connect() as db:
            run_ids = [row["run_id"] for row in db.execute(query, params).fetchall()]
        return [self.get(run_id) for run_id in run_ids]


class ControllerSelfEvolutionWorkflow:
    """Plan and evaluate changes to an operator-designated controller model.

    The model may be the controller's own local/open-weight reasoning model.
    Hosted API models cannot have their provider-managed internals modified.
    Successful candidates remain candidates; promotion is a separate operator
    action outside this workflow.
    """

    def __init__(self, *, intelligence_workflow: Any,
                 ledger: ControllerEvolutionLedger | None = None):
        if not callable(getattr(intelligence_workflow, "run", None)):
            raise TypeError("intelligence_workflow must provide run()")
        self.intelligence_workflow = intelligence_workflow
        self.ledger = ledger or ControllerEvolutionLedger()

    def run(self, *, controller_model_id: str, architecture: Any, objective: str,
            trusted_evaluation_dir: str | Path,
            max_experiments: int | None = None) -> ControllerEvolutionResult:
        if not isinstance(controller_model_id, str) or not controller_model_id.strip():
            raise ValueError("controller_model_id must be a non-empty string")
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be a non-empty string")
        architecture_value = architecture.to_dict() if hasattr(architecture, "to_dict") else architecture
        if not isinstance(architecture_value, dict):
            raise TypeError("controller architecture must be an object or expose to_dict()")
        architecture_sha256 = hashlib.sha256(_canonical(architecture_value)).hexdigest()
        workflow_result = self.intelligence_workflow.run(
            model_id=controller_model_id,
            architecture=architecture,
            objective=objective,
            trusted_evaluation_dir=trusted_evaluation_dir,
            max_experiments=max_experiments,
        )
        experiments = tuple(workflow_result.experiments)
        authorized = []
        for item in experiments:
            pipeline_result = item.get("pipeline_result", {})
            if pipeline_result.get("workflow_status") == "candidate_authorized":
                authorized.append(item["plan_id"])
        result = ControllerEvolutionResult(
            run_id=str(uuid4()),
            model_id=controller_model_id,
            objective=objective,
            architecture_sha256=architecture_sha256,
            proposal_count=len(workflow_result.report.plans),
            experiment_results=experiments,
            candidate_authorized_ids=tuple(authorized),
            promotion_performed=False,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.ledger.append(result)
        return result
