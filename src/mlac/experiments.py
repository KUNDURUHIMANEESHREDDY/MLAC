"""SQLite-backed append-only storage for MLAC evaluation records."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentStore:
    """Small durable experiment ledger using only the Python standard library.

    Records are immutable after insertion. The store persists evaluation
    evidence, not model weights or executable checkpoints.
    """

    def __init__(self, database: str | Path = "mlac-experiments.sqlite3") -> None:
        self.database = str(database)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS experiments (
                evaluation_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                model_id TEXT,
                candidate_id TEXT,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at)")

    def record(self, result: Any, *, model_id: str | None = None,
               candidate_id: str | None = None) -> dict[str, Any]:
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        evaluation_id = str(payload["evaluation_id"])
        created_at = str(payload.get("created_at") or datetime.now(timezone.utc).isoformat())
        status = ("eligible" if payload.get("promotion_eligible") else
                  "exploratory" if payload.get("passed") else "failed")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._connect() as db:
            db.execute(
                "INSERT INTO experiments(evaluation_id, created_at, model_id, candidate_id, status, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (evaluation_id, created_at, model_id, candidate_id, status, encoded),
            )
        return self.get(evaluation_id)

    def get(self, evaluation_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM experiments WHERE evaluation_id = ?", (evaluation_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown evaluation_id: {evaluation_id}")
        return {
            "evaluation_id": row["evaluation_id"], "created_at": row["created_at"],
            "model_id": row["model_id"], "candidate_id": row["candidate_id"],
            "status": row["status"], "result": json.loads(row["payload_json"]),
        }

    def list(self, *, limit: int = 100, model_id: str | None = None) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        query = "SELECT evaluation_id FROM experiments"
        params: list[Any] = []
        if model_id is not None:
            query += " WHERE model_id = ?"
            params.append(model_id)
        query += " ORDER BY created_at DESC, evaluation_id LIMIT ?"
        params.append(limit)
        with self._connect() as db:
            ids = [row[0] for row in db.execute(query, params).fetchall()]
        return [self.get(evaluation_id) for evaluation_id in ids]
