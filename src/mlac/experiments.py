"""SQLite-backed append-only storage for MLAC evaluation records."""
from __future__ import annotations
import json, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class ExperimentStore:
    """Persist immutable evaluation evidence and workflow outcomes, not model weights."""
    def __init__(self,database: str|Path="mlac-experiments.sqlite3"):
        self.database=str(database)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS experiments (
                evaluation_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                model_id TEXT, candidate_id TEXT, status TEXT NOT NULL, payload_json TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at)")
    def _connect(self):
        db=sqlite3.connect(self.database,timeout=30)
        db.row_factory=sqlite3.Row
        return db
    def record(self,result: Any,*,model_id: str|None=None,candidate_id: str|None=None):
        payload=result.to_dict() if hasattr(result,"to_dict") else dict(result)
        evaluation_id=str(payload["evaluation_id"])
        created_at=str(payload.get("created_at") or datetime.now(timezone.utc).isoformat())
        mapping={"candidate_authorized":"eligible","governance_denied":"governance_denied",
                 "evaluation_failed":"failed","evaluation_exploratory":"exploratory"}
        status=mapping.get(payload.get("workflow_status"),
            "eligible" if payload.get("promotion_eligible") else
            "exploratory" if payload.get("passed") else "failed")
        encoded=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False)
        with self._connect() as db:
            db.execute("INSERT INTO experiments(evaluation_id,created_at,model_id,candidate_id,status,payload_json) VALUES (?,?,?,?,?,?)",
                (evaluation_id,created_at,model_id,candidate_id,status,encoded))
        return self.get(evaluation_id)
    def get(self,evaluation_id: str):
        with self._connect() as db:
            row=db.execute("SELECT * FROM experiments WHERE evaluation_id=?",(evaluation_id,)).fetchone()
        if row is None: raise KeyError(f"Unknown evaluation_id: {evaluation_id}")
        return {"evaluation_id":row["evaluation_id"],"created_at":row["created_at"],
            "model_id":row["model_id"],"candidate_id":row["candidate_id"],
            "status":row["status"],"result":json.loads(row["payload_json"])}
    def list(self,*,limit: int=100,model_id: str|None=None):
        if limit<1 or limit>1000: raise ValueError("limit must be between 1 and 1000")
        query="SELECT evaluation_id FROM experiments"
        params=[]
        if model_id is not None: query+=" WHERE model_id=?"; params.append(model_id)
        query+=" ORDER BY created_at DESC,evaluation_id LIMIT ?"; params.append(limit)
        with self._connect() as db: ids=[r[0] for r in db.execute(query,params).fetchall()]
        return [self.get(i) for i in ids]
