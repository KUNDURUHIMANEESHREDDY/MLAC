"""Model registry with optional durable SQLite metadata storage."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from mlac.ir import Architecture


@dataclass
class ModelRecord:
    model_id: str
    architecture: Architecture
    status: str = "registered"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self, state_store=None):
        self._records = {}
        self._state_store = state_store

    @staticmethod
    def _from_stored(row):
        return ModelRecord(
            model_id=row["model_id"],
            architecture=Architecture.from_dict(row["architecture"]),
            status=row["status"],
            created_at=row["created_at"],
            metadata=row["metadata"],
        )

    def register(self, architecture, *, metadata=None):
        record = ModelRecord(str(uuid4()), architecture, metadata=metadata or {})
        if self._state_store is None:
            self._records[record.model_id] = record
        else:
            self._state_store.insert_model(record)
        return record

    def get(self, model_id):
        if self._state_store is None:
            if model_id not in self._records:
                raise KeyError(f"Unknown model_id: {model_id}")
            return self._records[model_id]
        row = self._state_store.get_model(model_id)
        if row is None:
            raise KeyError(f"Unknown model_id: {model_id}")
        return self._from_stored(row)

    def list(self, *, owner=None):
        if self._state_store is not None:
            return [
                self._from_stored(row)
                for row in self._state_store.list_models(owner=owner)
            ]
        records = sorted(self._records.values(), key=lambda item: (item.created_at, item.model_id))
        if owner is not None:
            records = [item for item in records if item.metadata.get("owner") == owner]
        return records

    def remove(self, model_id):
        self.get(model_id)
        if self._state_store is None:
            del self._records[model_id]
        else:
            self._state_store.delete_model(model_id)
