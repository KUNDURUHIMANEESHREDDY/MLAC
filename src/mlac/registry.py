"""In-memory model registry."""
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
    def __init__(self):
        self._records = {}
    def register(self, architecture, *, metadata=None):
        record = ModelRecord(str(uuid4()), architecture, metadata=metadata or {})
        self._records[record.model_id] = record
        return record
    def get(self, model_id):
        if model_id not in self._records:
            raise KeyError(f"Unknown model_id: {model_id}")
        return self._records[model_id]
    def list(self):
        return sorted(self._records.values(), key=lambda x: x.created_at)
    def remove(self, model_id):
        self.get(model_id)
        del self._records[model_id]
