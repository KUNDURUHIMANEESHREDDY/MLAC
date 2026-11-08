"""Framework-neutral MLAC intermediate representation."""
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int | str, ...]
    dtype: str = "unknown"

@dataclass(frozen=True)
class Component:
    name: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Architecture:
    name: str
    family: str
    version: int = 1
    framework: str = "unknown"
    dimensions: dict[str, int | float | str] = field(default_factory=dict)
    components: tuple[Component, ...] = ()
    tensors: tuple[TensorSpec, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name.strip() or not self.family.strip():
            raise ValueError("Architecture name and family must be non-empty")
        if self.version < 1:
            raise ValueError("version must be >= 1")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"], family=data["family"], version=int(data.get("version", 1)),
            framework=data.get("framework", "unknown"), dimensions=dict(data.get("dimensions", {})),
            components=tuple(Component(**x) for x in data.get("components", [])),
            tensors=tuple(TensorSpec(x["name"], tuple(x["shape"]), x.get("dtype", "unknown"))
                          for x in data.get("tensors", [])),
            metadata=dict(data.get("metadata", {})))
