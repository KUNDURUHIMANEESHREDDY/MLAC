"""Master LLM Architecture Controller."""
__version__ = "0.8.0"
from mlac.ir import Architecture, Component, TensorSpec
from mlac.registry import ModelRegistry, ModelRecord
from mlac.master_intelligence import MasterIntelligenceEngine, IntelligenceReport, OptimizationObjective
__all__ = ["Architecture", "Component", "TensorSpec", "ModelRegistry", "ModelRecord", "MasterIntelligenceEngine", "IntelligenceReport", "OptimizationObjective", "__version__"]
