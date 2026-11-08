"""Master LLM Architecture Controller."""
__version__ = "0.8.0"
from mlac.ir import Architecture, Component, TensorSpec
from mlac.registry import ModelRegistry, ModelRecord
from mlac.master_intelligence import MasterIntelligenceEngine, IntelligenceReport, OptimizationObjective
from mlac.openai_backend import OpenAIResponsesBackend, OpenAIProviderError
from mlac.intelligence_workflow import IntelligenceExperimentWorkflow, IntelligenceWorkflowResult, TorchTransformerCandidateBuilder
__all__ = [
    "Architecture", "Component", "TensorSpec", "ModelRegistry", "ModelRecord",
    "MasterIntelligenceEngine", "IntelligenceReport", "OptimizationObjective",
    "OpenAIResponsesBackend", "OpenAIProviderError", "IntelligenceExperimentWorkflow",
    "IntelligenceWorkflowResult", "TorchTransformerCandidateBuilder", "__version__",
]
