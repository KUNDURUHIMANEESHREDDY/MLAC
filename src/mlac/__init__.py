"""Master LLM Architecture Controller."""
__version__ = "0.8.0"
from mlac.ir import (
    Architecture, CompatibilityConstraint, Component, GraphEdge, GraphNode,
    InitializationStrategy, ParameterMapping, RuntimeCapabilities,
    RuntimeRequirements, TensorSpec, WeightTransformationRequirement,
    validate_compatibility,
)
from mlac.registry import ModelRegistry, ModelRecord
from mlac.master_intelligence import MasterIntelligenceEngine, IntelligenceReport, OptimizationObjective
from mlac.openai_backend import OpenAIResponsesBackend, OpenAIProviderError
from mlac.intelligence_workflow import IntelligenceExperimentWorkflow, IntelligenceWorkflowResult, TorchTransformerCandidateBuilder
from mlac.inspector import (
    ArchitectureInspection, AttentionAnalysis, GraphEdge, GraphNode,
    KVCacheEstimate, MemoryProfile, estimate_kv_cache_bytes,
    inspect_torch_model, inspect_torch_model_full,
)
__all__ = [
    "Architecture", "Component", "TensorSpec", "GraphNode", "GraphEdge",
    "ParameterMapping", "InitializationStrategy", "CompatibilityConstraint",
    "RuntimeRequirements", "RuntimeCapabilities", "WeightTransformationRequirement",
    "validate_compatibility", "ModelRegistry", "ModelRecord",
    "MasterIntelligenceEngine", "IntelligenceReport", "OptimizationObjective",
    "OpenAIResponsesBackend", "OpenAIProviderError", "IntelligenceExperimentWorkflow",
    "IntelligenceWorkflowResult", "TorchTransformerCandidateBuilder", "__version__",
    "ArchitectureInspection", "AttentionAnalysis", "GraphEdge", "GraphNode",
    "KVCacheEstimate", "MemoryProfile", "estimate_kv_cache_bytes",
    "inspect_torch_model", "inspect_torch_model_full",
]
