"""Master LLM Architecture Controller."""
__version__ = "0.8.0"
from mlac.ir import (
    IR_SCHEMA_VERSION, Architecture, CompatibilityConstraint, Component, GraphEdge as IRGraphEdge,
    GraphNode as IRGraphNode, InitializationStrategy, ParameterMapping,
    RuntimeCapabilities, RuntimeRequirements, TensorSpec,
    WeightTransformationRequirement, validate_compatibility,
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
def compile_architecture(*args, **kwargs):
    """Lazily compile an MLAC-IR graph; requires the optional torch extra."""
    from mlac.compiler import compile_architecture as _compile
    return _compile(*args, **kwargs)


def apply_weight_transformations(*args, **kwargs):
    """Lazily execute declared weight transforms; requires the optional torch extra."""
    from mlac.weight_transforms import apply_weight_transformations as _apply
    return _apply(*args, **kwargs)


__all__ = [
    "IR_SCHEMA_VERSION", "Architecture", "Component", "TensorSpec", "IRGraphNode", "IRGraphEdge",
    "GraphNode", "GraphEdge", "compile_architecture", "apply_weight_transformations",
    "ParameterMapping", "InitializationStrategy", "CompatibilityConstraint",
    "RuntimeRequirements", "RuntimeCapabilities", "WeightTransformationRequirement",
    "validate_compatibility", "ModelRegistry", "ModelRecord",
    "MasterIntelligenceEngine", "IntelligenceReport", "OptimizationObjective",
    "OpenAIResponsesBackend", "OpenAIProviderError", "IntelligenceExperimentWorkflow",
    "IntelligenceWorkflowResult", "TorchTransformerCandidateBuilder", "__version__",
    "ArchitectureInspection", "AttentionAnalysis", "KVCacheEstimate", "MemoryProfile", "estimate_kv_cache_bytes",
    "inspect_torch_model", "inspect_torch_model_full",
]
