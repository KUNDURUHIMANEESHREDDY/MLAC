import pytest

from mlac.designer import ArchitectureDesigner, DesignConstraints


def test_designer_generates_valid_deterministic_candidates():
    constraints = DesignConstraints(
        input_size=8, output_size=3, hidden_sizes=(8, 16),
        min_layers=1, max_layers=2, allowed_activations=("relu", "gelu"),
        candidate_limit=20,
    )
    designer = ArchitectureDesigner()
    first = designer.generate(constraints)
    second = designer.generate(constraints)
    assert first == second
    assert first
    assert all(candidate.architecture.validate() == () for candidate in first)
    assert all(candidate.parameter_count > 0 for candidate in first)


def test_designer_respects_parameter_budget_and_target_ranking():
    constraints = DesignConstraints(
        input_size=4, output_size=2, hidden_sizes=(4, 8, 16),
        min_layers=1, max_layers=2, allowed_activations=("relu",),
        max_parameters=100, target_parameters=50,
    )
    candidates = ArchitectureDesigner().generate(constraints)
    assert candidates
    assert all(item.parameter_count <= 100 for item in candidates)
    assert candidates[0].score >= candidates[-1].score


def test_designer_compiles_to_executable_torch_module():
    pytest.importorskip("torch")
    from mlac.compiler import compile_architecture
    candidate = ArchitectureDesigner().design(DesignConstraints(
        input_size=5, output_size=2, hidden_sizes=(8,),
        min_layers=1, max_layers=1, allowed_activations=("gelu",),
    ))
    module = compile_architecture(candidate.architecture)
    import torch
    output = module(torch.randn(4, 5))
    assert tuple(output.shape) == (4, 2)


def test_designer_rejects_invalid_constraints_and_empty_feasible_space():
    with pytest.raises(ValueError, match="hidden_sizes"):
        DesignConstraints(input_size=4, output_size=2, hidden_sizes=())
    with pytest.raises(ValueError, match="layer bounds"):
        DesignConstraints(input_size=4, output_size=2, min_layers=3, max_layers=2)
    constraints = DesignConstraints(
        input_size=64, output_size=64, hidden_sizes=(128,), max_parameters=1,
    )
    with pytest.raises(ValueError, match="no architecture"):
        ArchitectureDesigner().design(constraints)


def test_designer_emits_parameter_mappings_and_initialization_contracts():
    candidate = ArchitectureDesigner().design(DesignConstraints(
        input_size=4, output_size=2, hidden_sizes=(8,),
        min_layers=1, max_layers=1,
    ))
    architecture = candidate.architecture
    assert architecture.parameter_mappings
    assert architecture.initialization
    assert all(mapping.checkpoint_key is None for mapping in architecture.parameter_mappings)
    assert len(architecture.graph_nodes) == 4


def test_designer_bounds_combinatorial_search_before_enumeration():
    constraints = DesignConstraints(
        input_size=4, output_size=2,
        hidden_sizes=tuple(range(1, 101)),
        min_layers=1, max_layers=8,
        search_evaluation_limit=1000,
    )
    with pytest.raises(ValueError, match="exceeding search_evaluation_limit"):
        ArchitectureDesigner().generate(constraints)


def test_designer_rejects_fractional_dimensions_and_boolean_limits():
    with pytest.raises(ValueError, match="input_size"):
        DesignConstraints(input_size=4.5, output_size=2)
    with pytest.raises(ValueError, match="candidate_limit"):
        DesignConstraints(input_size=4, output_size=2, candidate_limit=True)
