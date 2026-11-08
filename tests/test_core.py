import pytest
from mlac.ir import Architecture, Component
from mlac.registry import ModelRegistry
from mlac.mutations import set_dropout_probability
from mlac.governance import GovernancePolicy
from mlac.evolution import EvolutionController

def arch():
    return Architecture("tiny", "mlp", components=(Component("drop", "Dropout", {"p": .1}),))

def test_ir_roundtrip():
    assert Architecture.from_dict(arch().to_dict()) == arch()

def test_registry():
    r = ModelRegistry()
    x = r.register(arch())
    assert r.get(x.model_id) == x
    r.remove(x.model_id)
    with pytest.raises(KeyError):
        r.get(x.model_id)

def test_mutation_is_versioned():
    x = set_dropout_probability(arch(), component_name="drop", probability=.2)
    assert x.version == 2 and x.components[0].config["p"] == .2

def test_mutation_rejects_invalid_probability():
    with pytest.raises(ValueError):
        set_dropout_probability(arch(), component_name="drop", probability=1)

def test_production_is_denied():
    with pytest.raises(PermissionError):
        GovernancePolicy().authorize(target="production", operation="mutate")

def test_evolution_is_proposal_only():
    c = EvolutionController()
    p = c.propose("m1", arch(), hypothesis="Test lower dropout", operation="dropout",
                  parameters={"probability": .05})
    assert p.status == "proposed" and len(c.list_proposals()) == 1
