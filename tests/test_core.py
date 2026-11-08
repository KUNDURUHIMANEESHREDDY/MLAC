import pytest
from datetime import datetime, timedelta, timezone
from mlac.ir import Architecture, Component
from mlac.registry import ModelRegistry
from mlac.mutations import set_dropout_probability
from mlac.governance import GovernancePolicy
from mlac.safety import GuardrailPolicy, OperatorApproval, approval_signature, guardrail_policy_digest
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


def test_governance_routes_candidate_authorization_through_safety_kernel():
    GovernancePolicy().authorize(target="candidate", operation="prune-head", parameter_growth_ratio=0.01)
    with pytest.raises(PermissionError, match="budget"):
        GovernancePolicy().authorize(target="candidate", operation="expand", parameter_growth_ratio=0.5)


def test_governance_production_requires_evaluation_and_signed_external_approval():
    guardrails = GuardrailPolicy(allow_production_mutations=True)
    policy = GovernancePolicy(
        allow_production_mutations=True,
        expected_policy_sha256=guardrail_policy_digest(guardrails),
    )
    key = b"operator-key-from-secret-manager"
    payload = {
        "approval_id": "governance-approval-1",
        "model_id": "model-1",
        "operation": "promote-candidate",
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat(),
    }
    approval = OperatorApproval(**payload, signature=approval_signature(payload, key))
    with pytest.raises(PermissionError, match="evaluation"):
        policy.authorize(
            target="production", operation="promote-candidate", model_id="model-1",
            approval=approval, verification_key=key,
        )
    policy.authorize(
        target="production", operation="promote-candidate", model_id="model-1",
        approval=approval, verification_key=key, evaluation_passed=True,
    )


def test_governance_cannot_enable_production_without_external_policy_pin():
    policy = GovernancePolicy(allow_production_mutations=True)
    with pytest.raises(PermissionError, match="externally pinned"):
        policy.authorize(target="production", operation="deploy", model_id="model-1")
