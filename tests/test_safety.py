from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from mlac.safety import (
    GuardrailPolicy,
    OperatorApproval,
    SafetyKernel,
    approval_signature,
)


def _approval(key=b"external-operator-secret", **overrides):
    payload = {
        "approval_id": "approval-001",
        "model_id": "model-123",
        "operation": "deploy-candidate-v2",
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    }
    payload.update(overrides)
    return OperatorApproval(**payload, signature=approval_signature(payload, key))


def test_default_safety_kernel_denies_production():
    kernel = SafetyKernel()
    with pytest.raises(PermissionError, match="disabled"):
        kernel.authorize_production(
            model_id="model-123", operation="deploy-candidate-v2",
            approval=_approval(), verification_key=b"external-operator-secret",
        )


def test_policy_digest_pins_trusted_configuration():
    policy = GuardrailPolicy()
    digest = SafetyKernel(policy).policy_digest
    assert SafetyKernel(policy, expected_policy_sha256=digest).policy == policy
    with pytest.raises(PermissionError, match="trusted policy digest"):
        SafetyKernel(policy, expected_policy_sha256="0" * 64)


def test_guardrails_can_only_be_tightened():
    kernel = SafetyKernel()
    tighter = kernel.tighten(max_parameter_growth_ratio=0.02, max_candidates_per_run=2)
    assert tighter.policy.max_parameter_growth_ratio == 0.02
    assert tighter.policy.max_candidates_per_run == 2
    with pytest.raises(PermissionError, match="only be tightened"):
        kernel.tighten(max_parameter_growth_ratio=0.5)
    disabled = kernel.tighten(allow_candidate_mutations=False)
    with pytest.raises(PermissionError, match="disabled"):
        disabled.authorize_candidate(operation="prune-head")


def test_candidate_authorization_checks_growth_and_operation():
    kernel = SafetyKernel()
    kernel.authorize_candidate(operation="prune-head", parameter_growth_ratio=0.01)
    with pytest.raises(PermissionError, match="budget"):
        kernel.authorize_candidate(operation="expand-mlp", parameter_growth_ratio=0.2)
    with pytest.raises(ValueError, match="operation"):
        kernel.authorize_candidate(operation=" ")


def test_production_requires_matching_unexpired_external_signature():
    policy = GuardrailPolicy(allow_production_mutations=True)
    digest = SafetyKernel(policy).policy_digest
    kernel = SafetyKernel(policy, expected_policy_sha256=digest)
    key = b"external-operator-secret"
    approval = _approval(key)
    kernel.authorize_production(
        model_id="model-123", operation="deploy-candidate-v2",
        approval=approval, verification_key=key,
    )
    with pytest.raises(PermissionError, match="scope"):
        kernel.authorize_production(
            model_id="other-model", operation="deploy-candidate-v2",
            approval=approval, verification_key=key,
        )
    with pytest.raises(PermissionError, match="signature"):
        kernel.authorize_production(
            model_id="model-123", operation="deploy-candidate-v2",
            approval=approval, verification_key=b"wrong-key",
        )


def test_expired_or_timezone_naive_approval_is_rejected():
    policy = GuardrailPolicy(allow_production_mutations=True)
    kernel = SafetyKernel(policy)
    key = b"external-operator-secret"
    expired_payload = {
        "approval_id": "old", "model_id": "model-123",
        "operation": "deploy-candidate-v2",
        "expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    }
    expired = OperatorApproval(**expired_payload, signature=approval_signature(expired_payload, key))
    with pytest.raises(PermissionError, match="expired"):
        kernel.authorize_production(
            model_id="model-123", operation="deploy-candidate-v2",
            approval=expired, verification_key=key,
        )
    naive_payload = dict(expired_payload, expires_at="2035-01-01T00:00:00")
    naive = OperatorApproval(**naive_payload, signature=approval_signature(naive_payload, key))
    with pytest.raises(PermissionError, match="timezone"):
        kernel.authorize_production(
            model_id="model-123", operation="deploy-candidate-v2",
            approval=naive, verification_key=key,
        )


def test_policy_tampering_fails_integrity_check():
    kernel = SafetyKernel()
    object.__setattr__(kernel._effective, "allow_production_mutations", True)
    with pytest.raises(PermissionError, match="integrity"):
        kernel.authorize_candidate(operation="prune-head")


def test_production_approval_cannot_be_disabled_by_tightening():
    policy = GuardrailPolicy(allow_production_mutations=True)
    kernel = SafetyKernel(policy, expected_policy_sha256=SafetyKernel(policy).policy_digest)
    with pytest.raises(PermissionError, match="only be tightened"):
        kernel.tighten(require_operator_approval=False)


def test_production_policy_requires_external_digest_pin():
    with pytest.raises(PermissionError, match="externally pinned"):
        SafetyKernel(GuardrailPolicy(allow_production_mutations=True))
