"""Integrity-checked guardrails and externally signed production approvals.

This is an application-level safety control, not a sandbox or a defense against
arbitrary code execution in the same Python process. Keep signing keys and the
policy trust root outside the MLAC process in a separate operator service/KMS.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite


def _canonical_digest(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GuardrailPolicy:
    """Pinned baseline policy; tightening is allowed, weakening is not."""

    allow_candidate_mutations: bool = True
    allow_production_mutations: bool = False
    require_operator_approval: bool = True
    max_parameter_growth_ratio: float = 0.10
    max_candidates_per_run: int = 8

    def __post_init__(self) -> None:
        if not isfinite(self.max_parameter_growth_ratio) or self.max_parameter_growth_ratio < 0:
            raise ValueError("max_parameter_growth_ratio must be finite and >= 0")
        if self.max_candidates_per_run < 1:
            raise ValueError("max_candidates_per_run must be >= 1")


@dataclass(frozen=True)
class OperatorApproval:
    """Approval envelope signed by a separate trusted operator service."""

    approval_id: str
    model_id: str
    operation: str
    expires_at: str
    signature: str

    def payload(self) -> dict[str, str]:
        return {
            "approval_id": self.approval_id,
            "model_id": self.model_id,
            "operation": self.operation,
            "expires_at": self.expires_at,
        }


def guardrail_policy_digest(policy: GuardrailPolicy) -> str:
    """Compute a canonical digest for pinning in an external deployment manifest."""
    return _canonical_digest(asdict(policy))


def approval_signature(payload: dict[str, str], signing_key: bytes) -> str:
    """Return the HMAC-SHA256 signature; call only in the external operator service."""
    if not signing_key:
        raise ValueError("signing_key must not be empty")
    message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(signing_key, message, hashlib.sha256).hexdigest()


class SafetyKernel:
    """Fail-closed guardrail verifier with monotonic policy tightening."""

    def __init__(
        self,
        policy: GuardrailPolicy | None = None,
        *,
        expected_policy_sha256: str | None = None,
    ) -> None:
        self._baseline = policy or GuardrailPolicy()
        self._baseline_digest = _canonical_digest(asdict(self._baseline))
        if self._baseline.allow_production_mutations and expected_policy_sha256 is None:
            raise PermissionError(
                "Production-enabled policy requires an externally pinned expected_policy_sha256"
            )
        if expected_policy_sha256 is not None and not hmac.compare_digest(
            self._baseline_digest, expected_policy_sha256
        ):
            raise PermissionError("Guardrail policy does not match the trusted policy digest")
        self._effective = self._baseline
        self._effective_digest = _canonical_digest(asdict(self._effective))

    @property
    def policy_digest(self) -> str:
        return self._baseline_digest

    @property
    def policy(self) -> GuardrailPolicy:
        self.verify_integrity()
        return self._effective

    def verify_integrity(self) -> None:
        baseline_digest = _canonical_digest(asdict(self._baseline))
        effective_digest = _canonical_digest(asdict(self._effective))
        if not hmac.compare_digest(baseline_digest, self._baseline_digest):
            raise PermissionError("Trusted baseline guardrail policy integrity check failed")
        if not hmac.compare_digest(effective_digest, self._effective_digest):
            raise PermissionError("Effective guardrail policy integrity check failed")
        if not self._is_tightening(self._baseline, self._effective):
            raise PermissionError("Effective guardrail policy is weaker than trusted baseline")

    @staticmethod
    def _is_tightening(base: GuardrailPolicy, candidate: GuardrailPolicy) -> bool:
        return (
            (not candidate.allow_candidate_mutations or base.allow_candidate_mutations)
            and (not candidate.allow_production_mutations or base.allow_production_mutations)
            and (candidate.require_operator_approval or not base.require_operator_approval)
            and candidate.max_parameter_growth_ratio <= base.max_parameter_growth_ratio
            and candidate.max_candidates_per_run <= base.max_candidates_per_run
        )

    def tighten(self, **changes: object) -> "SafetyKernel":
        """Return a stricter kernel; attempts to loosen any control fail closed."""
        self.verify_integrity()
        unknown = set(changes) - set(GuardrailPolicy.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown guardrail fields: {sorted(unknown)}")
        values = asdict(self._effective)
        values.update(changes)
        tightened = GuardrailPolicy(**values)
        if not self._is_tightening(self._effective, tightened):
            raise PermissionError("Guardrails may only be tightened, never weakened")
        result = object.__new__(SafetyKernel)
        result._baseline = self._baseline
        result._baseline_digest = self._baseline_digest
        result._effective = tightened
        result._effective_digest = _canonical_digest(asdict(tightened))
        result.verify_integrity()
        return result

    def authorize_candidate(self, *, operation: str, parameter_growth_ratio: float = 0.0) -> None:
        self.verify_integrity()
        policy = self._effective
        if not policy.allow_candidate_mutations:
            raise PermissionError("Candidate mutations are disabled by guardrail policy")
        if not operation.strip():
            raise ValueError("operation must be specified")
        if not isfinite(parameter_growth_ratio) or parameter_growth_ratio < 0:
            raise ValueError("parameter_growth_ratio must be finite and >= 0")
        if parameter_growth_ratio > policy.max_parameter_growth_ratio:
            raise PermissionError("Candidate parameter-growth budget exceeded")

    def authorize_production(
        self,
        *,
        model_id: str,
        operation: str,
        approval: OperatorApproval | None,
        verification_key: bytes,
        now: datetime | None = None,
    ) -> None:
        """Require policy enablement plus a valid, scoped, unexpired operator signature."""
        self.verify_integrity()
        policy = self._effective
        if not policy.allow_production_mutations:
            raise PermissionError("Production mutations are disabled by trusted guardrails")
        if not model_id.strip() or not operation.strip():
            raise ValueError("model_id and operation must be specified")
        if not policy.require_operator_approval:
            raise PermissionError("Production authorization must require operator approval")
        if approval is None:
            raise PermissionError("A signed external operator approval is required")
        if approval.model_id != model_id or approval.operation != operation:
            raise PermissionError("Operator approval scope does not match requested operation")
        if not approval.approval_id.strip() or not verification_key:
            raise PermissionError("Approval identity and verification key are required")
        try:
            expiry = datetime.fromisoformat(approval.expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PermissionError("Operator approval expiry is invalid") from exc
        if expiry.tzinfo is None:
            raise PermissionError("Operator approval expiry must include a timezone")
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise ValueError("now must include a timezone")
        if expiry <= current:
            raise PermissionError("Operator approval has expired")
        expected = approval_signature(approval.payload(), verification_key)
        if not hmac.compare_digest(expected, approval.signature):
            raise PermissionError("Operator approval signature is invalid")
