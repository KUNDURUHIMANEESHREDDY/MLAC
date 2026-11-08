"""Governance adapter that routes mutation decisions through the safety kernel."""
from dataclasses import dataclass
from datetime import datetime

from mlac.safety import GuardrailPolicy, OperatorApproval, SafetyKernel


@dataclass(frozen=True)
class GovernancePolicy:
    allow_candidate_mutations: bool = True
    allow_production_mutations: bool = False
    require_evaluation: bool = True
    max_parameter_growth_ratio: float = 0.10
    max_candidates_per_run: int = 8
    expected_policy_sha256: str | None = None

    def _kernel(self) -> SafetyKernel:
        policy = GuardrailPolicy(
            allow_candidate_mutations=self.allow_candidate_mutations,
            allow_production_mutations=self.allow_production_mutations,
            require_operator_approval=True,
            max_parameter_growth_ratio=self.max_parameter_growth_ratio,
            max_candidates_per_run=self.max_candidates_per_run,
        )
        return SafetyKernel(policy, expected_policy_sha256=self.expected_policy_sha256)

    def authorize(
        self,
        *,
        target: str,
        operation: str,
        parameter_growth_ratio: float = 0.0,
        model_id: str | None = None,
        approval: OperatorApproval | None = None,
        verification_key: bytes | None = None,
        evaluation_passed: bool = False,
        now: datetime | None = None,
    ) -> None:
        if target not in {"candidate", "production"}:
            raise ValueError("target must be candidate or production")
        kernel = self._kernel()
        if target == "candidate":
            kernel.authorize_candidate(
                operation=operation,
                parameter_growth_ratio=parameter_growth_ratio,
            )
            return
        if not self.allow_production_mutations:
            raise PermissionError("Production mutations are disabled by policy")
        if self.require_evaluation and not evaluation_passed:
            raise PermissionError("A passing candidate evaluation is required")
        if model_id is None:
            raise ValueError("model_id is required for production authorization")
        kernel.authorize_production(
            model_id=model_id,
            operation=operation,
            approval=approval,
            verification_key=verification_key or b"",
            now=now,
        )
