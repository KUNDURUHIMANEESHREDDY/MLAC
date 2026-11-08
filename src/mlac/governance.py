"""Default-deny governance policy."""
from dataclasses import dataclass

@dataclass(frozen=True)
class GovernancePolicy:
    allow_candidate_mutations: bool = True
    allow_production_mutations: bool = False
    require_evaluation: bool = True
    max_parameter_growth_ratio: float = 0.10
    max_candidates_per_run: int = 8

    def authorize(self, *, target, operation, parameter_growth_ratio=0.0):
        if target not in {"candidate", "production"}:
            raise ValueError("target must be candidate or production")
        if target == "production" and not self.allow_production_mutations:
            raise PermissionError("Production mutations are disabled by policy")
        if target == "candidate" and not self.allow_candidate_mutations:
            raise PermissionError("Candidate mutations are disabled by policy")
        if parameter_growth_ratio > self.max_parameter_growth_ratio:
            raise PermissionError("Parameter-growth budget exceeded")
        if not operation.strip():
            raise ValueError("operation must be specified")
