"""Proposal-only evolution controller. No automatic execution or promotion."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from mlac.governance import GovernancePolicy

@dataclass(frozen=True)
class EvolutionProposal:
    proposal_id: str
    model_id: str
    hypothesis: str
    operation: str
    parameters: dict = field(default_factory=dict)
    status: str = "proposed"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class EvolutionController:
    def __init__(self, policy=None):
        self.policy = policy or GovernancePolicy()
        self._proposals = []
    def propose(self, model_id, architecture, *, hypothesis, operation, parameters=None):
        self.policy.authorize(target="candidate", operation=operation)
        if not hypothesis.strip():
            raise ValueError("A testable hypothesis is required")
        proposal = EvolutionProposal(str(uuid4()), model_id, hypothesis, operation, parameters or {})
        self._proposals.append(proposal)
        return proposal
    def list_proposals(self):
        return [asdict(x) for x in self._proposals]
