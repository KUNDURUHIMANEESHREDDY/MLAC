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
    """Create and list governed proposals without executing or promoting them."""

    def __init__(self, policy=None, state_store=None):
        """Initialize the controller with an optional durable metadata store."""
        self.policy = policy or GovernancePolicy()
        self._proposals = []
        self._state_store = state_store

    def propose(self, model_id, architecture, *, hypothesis, operation, parameters=None):
        """Authorize and record a candidate proposal for a registered model."""
        self.policy.authorize(target="candidate", operation=operation)
        if not hypothesis.strip():
            raise ValueError("A testable hypothesis is required")
        proposal = EvolutionProposal(str(uuid4()), model_id, hypothesis, operation, parameters or {})
        if self._state_store is None:
            self._proposals.append(proposal)
        else:
            self._state_store.insert_proposal(proposal)
        return proposal

    def list_proposals(self, *, owner=None, model_ids=None):
        """List proposals scoped to a principal or a set of owned model IDs."""
        if self._state_store is not None:
            return self._state_store.list_proposals(owner=owner, model_ids=model_ids)
        proposals = self._proposals
        if model_ids is not None:
            proposals = [item for item in proposals if item.model_id in model_ids]
        return [asdict(item) for item in proposals]
