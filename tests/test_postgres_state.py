"""Integration tests for the tenant-scoped PostgreSQL state backend."""

import os
from uuid import uuid4

import pytest

from mlac.evolution import EvolutionController
from mlac.ir import Architecture, Component
from mlac.postgres_state import PostgresStateStore
from mlac.registry import ModelRegistry


def _architecture():
    """Build a minimal architecture for state-store integration coverage."""
    return Architecture(
        "postgres-test",
        "mlp",
        components=(Component("drop", "Dropout", {"p": 0.1}),),
    )


def test_postgres_state_persists_and_scopes_models_and_proposals():
    """Verify restart-safe state and owner/model-ID filtering in PostgreSQL."""
    dsn = os.getenv("MLAC_TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("MLAC_TEST_POSTGRES_URL is not configured")

    store = PostgresStateStore(dsn)
    registry = ModelRegistry(state_store=store)
    controller = EvolutionController(state_store=store)
    owner = f"alice-{uuid4().hex}"
    other_owner = f"bob-{uuid4().hex}"
    alice = registry.register(_architecture(), metadata={"owner": owner})
    bob = registry.register(_architecture(), metadata={"owner": other_owner})
    proposal = controller.propose(
        alice.model_id,
        alice.architecture,
        hypothesis="PostgreSQL proposal persists",
        operation="dropout",
        parameters={"probability": 0.05},
    )
    controller.propose(
        bob.model_id,
        bob.architecture,
        hypothesis="Other tenant proposal",
        operation="dropout",
    )

    reopened = PostgresStateStore(dsn)
    reopened_registry = ModelRegistry(state_store=reopened)
    reopened_controller = EvolutionController(state_store=reopened)
    assert reopened_registry.get(alice.model_id) == alice
    assert reopened_registry.list(owner=owner) == [alice]
    assert reopened_registry.list(owner=other_owner) == [bob]
    scoped = reopened_controller.list_proposals(
        owner=owner, model_ids={alice.model_id}
    )
    assert [item["proposal_id"] for item in scoped] == [proposal.proposal_id]
    assert reopened_controller.list_proposals(owner=owner, model_ids=set()) == []
    assert reopened_controller.list_proposals(owner=owner, model_ids={bob.model_id}) == []
    reopened.health_check()

    reopened_registry.remove(alice.model_id)
    reopened_registry.remove(bob.model_id)
