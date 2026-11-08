"""Integration tests for the tenant-scoped PostgreSQL state backend."""

import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mlac.evolution import EvolutionController
from mlac.ir import Architecture, Component
from mlac.postgres_state import PostgresStateStore
from mlac.registry import ModelRegistry
from mlac.sqlite_state import SQLiteStateStore
from scripts.migrate_sqlite_to_postgres import migrate


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
    store.close()
    reopened.close()


def test_sqlite_migration_copies_models_and_proposals_transactionally(tmp_path):
    """Migrate a SQLite snapshot into an empty PostgreSQL database."""
    dsn = os.getenv("MLAC_TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("MLAC_TEST_POSTGRES_URL is not configured")

    source = SQLiteStateStore(tmp_path / "legacy.sqlite3")
    registry = ModelRegistry(state_store=source)
    controller = EvolutionController(state_store=source)
    owner = f"migration-{uuid4().hex}"
    model = registry.register(_architecture(), metadata={"owner": owner})
    proposal = controller.propose(
        model.model_id,
        model.architecture,
        hypothesis="Legacy proposal",
        operation="dropout",
        parameters={"probability": 0.2},
    )

    model_count, proposal_count = migrate(str(tmp_path / "legacy.sqlite3"), dsn)
    assert (model_count, proposal_count) == (1, 1)

    target = PostgresStateStore(dsn)
    try:
        migrated = ModelRegistry(state_store=target)
        migrated_controller = EvolutionController(state_store=target)
        assert migrated.get(model.model_id).metadata["owner"] == owner
        assert migrated_controller.list_proposals(
            owner=owner, model_ids={model.model_id}
        )[0]["proposal_id"] == proposal.proposal_id
        migrated.remove(model.model_id)
    finally:
        target.close()



def test_postgres_json_object_rejects_non_object_json() -> None:
    """Reject valid JSON values that do not decode to an object."""
    with pytest.raises(TypeError, match="must be a JSON object"):
        PostgresStateStore._json_object("[]", field="model metadata")


def test_postgres_insert_model_rejects_non_string_owner() -> None:
    store = PostgresStateStore.__new__(PostgresStateStore)
    with pytest.raises(TypeError, match="model owner must be a string"):
        store.insert_model(SimpleNamespace(metadata={"owner": 42}))
