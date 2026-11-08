"""Tests for durable, owner-scoped API metadata storage."""
import sqlite3
from types import SimpleNamespace

import pytest

from mlac.evolution import EvolutionController
from mlac.ir import Architecture, Component, RuntimeRequirements
from mlac.registry import ModelRegistry
from mlac.sqlite_state import SQLiteStateStore


def architecture():
    return Architecture("tiny", "mlp", components=(Component("drop", "Dropout", {"p": 0.1}),))


def test_models_and_proposals_survive_store_recreation(tmp_path):
    database = tmp_path / "mlac-state.sqlite3"
    owner = "https://identity.example.test/|alice"

    first_store = SQLiteStateStore(database)
    first_registry = ModelRegistry(state_store=first_store)
    first_controller = EvolutionController(state_store=first_store)
    model = first_registry.register(architecture(), metadata={"owner": owner})
    proposal = first_controller.propose(
        model.model_id,
        model.architecture,
        hypothesis="Lower dropout may improve validation quality",
        operation="dropout",
        parameters={"probability": 0.05},
    )

    second_store = SQLiteStateStore(database)
    second_registry = ModelRegistry(state_store=second_store)
    second_controller = EvolutionController(state_store=second_store)

    assert second_registry.get(model.model_id) == model
    assert second_registry.list(owner=owner) == [model]
    assert second_registry.list(owner="https://identity.example.test/|bob") == []
    assert second_controller.list_proposals() == [
        {
            "proposal_id": proposal.proposal_id,
            "model_id": proposal.model_id,
            "hypothesis": proposal.hypothesis,
            "operation": proposal.operation,
            "parameters": proposal.parameters,
            "status": proposal.status,
            "created_at": proposal.created_at,
        }
    ]
    second_store.health_check()


def test_deleting_a_model_cascades_only_its_proposals(tmp_path):
    store = SQLiteStateStore(tmp_path / "mlac-state.sqlite3")
    registry = ModelRegistry(state_store=store)
    controller = EvolutionController(state_store=store)
    first = registry.register(architecture(), metadata={"owner": "alice"})
    second = registry.register(architecture(), metadata={"owner": "bob"})
    controller.propose(
        first.model_id,
        first.architecture,
        hypothesis="First proposal",
        operation="dropout",
    )
    controller.propose(
        second.model_id,
        second.architecture,
        hypothesis="Second proposal",
        operation="dropout",
    )

    registry.remove(first.model_id)

    assert [item["model_id"] for item in controller.list_proposals()] == [second.model_id]
    assert registry.get(second.model_id) == second
    with pytest.raises(KeyError):
        registry.get(first.model_id)


def test_future_schema_version_fails_closed(tmp_path):
    database = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 99")
    connection.close()

    with pytest.raises(RuntimeError, match="newer than supported"):
        SQLiteStateStore(database)


@pytest.mark.parametrize("path", [":memory:", "relative.sqlite3", "file:state.sqlite3"])
def test_store_rejects_non_file_backed_or_relative_paths(path):
    with pytest.raises(ValueError):
        SQLiteStateStore(path)


def test_architecture_round_trip_normalizes_runtime_requirement_sequences():
    """Preserve tuple-valued IR contracts when JSON converts sequences to lists."""
    original = Architecture(
        "tiny",
        "mlp",
        runtime_requirements=RuntimeRequirements(
            required_features=("cuda",),
            required_dtypes=("float16",),
            required_devices=("gpu",),
        ),
    )

    assert Architecture.from_dict(original.to_dict()) == original


def test_store_proposal_listing_applies_owner_and_model_id_filters(tmp_path):
    """Apply both principal and model-ID filters in the durable query path."""
    store = SQLiteStateStore(tmp_path / "scoped.sqlite3")
    registry = ModelRegistry(state_store=store)
    controller = EvolutionController(state_store=store)
    alice = registry.register(architecture(), metadata={"owner": "alice"})
    bob = registry.register(architecture(), metadata={"owner": "bob"})
    alice_proposal = controller.propose(
        alice.model_id, alice.architecture, hypothesis="Alice", operation="dropout"
    )
    controller.propose(
        bob.model_id, bob.architecture, hypothesis="Bob", operation="dropout"
    )

    assert [item["proposal_id"] for item in controller.list_proposals(
        owner="alice", model_ids={alice.model_id}
    )] == [alice_proposal.proposal_id]
    assert controller.list_proposals(owner="alice", model_ids=set()) == []
    assert controller.list_proposals(owner="alice", model_ids={bob.model_id}) == []



def test_json_object_rejects_non_object_json() -> None:
    """Reject valid JSON values that do not decode to an object."""
    with pytest.raises(TypeError, match="must be a JSON object"):
        SQLiteStateStore._json_object("[]", field="model metadata")


def test_insert_model_rejects_non_string_owner(tmp_path) -> None:
    """Reject model metadata with a non-string owner value."""
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    with pytest.raises(TypeError, match="model owner must be a string"):
        store.insert_model(SimpleNamespace(metadata={"owner": 42}))
