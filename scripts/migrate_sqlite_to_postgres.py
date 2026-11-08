"""Migrate an empty PostgreSQL MLAC state database from a SQLite snapshot."""

from __future__ import annotations

import os
from pathlib import Path

from mlac.postgres_state import PostgresStateStore
from mlac.sqlite_state import SQLiteStateStore


def migrate(source_path: str, target_dsn: str) -> tuple[int, int]:
    """Copy all model/proposal metadata in one PostgreSQL transaction."""
    source_file = Path(source_path)
    if not source_file.is_absolute() or not source_file.is_file() or source_file.is_symlink():
        raise RuntimeError("source SQLite path must be an existing absolute regular file")
    source = SQLiteStateStore(source_file)
    source.health_check()
    models = source.list_models()
    proposals = source.list_proposals()

    target = PostgresStateStore(target_dsn)
    try:
        if target.list_models() or target.list_proposals():
            raise RuntimeError("target PostgreSQL database must be empty before migration")
        expected_models = {row["model_id"] for row in models}
        expected_proposals = {row["proposal_id"] for row in proposals}

        with target._connect() as connection:
            for model in models:
                metadata = model["metadata"]
                owner = metadata.get("owner", "")
                if not isinstance(owner, str):
                    raise TypeError("source model owner must be a string")
                connection.execute(
                    """
                    INSERT INTO models (
                        model_id, owner, architecture_json, status, created_at, metadata_json
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        model["model_id"],
                        owner,
                        target._encode(model["architecture"], field="architecture"),
                        model["status"],
                        model["created_at"],
                        target._encode(metadata, field="model metadata"),
                    ),
                )
            for proposal in proposals:
                connection.execute(
                    """
                    INSERT INTO proposals (
                        proposal_id, model_id, hypothesis, operation,
                        parameters_json, status, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        proposal["proposal_id"],
                        proposal["model_id"],
                        proposal["hypothesis"],
                        proposal["operation"],
                        target._encode(
                            proposal["parameters"], field="proposal parameters"
                        ),
                        proposal["status"],
                        proposal["created_at"],
                    ),
                )
            migrated_models = {
                row["model_id"]
                for row in connection.execute("SELECT model_id FROM models").fetchall()
            }
            migrated_proposals = {
                row["proposal_id"]
                for row in connection.execute(
                    "SELECT proposal_id FROM proposals"
                ).fetchall()
            }
            if migrated_models != expected_models or migrated_proposals != expected_proposals:
                raise RuntimeError("PostgreSQL migration verification failed; transaction rolled back")

        target.health_check()
        return len(models), len(proposals)
    finally:
        target.close()


def main() -> None:
    """Read migration source/target from environment and print verified counts."""
    source_path = os.getenv("MLAC_MIGRATION_SQLITE_PATH", "").strip()
    target_dsn = os.getenv("MLAC_API_DATABASE_URL", "").strip()
    if not source_path or not Path(source_path).is_absolute():
        raise RuntimeError("MLAC_MIGRATION_SQLITE_PATH must be an absolute file path")
    if not target_dsn:
        raise RuntimeError("MLAC_API_DATABASE_URL must be configured")
    model_count, proposal_count = migrate(source_path, target_dsn)
    print(f"Migrated {model_count} models and {proposal_count} proposals")


if __name__ == "__main__":
    main()
