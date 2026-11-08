"""PostgreSQL-backed tenant-scoped API metadata storage."""

from __future__ import annotations

import json
from typing import Any

_SCHEMA_VERSION = 1
_CONNECT_TIMEOUT_SECONDS = 5


class PostgresStateStore:
    """Transactional PostgreSQL persistence for model metadata and proposals."""

    def __init__(self, dsn: str) -> None:
        """Validate the database URL and apply the versioned schema migration."""
        if not isinstance(dsn, str) or not dsn.startswith(
            ("postgresql://", "postgres://")
        ):
            raise ValueError("MLAC_API_DATABASE_URL must use postgresql://")
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError(
                "Install the production extra to use PostgreSQL state storage"
            ) from exc
        import os

        pool_size = int(os.getenv("MLAC_API_DB_POOL_SIZE", "10"))
        if not 1 <= pool_size <= 100:
            raise ValueError("MLAC_API_DB_POOL_SIZE must be between 1 and 100")
        self.dsn = dsn
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=pool_size,
            timeout=_CONNECT_TIMEOUT_SECONDS,
            kwargs={"connect_timeout": _CONNECT_TIMEOUT_SECONDS, "row_factory": dict_row},
            open=True,
        )
        self._initialize()

    def _connect(self):
        """Borrow a bounded-timeout PostgreSQL connection from the process pool."""
        return self._pool.connection()

    @staticmethod
    def _encode(value: Any, *, field: str) -> str:
        """Serialize JSON fields while rejecting NaN and non-serializable values."""
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be JSON serializable") from exc

    @staticmethod
    def _json_object(value: str, *, field: str) -> dict[str, Any]:
        """Decode a stored JSON object and reject malformed persisted state."""
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"stored {field} JSON is invalid") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(f"stored {field} must be a JSON object")
        return decoded

    def _initialize(self) -> None:
        """Apply schema version 1 under a transaction-scoped advisory lock."""
        with self._connect() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)",
                ("mlac-schema-migration",),
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mlac_schema_version (
                    id SMALLINT PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL
                )
                """
            )
            row = connection.execute(
                "SELECT version FROM mlac_schema_version WHERE id = 1 FOR UPDATE"
            ).fetchone()
            version = int(row["version"]) if row else 0
            if version > _SCHEMA_VERSION:
                raise RuntimeError(
                    f"state database schema {version} is newer than supported {_SCHEMA_VERSION}"
                )
            if version < 1:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS models (
                        model_id TEXT PRIMARY KEY,
                        owner TEXT NOT NULL,
                        architecture_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS models_owner_created
                    ON models(owner, created_at, model_id)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS proposals (
                        proposal_id TEXT PRIMARY KEY,
                        model_id TEXT NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
                        hypothesis TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        parameters_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS proposals_model_created
                    ON proposals(model_id, created_at, proposal_id)
                    """
                )
                connection.execute(
                    """
                    INSERT INTO mlac_schema_version(id, version) VALUES (1, %s)
                    ON CONFLICT (id) DO UPDATE SET version = EXCLUDED.version
                    """,
                    (_SCHEMA_VERSION,),
                )

    def close(self) -> None:
        """Close the process-local PostgreSQL connection pool."""
        self._pool.close()

    def health_check(self) -> None:
        """Verify connectivity and the currently supported schema version."""
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()
            row = connection.execute(
                "SELECT version FROM mlac_schema_version WHERE id = 1"
            ).fetchone()
            if row is None or int(row["version"]) != _SCHEMA_VERSION:
                raise RuntimeError("state database schema is not ready")

    def insert_model(self, record: Any) -> None:
        """Persist a model and its tenant owner in one transaction."""
        metadata = dict(record.metadata or {})
        owner = metadata.get("owner", "")
        if not isinstance(owner, str):
            raise ValueError("model owner must be a string")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO models (
                    model_id, owner, architecture_json, status, created_at, metadata_json
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    record.model_id,
                    owner,
                    self._encode(record.architecture.to_dict(), field="architecture"),
                    record.status,
                    record.created_at,
                    self._encode(metadata, field="model metadata"),
                ),
            )

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        """Read one model record by its stable identifier."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT model_id, architecture_json, status, created_at, metadata_json
                FROM models WHERE model_id = %s
                """,
                (model_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "model_id": row["model_id"],
            "architecture": self._json_object(row["architecture_json"], field="architecture"),
            "status": row["status"],
            "created_at": row["created_at"],
            "metadata": self._json_object(row["metadata_json"], field="model metadata"),
        }

    def list_models(self, *, owner: str | None = None) -> list[dict[str, Any]]:
        """List models, optionally restricting the result to one principal."""
        query = """
            SELECT model_id, architecture_json, status, created_at, metadata_json
            FROM models
        """
        parameters: tuple[str, ...] = ()
        if owner is not None:
            query += " WHERE owner = %s"
            parameters = (owner,)
        query += " ORDER BY created_at, model_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "model_id": row["model_id"],
                "architecture": self._json_object(row["architecture_json"], field="architecture"),
                "status": row["status"],
                "created_at": row["created_at"],
                "metadata": self._json_object(row["metadata_json"], field="model metadata"),
            }
            for row in rows
        ]

    def delete_model(self, model_id: str) -> None:
        """Delete a model and let the foreign key cascade its proposals."""
        with self._connect() as connection:
            connection.execute("DELETE FROM models WHERE model_id = %s", (model_id,))

    def insert_proposal(self, proposal: Any) -> None:
        """Persist a governed proposal associated with an existing model."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO proposals (
                    proposal_id, model_id, hypothesis, operation,
                    parameters_json, status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    proposal.proposal_id,
                    proposal.model_id,
                    proposal.hypothesis,
                    proposal.operation,
                    self._encode(proposal.parameters, field="proposal parameters"),
                    proposal.status,
                    proposal.created_at,
                ),
            )

    def list_proposals(
        self,
        *,
        owner: str | None = None,
        model_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List proposals filtered by principal and/or an explicit model-ID set."""
        if model_ids is not None and not model_ids:
            return []
        query = """
            SELECT p.proposal_id, p.model_id, p.hypothesis, p.operation,
                   p.parameters_json, p.status, p.created_at
            FROM proposals AS p
            JOIN models AS m ON m.model_id = p.model_id
        """
        clauses: list[str] = []
        parameters: list[Any] = []
        if owner is not None:
            clauses.append("m.owner = %s")
            parameters.append(owner)
        if model_ids is not None:
            clauses.append("p.model_id = ANY(%s)")
            parameters.append(sorted(model_ids))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY p.created_at, p.proposal_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "proposal_id": row["proposal_id"],
                "model_id": row["model_id"],
                "hypothesis": row["hypothesis"],
                "operation": row["operation"],
                "parameters": self._json_object(row["parameters_json"], field="proposal parameters"),
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
