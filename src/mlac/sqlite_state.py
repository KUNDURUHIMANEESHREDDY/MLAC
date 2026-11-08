"""Durable, owner-scoped SQLite state for the optional local API.

The store contains metadata only; model weights and candidate artifacts remain in
the content-addressed artifact registry. It is deliberately file-backed and does
not imply multi-host or production deployment readiness.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_BUSY_TIMEOUT_MS = 5_000


class SQLiteStateStore:
    """Transactional SQLite persistence for model metadata and proposals."""

    def __init__(self, path: str | os.PathLike[str]):
        raw_path = os.fspath(path)
        if not raw_path or raw_path == ":memory:" or raw_path.startswith("file:"):
            raise ValueError("state database must be a file-backed absolute path")
        database_path = Path(raw_path).expanduser()
        if not database_path.is_absolute():
            raise ValueError("state database path must be absolute")

        database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if database_path.is_symlink():
            raise ValueError("state database must not be a symbolic link")
        try:
            descriptor = os.open(
                database_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError:
            if database_path.is_symlink() or not database_path.is_file():
                raise ValueError("state database must be a regular file")
        else:
            os.close(descriptor)

        if os.name != "nt":
            os.chmod(database_path, 0o600)
        self.path = str(database_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_MS / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > _SCHEMA_VERSION:
                raise RuntimeError(
                    f"state database schema {version} is newer than supported {_SCHEMA_VERSION}"
                )
            if version == _SCHEMA_VERSION:
                return

            connection.execute("BEGIN IMMEDIATE")
            try:
                # Re-read after acquiring the write lock so concurrent first-start
                # processes do not both attempt to apply the initial migration.
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version > _SCHEMA_VERSION:
                    raise RuntimeError(
                        f"state database schema {version} is newer than supported {_SCHEMA_VERSION}"
                    )
                if version == 0:
                    connection.execute(
                        """
                        CREATE TABLE models (
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
                        "CREATE INDEX models_owner_created ON models(owner, created_at, model_id)"
                    )
                    connection.execute(
                        """
                        CREATE TABLE proposals (
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
                        "CREATE INDEX proposals_model_created ON proposals(model_id, created_at, proposal_id)"
                    )
                    connection.execute("PRAGMA user_version = 1")
                connection.commit()
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise
        finally:
            connection.close()

    @staticmethod
    def _json_object(value: str, *, field: str) -> dict[str, Any]:
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"stored {field} JSON is invalid") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(f"stored {field} must be a JSON object")
        return decoded

    @staticmethod
    def _encode(value: Any, *, field: str) -> str:
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

    def health_check(self) -> None:
        """Verify the database is readable and on the supported schema version."""
        connection = self._connect()
        try:
            connection.execute("SELECT 1").fetchone()
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != _SCHEMA_VERSION:
                raise RuntimeError("state database schema is not ready")
        finally:
            connection.close()

    def insert_model(self, record: Any) -> None:
        metadata = dict(record.metadata or {})
        owner = metadata.get("owner", "")
        if not isinstance(owner, str):
            raise ValueError("model owner must be a string")
        architecture_json = self._encode(record.architecture.to_dict(), field="architecture")
        metadata_json = self._encode(metadata, field="model metadata")
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    INSERT INTO models (
                        model_id, owner, architecture_json, status, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.model_id,
                        owner,
                        architecture_json,
                        record.status,
                        record.created_at,
                        metadata_json,
                    ),
                )
        finally:
            connection.close()

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT model_id, architecture_json, status, created_at, metadata_json
                FROM models WHERE model_id = ?
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
        finally:
            connection.close()

    def list_models(self, *, owner: str | None = None) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            if owner is None:
                rows = connection.execute(
                    """
                    SELECT model_id, architecture_json, status, created_at, metadata_json
                    FROM models ORDER BY created_at, model_id
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT model_id, architecture_json, status, created_at, metadata_json
                    FROM models WHERE owner = ? ORDER BY created_at, model_id
                    """,
                    (owner,),
                ).fetchall()
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
        finally:
            connection.close()

    def delete_model(self, model_id: str) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
        finally:
            connection.close()

    def insert_proposal(self, proposal: Any) -> None:
        parameters_json = self._encode(proposal.parameters, field="proposal parameters")
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    INSERT INTO proposals (
                        proposal_id, model_id, hypothesis, operation,
                        parameters_json, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        proposal.proposal_id,
                        proposal.model_id,
                        proposal.hypothesis,
                        proposal.operation,
                        parameters_json,
                        proposal.status,
                        proposal.created_at,
                    ),
                )
        finally:
            connection.close()

    def list_proposals(
        self,
        *,
        owner: str | None = None,
        model_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return proposals filtered by owner and/or an explicit model-ID set."""
        if model_ids is not None and not model_ids:
            return []
        connection = self._connect()
        try:
            clauses: list[str] = []
            parameters: list[str] = []
            query = """
                SELECT p.proposal_id, p.model_id, p.hypothesis, p.operation,
                       p.parameters_json, p.status, p.created_at
                FROM proposals AS p
                JOIN models AS m ON m.model_id = p.model_id
            """
            if owner is not None:
                clauses.append("m.owner = ?")
                parameters.append(owner)
            if model_ids is not None:
                ordered_ids = sorted(model_ids)
                clauses.append("p.model_id IN (" + ",".join("?" for _ in ordered_ids) + ")")
                parameters.extend(ordered_ids)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY p.created_at, p.proposal_id"
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
        finally:
            connection.close()
