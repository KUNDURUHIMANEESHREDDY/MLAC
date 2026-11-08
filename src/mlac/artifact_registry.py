"""Durable content-addressed artifact storage and evaluation associations."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class ArtifactRegistry:
    """Store immutable blobs by SHA-256 and append evaluation-to-artifact links."""

    def __init__(self, root: str | Path = "mlac-artifacts"):
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects" / "sha256"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "registry.sqlite3"
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS artifacts (
                digest TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, media_type TEXT NOT NULL,
                created_at TEXT NOT NULL, object_path TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS evaluation_artifacts (
                evaluation_id TEXT NOT NULL, role TEXT NOT NULL, digest TEXT NOT NULL,
                model_id TEXT NOT NULL, candidate_id TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(evaluation_id, role), FOREIGN KEY(digest) REFERENCES artifacts(digest))""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_artifact_links_digest ON evaluation_artifacts(digest)")
            db.execute("""CREATE TABLE IF NOT EXISTS worker_attestations (
                nonce TEXT PRIMARY KEY, key_id TEXT NOT NULL, report_sha256 TEXT NOT NULL,
                evaluation_id TEXT NOT NULL, created_at TEXT NOT NULL)""")
    def _connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db
    def put_bytes(self, data: bytes, *, media_type: str = "application/octet-stream") -> dict[str, Any]:
        if not isinstance(data, bytes):
            raise TypeError("artifact data must be bytes")
        digest = hashlib.sha256(data).hexdigest()
        target = self.objects / digest[:2] / digest[2:]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise RuntimeError("artifact object path must not be a symlink")
        if not target.exists():
            fd, temporary = tempfile.mkstemp(prefix=".incoming-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass
            finally:
                try: os.unlink(temporary)
                except FileNotFoundError: pass
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError("existing artifact object failed its content hash check")
        created = datetime.now(timezone.utc).isoformat()
        relative = target.relative_to(self.root).as_posix()
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?,?)",
                (digest, len(data), media_type, created, relative))
        return {"sha256": digest, "size_bytes": len(data), "media_type": media_type, "object_path": str(target)}
    def link_evaluation(self, *, evaluation_id: str, model_id: str, candidate_id: str,
                        artifacts: dict[str, bytes]) -> list[dict[str, Any]]:
        if not evaluation_id or not model_id or not candidate_id or not artifacts:
            raise ValueError("evaluation identifiers and artifacts must be non-empty")
        linked = []
        now = datetime.now(timezone.utc).isoformat()
        # Persist immutable blobs before opening the metadata transaction. This
        # avoids holding a SQLite write lock while put_bytes uses its own connection.
        for role, data in sorted(artifacts.items()):
            if not role or not isinstance(data, bytes):
                raise ValueError("artifact roles must be non-empty and values must be bytes")
            item = self.put_bytes(data, media_type="application/json" if role.endswith("json") else "application/octet-stream")
            linked.append((role, item))
        with self._connect() as db:
            for role, item in linked:
                db.execute("INSERT INTO evaluation_artifacts VALUES (?,?,?,?,?,?)",
                    (evaluation_id, role, item["sha256"], model_id, candidate_id, now))
        return [{"role": role, **item} for role, item in linked]
    def record_attestation(self, *, nonce: str, key_id: str, report_sha256: str, evaluation_id: str) -> None:
        """Persist a one-time nonce so a signed worker result cannot be replayed."""
        if not all((nonce, key_id, report_sha256, evaluation_id)):
            raise ValueError("attestation replay record fields must be non-empty")
        with self._connect() as db:
            try:
                db.execute("INSERT INTO worker_attestations VALUES (?,?,?,?,?)",
                    (nonce, key_id, report_sha256, evaluation_id, datetime.now(timezone.utc).isoformat()))
            except sqlite3.IntegrityError as exc:
                raise ValueError("worker attestation nonce has already been used") from exc
    def get(self, digest: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM artifacts WHERE digest=?", (digest,)).fetchone()
        if row is None: raise KeyError(f"Unknown artifact digest: {digest}")
        path = self.root / row["object_path"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("artifact blob integrity check failed")
        return {"sha256": digest, "size_bytes": row["size_bytes"], "media_type": row["media_type"], "data": data}
    def list_evaluation(self, evaluation_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT evaluation_artifacts.role, evaluation_artifacts.digest, artifacts.size_bytes, artifacts.media_type FROM evaluation_artifacts JOIN artifacts USING(digest) WHERE evaluation_id=? ORDER BY role", (evaluation_id,)).fetchall()
        return [{"role": r["role"], "sha256": r["digest"], "size_bytes": r["size_bytes"], "media_type": r["media_type"]} for r in rows]
