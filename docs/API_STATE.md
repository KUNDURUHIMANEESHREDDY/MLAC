# Optional SQLite API state

MLAC's optional FastAPI surface can persist model and proposal metadata across
process restarts. This is a local-development capability, not authorization to
expose the API on a network.

## Configure

Install the API extra and configure OIDC as described in the root README. Then
set `MLAC_API_STATE_DB` to an absolute path to a local SQLite file before
starting the API:

```text
MLAC_API_STATE_DB=/var/lib/mlac/api-state.sqlite3
```

If the variable is absent, the API keeps its existing process-local registry and
proposal controller. The `/ready` response reports `storage: sqlite` when
the file-backed store is active and `storage: process-local` otherwise.
The database path is never returned by the health endpoints.

The store creates missing parent directories and a new database file. On POSIX,
the new file is created with mode `0600`; newly created directories request
mode `0700`. Windows operators must apply an ACL that limits access to the
service identity because `chmod` does not provide equivalent Windows ACL
semantics. Keep the database on a local filesystem with reliable SQLite locking;
do not place it on an unverified network share.

## Stored data and transaction behavior

Schema version 1 contains:

- `models`: model ID, issuer/subject owner, serialized MLAC-IR architecture,
  status, creation time, and JSON metadata.
- `proposals`: proposal ID, model foreign key, hypothesis, operation,
  JSON parameters, status, and creation time.

Every write is a parameterized SQLite statement in a transaction. The database
uses WAL journaling, `synchronous=FULL`, foreign-key enforcement, and a
5-second busy timeout. An owner index supports issuer/subject-scoped model
listing. Deleting a model through the registry cascades to its proposals.

The store persists metadata only. Model weights, candidate bundles, attestations,
and content-addressed artifacts remain in their existing artifact/worker
boundaries. JSON values are encoded with non-finite numbers rejected.

The initial schema migration is transactional and sets `PRAGMA user_version=1`.
A database with a newer schema version fails closed at startup; the application
does not attempt a downgrade or destructive migration. Future schema changes
must use a new explicit, tested migration.

## Backup and restore

Backups are not performed automatically. Use SQLite's online backup API rather
than copying a live database file:

```python
import sqlite3

source = sqlite3.connect("/var/lib/mlac/api-state.sqlite3")
destination = sqlite3.connect("/secure-backups/mlac-api-state.sqlite3")
try:
    source.backup(destination)
finally:
    destination.close()
    source.close()
```

Protect backups with the same access controls as the source database and verify
restores in an isolated environment. Restore only while the API is stopped, then
start it and verify `/ready`, model ownership, and proposal visibility. A backup
is not a substitute for the separately configured artifact-store/WORM controls.

## Security and operational limits

- OIDC authentication remains mandatory for every non-health API route.
- Model reads are filtered by the authenticated issuer/subject principal.
- The API CLI remains loopback-only. This feature does not enable remote binds.
- Pre-authentication and principal rate limiters remain process-local. They do
  not coordinate across multiple workers or hosts.
- No TLS termination, distributed rate limiting, backup scheduler, restore
  automation, or production deployment is provided by this feature.
- Keep database files and backups out of Git, shared workspaces, and public
  artifact directories.

The API is not production-ready until independent TLS/reverse-proxy controls,
distributed rate limiting, backup/restore drills, resource authorization,
monitoring, and deployed acceptance are verified.
