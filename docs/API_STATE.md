# API state stores

MLAC's FastAPI surface supports PostgreSQL for durable production metadata and
SQLite for single-host development. Both backends persist model/proposal metadata
and enforce issuer/subject ownership at query time.

## Configure

Install the API and production extras and configure OIDC. For production, set
`MLAC_API_DATABASE_URL` to a PostgreSQL DSN. For local single-host development,
set `MLAC_API_STATE_DB` to an absolute SQLite path. Configure exactly one; setting
both fails startup. If neither is set, the API uses process-local state and is
not suitable for production. The `/ready` response reports `storage: postgres`,
`storage: sqlite`, or `storage: process-local`; it never returns a database URL or path.

For SQLite only, the store creates missing parent directories and a new database
file. On POSIX, the new file is created with mode `0600`; newly created
directories request mode `0700`. Windows operators must apply an ACL that limits
access to the service identity because `chmod` does not provide equivalent
Windows ACL semantics. Keep SQLite on a local filesystem with reliable locking.

## PostgreSQL production backend

The PostgreSQL store applies schema version 1 transactionally under a
PostgreSQL advisory lock. It uses parameterized statements, a principal owner
index, a model foreign key with proposal cascade deletion, bounded connection
timeouts, a configurable per-process connection pool (MLAC_API_DB_POOL_SIZE,
default 10), and fail-closed readiness. It supports multiple API processes
sharing the same database. PostgreSQL credentials must be supplied through a secret
manager or private environment file; use TLS (`sslmode=require`) for remote
managed databases. Do not publish the database port.

## SQLite development backend

Schema version 1 contains:

- `models`: model ID, issuer/subject owner, serialized MLAC-IR architecture,
  status, creation time, and JSON metadata.
- `proposals`: proposal ID, model foreign key, hypothesis, operation,
  JSON parameters, status, and creation time.

Every write is a parameterized SQLite statement in a transaction. The database
uses WAL journaling, `synchronous=FULL`, foreign-key enforcement, and a
5-second busy timeout. An owner index supports issuer/subject-scoped model
listing. Deleting a model through the registry cascades to its proposals.

Both stores persist metadata only. Model weights, candidate bundles,
attestations, and content-addressed artifacts remain in their existing
artifact/worker boundaries. JSON values are encoded with non-finite numbers rejected.

The SQLite schema migration is transactional and sets `PRAGMA user_version=1`;
the PostgreSQL backend uses a schema-version table and transaction-scoped
advisory lock. A database with a newer schema version fails closed at startup;
the application does not attempt a downgrade or destructive migration. Future
schema changes must use a new explicit, tested migration.

## Backup and restore

Backups are not performed automatically. The production backup utility uses
`pg_dump --format=custom` and validates the archive with `pg_restore --list` for
PostgreSQL. For SQLite, it uses the online backup API rather than copying a live
database file:

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
- The API CLI remains loopback-only. The production container starts Uvicorn behind the Caddy reverse proxy and trusts forwarded headers only from Caddy's pinned container IP.
- Set MLAC_RATE_LIMIT_REDIS_URL to enable atomic shared peer/principal rate limits across API processes. If Redis is unavailable, protected requests and readiness fail closed with HTTP 503. Without Redis, the in-process limiter is for localhost development only.
- The production-candidate Docker Compose topology provides Caddy TLS, a persistent PostgreSQL volume, an internal Redis service, and Object Lock backups. See docs/PRODUCTION_DEPLOYMENT.md for host controls and restore drills.
- Keep database files and backups out of Git, shared workspaces, and public
  artifact directories.

The repository contains a production-candidate deployment topology, but no host is considered production-deployed until the real OIDC provider, HTTPS endpoint, Redis health, S3 Object Lock backup, restore drill, monitoring, and end-to-end acceptance are verified.
