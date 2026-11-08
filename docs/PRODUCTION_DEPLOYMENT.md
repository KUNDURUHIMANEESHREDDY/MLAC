# MLAC 1.0 production deployment

This runbook deploys the OIDC-authenticated API as a single API instance behind Caddy TLS, with a persistent PostgreSQL metadata volume and Redis-backed shared rate limits. PostgreSQL and Redis are shared services; the API can be replicated behind an orchestrator when the deployment has sufficient resource and connection limits.

## Runtime boundaries

- Caddy is the only service with published ports. The API and Redis have no host-published ports.
- Caddy terminates HTTPS and forwards requests over a dedicated Docker network. Uvicorn trusts proxy headers only from Caddy's pinned address (172.30.0.2).
- The API runs as UID/GID 10001, with a read-only root filesystem, dropped Linux capabilities, and no-new-privileges.
- PostgreSQL persists tenant-scoped model/proposal metadata on the postgres-data volume. It is not exposed outside the private Docker network.
- Redis uses AOF persistence and is reachable only on an internal Docker network. Both pre-authentication peer limits and authenticated-principal limits use atomic Redis Lua operations and Redis server time.
- If Redis is unavailable, readiness fails and protected requests fail closed with HTTP 503.
- Model weights, evaluation datasets, worker images, and content-addressed artifacts remain separate from the API metadata database.

## Host prerequisites

- A supported Linux host with Docker Engine and the Docker Compose plugin.
- A public DNS A/AAAA record for the deployment hostname.
- Inbound TCP 80 and 443 (and optionally UDP 443) allowed to the host.
- A persistent, encrypted host disk with monitored free space and restricted administrator access.
- An OIDC provider with a production issuer, audience, and HTTPS JWKS endpoint.
- Strong PostgreSQL and Redis passwords and a private .env file.
- A tested backup bucket with S3 versioning and Object Lock COMPLIANCE enabled.
- A workload/instance IAM role for the backup process with narrowly scoped S3 permissions; avoid long-lived AWS access keys in .env.

The main-branch workflow publishes immutable image tags to ghcr.io/kunduruhimaneeshreddy/mlac, including sha-<commit>. Pin MLAC_IMAGE to the exact commit that passed CI; do not deploy the mutable main tag. The default Docker subnet is 172.30.0.0/24. Change the subnet and the pinned proxy IP together in docker-compose.yml, Dockerfile, and this runbook if it conflicts with the host network.

## Configure and start

1. Clone the repository and enter its directory.
2. Copy .env.example to .env and restrict it with chmod 600.
3. Set MLAC_IMAGE to the immutable sha-<commit> tag published by the main-branch container workflow. Authenticate to ghcr.io with a read:packages credential if the package is private. Set MLAC_POSTGRES_PASSWORD and MLAC_API_DATABASE_URL to the same URL-encoded password, and use sslmode=require for remote managed PostgreSQL.
4. Edit .env with the real domain and OIDC values. Generate a strong random MLAC_REDIS_PASSWORD; URL-encode reserved characters when placing the password in MLAC_RATE_LIMIT_REDIS_URL. The Redis server password and URL password must match exactly. Never commit .env.
5. Pull the pinned image and start:

    docker compose pull mlac
    docker compose config --quiet
    docker compose up -d
    docker compose ps
    docker compose logs --tail=100 caddy mlac postgres redis

Caddy obtains and renews the TLS certificate after DNS and inbound ports are correct. Confirm the public endpoint:

    curl --fail --show-error https://$MLAC_DOMAIN/health
    curl --fail --show-error https://$MLAC_DOMAIN/ready

/ready must report storage: postgres. The endpoint returns 503 if OIDC configuration, PostgreSQL, or Redis readiness checks fail. Verify an authenticated request with a real production-issued OIDC token and verify that a second principal cannot read the first principal's models or proposals.

## Operational controls

- PostgreSQL and Redis are the shared state backends for API replicas. Keep their ports private, size PostgreSQL connection limits for the number of API replicas multiplied by MLAC_API_DB_POOL_SIZE, and keep database/Redis volumes on encrypted persistent storage.
- Do not publish Redis or the API port. Restrict SSH and Docker administration to trusted operators.
- Keep .env, PostgreSQL data, Redis data, Caddy state, and backups out of source control and public artifacts.
- Configure host-level encrypted disk, unattended security updates, firewall policy, log collection, alerting, and a host restart policy.
- Monitor /health, /ready, disk capacity, container restarts, Redis persistence, certificate renewal, and backup age.
- Rotate OIDC signing keys and Redis credentials through the relevant provider/secret-management procedures. Restart services after controlled secret rotation.
- Do not widen Uvicorn's forwarded-IP trust. If the Caddy address changes, update the exact allowlist and static address together.
- The immutable authorization, governance, recovery, and deployment trust roots remain outside autonomous controller mutation.

## Immutable database backups

Create an S3 bucket with versioning and Object Lock enabled at bucket creation. Configure bucket-level default retention and access policies independently. The backup utility additionally verifies versioning and Object Lock are enabled. It creates a PostgreSQL custom-format dump, validates it with pg_restore, and uploads it in COMPLIANCE mode with a retention date and SHA-256 metadata. SQLite online backups remain supported for local deployments.

Run a backup manually:

    docker compose exec -T mlac python /app/scripts/backup_state.py

Schedule the same command from a restricted host scheduler (for example, a systemd timer) at the recovery-point objective approved by the operator. Alert on non-zero exit status and on a missing recent backup. The backup role should have only the required bucket inspection and object upload permissions; it must not be able to shorten Object Lock retention or delete protected versions. Provide credentials through the host/workload identity provider and verify the container can reach it; do not store long-lived AWS access keys in the repository or image.

Before starting a restore drill, verify every downloaded backup against the SHA-256 value read from trusted S3 object metadata (not a digest copied from the backup file itself):

Run these commands from the repository root on the host. The verifier requires Python 3.10 or newer and is invoked with `python3` so Debian/Ubuntu hosts do not need the optional `python` alias. PostgreSQL custom archives require `pg_restore` 16 or newer from the PostgreSQL client tools, matching the major version used by the backup image. SQLite verification uses Python's standard library.

    python3 scripts/verify_backup.py /secure/restore/api-state.sqlite3 --sha256 "$EXPECTED_SHA256"
    python3 scripts/verify_backup.py /secure/restore/api-state.dump --sha256 "$EXPECTED_SHA256" --format postgres-custom

The verifier rejects symlinks and non-regular files, checks the expected SHA-256 in bounded-memory chunks, and detects only the SQLite and PostgreSQL custom formats emitted by this utility. SQLite is opened read-only with immutable/query-only settings and checked with PRAGMA integrity_check. PostgreSQL custom archives are inspected with `pg_restore --list`, which validates the archive header and table of contents only; it does not read or validate the data blocks, and no SQL is executed. A successful result proves the digest and these structural checks only. It does not prove that a full restore will succeed. It does not restore data or prove application-level recovery; the isolated restore drill below remains mandatory.

Restore drills must be performed in an isolated environment. Download a retained backup and verify its SHA-256 against trusted backup metadata. For PostgreSQL, validate the archive with pg_restore --list, stop the API, and restore with pg_restore --no-owner --no-acl --clean --if-exists --exit-on-error. For SQLite, run PRAGMA integrity_check and restore while the API is stopped. Then verify /ready, tenant isolation, and representative model/proposal reads. Record the achieved recovery point and recovery time.

## Rollback and recovery

- Keep the previous immutable container image digest and the last known-good database backup.
- Roll back application code only to a version that supports the existing database schema.
- Do not delete or downgrade database files during rollback. A future schema migration must be additive, versioned, tested, and accompanied by a restore drill.
- For suspected compromise, isolate the host, preserve logs and immutable backups, revoke affected OIDC credentials, and follow the separately approved incident-response procedure.

## MLAC 1.0 acceptance checklist

- [ ] Real OIDC issuer/audience/JWKS configured; invalid and expired tokens rejected.
- [ ] Public HTTPS certificate valid; API and Redis ports not externally reachable.
- [ ] /ready reports PostgreSQL and Redis healthy.
- [ ] Cross-tenant model and proposal isolation verified with two real principals.
- [ ] Redis rate limiting verified across separate API processes.
- [ ] PostgreSQL data survives container recreation and host restart.
- [ ] S3 versioning and Object Lock COMPLIANCE backup uploaded and retained.
- [ ] Restore drill completed and recovery metrics recorded.
- [ ] Runtime image digest, host patching, monitoring, alerting, and incident-response ownership recorded.
- [ ] Fixed evaluation worker, trusted dataset signing key, held-out dataset, runtime attestor, and artifact-store end-to-end evaluation accepted separately.

A successful container build or CI run is not evidence that this host has been deployed. Production status is achieved only after the operator completes the checklist against the actual target environment.
