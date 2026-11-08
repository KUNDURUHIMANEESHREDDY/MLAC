# MLAC 1.0.0 release candidate

## Scope

MLAC 1.0.0 packages the governed architecture controller and the first production-candidate API runtime topology:

- OIDC-authenticated, issuer/subject-scoped model and proposal API.
- Versioned PostgreSQL metadata for production, with SQLite retained for local single-host development; transactional schema initialization and tenant-scoped queries.
- Shared Redis sliding-window rate limits for pre-authentication peers and authenticated principals; Redis outages fail closed.
- Single-instance Docker deployment behind Caddy automatic HTTPS, with the API and Redis ports unpublished.
- Non-root API container, read-only root filesystem, dropped capabilities, and pinned reverse-proxy trust.
- PostgreSQL custom-format backups (and SQLite development backups) uploaded to S3 with versioning and Object Lock COMPLIANCE retention.
- Signed dataset provenance enforced for training and evaluation, plus bounded recursive controller evaluation with immutable authorization and recovery boundaries.
- CI lint, unit/integration tests, Docker image build, package build/install, and CLI version smoke test.

## Build and verification

The main-branch workflow must pass on the exact main commit. It runs Ruff, the full pytest suite (including shared-Redis and PostgreSQL integration tests), Docker build, wheel/sdist build, wheel installation, and the mlac version smoke test. The container publication workflow publishes an immutable SHA-tagged image to GHCR. Do not deploy the mutable main tag.

## Production deployment status

The repository contains a production-candidate deployment topology, not evidence of a live deployment. Production acceptance still requires an operator-owned host and DNS, real OIDC configuration, private Redis credentials, persistent encrypted storage, S3 Object Lock configuration, backup/restore drill, monitoring, and security review. The production metadata backend is PostgreSQL and rate limits are shared through Redis. The supplied Compose stack runs a single API container by default; API replicas may share the database and Redis when deployed behind a suitably configured orchestrator with bounded connection pools and health checks.

The fixed evaluation worker, production dataset-signing key, held-out dataset, independent runtime attestor, and remote artifact store must also be exercised end-to-end before autonomous candidate operation is enabled in a target environment.

## Autonomous-operation boundary

Controller self-evolution remains evaluation-only. It records candidate outcomes but does not auto-promote controller weights or mutate authorization, governance, recovery, or deployment trust roots. Hosted API model weights cannot be modified by MLAC; recursive model evolution requires an operator-designated local/open-weight model.

## Release acceptance

- [ ] Exact-main CI run passes, including Redis integration and Docker build.
- [ ] GHCR image is published and deployment pins its immutable commit tag.
- [ ] OIDC, HTTPS, PostgreSQL persistence, Redis shared rate limits, and cross-tenant isolation verified on the target host.
- [ ] PostgreSQL S3 Object Lock backup uploaded and a restore drill completed.
- [ ] Fixed worker and signed held-out evaluation bundle pass end-to-end.
- [ ] Runtime identity and deployment authority verified independently.
- [ ] Monitoring, alerting, incident response, and rollback ownership recorded.

A package version of 1.0.0 or a successful CI build does not by itself mean the system is production-deployed.
