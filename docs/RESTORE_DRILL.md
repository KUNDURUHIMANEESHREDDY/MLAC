# MLAC backup restore drill

This runbook describes a safe, isolated restore of a downloaded SQLite state backup. It does not restore into a configured production database and does not prove recovery of a live MLAC deployment.

## Preconditions

- Download the backup object from the configured Object Lock bucket into an operator-controlled directory.
- Obtain the expected SHA-256 value from the trusted S3 object metadata or the separately retained backup record. Do not derive the expected value from the downloaded file.
- Keep the downloaded backup and restore destination in directories accessible only to the operator.
- The destination directory must already exist, must not be group- or world-writable, and must support POSIX directory fsync (the utility fails closed on non-POSIX platforms).
- The destination file and its SQLite sidecar paths (`-wal`, `-shm`, and `-journal`) must not exist. The restore utility refuses to overwrite any existing path or open SQLite when a destination-named sidecar is already present.

## SQLite restore

Create a private working directory and restore to a new file:

```bash
install -d -m 0700 ./restore-drill
python -m scripts.restore_backup \
  ./downloaded/api-state.sqlite3 \
  ./restore-drill/restored.sqlite3 \
  --sha256 '<trusted-64-character-sha256>'
```

The utility first copies the downloaded backup through a single no-follow file descriptor into a private temporary directory. It checks that the source file identity and mutation-sensitive metadata remain stable while copying. It then runs `scripts.verify_backup` against that snapshot and restores from the exact same snapshot; changing or replacing the original download after verification cannot change the restored bytes. It runs `PRAGMA integrity_check` on the restored file, records the table count, and leaves the result with mode `0600`. It creates the destination with exclusive-create semantics and never replaces an existing file. It fsyncs both the restored file and its parent directory before reporting success. If verification or copying fails, it removes only the file it created.

Inspect the restored database without enabling writes:

```bash
python - <<'PY'
import sqlite3
from pathlib import Path

path = Path("./restore-drill/restored.sqlite3")
with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
    print("integrity:", db.execute("PRAGMA integrity_check").fetchone()[0])
    print("tables:", db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall())
PY
```

## PostgreSQL restore drill

Provision a **new, empty disposable PostgreSQL database** on a test instance. The database name must begin with `mlac_restore_drill_`; the restore utility does not create or drop databases. Keep the source archive immutable and obtain its expected SHA-256 from trusted object metadata.

Read the disposable target DSN without echoing it or placing it in shell history. Do not enable shell tracing (set -x) while handling the credential:

```bash
read -r -s -p "Disposable PostgreSQL DSN: " MLAC_RESTORE_DATABASE_URL
printf '\n'
export MLAC_RESTORE_DATABASE_URL
python -m scripts.restore_postgres_backup \
  ./downloaded/api-state.dump \
  --sha256 '<trusted-64-character-sha256>'
unset MLAC_RESTORE_DATABASE_URL
```

For a local-only test service, `sslmode=disable` is accepted only for `localhost` or a loopback IP. Remote targets must use `sslmode=verify-full`; `require` and `verify-ca` are rejected because they do not verify the server hostname. The utility:

1. Verifies the archive SHA-256 and PostgreSQL custom-format table of contents before connecting.
2. Requires the dedicated `mlac_restore_drill_*` database-name prefix.
3. Rejects any target containing user schemas, relations, functions, or types.
4. Runs `pg_restore --exit-on-error --no-owner --no-acl --single-transaction`; it never passes `--clean` or `--create`.
5. Reports only the target database name, trusted digest and restored table count. Restore output is suppressed to avoid logging database contents.

A restore failure rolls back the restore transaction. The utility does not drop the database or attempt an automatic cleanup. Inspect and remove a disposable database only through the operator's separately approved database lifecycle procedure.

The CI integration test uses its ephemeral PostgreSQL 16 service to create a source database, dump one table, restore it into a fresh drill database, and read the restored row. A second test proves that a non-empty target is refused and its existing row remains unchanged.

A complete deployment-specific drill must additionally record the disposable server version, extension/ownership requirements, application-level read checks, measured recovery time and observed RPO/RTO. Never use the production DSN.

## What this proves—and does not prove

A successful SQLite drill proves that one downloaded backup passed its trusted SHA-256 check, was restored into a new local file, and passed SQLite structural integrity. A successful CI PostgreSQL drill proves that a trusted custom archive can be restored transactionally into an empty disposable PostgreSQL 16 database and that an existing non-empty target is not modified. Neither proves production S3 Object Lock enforcement, independent administration, encryption-key recovery, production-cluster compatibility, API startup, or application-level recovery. Production RPO/RTO remains unverified until the operator runs and records the full deployment-specific drill.
