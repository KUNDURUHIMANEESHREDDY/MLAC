# MLAC backup restore drill

This runbook describes a safe, isolated restore of a downloaded SQLite state backup. It does not restore into a configured production database and does not prove recovery of a live MLAC deployment.

## Preconditions

- Download the backup object from the configured Object Lock bucket into an operator-controlled directory.
- Obtain the expected SHA-256 value from the trusted S3 object metadata or the separately retained backup record. Do not derive the expected value from the downloaded file.
- Keep the downloaded backup and restore destination in directories accessible only to the operator.
- The destination directory must already exist and must not be group- or world-writable.
- The destination file must not exist. The restore utility refuses to overwrite any existing path.

## SQLite restore

Create a private working directory and restore to a new file:

```bash
install -d -m 0700 ./restore-drill
python -m scripts.restore_backup \
  ./downloaded/api-state.sqlite3 \
  ./restore-drill/restored.sqlite3 \
  --sha256 '<trusted-64-character-sha256>'
```

The utility first calls `scripts.verify_backup` to verify the trusted digest and SQLite integrity. It then copies the database through SQLite's online backup API, runs `PRAGMA integrity_check` on the restored file, records the table count, and leaves the result with mode `0600`. It creates the destination with exclusive-create semantics and never replaces an existing file. If verification or copying fails, it removes only the file it created.

Inspect the restored database without enabling writes:

```bash
python - <<'PY'
import sqlite3
from pathlib import Path

path = Path("./restore-drill/restored.sqlite3")
with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
    print("integrity:", db.execute("PRAGMA integrity_check").fetchone()[0])
    print("tables:", db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall())
PY
```

## PostgreSQL restore boundary

The current utility intentionally does not automate PostgreSQL restoration. A custom-format archive is verified with `pg_restore --list`, but restoring it requires a separately provisioned disposable PostgreSQL instance/database, reviewed ownership/extension requirements, and explicit operator-controlled credentials. Do not point a restore command at the production DSN.

A complete PostgreSQL drill must record the disposable server version, restore command and exit status, schema/table counts, application-level read checks, and measured recovery time. Keep the source backup immutable.

## What this proves—and does not prove

A successful SQLite drill proves that one downloaded backup passed its trusted SHA-256 check, was restored into a new local file, and passed SQLite structural integrity. It does not prove S3 Object Lock enforcement, independent administration, encryption-key recovery, PostgreSQL restore, API startup, or application-level recovery. Production RPO/RTO remains unverified until the operator runs and records the full deployment-specific drill.
