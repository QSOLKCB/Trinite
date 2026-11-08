# TRI-SQLITE-001 — local SQLite persistence and verified backups

**Status:** separate additive implementation PR, 9 November 2026. Its scheduling was first committed directly to [ROADMAP](ROADMAP.md). This code is an application-side storage component; it does not modify native Trinite models, historical fixtures/weights, MTC-001, scientific evidence or teacher data admission.

## Architecture and privacy

- Python 3.11+ standard-library sqlite3; zero extra dependencies, offline and local by default. No cloud database, HTTP, model call, Node, NPM or Notion API in this module.
- Default path: data/trinite.db. .gitignore excludes that file, its WAL, SHM and journal sidecars, and the data/backups/ directory. Never commit user conversations, backup files, journals or credentials.
- Persistence **requires explicit opt-in** at both initialization and programmatic write. No file is created by a read, and this PR does **not automatically wire** the existing TRI-CHAT-001 WebUI to disk.
- Version 1 schema with a PRAGMA user_version gate; explicit SQL transactions, parameterized statements, strict foreign keys, WAL and synchronous FULL, identity/size constraints. Tables: conversations, messages, route_decisions, tool_results, model_registry and receipt_refs.
- Exchange writes atomically store one bounded user + assistant pair; reading and logical deletion are explicit. Input is *application data*, not an independently verified research label or authorization token.
- SQLite is plaintext without native encryption in the standard-library build. Use a private local directory and OS/disk encryption for sensitive material. Deletion is not forensic erasure from previous SQLite pages, WAL, filesystem snapshots or retained backups.
- Notion remains an optional curated cloud knowledge index. Raw chat transcripts, private model outputs and local database files are never sent there by this adapter.

## Initialize only with explicit permission

From a trusted reviewed checkout and with owner approval:

~~~sh
python3 -I -B scripts/trinite_chat_sqlite_v1.py init --enable-persistence
~~~

The CLI creates data/ with private permissions if absent and refuses an existing database. The default app data file is not part of Git or the scientific evidence archive.

## Explicit online backup, offline verification and fresh restore

Make the local backup directory private before running. The example names are illustrative: use a unique new destination for every capture.

~~~sh
mkdir -m 700 -p data/backups
python3 -I -B scripts/trinite_chat_sqlite_v1.py backup \
  --db data/trinite.db \
  --output data/backups/trinite-2026-11-09.db \
  --authorize-backup
python3 -I -B scripts/trinite_chat_sqlite_v1.py verify-backup \
  --backup data/backups/trinite-2026-11-09.db \
  --manifest data/backups/trinite-2026-11-09.db.manifest.json
python3 -I -B scripts/trinite_chat_sqlite_v1.py restore \
  --backup data/backups/trinite-2026-11-09.db \
  --manifest data/backups/trinite-2026-11-09.db.manifest.json \
  --output data/trinite-restored.db \
  --authorize-restore
~~~

Backup uses SQLite Connection.backup() to obtain a consistent snapshot of a live WAL database rather than copying an open database file. A standalone backup is converted to DELETE journal mode, validated by SQLite integrity_check and foreign_key_check plus version and table inventory, and bound to an external JSON SHA-256/byte-count manifest. Backup and manifest are published without clobbering existing filenames; failures retain any already published partial output for audit.

Restore verifies the manifest, exact checksum and schema before making a fresh destination. It never overwrites an active DB. Stop the app and inspect the restored database before deciding on any separate replacement or switchover. Secure long-term retention, backup encryption and deleting old copies are owner-managed policies.

## Programmatic API (not integrated into TRI-CHAT-001 yet)

~~~python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "trinite_sqlite_v1", Path("scripts/trinite_chat_sqlite_v1.py")
)
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)

store.record_exchange(
    Path("data/trinite.db"), "session-1", "Hello", "Hi",
    classification="model_generated_unverified",
    enable_persistence=True,
)
messages = store.list_messages(Path("data/trinite.db"), "session-1")
~~~

Users must approve local persistence before invoking these writes. A future separately reviewed UI integration will need per-session retention/deletion controls; it must never use logged training or Notion synchronization as a silent side effect.

## Test and claim boundaries

~~~sh
python3 -I -B -m unittest discover -s tests/trinite_sqlite_v1 -v
~~~

Synthetic offline tests cover no-opt-in/no-file semantics, migration/version inventory, WAL operation, bounded atomic records, parameterized SQL injection-as-data, deletion, online backup while a writer holds a transaction, manifest corruption, no-clobber restore and symlink rejection.

Tests on GitHub-hosted ephemeral files demonstrate a scoped implementation contract, not an owner-machine durability guarantee, forensic secure erasure, whole-disk corruption resistance or user authorization for real private-data writes. Next independent step after TRI-CHAT-001 merge is a separate integration PR, not a silent branch substitution.
