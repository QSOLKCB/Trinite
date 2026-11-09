# TRI-CHAT-DB-001 — opt-in local SQLite session integration

**Status:** additive review PR, 9 November 2026. Based on merged TRI-CHAT-001 (PR #143), TRI-SQLITE-001 (PR #144) and independent security hardening TRI-SEC-001 (PR #145). This phase integrates the existing local chat UI and Python HTTP server with the existing version-1 SQLite database; it makes **no native Trinite research, training, fixture or evidence changes**.

## Two explicit modes

1. **Default mode — browser-tab memory only.** The existing \`trinite.chat.request.v1\` JSON contract and \`/api/chat\` endpoint remain unchanged. The browser never issues SQL calls or writes chat data to the server's disk when the checkbox is unchecked. No database is loaded or created at server startup unless the owner explicitly enables it.
2. **Opt-in mode — local SQLite.** Requires **both** the server's \`--enable-persistence\` flag and explicit browser selection of **Remember in local SQLite**. A preinitialized database must already exist. Server-side code—not teacher output and not a browser-supplied history—reads the most recent complete exchanges, applies the existing message and 16-KiB request budgets, calls the same rules-first chat controller, then atomically appends a successful exchange to SQLite. Failed/blocked outputs are displayed but are not misrepresented as saved.

**No silent fallback** from a requested persistent exchange to in-tab storage or a second teacher call after a database failure. UI persistence success is acknowledged only when \`record_exchange()\` returns successfully.

## Local setup (explicit authorization)

From an owner-reviewed checkout, before creating any real chat data:

~~~sh
# Create the default database exactly once (skip this if already initialized):
uv run --no-project python -I -B scripts/trinite_chat_sqlite_v1.py init --enable-persistence

# Run the loopback-only chat server, explicitly permitting user-chosen saving:
uv run --no-project python -I -B scripts/run_trinite_chat_v1.py --port 8765 \
  --enable-persistence --db-path data/trinite.db
~~~

If \`uv\` is not installed, the exact commands also work with \`python3 -I -B\` directly. Neither command downloads a model or sends data to Notion. The optional separately authorized local teacher requires the original \`--model-*\` flags and pinned weight/runtime identity; no model is included in this integration.

Open http://127.0.0.1:8765/ on the same machine. Browser checkbox selection is per user action, **unchecked by default on every page load**. With no checkbox or without \`--enable-persistence\`, the existing chat is memory-only. If the database preflight fails, server startup fails closed rather than quietly running a different mode.

## Session identity and local replay

- A new persisted session gets a cryptographically random **128-bit hexadecimal session token** only after a successful disk write. The browser stores this token in same-origin \`localStorage\` only after the server reports \`persisted: true\`; it is **not a password for the OS account**. Anyone with that token and access to the same local HTTP origin can request load/delete. Browser localStorage is not encrypted; protect the local OS account and browser profile.
- Existing sessions can be loaded through a POST to \`/api/session\` with action \`load\`. The response contains **at most the latest 8 complete exchanges** (16 messages); previous exchanges still exist in SQLite subject to the storage caps. It never exposes a listing of every database session or raw database paths.
- A new persistent POST to \`/api/chat\` contains exactly \`schema\`, \`message\` and \`session_id\`; user-supplied \`history\` is forbidden. The server constructs model context from trusted SQLite session rows, retaining a **contiguous suffix** of complete pairs fitting each 8-KiB entry and the 16-KiB canonical request budget. A stored oversized 8–64-KiB assistant response is displayable, but terminates context replay rather than entering a future model prompt.
- The only saved outcomes are completed \`verified_exact\` or \`model_generated_unverified\` exchanges. No model text gains authority, verified status or SQLite permission. Receipts remain controller records, not proof of fresh neural replay.
- **Forget saved conversation** issues an explicit server deletion command. **Clear conversation** only clears the display and disconnects opt-in mode; saved database content is preserved until Forget. Logical deletion does **not** erase old pages, WAL, filesystem snapshots or independently retained backups.
- If \`localStorage\` is unavailable, no new saved session is created. If a persisted HTTP response or browser token write is interrupted, orphaned database sessions may require owner-local remediation; this prototype does not claim transactional coordination across browser storage and SQLite.

## API contracts

| Endpoint | Request | Outcome |
| --- | --- | --- |
| \`POST /api/chat\` | \`trinite.chat.request.v1\`: \`message\` and client in-tab \`history\` | Original stateless contract |
| \`POST /api/chat\` | \`trinite.chat.persist.request.v1\`: \`message\`, \`session_id\` (null for first save) | Additive \`persisted\` boolean and generated opaque \`session_id\`, only after SQLite commit |
| \`POST /api/session\` | \`trinite.chat.session.request.v1\`: \`action: load\`, 32-char lowercase hex \`session_id\` | \`trinite.chat.session.result.v1\` with bounded message tail |
| \`POST /api/session\` | Same schema, \`action: delete\` | Logical deletion status, no stored messages returned |

All commands retain the chat server's strict loopback listener, Host/Origin/content-type/size checks and finite transport deadlines. No GET route exposes saved transcripts. Unknown or invalid session IDs fail closed. Database errors are redacted as \`storage_unavailable\`, with no automatic retry, cloud fallback or model/tool escalation. Owner-local processes sharing the OS account remain outside the current access-control boundary.

## Tests and review gates

~~~sh
uv run --no-project python -I -B -m unittest discover -s tests/trinite_chat_v1 -v
uv run --no-project python -I -B -m unittest discover -s tests/trinite_sqlite_v1 -v
uv run --no-project python -I -B -m unittest discover -s tests/trinite_chat_db_v1 -v
~~~

The third suite uses a synthetic model double and temporary SQLite files; it exercises opt-in, saved history reconstruction, session loading/deletion, invalid IDs, injection attempts, oversized assistant replies, storage-write rejection, Host/Origin denial and disabled-persistence behavior. The two original suites are required to remain green. These tests do **not** certify a real LFM3 runtime, browser end-to-end automation, secure file erasure, multi-user deployment or whole-system security.

The new storage integration still requires independent review, a controlled real-browser owner smoke test and explicit approval before merge/release. No cloud, Notion synchronization, knowledge retrieval, automatic training-data intake, d1 decision backend or native model-parameter modification is included.

## Next gate

**TRI-DECIDE-001**: freeze the baseline deterministic controller routing and only then evaluate the optional, explicitly authorized d1 advisor with fixed software doubles and later owner-local evidence. Treat abstention, bad advice, model ID mismatch, latency/cost and false tool execution independently. The controller—not d1—remains the final authority.
