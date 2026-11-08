#!/usr/bin/env python3
"""TRI-SQLITE-001: opt-in, offline SQLite state and verified snapshot backups.

No chat runtime imports, network access, model inference, native training or Notion.
This adapter is inert until explicitly opened by an owner-authorized caller.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import stat
import time

VERSION = 1
BACKUP_SCHEMA = "trinite.sqlite.backup.v1"
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "trinite.db"
MAX_USER = 8 * 1024
MAX_ASSISTANT = 64 * 1024
MAX_ID = 128
MAX_BACKUP_SECONDS = 30.0

TABLES = (
    """CREATE TABLE conversations (
        conversation_id TEXT PRIMARY KEY NOT NULL CHECK(length(conversation_id) BETWEEN 1 AND 128),
        created_utc TEXT NOT NULL,
        updated_utc TEXT NOT NULL,
        consent INTEGER NOT NULL CHECK(consent = 1)
    )""",
    """CREATE TABLE messages (
        message_id INTEGER PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        content TEXT NOT NULL CHECK(length(CAST(content AS BLOB)) BETWEEN 1 AND 65536),
        classification TEXT NOT NULL CHECK(classification IN
            ('user', 'verified_exact', 'model_generated_unverified', 'blocked', 'failed')),
        created_utc TEXT NOT NULL
    )""",
    """CREATE TABLE route_decisions (
        decision_id INTEGER PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        route TEXT NOT NULL CHECK(route IN ('chat', 'formal', 'unsupported', 'clarify')),
        outcome TEXT NOT NULL CHECK(outcome IN
            ('verified_exact', 'model_generated_unverified', 'blocked', 'failed')),
        request_sha256 TEXT NOT NULL CHECK(length(request_sha256) = 64),
        created_utc TEXT NOT NULL
    )""",
    """CREATE TABLE tool_results (
        result_id INTEGER PRIMARY KEY,
        decision_id INTEGER NOT NULL REFERENCES route_decisions(decision_id) ON DELETE CASCADE,
        operation TEXT NOT NULL,
        exact_value TEXT,
        verified INTEGER NOT NULL CHECK(verified IN (0, 1)),
        created_utc TEXT NOT NULL
    )""",
    """CREATE TABLE model_registry (
        model_id TEXT PRIMARY KEY NOT NULL,
        asset_sha256 TEXT NOT NULL CHECK(length(asset_sha256) = 64),
        runtime_version TEXT NOT NULL,
        quantization TEXT NOT NULL,
        recorded_utc TEXT NOT NULL
    )""",
    """CREATE TABLE receipt_refs (
        receipt_id INTEGER PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        request_sha256 TEXT NOT NULL CHECK(length(request_sha256) = 64),
        policy_sha256 TEXT NOT NULL CHECK(length(policy_sha256) = 64),
        outcome TEXT NOT NULL,
        recorded_utc TEXT NOT NULL
    )""",
)
INDEXES = (
    "CREATE INDEX messages_by_conversation ON messages(conversation_id, message_id)",
    "CREATE INDEX decisions_by_conversation ON route_decisions(conversation_id, decision_id)",
    "CREATE INDEX receipts_by_conversation ON receipt_refs(conversation_id, receipt_id)",
)
EXPECTED_TABLES = {"conversations", "messages", "route_decisions", "tool_results",
                   "model_registry", "receipt_refs"}


class StoreError(Exception):
    """A bounded storage or backup contract failure."""


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(path, *, require_file=False):
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise StoreError("path_traversal")
    absolute = raw.absolute()
    for node in (absolute, *absolute.parents):
        try:
            st = node.lstat()
        except FileNotFoundError:
            if node != absolute and node == absolute.parent:
                raise StoreError("missing_parent") from None
            continue
        if stat.S_ISLNK(st.st_mode):
            raise StoreError("symlink_path")
        if node == absolute and require_file and not stat.S_ISREG(st.st_mode):
            raise StoreError("not_regular_file")
        if node == absolute and not require_file and not stat.S_ISREG(st.st_mode):
            raise StoreError("not_regular_file")
    if require_file and not absolute.is_file():
        raise StoreError("missing_regular_file")
    if not absolute.parent.is_dir():
        raise StoreError("missing_parent")
    return absolute


def exclusive_empty_file(path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    os.close(fd)


def sqlite_connect(path, *, readonly):
    if readonly:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True,
                                     timeout=3.0, isolation_level=None)
    else:
        connection = sqlite3.connect(str(path), timeout=3.0, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=3000")
    return connection


def check_schema(conn):
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version != VERSION:
        raise StoreError("schema_version_mismatch")
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )}
    if tables != EXPECTED_TABLES:
        raise StoreError("schema_membership_mismatch")
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise StoreError("sqlite_integrity_failed")
    if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise StoreError("foreign_key_integrity_failed")


def initialize(path=DEFAULT_DB, *, enable_persistence=False):
    if not enable_persistence:
        raise StoreError("persistence_opt_in_required")
    path = Path(path)
    if path == DEFAULT_DB and not path.parent.exists():
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    path = checked_path(path)
    if path.exists():
        raise StoreError("database_already_exists")
    exclusive_empty_file(path)
    conn = sqlite_connect(path, readonly=False)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in TABLES + INDEXES:
                conn.execute(statement)
            conn.execute("PRAGMA user_version=1")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        check_schema(conn)
    finally:
        conn.close()
    return path


def open_store(path=DEFAULT_DB, *, enable_persistence=False):
    if not enable_persistence:
        raise StoreError("persistence_opt_in_required")
    path = checked_path(path, require_file=True)
    conn = sqlite_connect(path, readonly=False)
    try:
        check_schema(conn)
        return conn
    except Exception:
        conn.close()
        raise


def validated_text(value, limit):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise StoreError("invalid_text")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        raise StoreError("invalid_utf8") from None
    if size > limit:
        raise StoreError("text_limit")
    return value


def validated_id(value):
    if not isinstance(value, str) or not (0 < len(value) <= MAX_ID):
        raise StoreError("invalid_conversation_id")
    if "\x00" in value:
        raise StoreError("invalid_conversation_id")
    return value


def record_exchange(path, conversation_id, user_text, assistant_text, *,
                    classification, enable_persistence=False):
    """Atomically append one explicit-opt-in conversation exchange."""
    validated_id(conversation_id)
    validated_text(user_text, MAX_USER)
    validated_text(assistant_text, MAX_ASSISTANT)
    if classification not in ("verified_exact", "model_generated_unverified",
                              "blocked", "failed"):
        raise StoreError("invalid_classification")
    conn = open_store(path, enable_persistence=enable_persistence)
    try:
        stamp = now_utc()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("""INSERT INTO conversations
                (conversation_id, created_utc, updated_utc, consent)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(conversation_id) DO UPDATE SET updated_utc=excluded.updated_utc""",
                (conversation_id, stamp, stamp))
            conn.executemany("""INSERT INTO messages
                (conversation_id, role, content, classification, created_utc)
                VALUES (?, ?, ?, ?, ?)""",
                ((conversation_id, "user", user_text, "user", stamp),
                 (conversation_id, "assistant", assistant_text, classification, stamp)))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def list_messages(path, conversation_id):
    validated_id(conversation_id)
    path = checked_path(path, require_file=True)
    with sqlite_connect(path, readonly=True) as conn:
        check_schema(conn)
        return conn.execute(
            "SELECT role, content, classification FROM messages "
            "WHERE conversation_id=? ORDER BY message_id", (conversation_id,)
        ).fetchall()


def delete_conversation(path, conversation_id, *, enable_persistence=False):
    validated_id(conversation_id)
    conn = open_store(path, enable_persistence=enable_persistence)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            changed = conn.execute("DELETE FROM conversations WHERE conversation_id=?",
                                   (conversation_id,)).rowcount
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return changed
    finally:
        conn.close()


def private_temp_path(destination):
    path = destination.with_name(destination.name + ".partial-" + secrets.token_hex(8))
    exclusive_empty_file(path)
    return path


def sync_parent(path):
    if hasattr(os, "O_DIRECTORY"):
        fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def verify_backup(backup, manifest):
    backup = checked_path(backup, require_file=True)
    manifest = checked_path(manifest, require_file=True)
    if backup.name + ".manifest.json" != manifest.name:
        raise StoreError("invalid_manifest_location")
    if backup.stat().st_size > (1 << 34):
        raise StoreError("oversized_backup")
    if manifest.stat().st_size > 4096:
        raise StoreError("oversized_manifest")
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise StoreError("duplicate_manifest_key")
            result[key] = value
        return result
    try:
        raw = json.loads(manifest.read_text("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, ValueError, RecursionError):
        raise StoreError("invalid_manifest") from None
    if not isinstance(raw, dict) or set(raw) != {
        "schema", "backup_name", "bytes", "sha256", "user_version", "integrity"
    }:
        raise StoreError("invalid_manifest_fields")
    if (raw["schema"] != BACKUP_SCHEMA or raw["backup_name"] != backup.name or
            type(raw["bytes"]) is not int or raw["bytes"] != backup.stat().st_size or
            type(raw["user_version"]) is not int or raw["user_version"] != VERSION or
            raw["integrity"] != "ok" or not isinstance(raw["sha256"], str) or
            len(raw["sha256"]) != 64 or sha256_path(backup) != raw["sha256"]):
        raise StoreError("backup_identity_mismatch")
    conn = sqlite_connect(backup, readonly=True)
    try:
        check_schema(conn)
    finally:
        conn.close()
    return raw


def backup_database(database, target, *, authorize_backup=False):
    if not authorize_backup:
        raise StoreError("backup_authorization_required")
    source = checked_path(database, require_file=True)
    target = checked_path(target)
    manifest = checked_path(str(target) + ".manifest.json")
    if target.exists() or manifest.exists():
        raise StoreError("backup_destination_exists")
    source_conn = sqlite_connect(source, readonly=True)
    temp = None
    temp_manifest = None
    try:
        check_schema(source_conn)
        temp = private_temp_path(target)
        dest_conn = sqlite_connect(temp, readonly=False)
        try:
            started = time.monotonic()
            def progress(status, remaining, total):
                if time.monotonic() - started > MAX_BACKUP_SECONDS:
                    raise StoreError("backup_deadline")
            source_conn.backup(dest_conn, pages=64, progress=progress, sleep=0.02)
            # Offline standalone snapshot; make sure it does not depend on
            # the live source's WAL sidecars.
            dest_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            dest_conn.execute("PRAGMA journal_mode=DELETE")
            check_schema(dest_conn)
        finally:
            dest_conn.close()
        info = {
            "schema": BACKUP_SCHEMA, "backup_name": target.name,
            "bytes": temp.stat().st_size, "sha256": sha256_path(temp),
            "user_version": VERSION, "integrity": "ok",
        }
        temp_manifest = private_temp_path(manifest)
        with open(temp_manifest, "wb") as sink:
            sink.write(json_bytes(info))
            sink.flush()
            os.fsync(sink.fileno())
        with open(temp, "rb") as src:
            os.fsync(src.fileno())
        # Link is atomic and refuses an existing file (unlike os.replace).
        os.link(temp, target)
        os.link(temp_manifest, manifest)
        sync_parent(target)
        verify_backup(target, manifest)
        return target, manifest
    finally:
        source_conn.close()
        if temp is not None:
            temp.unlink(missing_ok=True)
        if temp_manifest is not None:
            temp_manifest.unlink(missing_ok=True)


def restore_backup(backup, manifest, output, *, authorize_restore=False):
    if not authorize_restore:
        raise StoreError("restore_authorization_required")
    info = verify_backup(backup, manifest)
    output = checked_path(output)
    if output.exists():
        raise StoreError("restore_destination_exists")
    temp = private_temp_path(output)
    try:
        with open(backup, "rb") as src, open(temp, "wb") as dest:
            shutil.copyfileobj(src, dest, length=1 << 20)
            dest.flush()
            os.fsync(dest.fileno())
        if temp.stat().st_size != info["bytes"] or sha256_path(temp) != info["sha256"]:
            raise StoreError("restore_source_changed")
        conn = sqlite_connect(temp, readonly=True)
        try:
            check_schema(conn)
        finally:
            conn.close()
        os.link(temp, output)
        sync_parent(output)
        return output
    finally:
        temp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="TRI-SQLITE-001: local-only state and snapshots")
    commands = parser.add_subparsers(dest="operation", required=True)
    init = commands.add_parser("init")
    init.add_argument("--db", default=str(DEFAULT_DB))
    init.add_argument("--enable-persistence", action="store_true")
    b = commands.add_parser("backup")
    b.add_argument("--db", default=str(DEFAULT_DB))
    b.add_argument("--output", required=True)
    b.add_argument("--authorize-backup", action="store_true")
    v = commands.add_parser("verify-backup")
    v.add_argument("--backup", required=True)
    v.add_argument("--manifest", required=True)
    r = commands.add_parser("restore")
    r.add_argument("--backup", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--authorize-restore", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.operation == "init":
            value = initialize(args.db, enable_persistence=args.enable_persistence)
            print("created", value)
        elif args.operation == "backup":
            out, manifest = backup_database(args.db, args.output,
                                            authorize_backup=args.authorize_backup)
            print("backup", out, "manifest", manifest)
        elif args.operation == "verify-backup":
            result = verify_backup(args.backup, args.manifest)
            print("verified", result["sha256"])
        elif args.operation == "restore":
            value = restore_backup(args.backup, args.manifest, args.output,
                                   authorize_restore=args.authorize_restore)
            print("restored", value)
    except (StoreError, sqlite3.Error, OSError, ValueError) as exc:
        label = str(exc) if isinstance(exc, StoreError) else type(exc).__name__
        parser.exit(2, "TRI-SQLITE-001: " + label + "\n")


if __name__ == "__main__":
    main()
