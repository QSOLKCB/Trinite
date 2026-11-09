#!/usr/bin/env python3
"""TRI-SQLITE-001: opt-in, offline SQLite state and verified snapshot backups.

No chat runtime imports, network access, model inference, native training or Notion.
This adapter is inert until explicitly opened by an owner-authorized caller.
"""
from __future__ import annotations

import argparse
import errno
from contextlib import closing, contextmanager
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
MAX_CONVERSATIONS = 1024
MAX_MESSAGES_PER_CONVERSATION = 256
MAX_TOTAL_MESSAGES = 4096

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


def _file_identity(info):
    if not stat.S_ISREG(info.st_mode):
        raise StoreError("not_regular_file")
    if os.name == "posix":
        if info.st_uid != os.geteuid():
            raise StoreError("unowned_file")
        if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise StoreError("unsafe_file_permissions")
        if info.st_nlink != 1:
            raise StoreError("unsafe_file_link_count")


@contextmanager
def pinned_parent(directory):
    """Walk from / using descriptor-relative, no-follow directory opens.

    Sticky ancestors (for example /tmp) are acceptable because other UIDs
    cannot rename an owner-owned child there. A non-sticky directory writable
    by other users is not acceptable even if the final parent is mode 0700.
    """
    if os.name != "posix" or not hasattr(os, "O_DIRECTORY"):
        raise StoreError("secure_directory_descriptors_unavailable")
    directory = Path(directory).absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    fd = os.open("/", flags)
    try:
        for segment in directory.parts[1:]:
            info = os.fstat(fd)
            if (info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                    and not info.st_mode & stat.S_ISVTX):
                raise StoreError("unsafe_writable_ancestor")
            new_fd = os.open(segment, flags, dir_fd=fd)
            os.close(fd)
            fd = new_fd
        info = os.fstat(fd)
        if info.st_uid != os.geteuid():
            raise StoreError("unowned_parent_directory")
        if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise StoreError("unsafe_parent_permissions")
        yield fd
    finally:
        os.close(fd)


def _checked_fd(parent_fd, name):
    """Open one regular owned private file without following a final symlink."""
    # A FIFO opened O_RDONLY normally waits indefinitely for a writer,
    # before fstat can reject the nonregular file. Nonblocking open lets
    # _file_identity reject it promptly; it does not affect regular files.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise StoreError("symlink_path") from None
        raise
    try:
        _file_identity(os.fstat(fd))
        return fd
    except Exception:
        os.close(fd)
        raise


@contextmanager
def checked_reader(path):
    """Validate and read through the same open file descriptor."""
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise StoreError("path_traversal")
    path = raw.absolute()
    with pinned_parent(path.parent) as directory_fd:
        fd = _checked_fd(directory_fd, path.name)
        with os.fdopen(fd, "rb") as source:
            yield source


def sha256_path(path):
    digest = hashlib.sha256()
    with checked_reader(path) as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(path, *, require_file=False):
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise StoreError("path_traversal")
    absolute = raw.absolute()
    with pinned_parent(absolute.parent) as parent_fd:
        try:
            info = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if require_file:
                raise StoreError("missing_regular_file") from None
        else:
            if stat.S_ISLNK(info.st_mode):
                raise StoreError("symlink_path")
            fd = _checked_fd(parent_fd, absolute.name)
            try:
                opened = os.fstat(fd)
                if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
                    raise StoreError("file_changed_during_check")
            finally:
                os.close(fd)
    return absolute


def exclusive_empty_file(path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    os.close(fd)


def path_present(path):
    """Include broken symlinks and special files; Path.exists() misses them."""
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return False
    return True


def sqlite_sidecars(path):
    return tuple(Path(str(path) + ext) for ext in ("-wal", "-shm", "-journal"))


def sqlite_connect(path, *, readonly, immutable_snapshot=False):
    """Keep SQLite locks for live databases; immutable is snapshot-only.

    The absence of WAL/SHM files does not make a live database immutable.
    Bypassing SQLite locking can mix old and new pages during backup if a
    writer updates the database and checkpoints between backup steps.
    """
    if immutable_snapshot and not readonly:
        raise StoreError("immutable_snapshot_requires_readonly")
    if readonly:
        wal, shm, journal = sqlite_sidecars(path)
        has_wal, has_shm = path_present(wal), path_present(shm)
        if immutable_snapshot:
            # Only the verified standalone backup and private restore copy
            # are eligible, not mutable application databases.
            if has_wal or has_shm or path_present(journal):
                raise StoreError("snapshot_sidecars_exist")
            mode = "mode=ro&immutable=1"
        else:
            if has_wal != has_shm:
                raise StoreError("incomplete_wal_sidecars")
            if has_wal:
                checked_path(wal, require_file=True)
                checked_path(shm, require_file=True)
            # For live sources use normal locking and change detection even
            # if no sidecars exist yet. SQLite may create coordination files.
            mode = "mode=ro"
        connection = sqlite3.connect(path.as_uri() + "?" + mode, uri=True,
                                     timeout=3.0, isolation_level=None)
    else:
        connection = sqlite3.connect(str(path), timeout=3.0, isolation_level=None)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=3000")
    return connection


def configure_writable_connection(connection):
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")


def check_schema(conn):
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version != VERSION:
        raise StoreError("schema_version_mismatch")
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT GLOB 'sqlite_*'"
    )}
    if tables != EXPECTED_TABLES:
        raise StoreError("schema_membership_mismatch")
    # Table names and PRAGMA integrity_check alone cannot detect a renamed
    # column, missing CHECK/foreign key, relaxed NOT NULL or missing index.
    # Schema v1 is a frozen contract; compare the exact stored definitions
    # of all application tables and indexes, rejecting extra views/triggers.
    expected_sql = {
        ("table", statement.split()[2]): statement for statement in TABLES
    }
    expected_sql.update({
        ("index", statement.split()[2]): statement for statement in INDEXES
    })
    actual_sql = {
        (kind, name): sql for kind, name, sql in conn.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'index', 'view', 'trigger') "
            "AND name NOT GLOB 'sqlite_*' AND sql IS NOT NULL"
        )
    }
    if actual_sql != expected_sql:
        raise StoreError("schema_structure_mismatch")
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise StoreError("sqlite_integrity_failed")
    if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise StoreError("foreign_key_integrity_failed")


def _fresh_destination_at(parent_fd, filename):
    for name in (filename, *(filename + ext for ext in ("-wal", "-shm", "-journal"))):
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise StoreError("database_already_exists")


def _parent_unchanged(path, parent_fd):
    """Detect parent substitution; publication still uses the pinned inode."""
    try:
        named = os.stat(path.parent, follow_symlinks=False)
    except FileNotFoundError:
        raise StoreError("parent_directory_changed") from None
    pinned = os.fstat(parent_fd)
    if not stat.S_ISDIR(named.st_mode) or (named.st_dev, named.st_ino) != (
            pinned.st_dev, pinned.st_ino):
        raise StoreError("parent_directory_changed")


def _unlink_at(parent_fd, name):
    try:
        os.unlink(name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def initialize(path=DEFAULT_DB, *, enable_persistence=False):
    if not enable_persistence:
        raise StoreError("persistence_opt_in_required")
    path = Path(path)
    if path == DEFAULT_DB and not path.parent.exists():
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    path = checked_path(path)
    with pinned_parent(path.parent) as parent_fd:
        if not Path("/proc/self/fd").is_dir():
            # sqlite3 accepts filenames, not a caller-owned file descriptor.
            # On Linux, the proc fd path keeps SQLite's main/WAL/SHM writes
            # rooted in the validated directory even after a rename.
            raise StoreError("fd_relative_sqlite_path_unavailable")
        _fresh_destination_at(parent_fd, path.name)
        temp_name = path.name + ".partial-" + secrets.token_hex(8)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
        os.close(fd)
        anchored_temp = Path("/proc/self/fd") / str(parent_fd) / temp_name
        try:
            conn = sqlite_connect(anchored_temp, readonly=False)
            try:
                configure_writable_connection(conn)
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
            _parent_unchanged(path, parent_fd)
            _fresh_destination_at(parent_fd, path.name)
            try:
                os.link(temp_name, path.name, src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd, follow_symlinks=False)
            except FileExistsError:
                raise StoreError("database_already_exists") from None
            # Remove the extra hard link before the caller can use the DB.
            _unlink_at(parent_fd, temp_name)
            os.fsync(parent_fd)
            return path
        finally:
            _unlink_at(parent_fd, temp_name)
            for ext in ("-wal", "-shm", "-journal"):
                _unlink_at(parent_fd, temp_name + ext)


def open_store(path=DEFAULT_DB, *, enable_persistence=False):
    if not enable_persistence:
        raise StoreError("persistence_opt_in_required")
    path = checked_path(path, require_file=True)
    # Reject an incompatible or attacker-replaced database before applying
    # persistent journal-mode changes to it. Recheck on the writable handle to
    # catch a replacement between the two opens.
    with closing(sqlite_connect(path, readonly=True)) as inspection:
        check_schema(inspection)
    conn = sqlite_connect(path, readonly=False)
    try:
        check_schema(conn)
        configure_writable_connection(conn)
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
            existing = conn.execute(
                "SELECT count(*) FROM messages WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()[0]
            if existing > MAX_MESSAGES_PER_CONVERSATION - 2:
                raise StoreError("conversation_message_limit")
            total = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
            if total > MAX_TOTAL_MESSAGES - 2:
                raise StoreError("total_message_limit")
            known = conn.execute(
                "SELECT 1 FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            if known is None:
                conversations = conn.execute(
                    "SELECT count(*) FROM conversations"
                ).fetchone()[0]
                if conversations >= MAX_CONVERSATIONS:
                    raise StoreError("conversation_count_limit")
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
    with closing(sqlite_connect(path, readonly=True)) as conn:
        check_schema(conn)
        rows = conn.execute(
            "SELECT role, content, classification FROM messages "
            "WHERE conversation_id=? ORDER BY message_id LIMIT ?",
            (conversation_id, MAX_MESSAGES_PER_CONVERSATION + 1),
        ).fetchall()
        if len(rows) > MAX_MESSAGES_PER_CONVERSATION:
            raise StoreError("conversation_message_limit")
        return rows


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


def cleanup_private_temp(path):
    Path(path).unlink(missing_ok=True)
    for sidecar in sqlite_sidecars(path):
        sidecar.unlink(missing_ok=True)


def publish_link_no_clobber(source, destination, error):
    try:
        os.link(source, destination)
    except FileExistsError:
        raise StoreError(error) from None


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
        with checked_reader(manifest) as stream:
            raw = json.loads(stream.read(4097).decode("utf-8"),
                             object_pairs_hook=reject_duplicates)
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
    conn = sqlite_connect(backup, readonly=True, immutable_snapshot=True)
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
        publish_link_no_clobber(temp, target, "backup_destination_exists")
        publish_link_no_clobber(temp_manifest, manifest, "backup_destination_exists")
        # No-clobber hard-link publication temporarily creates a second link.
        # Remove those temp names before verifying strict st_nlink == 1.
        temp.unlink()
        temp_manifest.unlink()
        sync_parent(target)
        verify_backup(target, manifest)
        return target, manifest
    finally:
        source_conn.close()
        if temp is not None:
            cleanup_private_temp(temp)
        if temp_manifest is not None:
            temp_manifest.unlink(missing_ok=True)


def check_fresh_database_destination(output):
    """An absent main .db file is insufficient if an old WAL still exists."""
    if path_present(output):
        raise StoreError("restore_destination_exists")
    if any(path_present(sidecar) for sidecar in sqlite_sidecars(output)):
        raise StoreError("restore_destination_sidecars_exist")


def restore_backup(backup, manifest, output, *, authorize_restore=False):
    if not authorize_restore:
        raise StoreError("restore_authorization_required")
    info = verify_backup(backup, manifest)
    output = checked_path(output)
    check_fresh_database_destination(output)
    temp = private_temp_path(output)
    try:
        with checked_reader(backup) as src, open(temp, "wb") as dest:
            shutil.copyfileobj(src, dest, length=1 << 20)
            dest.flush()
            os.fsync(dest.fileno())
        if temp.stat().st_size != info["bytes"] or sha256_path(temp) != info["sha256"]:
            raise StoreError("restore_source_changed")
        conn = sqlite_connect(temp, readonly=True, immutable_snapshot=True)
        try:
            check_schema(conn)
        finally:
            conn.close()
        # Detect WAL/SHM/journal artifacts added while copying, before the
        # no-clobber main-database publication. The target directory must
        # remain owner-controlled; this is not a hostile-host race guarantee.
        check_fresh_database_destination(output)
        publish_link_no_clobber(temp, output, "restore_destination_exists")
        sync_parent(output)
        return output
    finally:
        cleanup_private_temp(temp)


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
