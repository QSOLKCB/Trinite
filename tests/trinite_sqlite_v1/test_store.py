"""TRI-SQLITE-001: offline standard-library storage and backup tests."""
from contextlib import closing
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "trinite_chat_sqlite_v1.py"
spec = importlib.util.spec_from_file_location("trinite_sqlite_v1", SOURCE)
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)


class SQLiteContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "trinite.db"

    def tearDown(self):
        self.tmp.cleanup()

    def init(self):
        return store.initialize(self.db, enable_persistence=True)

    def write(self, conversation_id="conversation-1", user="Hello", answer="Hi",
              classification="model_generated_unverified"):
        return store.record_exchange(
            self.db, conversation_id, user, answer, classification=classification,
            enable_persistence=True
        )

    def test_zero_touch_without_explicit_opt_in(self):
        with self.assertRaisesRegex(store.StoreError, "persistence_opt_in_required"):
            store.initialize(self.db)
        self.assertFalse(self.db.exists())
        with self.assertRaises(store.StoreError):
            store.record_exchange(self.db, "s", "user", "answer",
                                  classification="verified_exact")
        self.assertFalse(self.db.exists())
        with self.assertRaises(store.StoreError):
            store.backup_database(self.db, self.root / "snapshot.db")
        self.assertFalse((self.root/"snapshot.db").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX directory mode contract")
    def test_attacker_writable_parent_is_rejected_before_creation(self):
        self.root.chmod(0o777)
        with self.assertRaisesRegex(store.StoreError, "unsafe_parent_permissions"):
            self.init()
        self.assertFalse(self.db.exists())

    def test_incompatible_database_is_not_changed_before_rejection(self):
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("CREATE TABLE unrelated(value TEXT)")
        self.db.chmod(0o600)
        before = self.db.read_bytes()
        with self.assertRaisesRegex(store.StoreError, "schema_version_mismatch"):
            store.open_store(self.db, enable_persistence=True)
        self.assertEqual(self.db.read_bytes(), before)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0], "delete")

    def test_init_publication_race_does_not_modify_substituted_victim(self):
        victim = self.root / "victim.db"
        with closing(sqlite3.connect(victim)) as conn:
            conn.execute("CREATE TABLE sentinel(value TEXT)")
            conn.execute("INSERT INTO sentinel VALUES ('preserve-me')")
        before = victim.read_bytes()
        real_link = store.os.link

        def substitute_before_publish(source, destination, **kwargs):
            # Destination is now relative to the pinned parent descriptor.
            self.db.symlink_to(victim)
            return real_link(source, destination, **kwargs)

        with mock.patch.object(store.os, "link", side_effect=substitute_before_publish):
            with self.assertRaisesRegex(store.StoreError, "database_already_exists"):
                self.init()
        self.assertTrue(self.db.is_symlink())
        self.assertEqual(victim.read_bytes(), before)
        with closing(sqlite3.connect(victim)) as conn:
            self.assertEqual(
                conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(),
                [("sentinel",)],
            )

    def test_message_and_conversation_caps_are_atomic(self):
        self.init()
        with mock.patch.object(store, "MAX_MESSAGES_PER_CONVERSATION", 4):
            self.write(user="u1", answer="a1")
            self.write(user="u2", answer="a2")
            with self.assertRaisesRegex(store.StoreError, "conversation_message_limit"):
                self.write(user="u3", answer="a3")
        self.assertEqual(len(store.list_messages(self.db, "conversation-1")), 4)
        with mock.patch.object(store, "MAX_MESSAGES_PER_CONVERSATION", 3):
            with self.assertRaisesRegex(store.StoreError, "conversation_message_limit"):
                store.list_messages(self.db, "conversation-1")

        with (mock.patch.object(store, "MAX_MESSAGES_PER_CONVERSATION", 100),
              mock.patch.object(store, "MAX_TOTAL_MESSAGES", 6),
              mock.patch.object(store, "MAX_CONVERSATIONS", 2)):
            self.write(conversation_id="second")
            with self.assertRaisesRegex(store.StoreError, "total_message_limit"):
                self.write(conversation_id="third")
        self.assertEqual(store.list_messages(self.db, "third"), [])
        with (mock.patch.object(store, "MAX_MESSAGES_PER_CONVERSATION", 100),
              mock.patch.object(store, "MAX_TOTAL_MESSAGES", 100),
              mock.patch.object(store, "MAX_CONVERSATIONS", 2)):
            with self.assertRaisesRegex(store.StoreError, "conversation_count_limit"):
                self.write(conversation_id="third")
        self.assertEqual(store.list_messages(self.db, "third"), [])


    @unittest.skipUnless(os.name == "posix" and Path("/proc/self/fd").is_dir(),
                         "Linux descriptor-relative SQLite initialization")
    def test_ancestor_substitution_is_rejected_without_retargeting_database(self):
        """Rename the parent after validation, before fd-relative publication."""
        outer = self.root / "safe-outer"
        outer.mkdir(mode=0o700)
        parent = outer / "private"
        parent.mkdir(mode=0o700)
        target = parent / "db.sqlite"
        replacement = outer / "replacement"
        replacement.mkdir(mode=0o700)
        check = store._parent_unchanged

        def swap_parent(path, parent_fd):
            parent.rename(outer / "old-private")
            replacement.rename(parent)
            return check(path, parent_fd)

        with mock.patch.object(store, "_parent_unchanged", side_effect=swap_parent):
            with self.assertRaisesRegex(store.StoreError, "parent_directory_changed"):
                store.initialize(target, enable_persistence=True)
        self.assertFalse(target.exists())
        self.assertEqual(list((outer / "old-private").iterdir()), [])
        self.assertEqual(list(parent.iterdir()), [])

    @unittest.skipUnless(os.name == "posix", "POSIX ancestor modes")
    def test_unsafe_nonsticky_writable_ancestor_rejected(self):
        ancestor = self.root / "unsafe-ancestor"
        ancestor.mkdir(mode=0o700)
        ancestor.chmod(0o777)
        child = ancestor / "private"
        child.mkdir(mode=0o700)
        try:
            with self.assertRaisesRegex(store.StoreError, "unsafe_writable_ancestor"):
                store.initialize(child / "test.db", enable_persistence=True)
        finally:
            ancestor.chmod(0o700)
        self.assertFalse((child / "test.db").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX file permissions")
    def test_existing_world_writable_database_rejected(self):
        self.init()
        self.db.chmod(0o666)
        with self.assertRaisesRegex(store.StoreError, "unsafe_file_permissions"):
            store.open_store(self.db, enable_persistence=True)
        with self.assertRaisesRegex(store.StoreError, "unsafe_file_permissions"):
            store.list_messages(self.db, "s")
        self.db.chmod(0o600)

    @unittest.skipUnless(os.name == "posix", "POSIX file permissions")
    def test_world_writable_backup_or_manifest_rejected(self):
        self.init()
        backup, manifest = store.backup_database(
            self.db, self.root / "snapshot.db", authorize_backup=True
        )
        for item in (backup, manifest):
            item.chmod(0o666)
            with self.assertRaisesRegex(store.StoreError, "unsafe_file_permissions"):
                store.verify_backup(backup, manifest)
            item.chmod(0o600)

    @unittest.skipUnless(os.name == "posix", "POSIX hardlink contract")
    def test_existing_database_and_manifest_extra_hardlinks_rejected(self):
        self.init()
        alias = self.root / "database-alias.db"
        os.link(self.db, alias)
        try:
            with self.assertRaisesRegex(store.StoreError, "unsafe_file_link_count"):
                store.list_messages(self.db, "s")
        finally:
            alias.unlink()
        backup, manifest = store.backup_database(
            self.db, self.root / "snapshot.db", authorize_backup=True
        )
        alias = self.root / "manifest-alias"
        os.link(manifest, alias)
        try:
            with self.assertRaisesRegex(store.StoreError, "unsafe_file_link_count"):
                store.verify_backup(backup, manifest)
        finally:
            alias.unlink()

    def test_schema_and_atomic_roundtrip(self):
        self.init()
        self.assertTrue(self.db.exists())
        if os.name == "posix":
            self.assertEqual(self.db.stat().st_mode & 0o077, 0)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        self.write()
        self.write(user="17*19",answer="323",classification="verified_exact")
        self.assertEqual(store.list_messages(self.db,"conversation-1"), [
            ("user","Hello","user"),("assistant","Hi","model_generated_unverified"),
            ("user","17*19","user"),("assistant","323","verified_exact"),
        ])
        self.assertEqual(store.delete_conversation(self.db,"conversation-1",
                               enable_persistence=True), 1)
        self.assertEqual(store.list_messages(self.db,"conversation-1"), [])
        self.assertEqual(store.delete_conversation(self.db,"conversation-1",
                               enable_persistence=True), 0)

    def test_limits_and_transaction_rejection(self):
        self.init()
        with self.assertRaisesRegex(store.StoreError, "text_limit"):
            self.write(user="x" * 8193)
        self.assertEqual(store.list_messages(self.db, "conversation-1"), [])
        with self.assertRaisesRegex(store.StoreError, "text_limit"):
            self.write(answer="a" * 65537)
        with self.assertRaisesRegex(store.StoreError, "invalid_classification"):
            self.write(classification="verified_by_teacher")
        self.assertEqual(store.list_messages(self.db, "conversation-1"), [])
        injected="x'); DROP TABLE messages;--"
        self.write(conversation_id=injected)
        self.assertEqual(len(store.list_messages(self.db,injected)), 2)

    def test_live_wal_backup_verify_and_fresh_restore(self):
        self.init()
        self.write(user="first",answer="verified",classification="verified_exact")
        writer = sqlite3.connect(self.db, isolation_level=None)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO conversations (conversation_id, created_utc, updated_utc, consent) "
            "VALUES (?, ?, ?, 1)",
            ("uncommitted", store.now_utc(), store.now_utc())
        )
        try:
            target = self.root / "snapshot.db"
            result, manifest = store.backup_database(
                self.db, target, authorize_backup=True
            )
            self.assertEqual(result, target)
            meta = store.verify_backup(result, manifest)
            self.assertEqual(meta["user_version"], 1)
            self.assertEqual(len(meta["sha256"]), 64)
        finally:
            writer.execute("ROLLBACK")
            writer.close()
        self.assertFalse((self.root/"snapshot.db-wal").exists())
        self.assertFalse((self.root/"snapshot.db-shm").exists())
        restored = self.root / "restored.db"
        store.restore_backup(result, manifest, restored, authorize_restore=True)
        self.assertEqual(store.list_messages(restored,"conversation-1"),
                         store.list_messages(self.db,"conversation-1"))
        with sqlite3.connect(restored) as restored_conn:
            self.assertIsNone(restored_conn.execute(
                "SELECT conversation_id FROM conversations WHERE conversation_id='uncommitted'"
            ).fetchone())
        self.write(user="after",answer="another")
        self.assertEqual(len(store.list_messages(restored,"conversation-1")), 2)
        with self.assertRaisesRegex(store.StoreError, "restore_destination_exists"):
            store.restore_backup(result, manifest, restored, authorize_restore=True)
        with self.assertRaisesRegex(store.StoreError, "backup_destination_exists"):
            store.backup_database(self.db, target, authorize_backup=True)

    def test_corrupt_manifest_backup_and_symlinks_fail_closed(self):
        self.init()
        self.write()
        target, manifest = store.backup_database(
            self.db, self.root/"snapshot.db",authorize_backup=True
        )
        raw = json.loads(manifest.read_text())
        raw["sha256"] = "0"*64
        manifest.write_text(json.dumps(raw))
        with self.assertRaisesRegex(store.StoreError, "backup_identity_mismatch"):
            store.verify_backup(target, manifest)
        with self.assertRaises(store.StoreError):
            store.restore_backup(target,manifest,self.root/"dest.db",
                                 authorize_restore=True)
        self.assertFalse((self.root/"dest.db").exists())
        link = self.root/"linked.db"
        link.symlink_to(self.db)
        with self.assertRaisesRegex(store.StoreError, "symlink_path"):
            store.list_messages(link,"conversation-1")


    def test_read_of_closed_wal_preserves_rows_and_main_file(self):
        self.init()
        self.assertEqual({p.name for p in self.root.iterdir()}, {"trinite.db"})
        before = store.sha256_path(self.db)
        self.assertEqual(store.list_messages(self.db, "s"), [])
        self.assertEqual(store.sha256_path(self.db), before)
        # A mutable WAL database needs SQLite locking and may create
        # -wal/-shm coordination files even for a logically read-only query.
        self.assertTrue(self.db.is_file())

    def test_live_wal_read_retains_uncheckpointed_committed_rows(self):
        self.init()
        writer = sqlite3.connect(self.db, isolation_level=None)
        try:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, 1)",
                ("live", store.now_utc(), store.now_utc())
            )
            writer.execute(
                "INSERT INTO messages (conversation_id, role, content, classification, created_utc) "
                "VALUES (?, ?, ?, ?, ?)",
                ("live", "user", "in WAL", "user", store.now_utc())
            )
            self.assertTrue(self.db.with_name("trinite.db-wal").exists())
            self.assertTrue(self.db.with_name("trinite.db-shm").exists())
            self.assertEqual(store.list_messages(self.db, "live"),
                             [("user", "in WAL", "user")])
            self.db.with_name("trinite.db-shm").unlink()
            with self.assertRaisesRegex(store.StoreError, "incomplete_wal_sidecars"):
                store.list_messages(self.db, "live")
            self.assertFalse(self.db.with_name("trinite.db-shm").exists())
        finally:
            writer.close()

    def test_restore_rejects_stale_wal_after_main_unlinked(self):
        self.init()
        self.write(user="source", answer="source answer")
        backup, manifest = store.backup_database(
            self.db, self.root / "snapshot.db", authorize_backup=True
        )
        destination = self.root / "restore.db"
        store.initialize(destination, enable_persistence=True)
        stale = sqlite3.connect(destination, isolation_level=None)
        try:
            stale.execute("PRAGMA journal_mode=WAL")
            stale.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, 1)",
                ("stale", store.now_utc(), store.now_utc())
            )
            stale.execute("INSERT INTO messages "
                          "(conversation_id, role, content, classification, created_utc) "
                          "VALUES (?, ?, ?, ?, ?)",
                          ("stale", "user", "stale WAL", "user", store.now_utc()))
            self.assertTrue(destination.with_name("restore.db-wal").exists())
            self.assertTrue(destination.with_name("restore.db-shm").exists())
            destination.unlink()
            with self.assertRaisesRegex(store.StoreError, "restore_destination_sidecars_exist"):
                store.restore_backup(backup, manifest, destination,
                                     authorize_restore=True)
            self.assertFalse(destination.exists())
            self.assertTrue(destination.with_name("restore.db-wal").exists())
        finally:
            stale.close()

    def test_restore_rejects_each_orphan_sidecar(self):
        self.init()
        backup, manifest = store.backup_database(
            self.db, self.root / "snapshot.db", authorize_backup=True
        )
        target = self.root / "restore.db"
        for extension in ("-wal", "-shm", "-journal"):
            with self.subTest(extension=extension):
                sidecar = self.root / ("restore.db" + extension)
                sidecar.write_bytes(b"orphan")
                with self.assertRaisesRegex(store.StoreError, "restore_destination_sidecars_exist"):
                    store.restore_backup(backup, manifest, target,
                                         authorize_restore=True)
                self.assertFalse(target.exists())
                sidecar.unlink()

    def test_renamed_column_and_missing_index_fail_schema_gate(self):
        self.init()
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("ALTER TABLE messages RENAME COLUMN content TO payload")
            conn.commit()
        with self.assertRaisesRegex(store.StoreError, "schema_structure_mismatch"):
            store.list_messages(self.db, "s")
        with self.assertRaisesRegex(store.StoreError, "schema_structure_mismatch"):
            store.backup_database(self.db, self.root / "bad-snapshot.db",
                                  authorize_backup=True)
        self.assertFalse((self.root / "bad-snapshot.db").exists())
        # A self-updated manifest must not turn structurally incompatible
        # bytes into an acceptable restored database.
        second = self.root / "second.db"
        store.initialize(second, enable_persistence=True)
        with closing(sqlite3.connect(second)) as conn:
            conn.execute("DROP INDEX messages_by_conversation")
            conn.commit()
        with self.assertRaisesRegex(store.StoreError, "schema_structure_mismatch"):
            store.list_messages(second, "s")

    def test_corrupted_backup_schema_rejected_even_with_matching_hash(self):
        self.init()
        backup, manifest = store.backup_database(
            self.db, self.root / "snapshot.db", authorize_backup=True
        )
        with closing(sqlite3.connect(backup)) as conn:
            conn.execute("ALTER TABLE messages RENAME COLUMN content TO payload")
            conn.commit()
        forged = json.loads(manifest.read_text("utf-8"))
        forged["bytes"] = backup.stat().st_size
        forged["sha256"] = store.sha256_path(backup)
        manifest.write_bytes(store.json_bytes(forged))
        with self.assertRaisesRegex(store.StoreError, "schema_structure_mismatch"):
            store.verify_backup(backup, manifest)
        with self.assertRaisesRegex(store.StoreError, "schema_structure_mismatch"):
            store.restore_backup(backup, manifest, self.root / "fresh.db",
                                 authorize_restore=True)
        self.assertFalse((self.root / "fresh.db").exists())


    def test_mutable_online_backup_never_mixes_checkpoint_versions(self):
        """Reproduce the immutable backup's 500-row mixed-page defect."""
        self.init()
        stamp = store.now_utc()
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, 1)",
                ("bulk", stamp, stamp)
            )
            conn.executemany(
                "INSERT INTO messages (conversation_id, role, content, classification, created_utc) "
                "VALUES (?, ?, ?, ?, ?)",
                [("bulk", "assistant", "a" * 4000,
                  "model_generated_unverified", stamp) for _ in range(500)]
            )
            conn.commit()
        self.assertFalse(self.db.with_name("trinite.db-wal").exists())

        source_path = self.db
        connect = store.sqlite_connect
        writer_ran = []

        class InterleavedSource:
            def __init__(self, original):
                self.original = original

            def __getattr__(self, name):
                return getattr(self.original, name)

            def backup(self, target, *, pages, progress, sleep):
                def after_step(status, remaining, total):
                    if not writer_ran and 0 < remaining < total:
                        with closing(sqlite3.connect(source_path, timeout=3.0)) as writer:
                            writer.execute(
                                "UPDATE messages SET content=? WHERE conversation_id='bulk'",
                                ("b" * 4000,)
                            )
                            writer.commit()
                            writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        writer_ran.append(True)
                    progress(status, remaining, total)
                return self.original.backup(
                    target, pages=pages, progress=after_step, sleep=sleep
                )

        def inject_progress(path, *, readonly, immutable_snapshot=False):
            connection = connect(
                path, readonly=readonly, immutable_snapshot=immutable_snapshot
            )
            if path == source_path and readonly:
                self.assertFalse(immutable_snapshot)
                return InterleavedSource(connection)
            return connection

        with mock.patch.object(store, "sqlite_connect", side_effect=inject_progress):
            backup, manifest = store.backup_database(
                self.db, self.root / "concurrent.db", authorize_backup=True
            )
        self.assertTrue(writer_ran, "writer must run between backup steps")
        store.verify_backup(backup, manifest)
        with closing(sqlite3.connect(backup)) as snapshot:
            groups = snapshot.execute(
                "SELECT substr(content,1,1),count(*) FROM messages "
                "GROUP BY substr(content,1,1)"
            ).fetchall()
        self.assertEqual(sum(count for _, count in groups), 500)
        self.assertEqual(len(groups), 1, "mixed old/new pages violate snapshot consistency")
        self.assertIn(groups[0][0], ("a", "b"))

    def test_immutable_mode_only_for_standalone_snapshots(self):
        self.init()
        with self.assertRaisesRegex(store.StoreError, "immutable_snapshot_requires_readonly"):
            store.sqlite_connect(self.db, readonly=False, immutable_snapshot=True)
        backup, manifest = store.backup_database(
            self.db, self.root / "snapshot.db", authorize_backup=True
        )
        store.verify_backup(backup, manifest)
        with closing(store.sqlite_connect(backup, readonly=True,
                                          immutable_snapshot=True)) as snapshot:
            store.check_schema(snapshot)
        sidecar = self.root / "snapshot.db-wal"
        sidecar.write_bytes(b"unexpected")
        with self.assertRaisesRegex(store.StoreError, "snapshot_sidecars_exist"):
            store.sqlite_connect(backup, readonly=True, immutable_snapshot=True)
        sidecar.unlink()

    def test_sqliteXerase_trigger_rejected_before_backup(self):
        self.init()
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute(
                "CREATE TRIGGER sqliteXerase AFTER INSERT ON messages "
                "BEGIN DELETE FROM messages; END"
            )
            conn.commit()
        with self.assertRaisesRegex(store.StoreError, "schema_structure_mismatch"):
            store.list_messages(self.db, "s")
        with self.assertRaisesRegex(store.StoreError, "schema_structure_mismatch"):
            store.backup_database(self.db, self.root / "trigger.db",
                                  authorize_backup=True)
        self.assertFalse((self.root / "trigger.db").exists())

    def test_trigger_with_literal_prefix_bypass_in_forged_backup(self):
        self.init()
        backup, manifest = store.backup_database(
            self.db, self.root / "snapshot.db", authorize_backup=True
        )
        with closing(sqlite3.connect(backup)) as conn:
            conn.execute(
                "CREATE TRIGGER sqliteXerase AFTER INSERT ON messages "
                "BEGIN DELETE FROM messages; END"
            )
            conn.commit()
        forged = json.loads(manifest.read_text("utf-8"))
        forged["bytes"] = backup.stat().st_size
        forged["sha256"] = store.sha256_path(backup)
        manifest.write_bytes(store.json_bytes(forged))
        with self.assertRaisesRegex(store.StoreError, "schema_structure_mismatch"):
            store.verify_backup(backup, manifest)
        with self.assertRaisesRegex(store.StoreError, "schema_structure_mismatch"):
            store.restore_backup(backup, manifest, self.root / "restored.db",
                                 authorize_restore=True)
        self.assertFalse((self.root / "restored.db").exists())

    def test_version_drift_rejected(self):
        self.init()
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("PRAGMA user_version=2")
        with self.assertRaisesRegex(store.StoreError, "schema_version_mismatch"):
            store.list_messages(self.db,"conversation-1")


if __name__=="__main__":
    unittest.main()
