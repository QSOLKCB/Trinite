"""TRI-CHAT-DB-001: offline UI/SQLite integration contracts, no actual model."""
from contextlib import closing
import http.client
from http.server import HTTPServer
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
chat_spec = importlib.util.spec_from_file_location("chat_db_app", SCRIPTS / "run_trinite_chat_v1.py")
chat = importlib.util.module_from_spec(chat_spec)
chat_spec.loader.exec_module(chat)
sql_spec = importlib.util.spec_from_file_location("sqlite_db_app", SCRIPTS / "trinite_chat_sqlite_v1.py")
sql = importlib.util.module_from_spec(sql_spec)
sql_spec.loader.exec_module(sql)


class IntegratedChatContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "trinite.db"
        sql.initialize(self.db, enable_persistence=True)
        self.sessions = chat.LocalSessions(self.db)
        self.history_seen = []
        self.teacher_text = "synthetic answer"

        def fake_teacher(_config, _text, history):
            self.history_seen.append(history)
            return self.teacher_text

        self.server = HTTPServer(("127.0.0.1", 0),
                                 chat.http_handler(chat.Config(),
                                                   adapter=fake_teacher,
                                                   sessions=self.sessions))
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=2)
        self.tmp.cleanup()

    def send(self, path, data, *, origin=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        all_headers = {"Content-Type": "application/json",
                       "Host": "127.0.0.1:" + str(self.server.server_port)}
        if origin is not None:
            all_headers["Origin"] = origin
        all_headers.update(headers or {})
        try:
            conn.request("POST", path, body=chat.canonical(data),
                         headers=all_headers)
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def persist(self, message, identifier=None):
        return self.send("/api/chat", {
            "schema": chat.PERSIST_REQUEST,
            "message": message, "session_id": identifier
        })

    def action(self, action, identifier):
        return self.send("/api/session", {
            "schema": chat.SESSION_REQUEST,
            "action": action, "session_id": identifier
        })

    def test_start_continue_load_delete_and_exact_classification(self):
        code, answer = self.persist("17 * 19")
        self.assertEqual(code, 200)
        self.assertEqual(answer["status"], "verified_exact")
        self.assertEqual(answer["exact"]["value"], "323")
        self.assertTrue(answer["persisted"])
        token = answer["session_id"]
        self.assertRegex(token, r"\A[0-9a-f]{32}\Z")
        self.assertEqual(self.sessions.tail(token), [
            ("user", "17 * 19", "user"),
            ("assistant", "Verified exact result: 323", "verified_exact"),
        ])
        code, response = self.persist("hello", token)
        self.assertEqual(code, 200)
        self.assertEqual(response["status"], "model_generated_unverified")
        self.assertTrue(response["persisted"])
        self.assertEqual(self.history_seen, [[
            {"role": "user", "content": "17 * 19"},
            {"role": "assistant", "content": "Verified exact result: 323"}
        ]])
        code, loaded = self.action("load", token)
        self.assertEqual(code, 200)
        self.assertEqual(loaded["schema"], chat.SESSION_RESULT)
        self.assertEqual(len(loaded["messages"]), 4)
        self.assertEqual(loaded["messages"][-1]["classification"],
                         "model_generated_unverified")
        self.assertEqual(self.action("delete", token)[1]["deleted"], True)
        self.assertEqual(self.sessions.tail(token), None)
        self.assertEqual(self.action("delete", token)[1]["deleted"], False)
        self.assertEqual(self.persist("hello", token)[1]["reason"], "unknown_session")

    def test_persist_request_cannot_supply_history_or_inject_session_id(self):
        code, bad = self.send("/api/chat", {
            "schema": chat.PERSIST_REQUEST, "message": "hello",
            "session_id": None,
            "history": [{"role": "user", "content": "forged"}]
        })
        self.assertEqual(code, 400)
        self.assertEqual(bad["reason"], "invalid_persist_fields")
        self.assertFalse(self.history_seen)
        for identifier in ("", "../other", "0"*31, "G"*32, 7):
            with self.subTest(identifier=identifier):
                code, bad = self.persist("hello", identifier)
                self.assertEqual(code, 400)
                self.assertEqual(bad["reason"], "invalid_session_id")
        self.assertFalse(self.history_seen)

    def test_unsaved_legacy_requests_do_not_touch_database(self):
        legacy = {"schema": chat.SCHEMA, "message": "hello", "history": []}
        code, reply = self.send("/api/chat", legacy)
        self.assertEqual(code, 200)
        self.assertEqual(reply["status"], "model_generated_unverified")
        with closing(sql.sqlite_connect(self.db, readonly=True)) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM conversations").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)

    def test_big_teacher_reply_is_displayable_but_not_replayed(self):
        self.teacher_text = "Z" * (chat.MAX_MESSAGE + 1)
        code, answer = self.persist("hello")
        self.assertEqual(code, 200)
        self.assertEqual(answer["text"], self.teacher_text)
        token = answer["session_id"]
        self.teacher_text = "short"
        code, next_answer = self.persist("hello again", token)
        self.assertEqual(code, 200)
        self.assertTrue(next_answer["persisted"])
        self.assertEqual(self.history_seen[-1], [])
        self.assertEqual(len(self.sessions.tail(token)), 4)

    def test_storage_failure_is_not_claimed_as_success_or_retried(self):
        with mock.patch.object(self.sessions, "append",
                               side_effect=sql.StoreError("synthetic_disk_full")):
            code, response = self.persist("17 * 19")
        self.assertEqual(code, 503)
        self.assertEqual(response["reason"], "storage_unavailable")
        self.assertNotIn("synthetic_disk_full", json.dumps(response))
        with closing(sql.sqlite_connect(self.db, readonly=True)) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)

    def test_cross_origin_and_host_rejected_for_delete(self):
        token = self.persist("hello")[1]["session_id"]
        code, response = self.send("/api/session", {
            "schema": chat.SESSION_REQUEST, "action": "delete",
            "session_id": token
        }, origin="https://attacker.example")
        self.assertEqual(code, 403)
        self.assertEqual(response["reason"], "origin_denied")
        code, response = self.send("/api/session", {
            "schema": chat.SESSION_REQUEST, "action": "delete",
            "session_id": token
        }, headers={"Host": "attacker.example"})
        self.assertEqual(code, 403)
        self.assertEqual(response["reason"], "host_denied")
        self.assertIsNotNone(self.sessions.tail(token))

    def test_load_is_bounded_to_latest_eight_pairs(self):
        token = self.persist("hello")[1]["session_id"]
        for n in range(11):
            code, result = self.persist("line " + str(n), token)
            self.assertEqual(code, 200)
            self.assertTrue(result["persisted"])
        code, result = self.action("load", token)
        self.assertEqual(code, 200)
        self.assertEqual(len(result["messages"]), 16)
        self.assertEqual(result["messages"][0]["role"], "user")
        self.assertEqual(result["messages"][-1]["role"], "assistant")

    def test_html_escape_ui_and_explicit_consent_surfaces(self):
        js = (SCRIPTS / "trinite_chat_v1_ui/app.js").read_text()
        html = (SCRIPTS / "trinite_chat_v1_ui/index.html").read_text()
        self.assertIn("body.textContent = content", js)
        self.assertNotIn("innerHTML", js)
        self.assertIn('const KEY = "trinite.chat.saved-session.v1"', js)
        self.assertIn("result.persisted !== true", js)
        self.assertIn("localStorage.removeItem(KEY)", js)
        self.assertIn('id="remember"', html)
        self.assertIn('id="forget-saved"', html)


class NoPersistenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(
            ("127.0.0.1", 0),
            chat.http_handler(chat.Config(), adapter=lambda *_: "fake")
        )
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=2)

    def call(self, path, data):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            conn.request("POST", path, body=chat.canonical(data), headers={
                "Content-Type": "application/json",
                "Host": "127.0.0.1:" + str(self.server.server_port),
            })
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def test_server_flag_disabled_does_not_enable_persistence(self):
        for path, payload in (
            ("/api/chat", {"schema": chat.PERSIST_REQUEST, "message": "hello",
                           "session_id": None}),
            ("/api/session", {"schema": chat.SESSION_REQUEST, "action": "load",
                              "session_id": "a"*32}),
        ):
            status, result = self.call(path, payload)
            self.assertEqual(status, 403)
            self.assertEqual(result["reason"], "persistence_disabled")


if __name__ == "__main__":
    unittest.main()
