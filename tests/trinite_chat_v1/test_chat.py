"""TRI-CHAT-001 software-double tests. No model/runtime acquisition."""
import hashlib
import http.client
import socket
import time
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.server import HTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "scripts" / "run_trinite_chat_v1.py"
spec = importlib.util.spec_from_file_location("chat_v1", SRC)
chat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat)


def sample(message="17 * 19", history=None):
    return {"schema": chat.SCHEMA, "message": message,
            "history": history if history is not None else []}


class CoreContracts(unittest.TestCase):
    def test_strict_json(self):
        self.assertEqual(chat.parse_request(chat.canonical(sample())), sample())
        for bad in (
            b'{"schema":"trinite.chat.request.v1","message":"A","history":[],"message":"B"}',
            b'{"schema":"trinite.chat.request.v1","message":"A","history":[],"extra":0}',
            b'{"schema":"trinite.chat.request.v1","message":"A","history":[],"x":NaN}',
            b'{"schema":"trinite.chat.request.v1","message":"","history":[]}',
            b'{"schema":"trinite.chat.request.v1","message":"A","history":[{"role":"assistant","content":"x"}]}',
            b'\xff',
            b'{"schema":"trinite.chat.request.v1","message":"Hi","history":' +
                b'[' * 1100 + b']' * 1100 + b'}',
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(chat.Invalid):
                    chat.parse_request(bad)
        with self.assertRaises(chat.Invalid):
            chat.parse_request(chat.canonical(sample("a" * (chat.MAX_MESSAGE + 1))))

    def test_exact_arithmetic_and_independent_oracle(self):
        cfg = chat.Config()
        for input_text, expected in (
            ("17 * 19", "323"), ("what is 6 x 7?", "42"),
            ("gf3(2 + 2)", "1"), ("gf(3)(-2 * 2)", "2"),
            ("calculate -17 + 19", "2"),
        ):
            with self.subTest(input_text=input_text):
                answer = chat.result_for(sample(input_text), cfg)
                self.assertEqual(answer["status"], "verified_exact")
                self.assertEqual(answer["exact"]["value"], expected)
        self.assertEqual(chat.choose_route("What is your name?")[0], "chat")
        self.assertEqual(chat.result_for(sample("gf3(eval(1))"), cfg)["status"], "blocked")
        self.assertEqual(chat.result_for(sample("calculate 3 / 0"), cfg)["status"], "blocked")

    def test_oracle_disagreement_and_prose_never_verified(self):
        cfg = chat.Config()
        x = chat.result_for(sample(), cfg, verifier=lambda _: -1)
        self.assertEqual(x["status"], "failed")
        self.assertIsNone(x["exact"])
        self.assertEqual(x["reason"], "independent_oracle_mismatch")
        y = chat.result_for(sample("hello"), cfg,
                            adapter=lambda *_: '<script>alert(1)</script>')
        self.assertEqual(y["status"], "model_generated_unverified")
        self.assertIsNone(y["exact"])
        self.assertNotIn("hello", json.dumps(y["receipt"]))
        self.assertEqual(chat.result_for(sample("hello"), cfg)["reason"],
                         "teacher_not_configured")

    def test_explicit_local_model_identity(self):
        with self.assertRaises(chat.Invalid):
            chat.Config(model_port=1234)
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "synthetic-model.gguf"
            model.write_bytes(b"synthetic local software double")
            sha = hashlib.sha256(model.read_bytes()).hexdigest()
            opts = dict(model_port=1234, model_id="fake", model_file=str(model),
                        model_sha256=sha, runtime_version="fake-v1",
                        chat_template="synthetic-template")
            with self.assertRaises(chat.Invalid):
                chat.Config(**opts)
            with self.assertRaises(chat.Invalid):
                chat.Config(**{**opts, "model_sha256": "0" * 64},
                            authorize_local_teacher=True)
            self.assertTrue(chat.Config(**opts, authorize_local_teacher=True).active)


class HTTPContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0),
                                chat.http_handler(chat.Config(),
                                                  adapter=lambda *_: "synthetic answer"))
        cls.port = cls.server.server_port
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=2)

    def request(self, method, path, data=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        hdr = {"Host": "127.0.0.1:" + str(self.port)}
        hdr.update(headers or {})
        conn.request(method, path, body=data, headers=hdr)
        response = conn.getresponse()
        out = (response.status, dict(response.getheaders()), response.read())
        conn.close()
        return out

    def test_assets_and_csp(self):
        for asset in ("/", "/style.css", "/app.js"):
            status, headers, body = self.request("GET", asset)
            self.assertEqual(status, 200)
            self.assertTrue(body)
            self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        self.assertEqual(self.request("GET", "/../../AGENTS.md")[0], 404)
        self.assertEqual(self.request("GET", "/style.css?bad")[0], 404)

    def test_local_json_route(self):
        status, _, content = self.request("POST", "/api/chat",
                                          chat.canonical(sample("gf3(2 + 2)")),
                                          {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        result = json.loads(content)
        self.assertEqual(result["status"], "verified_exact")
        self.assertEqual(result["exact"]["value"], "1")

    def test_transport_attack_cases(self):
        body = chat.canonical(sample("hello"))
        cases = [
            ("/api/chat", body, {"Content-Type":"text/plain"}, 415),
            ("/api/chat", body, {"Content-Type":"application/json",
                                 "Origin":"https://evil.example"}, 403),
            ("/api/chat", body, {"Content-Type":"application/json",
                                 "Host":"evil.example"}, 403),
            ("/api/chat", b'{"message":', {"Content-Type":"application/json"}, 400),
            ("/no-route", body, {"Content-Type":"application/json"}, 404),
            ("/api/chat", b"x" * (chat.MAX_REQUEST + 1),
             {"Content-Type":"application/json"}, 413)
        ]
        for path, data, headers, code in cases:
            with self.subTest(code=code):
                self.assertEqual(self.request("POST", path, data, headers)[0], code)
        self.assertEqual(self.request("PUT", "/api/chat")[0], 405)

    def test_unfinished_header_does_not_starve_server(self):
        # The single synchronous server must expire an accepted connection
        # even before BaseHTTPRequestHandler parses its request line.
        idle = socket.create_connection(("127.0.0.1", self.port), timeout=2)
        try:
            start = time.monotonic()
            status, _, _ = self.request("GET", "/")
            elapsed = time.monotonic() - start
            self.assertEqual(status, 200)
            self.assertLess(elapsed, chat.HEADER_DEADLINE + 1.5)
        finally:
            idle.close()

    def test_deep_json_returns_structured_400(self):
        raw = (b'{"schema":"trinite.chat.request.v1","message":"Hi","history":' +
               b'[' * 1100 + b']' * 1100 + b'}')
        status, _, data = self.request(
            "POST", "/api/chat", raw, {"Content-Type": "application/json"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(data)["reason"], "malformed_json")

    def test_escape_only_webui(self):
        js = (ROOT/"scripts/trinite_chat_v1_ui/app.js").read_text()
        self.assertIn("body.textContent = content", js)
        self.assertNotIn("innerHTML", js)
        self.assertNotIn("eval(", js)
        self.assertIn("TextEncoder", js)
        self.assertIn("MAX_REQUEST_BYTES = 16384", js)
        self.assertIn("byteLength(answer) <= MAX_MESSAGE_BYTES", js)
        self.assertIn("pruneHistory(text)", js)
        self.assertIn("history.splice(0, 2)", js)


class AdversarialTeacherDeadline(unittest.TestCase):
    """A byte-drip server must not extend an absolute teacher deadline."""

    def verify_trickle(self, mode):
        class SlowTeacher(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                size = int(self.headers["Content-Length"])
                self.rfile.read(size)
                body = json.dumps({"model": "fake", "choices": [
                    {"finish_reason": "stop",
                     "message": {"role": "assistant", "content": "ok"}}
                ]}).encode()
                try:
                    if mode == "headers":
                        # Header trickle begins before getresponse() completes.
                        for byte in b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n":
                            self.connection.sendall(bytes([byte]))
                            time.sleep(0.04)
                    else:
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        for byte in body:
                            self.connection.sendall(bytes([byte]))
                            time.sleep(0.04)
                except (OSError, BrokenPipeError):
                    pass

        with ThreadingHTTPServer(("127.0.0.1", 0), SlowTeacher) as server:
            server.daemon_threads = True
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            saved = chat.TEACHER_DEADLINE
            chat.TEACHER_DEADLINE = 0.25
            try:
                config = SimpleNamespace(active=True, model_id="fake", port=server.server_port)
                started = time.monotonic()
                with self.assertRaises(chat.TeacherFault) as captured:
                    chat.teacher_chat(config, "hello", [])
                elapsed = time.monotonic() - started
                self.assertEqual(captured.exception.code, "teacher_timeout")
                self.assertLess(elapsed, 0.9)
            finally:
                chat.TEACHER_DEADLINE = saved
                server.shutdown()
                worker.join(timeout=2)

    def test_slow_response_headers(self):
        self.verify_trickle("headers")

    def test_slow_response_body(self):
        self.verify_trickle("body")


if __name__ == "__main__":
    unittest.main()
