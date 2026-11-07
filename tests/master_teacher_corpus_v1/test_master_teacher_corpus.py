"""Offline MTC-001 capture conformance; no downloaded models or hosted services."""
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "master_teacher_corpus.py"
spec = importlib.util.spec_from_file_location("mtc", SOURCE)
mtc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtc)


class MockOllama(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path != "/api/tags":
            self.send_error(404); return
        data = {"models": [{"name": "mock-text", "digest": "a" * 64},
                           {"name": "mock-vision", "digest": "b" * 64}]}
        self.answer(data)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404); return
        n = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(n))
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0
        if payload["model"] == "mock-vision":
            assert len(payload["messages"][0]["images"]) == 1
        else:
            assert "images" not in payload["messages"][0]
        self.answer({"message": {"content": '{"answer":"0"}'}, "done_reason": "stop", "eval_count": 10})

    def answer(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class MockMalformedOllama(MockOllama):
    def do_POST(self):
        self.answer({"message": {"content": "NOT JSON"}, "done_reason": "stop"})


class MTCTest(unittest.TestCase):
    def test_corpus_task_ledger_and_splits(self):
        rows = mtc.generate_items()
        self.assertEqual(28, len(rows))
        self.assertTrue(all(r["answer"] == mtc.independent_oracle(r) for r in rows))
        self.assertEqual(28, len(set(r["id"] for r in rows)))
        family_splits = {}
        for row in rows:
            self.assertIn(row["split"], ("train", "development", "locked_test"))
            self.assertEqual(family_splits.setdefault(row["family"], row["split"]), row["split"])
            self.assertNotIn("answer", mtc.public_item(row))
        self.assertEqual({"0", "1"}, {r["answer"] for r in rows if r["family"] == "context-0"})
        self.assertEqual(b"\x89PNG\r\n\x1a\n", mtc.png_grid([0])[:8])

    def test_strict_model_json(self):
        self.assertEqual("1", mtc.strict_answer('{"answer":"1"}'))
        for bad in ['{"answer":1}', '{"answer":"1","answer":"2"}',
                    'text {"answer":"1"}', '{"answer":"1","reason":"foo"}',
                    '{"answer":true}', '["1"]']:
            self.assertIsNone(mtc.strict_answer(bad))
        for bad_url in ["https://127.0.0.1:11434", "http://192.168.0.1:11434", "http://127.0.0.1:11434/remote", "http://127.0.0.1:11434@evil.com"]:
            with self.assertRaises(ValueError):
                mtc.local_url(bad_url)

    def test_malformed_teacher_content_is_retained_but_not_admitted(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockMalformedOllama)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                url = f"http://127.0.0.1:{server.server_port}"
                mtc.main(["plan", "--url", url, "--formal-model", "mock-text", "--vision-model", "mock-vision", "--out", str(root / "plan")])
                request = root / "plan" / "request.json"
                request_identity = mtc.digest(request.read_bytes())
                mtc.main(["capture", "--request", str(request), "--request-identity", request_identity,
                          "--authorize-local-capture", "--out", str(root / "run")])
                rows = [json.loads(line) for line in (root / "run" / "attempts.jsonl").read_bytes().splitlines()]
                self.assertEqual(28, len(rows))
                self.assertTrue(all(x["status"] == "completed" and x["parsed_answer"] is None and not x["oracle_match"] for x in rows))
                self.assertTrue(all(x["raw_response_base64"] for x in rows))
                self.assertEqual(b"", (root / "run" / "train-candidates.jsonl").read_bytes())
                manifest_identity = mtc.digest((root / "run" / "manifest.json").read_bytes())
                mtc.main(["verify", "--run", str(root / "run"), "--manifest-identity", manifest_identity])
                bad = json.loads(request.read_bytes())
                bad["contract_closure"]["docs/MASTER-TEACHER-CORPUS-V1.md"] = "sha256:" + "0" * 64
                with self.assertRaises(ValueError):
                    mtc.check_request(bad)
        finally:
            server.shutdown(); server.server_close(); worker.join()

    def test_fake_local_capture_and_manifest_tamper(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockOllama)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan = root / "plan"
                run = root / "run"
                url = f"http://127.0.0.1:{server.server_port}"
                mtc.main(["plan", "--url", url, "--formal-model", "mock-text", "--vision-model", "mock-vision", "--out", str(plan)])
                identity = mtc.digest((plan / "request.json").read_bytes())
                mtc.main(["capture", "--request", str(plan / "request.json"), "--request-identity", identity,
                          "--authorize-local-capture", "--out", str(run)])
                manifest = mtc.digest((run / "manifest.json").read_bytes())
                mtc.main(["verify", "--run", str(run), "--manifest-identity", manifest])
                raw = (run / "attempts.jsonl").read_text()
                (run / "attempts.jsonl").write_text(raw.replace('"model":"mock-vision"', '"model":"spoofed"', 1))
                with self.assertRaises(ValueError):
                    mtc.verify(type("Args", (), {"run": run, "manifest_identity": manifest})())
        finally:
            server.shutdown(); server.server_close(); worker.join()


if __name__ == "__main__":
    unittest.main()
