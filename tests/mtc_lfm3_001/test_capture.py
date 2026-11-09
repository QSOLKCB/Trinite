"""MTC-LFM3-002 offline 1-5 teacher conformance, never real inference."""
import base64
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "scripts" / "mtc_lfm3_001.py"
spec = importlib.util.spec_from_file_location("mtc_lfm3", SRC)
mtc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtc)
OWNER = ROOT / "scripts" / "mtc_lfm3_owner_job.py"
owner_spec = importlib.util.spec_from_file_location("owner_launcher", OWNER)
owner = importlib.util.module_from_spec(owner_spec)
owner_spec.loader.exec_module(owner)
MODELS = ["lfm3-mock-" + str(i) + ":test" for i in range(5)]
DIGESTS = {m: f"{i + 1:064x}" for i, m in enumerate(MODELS)}
ANSWERS = {task["prompt"]: mtc.oracle(task) for task in mtc.tasks()}


class FakeOllama(BaseHTTPRequestHandler):
    counts = 0
    fail = False
    wrong_model = False
    duplicate_digest = False

    def log_message(self, *_):
        pass

    def respond(self, data, status=200):
        body = mtc.canonical(data)
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self.respond({"models": [
                {"name": m, "digest": DIGESTS[MODELS[0]] if self.duplicate_digest and m == MODELS[1] else DIGESTS[m]}
                for m in MODELS
            ]})
        elif self.path == "/api/version":
            self.respond({"version": "fake-local"})
        else:
            self.respond({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/chat":
            self.respond({"error": "not found"}, 404)
            return
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        data = json.loads(raw)
        assert data["stream"] is False
        assert data["options"] == mtc.SETTINGS
        assert data["model"] in MODELS
        assert len(data["messages"]) == 1
        assert "images" not in data["messages"][0]
        FakeOllama.counts += 1
        if FakeOllama.fail:
            self.respond({"error": "synthetic_failure"}, 503)
        else:
            expected = ANSWERS[data["messages"][0]["content"]]
            self.respond({
                "model": "unexpected-other-model" if FakeOllama.wrong_model else data["model"],
                "done": True,
                "message": {"role": "assistant",
                            "content": json.dumps({"answer": expected})},
                "eval_count": 12
            })


class CaptureContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.old_port = mtc.URL_PORT
        mtc.URL_PORT = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        mtc.URL_PORT = cls.old_port
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=2)

    def setUp(self):
        FakeOllama.counts = 0
        FakeOllama.fail = False
        FakeOllama.wrong_model = False
        FakeOllama.duplicate_digest = False
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.plan_dir = self.root / "plan"
        self.run_dir = self.root / "run"

    def tearDown(self):
        self.tmp.cleanup()

    def freeze(self, names=None):
        selected = MODELS if names is None else names
        mtc.plan_command(type("A", (), {"models": ",".join(selected),
                                        "out": self.plan_dir})())
        return mtc.sha((self.plan_dir / "plan.json").read_bytes())

    def capture(self, identity, resume=False, authorize=True):
        return mtc.capture_command(type("A", (), {
            "plan": self.plan_dir / "plan.json",
            "request_identity": identity,
            "out": self.run_dir,
            "resume": resume,
            "authorize_local_capture": authorize
        })())

    def verify(self):
        identity = mtc.sha((self.run_dir / "manifest.json").read_bytes())
        mtc.verify_command(type("A", (), {"run": self.run_dir,
                                         "manifest_identity": identity})())

    def test_frozen_five_model_capture_and_independent_verify(self):
        request_id = self.freeze()
        self.assertEqual(len(mtc.tasks()), 6)
        plan = json.loads((self.plan_dir / "plan.json").read_bytes())
        self.assertEqual(len(plan["models"]), 5)
        self.assertEqual(plan["status"], "QUARANTINED_NOT_ADMITTED")
        self.assertEqual(plan["admission"], False)
        self.assertEqual(plan["source_closure"], mtc.closure())
        self.capture(request_id)
        self.assertEqual(FakeOllama.counts, 30)
        rows = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        self.assertEqual(len(rows), 30)
        self.assertEqual(sum(row["oracle_match"] for row in rows), 30)
        self.assertEqual(mtc.summary(rows, mtc.strict_json((self.plan_dir / "plan.json").read_bytes()))["admitted_rows"], 0)
        self.assertEqual({p.name for p in self.run_dir.iterdir()},
                         {"plan.json", "attempts.jsonl", "summary.json", "manifest.json"})
        self.verify()
        self.assertEqual(len({row["model"] for row in rows}), 5)
        for directory in (self.plan_dir, self.run_dir):
            self.assertEqual(directory.stat().st_mode & 0o077, 0)
            self.assertTrue(all((f.stat().st_mode & 0o077) == 0
                                for f in directory.iterdir()))

    def test_one_teacher_plan_capture_verify_and_principal_selection(self):
        identity = self.freeze(MODELS[:1])
        plan = mtc.strict_json((self.plan_dir / "plan.json").read_bytes())
        self.assertEqual(plan["count"], 6)
        self.assertEqual(plan["schema"], mtc.SCHEMA)
        self.assertEqual([model["name"] for model in plan["models"]], MODELS[:1])
        self.capture(identity)
        self.assertEqual(FakeOllama.counts, 6)
        rows = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        self.assertEqual(len(rows), 6)
        self.assertEqual(mtc.summary(rows, plan)["planned"], 6)
        self.assertEqual(mtc.summary(rows, plan)["admitted_rows"], 0)
        self.verify()
        # Principal must still find exactly four frozen TRAIN examples when
        # the five-model baseline has been replaced with one selected teacher.
        spec = importlib.util.spec_from_file_location(
            "principal_single_teacher",
            ROOT / "scripts" / "mtc_principal_001.py"
        )
        principal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(principal)
        selected, model_digest = principal.selections(plan, rows, MODELS[0])
        self.assertEqual(len(selected), 4)
        self.assertEqual(model_digest, "sha256:" + DIGESTS[MODELS[0]])

    def test_three_teacher_count_is_frozen_and_verified(self):
        identity = self.freeze(MODELS[:3])
        plan = mtc.strict_json((self.plan_dir / "plan.json").read_bytes())
        self.assertEqual(plan["count"], 18)
        self.capture(identity)
        self.assertEqual(FakeOllama.counts, 18)
        rows = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        self.assertEqual(mtc.summary(rows, plan)["planned"], 18)
        self.verify()

    def test_aliased_model_digests_fail_before_plan_creation(self):
        FakeOllama.duplicate_digest = True
        with self.assertRaisesRegex(ValueError, "duplicate installed model digest"):
            self.freeze(MODELS[:2])
        self.assertFalse(self.plan_dir.exists())
        self.assertEqual(FakeOllama.counts, 0)

    def test_forged_model_digest_or_request_count_is_rejected(self):
        self.freeze(MODELS[:2])
        plan = mtc.strict_json((self.plan_dir / "plan.json").read_bytes())
        self.assertEqual(plan["count"], 12)
        wrong_count = dict(plan, count=30)
        with self.assertRaisesRegex(ValueError, "frozen request count mismatch"):
            mtc.validate_plan(wrong_count)
        wrong_count["count"] = True
        with self.assertRaisesRegex(ValueError, "frozen request count mismatch"):
            mtc.validate_plan(wrong_count)
        collision = {**plan, "models": [dict(x) for x in plan["models"]]}
        collision["models"][1]["digest"] = collision["models"][0]["digest"]
        with self.assertRaisesRegex(ValueError, "duplicate frozen model digest"):
            mtc.validate_plan(collision)

    def test_single_teacher_resume_preserves_recorded_prefix(self):
        request_id = self.freeze(MODELS[:1])
        self.capture(request_id)
        rows = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        (self.run_dir / "manifest.json").unlink()
        (self.run_dir / "summary.json").unlink()
        (self.run_dir / "attempts.jsonl").write_bytes(
            b"".join(mtc.canonical(x) for x in rows[:2])
        )
        self.capture(request_id, resume=True)
        self.assertEqual(FakeOllama.counts, 6 + 4)
        self.assertEqual(mtc.read_attempts(self.run_dir / "attempts.jsonl")[:2],
                         rows[:2])
        self.verify()

    def test_freeze_refuses_duplicate_uninstalled_or_decision_models(self):
        for names in ([], MODELS + ["lfm3-extra:test"],
                      MODELS[:4] + [MODELS[0]],
                      MODELS[:4] + ["other:test"],
                      MODELS[:4] + ["lfm3-d1:test"]):
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    mtc.validate_models(",".join(names))
        bad = MODELS[:4] + ["lfm3-not-installed:test"]
        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            mtc.build_plan(bad)
        self.assertFalse(self.plan_dir.exists())

    def test_capture_needs_separate_authorization_and_exact_plan_identity(self):
        digest = self.freeze()
        with self.assertRaisesRegex(ValueError, "explicit owner authorization"):
            self.capture(digest, authorize=False)
        with self.assertRaisesRegex(ValueError, "request identity mismatch"):
            self.capture("sha256:" + "0" * 64)
        self.assertFalse(self.run_dir.exists())
        self.assertEqual(FakeOllama.counts, 0)

    def test_failed_http_outputs_are_quarantined_and_verified(self):
        digest = self.freeze()
        FakeOllama.fail = True
        self.capture(digest)
        rows = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(x["status"] == "failed" for x in rows))
        self.assertTrue(all(x["http_status"] == 503 for x in rows))
        self.assertEqual(mtc.summary(rows, mtc.strict_json((self.plan_dir / "plan.json").read_bytes()))["admitted_rows"], 0)
        self.assertIn("synthetic_failure",
                      base64.b64decode(rows[0]["response_base64"]).decode())
        self.verify()

    def test_mismatched_model_response_is_recorded_failure(self):
        request_id = self.freeze()
        FakeOllama.wrong_model = True
        self.capture(request_id)
        rows = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(row["status"] == "failed" for row in rows))
        self.assertTrue(all(row["error"] == "model_identity_mismatch" for row in rows))
        self.assertTrue(all(not row["oracle_match"] for row in rows))
        self.verify()

    def test_explicit_resume_keeps_recorded_attempts(self):
        digest = self.freeze()
        self.capture(digest)
        rows = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        (self.run_dir / "manifest.json").unlink()
        (self.run_dir / "summary.json").unlink()
        (self.run_dir / "attempts.jsonl").write_bytes(
            b"".join(mtc.canonical(x) for x in rows[:3])
        )
        # Resumed capture must not redo the three retained attempts.
        self.capture(digest, resume=True)
        self.assertEqual(FakeOllama.counts, 30 + 27)
        self.assertEqual(mtc.read_attempts(self.run_dir / "attempts.jsonl")[:3], rows[:3])
        self.verify()

    def test_incomplete_journal_cannot_be_silently_repaired(self):
        digest = self.freeze()
        self.run_dir.mkdir(mode=0o700)
        shutil.copy2(self.plan_dir / "plan.json", self.run_dir / "plan.json")
        (self.run_dir / "attempts.jsonl").write_bytes(b'{"truncated":')
        (self.run_dir / "attempts.jsonl").chmod(0o600)
        with self.assertRaisesRegex(ValueError, "incomplete attempt"):
            self.capture(digest, resume=True)
        self.assertEqual(FakeOllama.counts, 0)

    def test_manifest_tamper_and_new_output_rejected(self):
        digest = self.freeze()
        self.capture(digest)
        with self.assertRaises(FileExistsError):
            self.capture(digest)
        with self.assertRaisesRegex(ValueError, "no resumable incomplete run"):
            self.capture(digest, resume=True)
        self.verify()
        (self.run_dir / "summary.json").write_bytes(b'{"admitted_rows":999}')
        with self.assertRaisesRegex(ValueError, "member hash mismatch"):
            self.verify()

    def test_frozen_source_protocol_drift_fails_before_capture(self):
        digest = self.freeze()
        plan = json.loads((self.plan_dir / "plan.json").read_bytes())
        plan["source_closure"]["scripts/mtc_lfm3_001.py"] = "sha256:" + "0" * 64
        (self.plan_dir / "plan.json").write_bytes(mtc.canonical(plan))
        with self.assertRaisesRegex(ValueError, "source or plan closure drift"):
            self.capture(mtc.sha((self.plan_dir / "plan.json").read_bytes()))
        self.assertEqual(FakeOllama.counts, 0)

    def test_no_script_or_model_runs_from_false_input(self):
        self.assertFalse(owner.bool_arg("false"))
        self.assertTrue(owner.bool_arg("true"))
        for bad in ("yes", "True", "1"):
            with self.assertRaises(ValueError):
                owner.bool_arg(bad)
        with self.assertRaises(SystemExit):
            owner.main(["--reviewed-sha", "0" * 40, "--operation", "plan",
                        "--run-id", "safe"])
        with self.assertRaises(ValueError):
            mtc.api("/api/pull")

    def test_no_parent_symlink_or_reused_capture_folder(self):
        link = self.root / "link"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            mtc.create_private(link)
        self.freeze()
        with self.assertRaises(FileExistsError):
            self.freeze()
