"""MTC-PRINCIPAL-001 tests: fake loopback Qwen and LFM, no real inference."""
import base64
import http.server
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "mtc_principal_001.py"
spec = importlib.util.spec_from_file_location("principal", SOURCE)
principal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(principal)
mtc = principal.mtc

TEACHERS = ["lfm3-test-" + str(i) + ":small" for i in range(5)]
QWEN = "qwen3.5:27b"
NAMES = TEACHERS + [QWEN]
DIGESTS = {name: f"{i + 1:064x}" for i, name in enumerate(NAMES)}
TASKS = {task["prompt"]: task for task in mtc.tasks()}


class FakeOllama(http.server.BaseHTTPRequestHandler):
    counts = {"original": 0, "principal": 0, "revision": 0}
    malformed_principal = False
    wrong_principal_model = False
    fail_revisions = False
    thinking_only_principal = False
    thinking_only_teacher = False

    def log_message(self, *_):
        pass

    def reply(self, obj, code=200):
        raw = mtc.canonical(obj)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/tags":
            self.reply({"models": [
                {"name": name, "digest": DIGESTS[name]} for name in NAMES
            ]})
        else:
            self.reply({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/chat":
            self.reply({"error": "not found"}, 404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body["model"]
        if model == QWEN:
            FakeOllama.counts["principal"] += 1
            assert body["keep_alive"] == 0
            assert body["options"] == principal.PRINCIPAL_OPTIONS
            assert body["think"] is False
            assert body["format"] == "json"
            assert len(body["messages"]) == 2
            content = json.loads(body["messages"][1]["content"])
            assert set(content) == {"task", "teacher_output"}
            if FakeOllama.thinking_only_principal:
                self.reply({
                    "model": model, "done": True, "done_reason": "length",
                    "eval_count": principal.PRINCIPAL_OPTIONS["num_predict"],
                    "message": {"role": "assistant", "content": "",
                                "thinking": "synthetic principal token exhaustion"},
                })
                return
            if FakeOllama.malformed_principal:
                review = "not structured"
            else:
                # Deliberately fallible advice. No code can promote it into oracle truth.
                review = json.dumps({
                    "verdict": "incorrect", "feedback": "Recompute independently.",
                    "suggested_answer": "999"
                })
            returned_model = "different-model" if FakeOllama.wrong_principal_model else model
            self.reply({"model": returned_model, "done": True,
                        "message": {"role": "assistant", "content": review}})
            return
        assert model in TEACHERS
        if len(body["messages"]) == 1:
            FakeOllama.counts["original"] += 1
            task = TASKS[body["messages"][0]["content"]]
            answer = "999" if task["id"] == "multiply-17-19" else mtc.oracle(task)
        else:
            FakeOllama.counts["revision"] += 1
            assert body["keep_alive"] == 0
            assert body["options"] == mtc.SETTINGS
            assert body["think"] is False
            assert body["format"] == "json"
            assert len(body["messages"]) == 2
            msg = json.loads(body["messages"][1]["content"])
            assert set(msg) == {"task", "previous_answer", "principal_feedback"}
            if FakeOllama.fail_revisions:
                self.reply({"error": "synthetic_fail"}, 503)
                return
            task = TASKS[msg["task"]]
            answer = mtc.oracle(task)
        if FakeOllama.thinking_only_teacher:
            self.reply({
                "model": model, "done": True, "done_reason": "length",
                "eval_count": mtc.SETTINGS["num_predict"],
                "message": {"role": "assistant", "content": "",
                            "thinking": "synthetic teacher token exhaustion"},
            })
            return
        self.reply({"model": model, "done": True,
                    "message": {"role": "assistant",
                                "content": json.dumps({"answer": answer})}})


class PrincipalContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.old_port = mtc.URL_PORT
        mtc.URL_PORT = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        mtc.URL_PORT = cls.old_port
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        FakeOllama.malformed_principal = False
        FakeOllama.wrong_principal_model = False
        FakeOllama.fail_revisions = False
        FakeOllama.thinking_only_principal = False
        FakeOllama.thinking_only_teacher = False
        FakeOllama.counts = {"original": 0, "principal": 0, "revision": 0}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.teacher_plan = self.root / "teacher-plan"
        self.teacher_run = self.root / "teacher-run"
        mtc.plan_command(SimpleNamespace(
            models=",".join(TEACHERS), out=self.teacher_plan
        ))
        mtc.capture_command(SimpleNamespace(
            plan=self.teacher_plan / "plan.json",
            request_identity=mtc.sha((self.teacher_plan / "plan.json").read_bytes()),
            out=self.teacher_run,
            resume=False,
            authorize_local_capture=True,
        ))
        self.teacher_manifest = mtc.sha((self.teacher_run / "manifest.json").read_bytes())
        self.plan_dir = self.root / "principal-plan"
        self.run_dir = self.root / "principal-run"
        FakeOllama.counts = {"original": 0, "principal": 0, "revision": 0}

    def freeze(self):
        principal.plan_command(SimpleNamespace(
            teacher_run=self.teacher_run,
            teacher_manifest_identity=self.teacher_manifest,
            teacher_model=TEACHERS[0], principal_model=QWEN,
            out=self.plan_dir
        ))
        return mtc.sha((self.plan_dir / "plan.json").read_bytes())

    def capture(self, identity, authorize=True):
        principal.run_command(SimpleNamespace(
            plan=self.plan_dir / "plan.json", teacher_run=self.teacher_run,
            request_identity=identity, out=self.run_dir,
            authorize_principal=authorize
        ))

    def verify(self):
        manifest_id = mtc.sha((self.run_dir / "manifest.json").read_bytes())
        principal.verify_command(SimpleNamespace(
            run=self.run_dir, teacher_run=self.teacher_run,
            manifest_identity=manifest_id
        ))

    def test_principal_gives_feedback_and_teacher_revises_without_training(self):
        request_identity = self.freeze()
        plan = json.loads((self.plan_dir / "plan.json").read_bytes())
        self.assertEqual(len(plan["items"]), 4)
        self.assertEqual(plan["principal"]["name"], QWEN)
        self.assertEqual(plan["training_admission"], False)
        self.assertEqual({x["item_id"] for x in plan["items"]},
                         {x["id"] for x in mtc.tasks() if x["split"] == "train"})
        self.capture(request_identity)
        self.assertEqual(FakeOllama.counts,
                         {"original": 0, "principal": 4, "revision": 4})
        attempts = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        self.assertEqual(len(attempts), 4)
        self.assertEqual(sum(x["original_match"] for x in attempts), 3)
        self.assertEqual(sum(x["revised_match"] for x in attempts), 4)
        self.assertEqual(attempts[0]["principal_feedback"]["suggested_answer"], "999")
        self.assertEqual(attempts[0]["revised_answer"], "323")
        summary = json.loads((self.run_dir / "summary.json").read_bytes())
        self.assertEqual(summary["training_admission"], False)
        self.assertEqual(summary["rights_status"], "QUARANTINED_NOT_ADMITTED")
        self.verify()
        for d in (self.plan_dir, self.run_dir):
            self.assertEqual(d.stat().st_mode & 0o077, 0)
            self.assertTrue(all((f.stat().st_mode & 0o077) == 0
                                for f in d.iterdir()))


    def test_principal_thinking_only_review_is_recorded_and_never_sent_as_feedback(self):
        identity = self.freeze()
        FakeOllama.thinking_only_principal = True
        self.capture(identity)
        self.assertEqual(FakeOllama.counts["principal"], 4)
        self.assertEqual(FakeOllama.counts["revision"], 0)
        records = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        self.assertEqual(len(records), 4)
        self.assertTrue(all(x["principal_feedback"] is None and
                            x["status"] == "principal_failed_or_invalid" and
                            x["teacher_call"] is None for x in records))
        self.verify()

    def test_teacher_thinking_only_revision_cannot_inflate_oracle_matches(self):
        identity = self.freeze()
        FakeOllama.thinking_only_teacher = True
        self.capture(identity)
        self.assertEqual(FakeOllama.counts["principal"], 4)
        self.assertEqual(FakeOllama.counts["revision"], 4)
        records = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        self.assertTrue(all(x["status"] == "completed" and
                            x["revised_answer"] is None and
                            x["revised_match"] is False for x in records))
        self.verify()

    def test_missing_or_invalid_baseline_answer_skips_principal_calls(self):
        # A true thinking-only baseline has no answer content. No Principal
        # inference should be wasted reviewing the empty string.
        self.assertIsNone(principal.bound_original({
            "status": "completed",
            "response_base64": base64.b64encode(mtc.canonical({
                "done": True, "done_reason": "length",
                "message": {"content": "", "thinking": "exhausted"}
            })).decode("ascii"),
        }))
        self.assertIsNone(principal.bound_original({
            "status": "completed",
            "response_base64": base64.b64encode(mtc.canonical({
                "done": True, "done_reason": "stop",
                "message": {"content": "not_json"}
            })).decode("ascii"),
        }))

    def test_principal_ignores_truncated_json_even_if_it_looks_valid(self):
        response = {
            "model": QWEN, "done": True, "done_reason": "length",
            "message": {"content": json.dumps({
                "verdict": "correct", "feedback": "okay", "suggested_answer": "323"
            })}
        }
        self.assertIsNone(principal.feedback_of(response, QWEN))
        self.assertIsNotNone(principal.feedback_of({
            **response, "done_reason": "stop"
        }, QWEN))
        identity = self.freeze()
        plan = mtc.strict_json((self.plan_dir / "plan.json").read_bytes())
        self.assertEqual(plan["response_policy"], {"think": False, "format": "json"})
        self.assertEqual(plan["schema"], "trinite.mtc-principal.plan.v2")
        with self.assertRaisesRegex(ValueError, "source/policy drift"):
            principal.check_plan({**plan, "response_policy":
                                  {"think": True, "format": "json"}})
        with self.assertRaises(ValueError):
            principal.check_plan({k: v for k, v in plan.items()
                                  if k != "response_policy"})

    def test_authorization_denied_before_any_principal_model_call(self):
        identity = self.freeze()
        with self.assertRaisesRegex(ValueError, "explicit Principal run authorization"):
            self.capture(identity, authorize=False)
        self.assertEqual(FakeOllama.counts["principal"], 0)
        self.assertFalse(self.run_dir.exists())

    def test_wrong_baseline_manifest_prevents_plan(self):
        with self.assertRaisesRegex(ValueError, "external manifest identity mismatch"):
            principal.plan_command(SimpleNamespace(
                teacher_run=self.teacher_run,
                teacher_manifest_identity="sha256:" + "0"*64,
                teacher_model=TEACHERS[0], principal_model=QWEN,
                out=self.plan_dir,
            ))
        self.assertFalse(self.plan_dir.exists())

    def test_malformed_principal_feedback_never_reaches_teacher(self):
        request_identity = self.freeze()
        FakeOllama.malformed_principal = True
        self.capture(request_identity)
        self.assertEqual(FakeOllama.counts["principal"], 4)
        self.assertEqual(FakeOllama.counts["revision"], 0)
        rows = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        self.assertTrue(all(x["status"] == "principal_failed_or_invalid" for x in rows))
        self.verify()

    def test_principal_model_identity_mismatch_leaves_partial_archive(self):
        request_identity = self.freeze()
        FakeOllama.wrong_principal_model = True
        with self.assertRaisesRegex(ValueError, "partial capture preserved"):
            self.capture(request_identity)
        self.assertFalse((self.run_dir / "manifest.json").exists())
        rows = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["principal_call"]["error"], "model_identity_mismatch")
        self.assertEqual(FakeOllama.counts["revision"], 0)

    def test_teacher_revision_http_failure_retains_bytes_and_fails_oracle(self):
        identity = self.freeze()
        FakeOllama.fail_revisions = True
        self.capture(identity)
        rows = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        self.assertTrue(all(x["status"] == "revision_transport_failed" for x in rows))
        self.assertTrue(all(x["revised_match"] is False for x in rows))
        self.assertEqual(rows[0]["teacher_call"]["http_status"], 503)
        self.assertIn("synthetic_fail",
                      base64.b64decode(rows[0]["teacher_call"]["response_base64"]).decode())
        self.verify()

    def test_plan_rejects_non27b_or_missing_model(self):
        for bad in ("qwen3.5:9b", "qwen3.5:35b", "../qwen", "qwen3:27b"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "Qwen3.5 27B"):
                    principal.plan_value(self.teacher_run, self.teacher_manifest,
                                         TEACHERS[0], bad)
        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            principal.plan_value(self.teacher_run, self.teacher_manifest,
                                 TEACHERS[0], "qwen3.5:27b-does-not-exist")

    def test_tampering_and_no_clobber_rejected(self):
        digest = self.freeze()
        self.capture(digest)
        with self.assertRaises(FileExistsError):
            self.capture(digest)
        self.verify()
        (self.run_dir / "summary.json").write_bytes(b'{"training_admission":true}')
        with self.assertRaisesRegex(ValueError, "Principal member mismatch"):
            self.verify()

    def test_source_closure_or_request_identity_drift_fails_closed(self):
        digest = self.freeze()
        with self.assertRaisesRegex(ValueError, "principal request identity mismatch"):
            self.capture("sha256:" + "0"*64)
        self.assertFalse(self.run_dir.exists())
        with patch.object(principal, "closure", return_value={"drift": "bad"}):
            with self.assertRaisesRegex(ValueError, "source/policy drift"):
                self.capture(digest)
        self.assertEqual(FakeOllama.counts["principal"], 0)
        self.assertFalse(self.run_dir.exists())

    def test_resumption_is_not_exposed_as_implicit_retry(self):
        parser_help_source = SOURCE.read_text()
        self.assertNotIn('capture.add_argument("--resume"', parser_help_source)
        self.assertIn("never silently send it again", parser_help_source)
        self.assertEqual(principal.LIMIT, 4)


if __name__ == "__main__":
    unittest.main()
