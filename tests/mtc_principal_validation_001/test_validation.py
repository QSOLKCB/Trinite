"""Offline tests for MTC-PRINCIPAL-VALIDATION-001; zero real Ollama access."""
from __future__ import annotations

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
SPEC = importlib.util.spec_from_file_location(
    "principal_validation", ROOT / "scripts/mtc_principal_validation_001.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
r2, principal, mtc = audit.r2, audit.principal, audit.mtc

TEACHER = "lfm3:1.7b"
PRINCIPAL = "qwen3.5:27b"
DIGESTS = {TEACHER: "1" * 64, PRINCIPAL: "2" * 64}
BASE_TASKS = {t["prompt"]: t for t in mtc.tasks()}


class FakeOllama(http.server.BaseHTTPRequestHandler):
    phase = "setup"
    counts = {"teacher_original": 0, "principal": 0, "teacher_revision": 0,
              "control": 0}
    thinking_only_exam = False
    drift_during_exam = False
    inventory_digests = dict(DIGESTS)
    requested_models = []

    def log_message(self, *_):
        pass

    def reply(self, obj, status=200):
        raw = mtc.canonical(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path != "/api/tags":
            self.reply({"error": "not found"}, 404)
        else:
            self.reply({"models": [
                {"name": name, "digest": d}
                for name, d in FakeOllama.inventory_digests.items()
            ]})

    def do_POST(self):
        if self.path != "/api/chat":
            self.reply({"error": "not found"}, 404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["model"] in DIGESTS
        assert body["think"] is False and body["format"] == "json"
        assert body["stream"] is False
        model = body["model"]
        FakeOllama.requested_models.append(model)
        if model == PRINCIPAL:
            FakeOllama.counts["principal"] += 1
            assert body["options"] == principal.PRINCIPAL_OPTIONS
            if FakeOllama.phase == "exam":
                claim = json.loads(body["messages"][1]["content"])
                assert set(claim) == {"problem", "claimed_answer"}
                # Deliberate false positive for the repeatedly missed mod-11 case.
                if claim["problem"] == "(8 + 15) modulo 11":
                    verdict = "correct"
                elif claim["problem"] == "(12 * 7) modulo 13":
                    verdict = "correct" if claim["claimed_answer"] == "6" else "incorrect"
                elif claim["problem"] == "(37 + 42) modulo 19":
                    verdict = "correct" if claim["claimed_answer"] == "3" else "incorrect"
                elif claim["problem"] == "(9 * 8) modulo 17":
                    verdict = "correct" if claim["claimed_answer"] == "4" else "incorrect"
                else:
                    raise AssertionError("unknown frozen exam")
                if FakeOllama.drift_during_exam:
                    FakeOllama.inventory_digests[PRINCIPAL] = "3" * 64
                if FakeOllama.thinking_only_exam:
                    self.reply({"model": model, "done": True,
                                "done_reason": "length", "eval_count": 768,
                                "message": {"content": "",
                                            "thinking": "synthetic budget spent"}})
                    return
                output = {"verdict": verdict, "feedback": "Check the arithmetic.",
                          "suggested_answer": "0"}
            else:
                output = {"verdict": "correct", "feedback": "Please recompute.",
                          "suggested_answer": "999"}
            text = json.dumps(output)
        else:
            assert body["options"] == mtc.SETTINGS
            task = BASE_TASKS[body["messages"][-1]["content"]] if len(
                body["messages"]
            ) == 1 else BASE_TASKS[json.loads(body["messages"][1]["content"])["task"]]
            if len(body["messages"]) == 1:
                if FakeOllama.phase == "exam":
                    FakeOllama.counts["control"] += 1
                    assert body.get("keep_alive") == 0
                else:
                    FakeOllama.counts["teacher_original"] += 1
                value = {"mod11-add-8-15": "2", "mod13-mul-12-7": "3"}.get(
                    task["id"], mtc.oracle(task)
                )
            else:
                FakeOllama.counts["teacher_revision"] += 1
                assert FakeOllama.phase == "setup"
                value = "2" if task["id"] == "mod11-add-8-15" else mtc.oracle(task)
            text = json.dumps({"answer": value})
        self.reply({"model": model, "done": True, "done_reason": "stop",
                    "message": {"role": "assistant", "content": text}})


class ExaminerControlContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.old_port = mtc.URL_PORT
        mtc.URL_PORT = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        mtc.URL_PORT = cls.old_port
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        FakeOllama.phase = "setup"
        FakeOllama.counts = {
            "teacher_original": 0, "principal": 0,
            "teacher_revision": 0, "control": 0
        }
        FakeOllama.inventory_digests = dict(DIGESTS)
        FakeOllama.requested_models = []
        FakeOllama.thinking_only_exam = False
        FakeOllama.drift_during_exam = False
        self.teacher_plan = self.home / "teacher-plan"
        self.teacher_run = self.home / "teacher-run"
        mtc.plan_command(SimpleNamespace(models=TEACHER, out=self.teacher_plan))
        mtc.capture_command(SimpleNamespace(
            plan=self.teacher_plan / "plan.json",
            request_identity=mtc.sha((self.teacher_plan / "plan.json").read_bytes()),
            out=self.teacher_run, authorize_local_capture=True, resume=False
        ))
        self.teacher_manifest = mtc.sha((self.teacher_run / "manifest.json").read_bytes())
        self.principal_plan = self.home / "principal-plan"
        self.principal_run = self.home / "principal-run"
        principal.plan_command(SimpleNamespace(
            teacher_run=self.teacher_run,
            teacher_manifest_identity=self.teacher_manifest,
            teacher_model=TEACHER, principal_model=PRINCIPAL,
            out=self.principal_plan
        ))
        principal.run_command(SimpleNamespace(
            plan=self.principal_plan / "plan.json",
            teacher_run=self.teacher_run,
            request_identity=mtc.sha((self.principal_plan / "plan.json").read_bytes()),
            out=self.principal_run, authorize_principal=True,
        ))
        self.principal_manifest = mtc.sha((self.principal_run / "manifest.json").read_bytes())
        rows = mtc.read_attempts(self.principal_run / "attempts.jsonl")
        self.assertEqual(sum(x["revised_match"] for x in rows), 3)
        self.assertEqual(len(rows), 4)
        FakeOllama.phase = "exam"
        FakeOllama.counts = {
            "teacher_original": 0, "principal": 0,
            "teacher_revision": 0, "control": 0
        }
        FakeOllama.requested_models = []
        self.plan_path = self.home / "validation-plan"
        self.run_path = self.home / "validation-run"
        self.approval_path = self.home / "owner-approval.json"

    def freeze(self):
        audit.plan_command(SimpleNamespace(
            teacher_run=self.teacher_run,
            baseline_run=self.principal_run,
            baseline_manifest_identity=self.principal_manifest,
            out=self.plan_path,
        ))
        return mtc.sha((self.plan_path / "plan.json").read_bytes())

    def approval(self, digest):
        plan = mtc.strict_json((self.plan_path / "plan.json").read_bytes())
        value = audit.authorization_value(
            plan, digest, self.approval_path, self.run_path
        )
        mtc.write_new(self.approval_path, mtc.canonical(value))

    def capture(self, digest):
        audit.capture_command(SimpleNamespace(
            plan=self.plan_path / "plan.json", request_identity=digest,
            out=self.run_path, owner_approval=self.approval_path,
            teacher_run=self.teacher_run, baseline_run=self.principal_run,
        ))

    def verify(self):
        digest = mtc.sha((self.run_path / "manifest.json").read_bytes())
        audit.verify_command(SimpleNamespace(
            run=self.run_path, manifest_identity=digest,
            teacher_run=self.teacher_run, baseline_run=self.principal_run,
        ))

    def test_balanced_exam_confusion_matrix_and_no_feedback_control(self):
        digest = self.freeze()
        plan = mtc.strict_json((self.plan_path / "plan.json").read_bytes())
        self.assertEqual(plan["planned_calls"], 12)
        self.assertEqual([audit.truth(x) for x in plan["exams"]].count("correct"), 4)
        self.assertEqual([audit.truth(x) for x in plan["exams"]].count("incorrect"), 4)
        self.assertEqual(plan["baseline_rows"][2]["previous_revised_match"], False)
        self.approval(digest)
        self.capture(digest)
        result = mtc.strict_json((self.run_path / "summary.json").read_bytes())
        self.assertEqual(result["examiner_valid_verdicts"], 8)
        self.assertEqual(result["true_positive"], 4)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["true_negative"], 3)
        self.assertEqual(result["false_negative"], 0)
        self.assertEqual(result["no_feedback_scored"], 4)
        self.assertEqual(result["no_feedback_matches"], 2)
        self.assertEqual(result["previous_supervised_baseline_matches"], 3)
        self.assertEqual(result["training_admission"], False)
        self.assertEqual(FakeOllama.counts, {
            "teacher_original": 0, "principal": 8,
            "teacher_revision": 0, "control": 4
        })
        self.assertEqual(FakeOllama.requested_models, [PRINCIPAL] * 8 + [TEACHER] * 4)
        self.verify()
        for d in (self.plan_path, self.run_path):
            self.assertEqual(d.stat().st_mode & 0o077, 0)
            self.assertTrue(all((f.stat().st_mode & 0o077) == 0 for f in d.iterdir()))

    def test_no_owner_approval_means_zero_new_inferences(self):
        digest = self.freeze()
        with self.assertRaises(ValueError):
            self.capture(digest)
        self.assertEqual(FakeOllama.counts["principal"], 0)
        self.assertFalse(self.run_path.exists())

    def test_wrong_identity_fails_without_consuming_authorization(self):
        digest = self.freeze()
        self.approval(digest)
        with self.assertRaisesRegex(ValueError, "request identity mismatch"):
            self.capture("sha256:" + "0"*64)
        self.assertEqual(FakeOllama.counts["principal"], 0)
        self.assertFalse(self.run_path.exists())
        self.assertFalse(audit.claim_path(self.approval_path).exists())

    def test_invalid_parent_digest_rejected_before_plan(self):
        with self.assertRaisesRegex(ValueError, "external manifest identity mismatch"):
            audit.plan_command(SimpleNamespace(
                teacher_run=self.teacher_run, baseline_run=self.principal_run,
                baseline_manifest_identity="sha256:" + "0"*64,
                out=self.plan_path,
            ))
        self.assertFalse(self.plan_path.exists())

    def test_frozen_exam_answer_tampering_rejected(self):
        self.freeze()
        plan = mtc.strict_json((self.plan_path / "plan.json").read_bytes())
        plan["exams"][0]["claimed_answer"] = "1"
        with self.assertRaisesRegex(ValueError, "source or policy drift"):
            audit.validate_plan(plan)

    def test_thinking_only_exam_ungrounded_not_silently_promoted(self):
        digest = self.freeze()
        FakeOllama.thinking_only_exam = True
        self.approval(digest)
        self.capture(digest)
        result = mtc.strict_json((self.run_path / "summary.json").read_bytes())
        self.assertEqual(result["invalid_examiner"], 8)
        self.assertEqual(result["true_positive"], 0)
        self.assertEqual(result["false_positive"], 0)
        self.assertEqual(result["no_feedback_scored"], 4)
        self.verify()

    def test_mid_call_digest_drift_is_logged_and_terminal(self):
        digest = self.freeze()
        self.approval(digest)
        FakeOllama.drift_during_exam = True
        with self.assertRaisesRegex(ValueError, "model identity drift"):
            self.capture(digest)
        records = mtc.read_attempts(self.run_path / "attempts.jsonl")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["call"]["error"], "model_digest_drift")
        self.assertEqual(records[0]["call"]["http_status"], 200)
        self.assertEqual(records[0]["call"]["model_digest_after"], "sha256:" + "3"*64)
        self.assertFalse((self.run_path / "manifest.json").exists())
        self.assertEqual(FakeOllama.counts["principal"], 1)
        self.assertEqual(FakeOllama.counts["control"], 0)

    def test_one_use_archive_tamper_and_no_clobber(self):
        digest = self.freeze()
        self.approval(digest)
        self.capture(digest)
        with self.assertRaises(FileExistsError):
            self.capture(digest)
        self.verify()
        (self.run_path / "summary.json").write_bytes(b'{"training_admission":true}')
        with self.assertRaisesRegex(ValueError, "examiner manifest member mismatch"):
            self.verify()

    def test_source_closure_or_prior_data_drift_fails_closed(self):
        digest = self.freeze()
        self.approval(digest)
        with patch.object(audit, "closure", return_value={"tampered": True}):
            with self.assertRaisesRegex(ValueError, "source or policy drift"):
                self.capture(digest)
        self.assertEqual(FakeOllama.counts["principal"], 0)
        self.assertFalse(self.run_path.exists())

    def test_no_feedback_request_has_no_principal_feedback_or_hidden_context(self):
        digest = self.freeze()
        plan = mtc.strict_json((self.plan_path / "plan.json").read_bytes())
        for task in audit.control_tasks():
            body = audit.control_body(plan, task)
            self.assertEqual(body["model"], TEACHER)
            self.assertEqual(len(body["messages"]), 1)
            self.assertEqual(body["messages"][0]["content"], task["prompt"])
            self.assertEqual(body["options"], mtc.SETTINGS)
            self.assertEqual(body["think"], False)
            self.assertEqual(body["format"], "json")
        self.assertEqual(FakeOllama.counts["principal"], 0)
        self.assertFalse(self.run_path.exists())


if __name__ == "__main__":
    unittest.main()
