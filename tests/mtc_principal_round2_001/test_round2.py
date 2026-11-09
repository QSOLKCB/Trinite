"""Synthetic evidence-chain checks for MTC-PRINCIPAL-R2-001.

Never connects to real models or owner data; all calls hit an ephemeral loopback server.
"""
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
spec = importlib.util.spec_from_file_location(
    "principal_round2", ROOT / "scripts" / "mtc_principal_round2_001.py"
)
round2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(round2)
principal, mtc = round2.principal, round2.mtc

TEACHER = "lfm3:1.7b"
PRINCIPAL = "qwen3.5:27b"
DIGESTS = {TEACHER: "1" * 64, PRINCIPAL: "2" * 64}
TASKS = {t["prompt"]: t for t in mtc.tasks()}


class FakeOllama(http.server.BaseHTTPRequestHandler):
    phase = 1
    thinking_only = False
    counts = {"baseline": 0, "principal": 0, "revisions": 0}

    def log_message(self, *_):
        pass

    def reply(self, data, code=200):
        raw = mtc.canonical(data)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/tags":
            self.reply({"models": [
                {"name": m, "digest": d} for m, d in DIGESTS.items()
            ]})
        else:
            self.reply({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/chat":
            self.reply({"error": "not found"}, 404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["think"] is False
        assert body["format"] == "json"
        assert body["stream"] is False
        assert body["model"] in DIGESTS
        model = body["model"]
        if model == PRINCIPAL:
            FakeOllama.counts["principal"] += 1
            assert body["keep_alive"] == 0
            assert body["options"] == principal.PRINCIPAL_OPTIONS
            prompt = json.loads(body["messages"][1]["content"])
            assert set(prompt) == {"task", "teacher_output"}
            if FakeOllama.thinking_only and FakeOllama.phase == 2:
                self.reply({
                    "model": model, "done": True, "done_reason": "length",
                    "message": {"content": "", "thinking": "synthetic 768 tokens"}
                })
                return
            # Qwen may falsely pronounce an incorrect answer "correct".
            output = json.dumps({
                "verdict": "correct", "feedback": "Please check arithmetic again.",
                "suggested_answer": "999"
            })
        else:
            assert body["options"] == mtc.SETTINGS
            if len(body["messages"]) == 1:
                FakeOllama.counts["baseline"] += 1
                task = TASKS[body["messages"][0]["content"]]
                # The first frozen baseline had two of four TRAIN tasks correct.
                output = json.dumps({"answer": {
                    "mod11-add-8-15": "2", "mod13-mul-12-7": "3"
                }.get(task["id"], mtc.oracle(task))})
            else:
                FakeOllama.counts["revisions"] += 1
                assert body["keep_alive"] == 0
                prompt = json.loads(body["messages"][1]["content"])
                assert set(prompt) == {"task", "previous_answer", "principal_feedback"}
                task = TASKS[prompt["task"]]
                if FakeOllama.phase == 1 and task["id"] == "mod11-add-8-15":
                    answer = "2"
                else:
                    answer = mtc.oracle(task)
                output = json.dumps({"answer": answer})
        self.reply({"model": model, "done": True, "done_reason": "stop",
                    "message": {"role": "assistant", "content": output}})


class Round2Contracts(unittest.TestCase):
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
        cls.thread.join(timeout=3)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        FakeOllama.phase = 1
        FakeOllama.thinking_only = False
        FakeOllama.counts = {"baseline": 0, "principal": 0, "revisions": 0}
        self.teacher_plan = self.root / "teacher-plan"
        self.teacher_run = self.root / "teacher-run"
        mtc.plan_command(SimpleNamespace(models=TEACHER, out=self.teacher_plan))
        mtc.capture_command(SimpleNamespace(
            plan=self.teacher_plan / "plan.json", out=self.teacher_run,
            request_identity=mtc.sha((self.teacher_plan / "plan.json").read_bytes()),
            authorize_local_capture=True, resume=False,
        ))
        self.teacher_anchor = mtc.sha((self.teacher_run / "manifest.json").read_bytes())
        self.principal_plan = self.root / "principal-plan"
        self.principal_run = self.root / "principal-run"
        principal.plan_command(SimpleNamespace(
            teacher_run=self.teacher_run,
            teacher_manifest_identity=self.teacher_anchor,
            teacher_model=TEACHER, principal_model=PRINCIPAL,
            out=self.principal_plan
        ))
        principal.run_command(SimpleNamespace(
            teacher_run=self.teacher_run, plan=self.principal_plan / "plan.json",
            request_identity=mtc.sha((self.principal_plan / "plan.json").read_bytes()),
            out=self.principal_run, authorize_principal=True,
        ))
        self.principal_anchor = mtc.sha((self.principal_run / "manifest.json").read_bytes())
        prev = mtc.read_attempts(self.principal_run / "attempts.jsonl")
        self.assertEqual(sum(x["revised_match"] for x in prev), 3)
        self.assertEqual(len(prev), 4)
        FakeOllama.phase = 2
        FakeOllama.counts = {"baseline": 0, "principal": 0, "revisions": 0}
        self.plan_dir = self.root / "plan-r2"
        self.run_dir = self.root / "run-r2"

    def freeze(self):
        round2.plan_command(SimpleNamespace(
            teacher_run=self.teacher_run, previous_run=self.principal_run,
            previous_manifest_identity=self.principal_anchor,
            out=self.plan_dir
        ))
        return mtc.sha((self.plan_dir / "plan.json").read_bytes())

    def capture(self, identity, authorized=True):
        round2.capture_command(SimpleNamespace(
            teacher_run=self.teacher_run, previous_run=self.principal_run,
            plan=self.plan_dir / "plan.json", request_identity=identity,
            out=self.run_dir, authorize_round2=authorized
        ))

    def verify(self):
        identity = mtc.sha((self.run_dir / "manifest.json").read_bytes())
        round2.verify_command(SimpleNamespace(
            teacher_run=self.teacher_run, previous_run=self.principal_run,
            run=self.run_dir, manifest_identity=identity
        ))

    def test_cumulative_round2_corrects_remaining_error_and_preserves_three(self):
        identity = self.freeze()
        plan = mtc.strict_json((self.plan_dir / "plan.json").read_bytes())
        self.assertEqual(sum(x["baseline_match"] for x in plan["items"]), 3)
        self.assertEqual(plan["prior_principal_manifest_identity"],
                         self.principal_anchor)
        self.assertEqual(plan["prior_teacher_manifest_identity"], self.teacher_anchor)
        self.assertEqual(plan["planned_calls_max"], 8)
        self.assertEqual(plan["training_admission"], False)
        self.capture(identity)
        self.assertEqual(FakeOllama.counts,
                         {"baseline": 0, "principal": 4, "revisions": 4})
        summary = mtc.strict_json((self.run_dir / "summary.json").read_bytes())
        self.assertEqual(summary["baseline_oracle_matches"], 3)
        self.assertEqual(summary["revised_oracle_matches"], 4)
        self.assertEqual(summary["corrected"], 1)
        self.assertEqual(summary["retained"], 3)
        self.assertEqual(summary["regressions"], 0)
        self.assertEqual(summary["training_admission"], False)
        self.assertEqual(summary["principal_verdict_oracle_disagreements"], 1)
        self.verify()
        for folder in (self.plan_dir, self.run_dir):
            self.assertEqual(folder.stat().st_mode & 0o077, 0)
            self.assertTrue(all((f.stat().st_mode & 0o077) == 0
                                for f in folder.iterdir()))

    def test_old_evidence_tampering_rejected_before_freeze_or_inference(self):
        (self.principal_run / "summary.json").write_bytes(b'{"forged":true}')
        with self.assertRaisesRegex(ValueError, "Principal member mismatch"):
            self.freeze()
        self.assertFalse(self.plan_dir.exists())
        self.assertEqual(FakeOllama.counts["principal"], 0)

    def test_request_identity_and_authorization_are_separate(self):
        identity = self.freeze()
        with self.assertRaisesRegex(ValueError, "authorization required"):
            self.capture(identity, authorized=False)
        with self.assertRaisesRegex(ValueError, "request identity mismatch"):
            self.capture("sha256:" + "0" * 64)
        self.assertFalse(self.run_dir.exists())
        self.assertEqual(FakeOllama.counts["principal"], 0)

    def test_rejects_source_or_policy_drift_without_model_calls(self):
        identity = self.freeze()
        data = mtc.strict_json((self.plan_dir / "plan.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "source or frozen policy drift"):
            round2.validate_plan({**data, "response_policy": {
                "think": True, "format": "json"
            }})
        with patch.object(round2, "closure", return_value={"invalid": True}):
            with self.assertRaisesRegex(ValueError, "source or frozen policy drift"):
                self.capture(identity)
        self.assertEqual(FakeOllama.counts["principal"], 0)
        self.assertFalse(self.run_dir.exists())

    def test_thinking_only_reviews_preserved_without_teacher_revision(self):
        identity = self.freeze()
        FakeOllama.thinking_only = True
        self.capture(identity)
        self.assertEqual(FakeOllama.counts,
                         {"baseline": 0, "principal": 4, "revisions": 0})
        rows = mtc.read_attempts(self.run_dir / "attempts.jsonl")
        self.assertTrue(all(x["principal_feedback"] is None for x in rows))
        self.assertTrue(all(x["teacher_call"] is None for x in rows))
        self.assertTrue(all(x["revised_answer"] is None for x in rows))
        result = mtc.strict_json((self.run_dir / "summary.json").read_bytes())
        self.assertEqual(result["ungraded"], 4)
        self.assertEqual(result["regressions"], 0)
        self.assertEqual(result["revisions_scored"], 0)
        self.verify()

    def test_no_overwrite_no_implicit_resume_and_manifest_tamper(self):
        identity = self.freeze()
        self.capture(identity)
        with self.assertRaises(FileExistsError):
            self.capture(identity)
        self.verify()
        (self.run_dir / "summary.json").write_bytes(b'{"forged":true}')
        with self.assertRaisesRegex(ValueError, "Round 2 member hash mismatch"):
            self.verify()

    def test_outputs_cannot_modify_parent_evidence_archives(self):
        for parent in (self.teacher_run, self.principal_run):
            self.plan_dir = parent / "nested-round2-plan"
            with self.assertRaisesRegex(ValueError, "outside parent evidence archives"):
                self.freeze()
            self.assertFalse(self.plan_dir.exists())

        self.plan_dir = self.root / "plan-r2"
        identity = self.freeze()
        for parent in (self.teacher_run, self.principal_run):
            self.run_dir = parent / "nested-round2-capture"
            with self.assertRaisesRegex(ValueError, "outside parent evidence archives"):
                self.capture(identity)
            self.assertFalse(self.run_dir.exists())

        self.assertEqual(FakeOllama.counts,
                         {"baseline": 0, "principal": 0, "revisions": 0})
        principal.verify_command(SimpleNamespace(
            teacher_run=self.teacher_run, run=self.principal_run,
            manifest_identity=self.principal_anchor
        ))

    def test_round2_is_not_replay_of_original_teacher_baseline(self):
        identity = self.freeze()
        plan = mtc.strict_json((self.plan_dir / "plan.json").read_bytes())
        prior = mtc.read_attempts(self.principal_run / "attempts.jsonl")
        for row, selected in zip(prior, plan["items"]):
            self.assertEqual(row["revised_answer"], selected["baseline_answer"])
            self.assertEqual(mtc.sha(mtc.canonical(row)),
                             selected["previous_row_sha256"])
        self.assertEqual(FakeOllama.counts["baseline"], 0)
        self.assertFalse(self.run_dir.exists())


if __name__ == "__main__":
    unittest.main()
