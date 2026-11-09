"""Offline software-double contracts for MTC-EXAMINER-CAUSE-001.

No real Ollama calls, trained models, student updates or original archive writes.
"""
from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "examiner_cause", ROOT / "scripts" / "mtc_examiner_cause_001.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
mtc = m.mtc
DIGEST = "sha256:" + "6" * 64


class DiagnosticContracts(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.teacher_run = self.home / "teacher"
        self.baseline_run = self.home / "baseline"
        self.validation_run = self.home / "validation"
        for item in (self.teacher_run, self.baseline_run, self.validation_run):
            item.mkdir(mode=0o700)
        self.plan_path = self.home / "plan"
        self.run_path = self.home / "capture"
        self.approval_path = self.home / "approval.json"
        self.hist = {
            "principal": {"name": "qwen3.5:27b", "digest": DIGEST},
            "summary_sha256": "sha256:" + "9" * 64,
            "prior_teacher_manifest_identity": "sha256:" + "1" * 64,
            "prior_principal_manifest_identity": "sha256:" + "2" * 64,
        }
        self.calls = []
        self.drift = False
        self.length_reply = False
        self.hist_patch = patch.object(m, "historical", return_value=self.hist)
        self.inventory_patch = patch.object(
            mtc, "inventory", return_value=[
                {"name": "qwen3.5:27b", "digest": "6"*64}])
        self.commit_patch = patch.object(
            m, "source_commit", return_value="a" * 40)
        self.record_patch = patch.object(m.r2, "record_call", side_effect=self.fake_call)
        for p in (self.hist_patch, self.inventory_patch,
                  self.commit_patch, self.record_patch):
            p.start()
            self.addCleanup(p.stop)

    def fake_call(self, body, model):
        self.calls.append(body)
        self.assertEqual(body["model"], "qwen3.5:27b")
        self.assertFalse(body["think"])
        self.assertEqual(body["format"], "json")
        self.assertEqual(body["keep_alive"], 0)
        self.assertEqual(model["digest"], DIGEST)
        args = json.loads(body["messages"][1]["content"])
        case = next(x for x in m.probes()
                    if m.problem(x) == args["problem"])
        actual = m.v.arithmetic_answer(case)
        if "claimed_answer" in args:
            self.assertEqual(set(args), {"problem", "claimed_answer"})
            # Simulate an always-accepting examiner; parser/oracle must not.
            output = {"verdict": "correct", "feedback": "Synthetic review.",
                      "suggested_answer": actual}
        elif "candidate_A" in args:
            self.assertEqual(set(args), {"problem", "candidate_A", "candidate_B"})
            output = {"choice": "A" if args["candidate_A"] == actual else "B",
                      "answer": actual}
        else:
            self.assertEqual(set(args), {"problem"})
            output = {"answer": actual}
        response = {
            "model": "qwen3.5:27b", "done": True,
            "done_reason": "length" if self.length_reply else "stop",
            "message": {"role": "assistant", "content": json.dumps(output)}
        }
        raw = mtc.canonical(response)
        return {"request_sha256": mtc.sha(mtc.canonical(body)),
                "response_base64": base64.b64encode(raw).decode("ascii"),
                "http_status": 200, "error": "model_digest_drift" if self.drift else None,
                "model_digest_after": "sha256:" + "7"*64 if self.drift else DIGEST}

    def paths(self):
        return dict(teacher_run=self.teacher_run, baseline_run=self.baseline_run,
                    validation_run=self.validation_run)

    def freeze(self):
        m.plan_command(SimpleNamespace(**self.paths(), out=self.plan_path))
        return mtc.sha((self.plan_path / "plan.json").read_bytes())

    def approve(self, identity):
        plan = m.validate_plan(mtc.strict_json(
            mtc.checked_file(self.plan_path / "plan.json")))
        auth = m.approval_value(plan, identity, self.approval_path, self.run_path)
        mtc.write_new(self.approval_path, mtc.canonical(auth))

    def capture(self, identity):
        m.capture_command(SimpleNamespace(
            **self.paths(), plan=self.plan_path / "plan.json",
            request_identity=identity, out=self.run_path,
            owner_approval=self.approval_path))

    def verify(self, identity):
        m.verify_command(SimpleNamespace(
            **self.paths(), run=self.run_path, manifest_identity=identity))

    def manifest_digest(self):
        return mtc.sha(mtc.checked_file(self.run_path / "manifest.json"))

    def test_pauses_after_frozen_plan_no_model_calls(self):
        digest = self.freeze()
        plan = mtc.strict_json(mtc.checked_file(self.plan_path / "plan.json"))
        self.assertEqual(plan["planned_calls"], 16)
        self.assertEqual(plan["source_commit"], "a"*40)
        self.assertEqual(plan["historical_validation_identity"], m.VALIDATION_IDENTITY)
        self.assertEqual(plan["milestone_log_receipt_identity"], m.LOG_RECEIPT_IDENTITY)
        self.assertEqual([r["lane"] for r in plan["schedule"]],
                         ["solve"]*4 + ["review"]*8 + ["choice"]*4)
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(digest, mtc.sha(mtc.canonical(plan)))

    def test_reviewer_false_positives_with_independent_solve_and_choice(self):
        identity = self.freeze()
        self.approve(identity)
        self.capture(identity)
        self.assertEqual(len(self.calls), 16)
        summary = mtc.strict_json(mtc.checked_file(self.run_path / "summary.json"))
        self.assertEqual(summary["solve_matches"], 4)
        self.assertEqual(summary["solve_valid_answers"], 4)
        self.assertEqual(summary["true_positive"], 4)
        self.assertEqual(summary["false_positive"], 4)
        self.assertEqual(summary["true_negative"], 0)
        self.assertEqual(summary["false_negative"], 0)
        self.assertEqual(summary["paired_both_claims_approved"], 4)
        self.assertEqual(summary["choice_correct"], 4)
        self.assertEqual(summary["choice_answer_correct"], 4)
        self.assertEqual(summary["review_suggested_oracle_matches"], 8)
        self.assertEqual(summary["review_suggested_claim_matches"], 4)
        self.assertFalse(summary["training_admission"])
        self.assertEqual([x["candidate_a"] == m.v.arithmetic_answer(x)
                          for x in m.choice_cases()], [False, True, False, True])
        with patch.object(mtc, "inventory", side_effect=AssertionError(
                "verification contacted Ollama")):
            self.verify(self.manifest_digest())

    def test_owner_approval_is_mandatory(self):
        ident = self.freeze()
        with self.assertRaises((ValueError, OSError)):
            self.capture(ident)
        self.assertFalse(self.run_path.exists())
        self.assertFalse(self.approval_path.with_name(
            "approval.json.execution-claim.json").exists())
        self.assertEqual(self.calls, [])

    def test_approval_cannot_be_reused(self):
        ident = self.freeze()
        self.approve(ident)
        self.capture(ident)
        with self.assertRaises(FileExistsError):
            self.capture(ident)
        self.assertEqual(len(self.calls), 16)

    def test_model_drift_preserves_failed_first_call_and_consumes_authorization(self):
        ident = self.freeze()
        self.approve(ident)
        self.drift = True
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.capture(ident)
        records = mtc.read_attempts(self.run_path / "attempts.jsonl")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["call"]["error"], "model_digest_drift")
        self.assertFalse((self.run_path / "manifest.json").exists())
        self.assertTrue(self.approval_path.with_name(
            "approval.json.execution-claim.json").exists())
        self.drift = False
        with self.assertRaises(FileExistsError):
            self.capture(ident)
        self.assertEqual(len(self.calls), 1)

    def test_answer_truncation_is_invalid_not_correct(self):
        ident = self.freeze()
        self.approve(ident)
        self.length_reply = True
        self.capture(ident)
        s = mtc.strict_json(mtc.checked_file(self.run_path / "summary.json"))
        self.assertEqual(s["solve_valid_answers"], 0)
        self.assertEqual(s["solve_matches"], 0)
        self.assertEqual(s["review_valid_verdicts"], 0)
        self.assertEqual(s["review_invalid"], 8)
        self.assertEqual(s["choice_valid"], 0)
        self.verify(self.manifest_digest())

    def test_mutated_attempts_are_rejected(self):
        ident = self.freeze()
        self.approve(ident)
        self.capture(ident)
        attempts = self.run_path / "attempts.jsonl"
        with attempts.open("ab") as out:
            out.write(b'{"extra":true}\n')
        with self.assertRaisesRegex(ValueError, "member digest mismatch"):
            self.verify(self.manifest_digest())

    def test_wrong_manifest_external_identity_fails(self):
        ident = self.freeze()
        self.approve(ident)
        self.capture(ident)
        with self.assertRaisesRegex(ValueError, "external diagnostic manifest"):
            self.verify("sha256:" + "0"*64)

    def test_historical_validation_drift_prevents_execution(self):
        ident = self.freeze()
        self.approve(ident)
        with patch.object(m, "historical", return_value={
                **self.hist, "summary_sha256": "sha256:" + "f"*64}):
            with self.assertRaisesRegex(ValueError, "historical lineage mismatch"):
                self.capture(ident)
        self.assertFalse(self.run_path.exists())
        self.assertFalse(self.approval_path.with_name(
            "approval.json.execution-claim.json").exists())

    def test_tampered_frozen_schedule_is_rejected(self):
        ident = self.freeze()
        self.approve(ident)
        plan = mtc.strict_json(mtc.checked_file(self.plan_path / "plan.json"))
        plan["schedule"][0]["case"]["a"] = 900
        self.assertRaisesRegex(ValueError, "protocol drift", m.validate_plan, plan)
        with self.assertRaisesRegex(ValueError, "frozen diagnostic request identity"):
            changed = mtc.canonical(plan)
            (self.plan_path / "plan.json").write_bytes(changed)
            self.capture(ident)

    def test_source_closure_drift_invalidates_frozen_request(self):
        self.freeze()
        plan = mtc.strict_json(mtc.checked_file(self.plan_path / "plan.json"))
        with patch.object(m, "closure", return_value={"drift": "yes"}):
            with self.assertRaisesRegex(ValueError, "protocol drift"):
                m.validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
