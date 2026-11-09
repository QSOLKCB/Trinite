"""Synthetic offline contracts: NO real model download, inference, trainer or GPU."""
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
    "recruitment", ROOT / "scripts" / "mtc_teacher_recruitment_001.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
mtc = m.mtc
HASHES = {m.CANDIDATES[0]: "1"*64, m.CANDIDATES[1]: "2"*64}


class RecruitmentContracts(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.teacher_run = self.home / "teacher"
        self.baseline_run = self.home / "principal"
        self.validation_run = self.home / "validation"
        self.cause_run = self.home / "examiner-cause"
        for p in (self.teacher_run, self.baseline_run, self.validation_run, self.cause_run):
            p.mkdir(mode=0o700)
        self.parent = {
            "principal": {"name": "qwen3.5:27b", "digest": "sha256:"+"3"*64},
            "summary_sha256": "sha256:"+"4"*64,
            "prior_teacher_manifest_identity": "sha256:"+"5"*64,
            "prior_principal_manifest_identity": "sha256:"+"6"*64,
            "examiner_cause_summary_sha256": "sha256:"+"7"*64,
        }
        self.candidate = m.CANDIDATES[0]
        self.folder = self.home / "phi4"
        self.folder.mkdir(mode=0o700)
        self.plan_path = self.folder / "plan"
        self.run_path = self.folder / "capture"
        self.approval_path = self.folder / "owner-approval.json"
        self.calls = []
        self.drift = False
        self.truncated = False
        self.patches = [
            patch.object(m, "parent_evidence", return_value=self.parent),
            patch.object(m.cause, "source_commit", return_value="a"*40),
            patch.object(mtc, "inventory", return_value=[
                {"name": x, "digest": dg} for x, dg in HASHES.items()
            ]),
            patch.object(m.r2, "record_call", side_effect=self.fake_call),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def parents(self):
        return dict(teacher_run=self.teacher_run,
                    baseline_run=self.baseline_run,
                    validation_run=self.validation_run, cause_run=self.cause_run)

    def fake_call(self, body, model):
        self.assertEqual(body["model"], self.candidate)
        self.assertEqual(body["format"], "json")
        self.assertNotIn("think", body)  # No silent reasoning disable.
        self.assertEqual(body["options"], m.OPTIONS)
        self.assertEqual(body["keep_alive"], 0)
        self.assertEqual(model["digest"], "sha256:"+HASHES[self.candidate])
        self.calls.append(body)
        user = json.loads(body["messages"][1]["content"])
        task = next(p for p in m.cause.probes()
                    if m.cause.problem(p) == user["problem"])
        exact = m.v.arithmetic_answer(task)
        if "claimed_answer" in user:
            # Synthetic over-approving examiner must count incorrect as FP.
            result = {"verdict":"correct", "feedback":"Recompute.",
                      "suggested_answer": exact}
        elif "candidate_A" in user:
            result = {"choice": "A" if user["candidate_A"] == exact else "B",
                      "answer": exact}
        else:
            result = {"answer": exact}
        obj = {
            "model": self.candidate, "done": True,
            "done_reason": "length" if self.truncated else "stop",
            "message": {"role": "assistant", "content": json.dumps(result),
                        "thinking": "synthetic reasoning recorded separately"},
        }
        return {
            "request_sha256": mtc.sha(mtc.canonical(body)),
            "response_base64": base64.b64encode(mtc.canonical(obj)).decode("ascii"),
            "http_status": 200,
            "error": "model_digest_drift" if self.drift else None,
            "model_digest_after": "sha256:"+"f"*64 if self.drift else model["digest"],
        }

    def freeze(self):
        m.plan_command(SimpleNamespace(
            **self.parents(), candidate=self.candidate, out=self.plan_path))
        return mtc.sha(mtc.checked_file(self.plan_path/"plan.json"))

    def approve(self, identity):
        plan = m.validate_plan(mtc.strict_json(
            mtc.checked_file(self.plan_path/"plan.json")))
        mtc.write_new(self.approval_path, mtc.canonical(
            m.authorization_value(plan, identity, self.approval_path, self.run_path)))

    def capture(self, identity):
        m.capture_command(SimpleNamespace(
            **self.parents(), plan=self.plan_path/"plan.json",
            request_identity=identity, owner_approval=self.approval_path,
            out=self.run_path))

    def receipt(self):
        return mtc.sha(mtc.checked_file(self.run_path/"manifest.json"))

    def verify(self, identity):
        m.verify_command(SimpleNamespace(
            **self.parents(), run=self.run_path, manifest_identity=identity))

    def test_plan_only_and_no_model_calls(self):
        identity = self.freeze()
        plan = mtc.strict_json(mtc.checked_file(self.plan_path/"plan.json"))
        self.assertEqual(plan["candidate"]["name"], m.CANDIDATES[0])
        self.assertEqual(plan["candidate"]["digest"], "sha256:"+HASHES[self.candidate])
        self.assertEqual(identity, mtc.sha(mtc.canonical(plan)))
        self.assertEqual([x["lane"] for x in plan["schedule"]],
                         ["solve"]*4 + ["review"]*8 + ["choice"]*4)
        self.assertEqual(len(self.calls), 0)
        self.assertFalse(plan["training_admission"])

    def test_phi4_full_synthetic_capture_and_offline_verify(self):
        identity = self.freeze()
        self.approve(identity)
        self.capture(identity)
        self.assertEqual(len(self.calls), 16)
        record = mtc.strict_json(mtc.checked_file(self.run_path/"summary.json"))
        self.assertEqual(record["solve_correct"], 4)
        self.assertEqual(record["false_positive"], 4)
        self.assertEqual(record["true_positive"], 4)
        self.assertEqual(record["true_negative"], 0)
        self.assertEqual(record["pair_both_approved"], 4)
        self.assertEqual(record["choice_correct"], 4)
        self.assertFalse(record["automatic_promotion"])
        self.assertFalse(record["training_admission"])
        with patch.object(mtc, "inventory", side_effect=AssertionError(
                "offline verifier must never query Ollama")):
            self.verify(self.receipt())

    def test_glm_community_quant_is_distinct_candidate_and_digest(self):
        self.candidate = m.CANDIDATES[1]
        self.folder = self.home / "glm-q3"
        self.folder.mkdir(mode=0o700)
        self.plan_path = self.folder/"plan"
        self.run_path = self.folder/"capture"
        self.approval_path = self.folder/"owner-approval.json"
        ident = self.freeze()
        plan = mtc.strict_json(mtc.checked_file(self.plan_path/"plan.json"))
        self.assertEqual(plan["candidate_family"], "glm47_flash_community_q3")
        self.assertEqual(plan["candidate"]["digest"], "sha256:"+HASHES[self.candidate])
        self.approve(ident)
        self.capture(ident)
        self.verify(self.receipt())
        self.assertEqual(len(self.calls), 16)

    def test_reject_unapproved_model_or_synthetic_unsupported_tag(self):
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            m.value_of_plan("qwen3.5:27b", **self.parents())
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            m.value_of_plan("glm-4.7-flash:q4_K_M", **self.parents())

    def test_no_implicit_owner_permission(self):
        ident = self.freeze()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.capture(ident)
        self.assertEqual(len(self.calls), 0)
        self.assertFalse(self.run_path.exists())

    def test_approval_single_use_and_no_rerun(self):
        ident = self.freeze()
        self.approve(ident)
        self.capture(ident)
        with self.assertRaises(FileExistsError):
            self.capture(ident)
        self.assertEqual(len(self.calls), 16)

    def test_model_digest_drift_preserves_single_failure_and_one_use_claim(self):
        ident = self.freeze()
        self.approve(ident)
        self.drift = True
        with self.assertRaisesRegex(ValueError, "identity drift"):
            self.capture(ident)
        self.assertEqual(len(mtc.read_attempts(self.run_path/"attempts.jsonl")), 1)
        self.assertTrue((self.folder/"owner-approval.json.execution-claim.json").exists())
        self.assertFalse((self.run_path/"manifest.json").exists())
        with self.assertRaises(FileExistsError):
            self.capture(ident)
        self.assertEqual(len(self.calls), 1)

    def test_truncated_replies_not_scored_as_valid(self):
        ident = self.freeze()
        self.approve(ident)
        self.truncated = True
        self.capture(ident)
        result = mtc.strict_json(mtc.checked_file(self.run_path/"summary.json"))
        self.assertEqual(result["solve_valid"], 0)
        self.assertEqual(result["review_valid"], 0)
        self.assertEqual(result["choice_valid"], 0)
        self.assertEqual(result["review_invalid"], 8)
        self.verify(self.receipt())

    def test_wrong_receipt_identity_rejected(self):
        ident = self.freeze()
        self.approve(ident)
        self.capture(ident)
        with self.assertRaisesRegex(ValueError, "external recruitment manifest"):
            self.verify("sha256:"+"0"*64)

    def test_attempt_tampering_rejected(self):
        ident = self.freeze()
        self.approve(ident)
        self.capture(ident)
        with (self.run_path/"attempts.jsonl").open("ab") as stream:
            stream.write(b'{"mutated":true}\n')
        with self.assertRaisesRegex(ValueError, "member digest mismatch"):
            self.verify(self.receipt())

    def test_historical_parent_drift_blocks_capture_before_claim(self):
        ident = self.freeze()
        self.approve(ident)
        changed = {**self.parent, "summary_sha256": "sha256:"+"d"*64}
        with patch.object(m, "parent_evidence", return_value=changed):
            with self.assertRaisesRegex(ValueError, "lineage drift"):
                self.capture(ident)
        self.assertFalse(self.run_path.exists())
        self.assertFalse((self.folder/"owner-approval.json.execution-claim.json").exists())

    def test_source_closure_drift_detected(self):
        self.freeze()
        plan = mtc.strict_json(mtc.checked_file(self.plan_path/"plan.json"))
        with patch.object(m, "closure", return_value={"unreviewed_change": "sha256:0"}):
            with self.assertRaisesRegex(ValueError, "policy drift"):
                m.validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
