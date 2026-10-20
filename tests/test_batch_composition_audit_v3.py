"""Fast auditor regressions; software-double CLI inputs are not model evidence."""
import ast
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from trinite.contracts import identity, json_bytes, parse_json
import audit_batch_composition_v3 as auditor

SCRIPT = ROOT / "scripts/audit_batch_composition_v3.py"
REVIEW = ROOT / "fixtures/batch-composition-v2/prepublication-review.json"


class AuditV3Tests(unittest.TestCase):
    def status(self, outcome="completed", code=0):
        return {"schema": "trinite.batch-composition-status.v2", "outcome": outcome,
                "exit_code": code, "wall_seconds_hex": (1.0).hex()}

    def test_no_optimization_sensitive_assertions(self):
        tree = ast.parse(SCRIPT.read_text())
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))

    def test_exact_status_types_and_runner_pairings(self):
        self.assertTrue(auditor.validate_status(self.status(), True))
        self.assertTrue(auditor.validate_status(self.status("completed_nonconformant", 2), False))
        for outcome, code in (("timeout", 124), ("launch_failed", 1), ("controller_failed", 1),
                              ("interrupted", 130), ("failed", 1), ("failed", -9)):
            self.assertFalse(auditor.validate_status(self.status(outcome, code), True))
        invalid = [self.status("timeout", 0), self.status("completed", 2),
                   self.status("failed", 0), self.status("failed", 2),
                   self.status("completed_nonconformant", 0), self.status("unknown", 1)]
        for code in (False, True, "0", 0.0):
            invalid.append(self.status(code=code))
        for wall in ("inf", "-0x1.0p+0", "0x1p+99999", 1, "1.0"):
            value = self.status()
            value["wall_seconds_hex"] = wall
            invalid.append(value)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(auditor.AuditError):
                auditor.validate_status(value, True)
        with self.assertRaises(auditor.AuditError):
            auditor.validate_status(self.status(), False)
        with self.assertRaises(auditor.AuditError):
            auditor.validate_status(self.status("completed_nonconformant", 2), True)

    def test_output_boundary_rejects_inside_alias_existing_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifact"
            root.mkdir()
            original = root / "existing"
            original.write_bytes(b"unchanged")
            alias = Path(tmp) / "alias"
            alias.symlink_to(root, target_is_directory=True)
            outside = Path(tmp) / "outside.json"
            self.assertEqual(auditor.output_destination(root, outside), outside)
            for output in (root / "new.json", alias / "new.json", original):
                with self.subTest(output=output), self.assertRaises(auditor.AuditError):
                    auditor.output_destination(root, output)
            outside.write_bytes(b"previous receipt")
            with self.assertRaises(auditor.AuditError):
                auditor.output_destination(root, outside)
            dangling = Path(tmp) / "dangling"
            dangling.symlink_to(Path(tmp) / "absent")
            with self.assertRaises(auditor.AuditError):
                auditor.output_destination(root, dangling)
            self.assertEqual(list(root.iterdir()), [original])
            self.assertEqual(original.read_bytes(), b"unchanged")
            self.assertEqual(outside.read_bytes(), b"previous receipt")

    def packet(self, root):
        # Deliberately minimal report: malformed transport must reject before the
        # strict numerical validator; this double cannot establish conformance.
        approved = parse_json(REVIEW.read_bytes(), canonical=True)
        original_request = parse_json((ROOT / "fixtures/batch-composition-v2/request.json").read_bytes(),
                                      canonical=True)
        request = {"schema": "trinite.batch-composition-request.v2",
                   "source": approved["source"], "protocol": original_request["protocol"]}
        report = {"schema": "trinite.batch-composition-report.v2", "conforms": True, "cells": []}
        request_raw, report_raw = json_bytes(request), json_bytes(report)
        (root / "batch-composition-request.json").write_bytes(request_raw)
        (root / "batch-composition-report.json").write_bytes(report_raw)
        attempt = root / "verify-attempt"
        attempt.mkdir()
        request_id, report_id = identity(request_raw), identity(report_raw)
        invocation = {"schema": "trinite.batch-composition-invocation.v2", "action": "verify",
                      "request_identity": request_id, "expected_identity": report_id,
                      "deadline_seconds": 300, "final_name": "batch-composition-report.json"}
        receipt = {"schema": "trinite.batch-composition-replay-receipt.v2",
                   "request_identity": request_id, "expected_identity": report_id,
                   "actual_identity": report_id, "exact_match": True, "conforms": True,
                   "scope": "same-environment complete inference replay, not optimizer replay"}
        (attempt / "invocation.json").write_bytes(json_bytes(invocation))
        (attempt / "status.json").write_bytes(json_bytes(self.status()))
        (attempt / "replay-receipt.json").write_bytes(json_bytes(receipt))
        return attempt, receipt, invocation

    def test_replay_gate_rejects_false_match_wrong_identity_and_boolean_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, receipt, invocation = self.packet(Path(tmp))
            args = (invocation, receipt["request_identity"], receipt["expected_identity"], True)
            auditor.validate_replay_receipt(receipt, *args)
            for field, value in (("exact_match", False), ("exact_match", 1),
                                 ("actual_identity", "sha256:" + "0" * 64), ("conforms", 1)):
                bad = copy.deepcopy(receipt)
                bad[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(auditor.AuditError):
                    auditor.validate_replay_receipt(bad, *args)

    def test_corrected_receipts_bind_unchanged_hosted_evidence_and_auditor(self):
        folder = ROOT / "fixtures/batch-composition-audit-v3"
        raw = (folder / "commitment.json").read_bytes()
        self.assertEqual(identity(raw), "sha256:b84dd07bcc650c134ebac330068ffc5423cb8cd58056ea5540c9cc2dbc70c98f")
        commitment = parse_json(raw, canonical=True)
        self.assertEqual(commitment["schema"], "trinite.batch-offline-audit-correction.v3")
        self.assertIs(commitment["model_inference_executed"], False)
        self.assertIs(commitment["numerical_producer_unchanged"], True)
        self.assertEqual(set(p.name for p in folder.iterdir()), set(commitment["files"]) | {"commitment.json"})
        for label in ("auditor", "approved_review", "historical_auditor"):
            self.assertEqual(identity((ROOT / commitment[label + "_path"]).read_bytes()),
                             commitment[label + "_identity"])
        historical_raw = (ROOT / "fixtures/batch-composition-v2/commitment.json").read_bytes()
        self.assertEqual(identity(historical_raw), commitment["historical_v2_commitment_identity"])
        historical = parse_json(historical_raw, canonical=True)
        self.assertEqual(set(commitment["matrix"]), {"3.11", "3.12", "3.13"})
        for version, entry in commitment["matrix"].items():
            original = historical["matrix"][version]
            raw = (folder / entry["receipt_path"]).read_bytes()
            self.assertEqual(identity(raw), entry["receipt_identity"])
            self.assertEqual(identity(raw), commitment["files"][entry["receipt_path"]])
            receipt = parse_json(raw, canonical=True)
            self.assertEqual(receipt["schema"], "trinite.independent-batch-ci-audit.v3")
            self.assertEqual(receipt["auditor_identity"], commitment["auditor_identity"])
            self.assertEqual(receipt["approved_review_identity"], commitment["approved_review_identity"])
            for name in ("report_identity", "request_identity"):
                self.assertEqual(receipt[name], original[name])
                self.assertEqual(entry[name], original[name])
            self.assertEqual(receipt["all_artifact_file_hashes"],
                             {name: info["identity"] for name, info in original["members"].items()})
            self.assertIs(receipt["read_only_inputs_unchanged"], True)
            self.assertIs(receipt["local_inference_executed"], False)
            self.assertIs(receipt["strict_recorded_report_validation"]["conforms"], True)
            self.assertEqual(receipt["same_prefix_vector_comparisons"], 10920)
            self.assertEqual(receipt["violating_coordinates"], 0)
            self.assertEqual(len(receipt["attempts"]), 2)
            for attempt in receipt["attempts"]:
                self.assertEqual((attempt["outcome"], attempt["exit_code"]), ("completed", 0))
                if attempt["action"] == "verify":
                    self.assertIs(attempt["exact_replay"], True)

    def test_real_cli_rejects_transport_tampering_under_all_optimization_modes(self):
        for flags, optimization in (([], None), (["-O"], None), ([], "1")):
            for defect, message in (("receipt", "replay report identity mismatch"),
                                    ("status", "inconsistent status outcome/exit-code pairing"),
                                    ("output", "output must be outside artifact directory")):
                with self.subTest(flags=flags, env=optimization, defect=defect):
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp) / "artifact"
                        root.mkdir()
                        attempt, receipt, _ = self.packet(root)
                        output = Path(tmp) / "audit.json"
                        if defect == "receipt":
                            receipt["exact_match"] = False
                            receipt["actual_identity"] = "sha256:" + "0" * 64
                            (attempt / "replay-receipt.json").write_bytes(json_bytes(receipt))
                        elif defect == "status":
                            (attempt / "status.json").write_bytes(json_bytes(self.status("timeout", 0)))
                        else:
                            output = root / "audit.json"
                        before = auditor.inventory(root)
                        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                        env.pop("PYTHONOPTIMIZE", None)
                        if optimization is not None:
                            env["PYTHONOPTIMIZE"] = optimization
                        result = subprocess.run(
                            [sys.executable, "-B", *flags, str(SCRIPT), "--artifact-dir", str(root),
                             "--source-root", str(ROOT), "--approved-review", str(REVIEW), "--output", str(output)],
                            env=env, capture_output=True, text=True, timeout=30)
                        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                        self.assertIn(message, result.stderr)
                        self.assertFalse(output.exists())
                        self.assertEqual(auditor.inventory(root), before)


if __name__ == "__main__":
    unittest.main()
