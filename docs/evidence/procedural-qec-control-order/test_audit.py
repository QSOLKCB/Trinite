"""Standard-library evidence checks; never mutate either supplied producer.

Set TRINITE_HISTORICAL_SOURCE and TRINITE_CURRENT_SOURCE to clean pinned Git
checkouts, then run unittest discover here with -I -S -B (see README).
"""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("control_order_evidence_audit", HERE / "audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class ControlOrderAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("TRINITE_HISTORICAL_SOURCE", "TRINITE_CURRENT_SOURCE"):
            if not os.environ.get(name):
                raise RuntimeError("Required clean-checkout test input missing: " + name)
        cls.old = Path(os.environ["TRINITE_HISTORICAL_SOURCE"]).resolve()
        cls.new = Path(os.environ["TRINITE_CURRENT_SOURCE"]).resolve()
        audit.verify_source(cls.old, "historical")
        audit.verify_source(cls.new, "current")
        cls.historical = audit.run_worker(cls.old, "historical")
        cls.current = audit.run_worker(cls.new, "current")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="trinite-control-audit-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def disposable_source(self):
        # Local object sharing reads the originals; sparse checkout materializes
        # only src into a disposable repo, so mutation tests never edit them.
        root = self.root / "producer"
        subprocess.run(["git", "clone", "--shared", "--no-checkout", str(self.old), str(root)],
                       check=True, capture_output=True, timeout=60)
        audit.git(root, "sparse-checkout", "set", "--no-cone", "/src/", "/.gitignore", "/.gitattributes")
        audit.git(root, "checkout", "--detach", audit.PINS["historical"]["commit"])
        audit.verify_source(root, "historical")
        return root

    def retained_copy(self):
        path = self.root / "comparison.json"
        path.write_bytes((HERE / "comparison.json").read_bytes())
        path.with_name(path.name + ".sha256").write_bytes(
            (HERE / "comparison.json.sha256").read_bytes())
        return path

    def test_actual_two_checkout_audit(self):
        # Fresh source checks, both generators/admission operations and every
        # schedule are rerun; the class fixtures do not certify this test.
        reproduced = audit.audit(self.old, self.new)
        self.assertEqual(reproduced, audit.read_receipt(HERE / "comparison.json"))
        self.assertEqual(audit.identity(reproduced), "sha256:" + audit.RECEIPT_SHA256)

    def test_exact_retained_findings_and_blind_spot(self):
        result = audit.parse(audit.comparison(self.historical, self.current))
        self.assertEqual(result["dataset_rows"], 3876)
        self.assertEqual(result["changed_example_identity_count"], 3876)
        self.assertTrue(result["numerical_rows_equal"])
        self.assertTrue(result["only_example_and_generator_id_differ"])
        for index, expected in ((0, 32576), (2, 32521)):
            cell = result["cells"][index]
            self.assertEqual(cell["visits"], 32768)
            self.assertEqual(cell["numeric_visit_differences"], expected)
            self.assertTrue(cell["current_v2_function_matches_current_v3"])
        for index in (1, 3):
            self.assertFalse(result["cells"][index]["current_v2_function_matches_current_v3"])

    def test_wrong_commit_rejected_before_any_producer_execution(self):
        with patch.object(audit, "run_worker") as run:
            with self.assertRaisesRegex(audit.AuditError, "current source commit mismatch"):
                audit.audit(self.old, self.old)
            run.assert_not_called()

    def test_dirty_source_rejected_before_execution(self):
        root = self.disposable_source()
        path = root / "src/trinite/procedural_qec/data.py"
        path.write_bytes(path.read_bytes() + b"\n# changed\n")
        with patch.object(audit, "run_worker") as run:
            with self.assertRaisesRegex(audit.AuditError, "dirty"):
                audit.audit(root, self.new)
            run.assert_not_called()

    def test_assume_unchanged_cannot_hide_tampered_bytes(self):
        root = self.disposable_source()
        name = "src/trinite/contracts.py"
        audit.git(root, "update-index", "--assume-unchanged", name)
        path = root / name
        path.write_bytes(path.read_bytes() + b"\n# hidden edit\n")
        self.assertEqual(audit.git(root, "status", "--porcelain=v1"), b"")
        with self.assertRaisesRegex(audit.AuditError, "source bytes differ"):
            audit.verify_source(root, "historical")

    def test_ignored_bytecode_or_import_shadow_rejected(self):
        root = self.disposable_source()
        cache = root / "src/trinite/__pycache__"
        cache.mkdir()
        (cache / "contracts.cpython-311.pyc").write_bytes(b"untrusted ignored bytecode")
        self.assertEqual(audit.git(root, "status", "--porcelain=v1"), b"")
        with self.assertRaisesRegex(audit.AuditError, "unexpected or missing source"):
            audit.verify_source(root, "historical")

    def test_symlink_source_rejected(self):
        root = self.disposable_source()
        name = "src/trinite/contracts.py"
        path = root / name
        target = self.root / "contracts-copy.py"
        target.write_bytes(path.read_bytes())
        audit.git(root, "update-index", "--assume-unchanged", name)
        path.unlink()
        path.symlink_to(target)
        with self.assertRaisesRegex(audit.AuditError, "symlink"):
            audit.verify_source(root, "historical")

    def test_tampered_receipt_rejected_before_execution(self):
        receipt = self.retained_copy()
        receipt.write_bytes(receipt.read_bytes() + b"\n")
        with patch.object(audit, "run_worker") as run:
            with self.assertRaisesRegex(audit.AuditError, "receipt identity mismatch"):
                audit.audit(self.old, self.new, receipt)
            run.assert_not_called()

    def test_tampered_sidecar_rejected(self):
        receipt = self.retained_copy()
        receipt.with_name(receipt.name + ".sha256").write_bytes(b"0" * 64 + b"  comparison.json\n")
        with self.assertRaisesRegex(audit.AuditError, "sidecar mismatch"):
            audit.read_receipt(receipt)

    def test_changed_dataset_cannot_supply_its_own_identity(self):
        value = copy.deepcopy(self.historical)
        value["dataset"]["examples"][0]["answer"] = "wrong"
        value["manifest"]["dataset_identity"] = audit.identity(audit.canonical(value["dataset"]))
        with self.assertRaisesRegex(audit.AuditError, "dataset identity mismatch"):
            audit.validate_payload(value, "historical")

    def test_tampered_manifest_rejected(self):
        value = copy.deepcopy(self.historical)
        value["manifest"]["admission"]["outcome"] = "rejected"
        with self.assertRaisesRegex(audit.AuditError, "manifest identity mismatch"):
            audit.validate_payload(value, "historical")

    def test_duplicate_math_mapping_rejected(self):
        value = copy.deepcopy(self.historical["dataset"])
        row, other = value["examples"][:2]
        row.update(prompt=other["prompt"], answer=other["answer"], text=other["text"])
        with self.assertRaisesRegex(audit.AuditError, "non-bijective"):
            audit.row_map(value)

    def test_duplicate_identity_rejected(self):
        value = copy.deepcopy(self.historical["dataset"])
        value["examples"][0]["example_identity"] = value["examples"][1]["example_identity"]
        with self.assertRaisesRegex(audit.AuditError, "duplicate example identity"):
            audit.row_map(value)

    def test_schema_is_not_narrowed_to_prompt_answer(self):
        for change in ("add", "remove", "wrong-schema", "missing-row", "bad-key"):
            with self.subTest(change=change):
                value = copy.deepcopy(self.historical["dataset"])
                if change == "add":
                    value["examples"][0]["unexpected_metadata"] = "hidden"
                elif change == "remove":
                    del value["examples"][0]["family_id"]
                elif change == "wrong-schema":
                    value["schema"] = "other"
                elif change == "missing-row":
                    value["examples"].pop()
                else:
                    value["examples"][0]["prompt"] = ["unhashable"]
                with self.assertRaises(audit.AuditError):
                    audit.row_map(value)

    def test_normalization_ignores_exactly_two_top_level_fields(self):
        row = self.historical["dataset"]["examples"][0]
        self.assertEqual(set(row) - set(audit.normalized(row)), audit.IGNORED_ROW_FIELDS)
        for key in set(row) - audit.IGNORED_ROW_FIELDS:
            with self.subTest(key=key):
                altered = copy.deepcopy(row)
                altered[key] = {"changed": altered[key]}
                self.assertNotEqual(audit.normalized(row), audit.normalized(altered))
        altered = copy.deepcopy(row)
        altered["transformation_lineage"]["example_identity"] = "must-not-be-ignored"
        self.assertNotEqual(audit.normalized(row), audit.normalized(altered))

    def test_schedule_lengths_and_indices_fail_closed(self):
        for change in ("truncate", "extra", "negative-index", "bool-index", "out-of-range", "cell"):
            with self.subTest(change=change):
                value = copy.deepcopy(self.historical)
                selected = value["schedules"]["v2"][0]
                if change == "truncate":
                    selected.pop()
                elif change == "extra":
                    selected.append(0)
                elif change == "cell":
                    value["schedules"]["v2"].pop()
                else:
                    selected[0] = {"negative-index": -1, "bool-index": True,
                                   "out-of-range": audit.ROWS}[change]
                with self.assertRaises(audit.AuditError):
                    audit.validate_payload(value, "historical")

    def test_altered_valid_schedule_cannot_match_retained_receipt(self):
        value = copy.deepcopy(self.historical)
        value["schedules"]["v2"][0][0] = (value["schedules"]["v2"][0][0] + 1) % audit.ROWS
        self.assertNotEqual(audit.comparison(value, self.current),
                            audit.read_receipt(HERE / "comparison.json"))
        # This fake tests only the controller's rejection, never reconstruction
        # success; test_actual_two_checkout_audit executes both real producers.
        with patch.object(audit, "run_worker", side_effect=[value, self.current]):
            with self.assertRaisesRegex(audit.AuditError, "differs from retained receipt"):
                audit.audit(self.old, self.new)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(audit.AuditError, "duplicate JSON key"):
            audit.parse(b'{"schema":"a","schema":"b"}\n')

    def test_optimized_python_still_rejects_wrong_commit_without_success_output(self):
        process = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-O", str(HERE / "audit.py"),
             "--historical-source", str(self.new), "--current-source", str(self.new)],
            capture_output=True, timeout=60)
        self.assertEqual(process.returncode, 1)
        self.assertEqual(process.stdout, b"")
        self.assertIn(b"CONTROL AUDIT REJECTED", process.stderr)
        self.assertIn(b"historical source commit mismatch", process.stderr)


if __name__ == "__main__":
    unittest.main()
