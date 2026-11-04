"""TT-001 local, standard-library-only admission and custody regressions."""
import copy
from pathlib import Path
import tempfile
import unittest

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from teacher_lab_v1 import core


class TeacherLabTests(unittest.TestCase):
    def test_default_preflight_reuses_authoritative_reference(self):
        plan, config, catalog = core.inspect()
        self.assertEqual(core.ModelConfig.from_dict(config).parameter_count, 1_247_232)
        self.assertEqual(plan["parameter_budget"], 2_000_000)
        self.assertEqual(len(catalog["sources"]), 4)

    def test_no_train_or_admission_authority(self):
        plan = core.source_json(core.DEFAULT_PLAN)
        for field in ("training_authorized", "data_admitted"):
            bad = dict(plan)
            bad[field] = True
            with self.assertRaises(ContractError):
                core.validate_plan(bad)

    def test_no_deceptive_dtype_or_model_lanes(self):
        plan = core.source_json(core.DEFAULT_PLAN)
        for value in ("ternary", "four-state", "", "dense-gpu"):
            with self.subTest(value=value):
                with self.assertRaises(ContractError):
                    core.validate_plan(dict(plan, lane=value))

    def test_budget_and_type_rejection(self):
        plan = core.source_json(core.DEFAULT_PLAN)
        for val in (True, 0, 2_000_001, "2000000", 0.5):
            with self.subTest(val=val):
                with self.assertRaises(ContractError):
                    core.validate_plan(dict(plan, parameter_budget=val))

    def test_catalog_rejects_implied_rights_admission(self):
        catalog = core.source_json("configs/teacher/source-candidates-v1.json")
        for status in ("ADMITTED", "APPROVED", "UNKNOWN"):
            bad = copy.deepcopy(catalog)
            bad["sources"][0]["rights_status"] = status
            with self.assertRaises(ContractError):
                core.validate_catalog(bad)

    def test_catalog_rejects_duplicates_and_unpinned_refs(self):
        catalog = core.source_json("configs/teacher/source-candidates-v1.json")
        bad = copy.deepcopy(catalog)
        bad["sources"][1]["id"] = bad["sources"][0]["id"]
        with self.assertRaises(ContractError):
            core.validate_catalog(bad)
        bad = copy.deepcopy(catalog)
        bad["sources"][0]["revision"] = "main"
        with self.assertRaises(ContractError):
            core.validate_catalog(bad)

    def test_source_path_traversal_and_syntax(self):
        for path in ("../secrets", "/etc/passwd", "docs//DATA.md",
                     "docs/./DATA.md", "docs/../DATA.md", "docs\\DATA.md"):
            with self.subTest(path=path):
                with self.assertRaises(ContractError):
                    core.repo_file(path)

    def test_source_canonical_rejects_duplicate_json_keys(self):
        with self.assertRaises(ContractError):
            parse_json(b'{"schema":"x","schema":"y"}')

    def test_preview_is_exact_but_explicitly_not_training_data(self):
        obj = core.preview_formal()
        self.assertEqual(len(obj["examples"]), 9)
        self.assertEqual(obj["split"], "NONE")
        self.assertIn("NOT_ADMITTED", obj["status"])
        self.assertEqual(len({v["family"] for v in obj["examples"]}), 9)
        for e in obj["examples"]:
            self.assertEqual((e["sum"] - e["a"] - e["b"]) % 3, 0)

    def test_fresh_freeze_and_independent_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "frozen"
            digest = core.freeze(target)
            self.assertEqual(digest, identity((target / "request.json").read_bytes()))
            result = core.verify(target, digest)
            self.assertTrue(result["verified"])
            self.assertFalse(result["training_authorized"])
            self.assertFalse(result["evidence_ready"])
            with self.assertRaises(ContractError):
                core.freeze(target)
            with self.assertRaises(ContractError):
                core.verify(target, "sha256:" + "0" * 64)

    def test_tampered_rights_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "frozen"
            digest = core.freeze(target)
            path = target / "dataset-rights.json"
            record = parse_json(path.read_bytes(), canonical=True)
            record["status"] = "ADMITTED"
            path.write_bytes(json_bytes(record))
            with self.assertRaises(ContractError):
                core.verify(target, digest)

    def test_unexpected_packet_members_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "frozen"
            digest = core.freeze(target)
            (target / "extra").write_text("injected")
            with self.assertRaises(ContractError):
                core.verify(target, digest)

    def test_rights_are_separate_from_code_licence(self):
        req, dataset, weights = core.request_and_rights()
        self.assertEqual(weights["code_license"], "MPL-2.0")
        self.assertEqual(weights["weights_license"], "UNDECIDED")
        self.assertEqual(dataset["status"], "NOT_CREATED_NOT_ADMITTED")
        self.assertEqual(req["optimizer_updates"], 0)
        self.assertEqual(req["fp32_adamw_state_bytes_lower_bound"],
                         16 * req["parameter_count"])

    def test_source_revalidation_detects_changed_closure(self):
        req, _, _ = core.request_and_rights()
        self.assertIn("src/trinite/contracts.py",
                      {item["path"] for item in req["source_closure"]})
        self.assertIn("docs/DATA.md",
                      {item["path"] for item in req["source_closure"]})
        self.assertEqual(len(req["source_closure"]),
                         len({item["path"] for item in req["source_closure"]}))


if __name__ == "__main__":
    unittest.main()
