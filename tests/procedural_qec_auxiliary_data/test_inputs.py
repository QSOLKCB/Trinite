"""Actual historical-source admission and data-only non-displacing boundaries.

No numerical backend is imported. Set TRINITE_HISTORICAL_SOURCE to an original
clean 93522bf checkout. Without it, the tests create a disposable sparse
worktree from the same local Git object; missing objects fail instead of
silently replacing real source admission with a mocked success.
"""
import copy
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.procedural_qec_auxiliary import inputs
from trinite.procedural_qec_auxiliary import scheduler

ROOT = Path(__file__).resolve().parents[2]
PLANNING = ROOT / "docs/evidence/procedural-qec-nondisplacing-proposal/planning-counts.json"
PLANNING_IDENTITY = "sha256:e8cc15200ed9bf6ae3820674f4383d3c67b54ab608048b1309d861598b232555"


def historical_fixture(owner):
    configured = os.environ.get("TRINITE_HISTORICAL_SOURCE")
    if configured:
        return Path(configured)
    temporary = tempfile.TemporaryDirectory(prefix="trinite-auxiliary-history-test-")
    owner.addClassCleanup(temporary.cleanup)
    root = Path(temporary.name) / "historical"
    command = ["git", "--no-replace-objects", "-C", str(ROOT)]
    result = subprocess.run(command + ["cat-file", "-t", inputs.HISTORICAL_COMMIT],
                            capture_output=True, timeout=30)
    if result.returncode or result.stdout.strip() != b"commit":
        raise RuntimeError("Real pinned historical source required: fetch 93522bf or set "
                           "TRINITE_HISTORICAL_SOURCE to its clean checkout")
    subprocess.run(command + ["worktree", "add", "--detach", "--no-checkout", str(root),
                               inputs.HISTORICAL_COMMIT], check=True,
                   capture_output=True, timeout=60)
    owner.addClassCleanup(subprocess.run, command + ["worktree", "remove", "--force", str(root)],
                          check=False, capture_output=True, timeout=60)
    subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", "--no-cone",
                    "/src/", "/.gitignore", "/.gitattributes"], check=True,
                   capture_output=True, timeout=60)
    subprocess.run(["git", "-C", str(root), "checkout", "--detach", inputs.HISTORICAL_COMMIT],
                   check=True, capture_output=True, timeout=60)
    return root


class AuxiliaryHistoricalInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.historical = historical_fixture(cls)
        cls.materialized = inputs.materialize_inputs(cls.historical)
        cls.audit, _, cls.comparison_raw = inputs._load_audit(ROOT)
        cls.historical_dataset = parse_json(cls.materialized.historical_dataset_raw, canonical=True)
        cls.prospective_dataset = parse_json(cls.materialized.dataset_raw, canonical=True)

    def test_actual_source_bound_admission_and_distinct_original_identities(self):
        value = self.materialized
        inputs.revalidate(value)
        self.assertEqual(identity(value.historical_dataset_raw), inputs.HISTORICAL_DATASET)
        self.assertEqual(identity(value.historical_manifest_raw), inputs.HISTORICAL_MANIFEST)
        self.assertEqual(len(value.items), 3876)
        self.assertNotEqual(value.dataset_raw, value.historical_dataset_raw)
        receipt = value.receipt
        self.assertEqual(receipt["audit_script_identity"], inputs.AUDIT_IDENTITY)
        self.assertEqual(receipt["original_comparison_identity"], inputs.COMPARISON_IDENTITY)
        self.assertEqual(receipt["historical_git_tree"], inputs.HISTORICAL_TREE)
        self.assertEqual(receipt["ignored_top_level_row_fields"],
                         ["example_identity", "generator_revision"])
        self.assertEqual(len(receipt["mapping"]), 3876)
        self.assertEqual(len({m["prospective_index"] for m in receipt["mapping"]}), 3876)
        self.assertTrue(all(m["historical_identity"] != m["prospective_identity"]
                            for m in receipt["mapping"]))
        self.assertIn("src/trinite/procedural_qec_auxiliary/inputs.py",
                      receipt["prospective_source"])

    def test_bijection_compares_every_field_and_the_full_schema(self):
        row = self.prospective_dataset["examples"][0]
        for key in set(row) - inputs.IGNORED_ROW_FIELDS:
            changed = copy.deepcopy(self.prospective_dataset)
            changed["examples"][0][key] = {"changed": row[key]}
            with self.subTest(field=key), self.assertRaises(ContractError):
                inputs._bijection(self.audit, self.historical_dataset, changed)
        for mutation in ("extra-field", "missing-field", "wrong-schema", "missing-row",
                         "extra-row", "duplicate-row", "duplicate-identity", "nested-identity"):
            changed = copy.deepcopy(self.prospective_dataset)
            row = changed["examples"][0]
            if mutation == "extra-field":
                row["extra"] = 0
            elif mutation == "missing-field":
                del row["family_id"]
            elif mutation == "wrong-schema":
                changed["schema"] = "other"
            elif mutation == "missing-row":
                changed["examples"].pop()
            elif mutation == "extra-row":
                changed["examples"].append(copy.deepcopy(row))
            elif mutation == "duplicate-row":
                changed["examples"][1] = copy.deepcopy(row)
            elif mutation == "duplicate-identity":
                row["example_identity"] = changed["examples"][1]["example_identity"]
            else:
                row["transformation_lineage"]["example_identity"] = "not-ignored"
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                inputs._bijection(self.audit, self.historical_dataset, changed)

    def test_materialized_accessors_cannot_mutate_retained_inputs(self):
        value = self.materialized
        changed = value.items
        changed[0]["values"].append(999)
        changed[0]["split"] = "test"
        receipt = value.receipt
        receipt["mapping"][0]["prospective_index"] = -1
        self.assertNotEqual(changed, value.items)
        self.assertNotEqual(receipt, value.receipt)
        with self.assertRaises(FrozenInstanceError):
            value.dataset_raw = b"forged"
        with self.assertRaisesRegex(ContractError, "materialize_inputs"):
            inputs.MaterializedInputs()
        for supplied in ({"admitted": True}, value.receipt, value.items, None):
            with self.assertRaisesRegex(ContractError, "actual admitted"):
                scheduler.schedule(supplied, 0, "extra-final", 1)

    def test_revalidation_rejects_source_changes_without_reusing_a_pass(self):
        # Controller failure only; actual admission above is never mocked.
        snapshot = inputs._source_snapshot()
        changed = dict(snapshot)
        changed["src/trinite/procedural_qec/data.py"] += b"\n# mutation\n"
        with patch.object(inputs, "_source_snapshot", return_value=changed), \
                self.assertRaisesRegex(ContractError, "source changed"):
            inputs.revalidate(self.materialized)
        with patch.object(inputs, "materialize_inputs", side_effect=AssertionError("reconstruction")):
            inputs.revalidate(self.materialized)

    def test_original_payload_order_cannot_be_replaced_with_current_order(self):
        payload = parse_json(self.materialized.historical_payload_raw, canonical=True)
        payload["schedules"]["v2"][0][0:2] = reversed(payload["schedules"]["v2"][0][0:2])
        with self.assertRaisesRegex(ContractError, "historical sequence"):
            inputs._validate_historical(self.audit, payload,
                                       parse_json(self.comparison_raw, canonical=True))

    def test_changed_audit_rejected_before_any_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            path = repository / inputs.AUDIT_RELATIVE_PATH
            path.parent.mkdir(parents=True)
            path.write_bytes((ROOT / inputs.AUDIT_RELATIVE_PATH).read_bytes() + b"\n# changed\n")
            with patch.object(inputs.subprocess, "run") as process, \
                    self.assertRaisesRegex(ContractError, "pinned historical audit changed"):
                inputs.materialize_inputs(self.historical, repository=repository)
            process.assert_not_called()

    def test_wrong_historical_producer_rejects_before_current_execution(self):
        with patch.object(inputs, "_run_current") as worker, \
                self.assertRaisesRegex(ContractError, "historical source commit mismatch"):
            inputs.materialize_inputs(ROOT)
        worker.assert_not_called()

    def test_import_and_actual_materialization_need_no_numerical_backend(self):
        program = (
            "import sys;sys.path.insert(0,sys.argv[1]);"
            "from trinite.procedural_qec_auxiliary.inputs import materialize_inputs;"
            "from trinite.procedural_qec_auxiliary.scheduler import schedule;"
            "x=materialize_inputs(sys.argv[2]);schedule(x,0,'extra-final',1);"
            "bad=[n for n in sys.modules if n.split('.')[0] in ('torch','numpy')];"
            "print(bad);sys.exit(bool(bad))")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", program,
                                 str(ROOT / "src"), str(self.historical)],
                                capture_output=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout, b"[]\n")


class AuxiliaryHistoricalScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.historical = historical_fixture(cls)
        cls.materialized = inputs.materialize_inputs(cls.historical)
        planning_raw = PLANNING.read_bytes()
        if identity(planning_raw) != PLANNING_IDENTITY:
            raise ContractError("independently retained planning receipt changed")
        cls.planning = parse_json(planning_raw, canonical=True)

    def test_all_four_complete_schedules_reproduce_preoutcome_planning(self):
        for expected in self.planning["cells"]:
            seed, arm = expected["seed"], expected["arm"]
            with self.subTest(seed=seed, arm=arm):
                actual = scheduler.schedule_receipt(self.materialized, seed, arm)
                for group in ("primary", "extra", "combined"):
                    for key in ("totals", "by_kind_and_mode", "original_row_sequence_identity",
                                "normalized_row_sequence_identity"):
                        self.assertEqual(actual[group][key], expected[group][key])
                for key in ("native_updates", "original_audit_primary_sequence_identity",
                            "padded_encoded_positions", "model_input_positions",
                            "within_existing_target_ceiling"):
                    self.assertEqual(actual[key], expected[key])
                self.assertEqual(actual["selected_primary_sequence_identity"],
                                 actual["original_audit_primary_sequence_identity"])

    def test_each_exact_primary_batch_is_shared_and_historical(self):
        for seed in (0, 1):
            control = scheduler.schedule(self.materialized, seed, "extra-final")
            treatment = scheduler.schedule(self.materialized, seed, "extra-procedural")
            indices = self.materialized.primary_indices(seed)
            rows = self.materialized.items
            for macro in range(4096):
                original = [rows[index] for index in indices[macro * 8:(macro + 1) * 8]]
                self.assertEqual(control[macro * 16:macro * 16 + 8], original)
                self.assertEqual(treatment[macro * 16:macro * 16 + 8], original)
                self.assertTrue(all(row["mode"] == "final" for row in original))

    def test_historical_ranks_and_independent_cursors_cover_then_wrap(self):
        for arm in scheduler.ARMS:
            selected = scheduler.schedule(self.materialized, 0, arm)
            seen = {}
            original = self.materialized.historical_items
            mapping = {m["historical_identity"]: m["prospective_identity"]
                       for m in self.materialized.receipt["mapping"]}
            for macro in range(4096):
                kind, mode = scheduler.auxiliary_stream(macro)
                key = (kind, "final") if arm == "extra-final" else (kind, mode)
                seen.setdefault(key, []).extend(row["example_identity"]
                    for row in selected[macro * 16 + 8:macro * 16 + 16])
            for key, sequence in seen.items():
                ranked = sorted([row for row in original if (row["kind"], row["mode"]) == key],
                    key=lambda row: identity(json_bytes({"seed": 0, "item": row["example_identity"]})))
                expected = [mapping[row["example_identity"]] for row in ranked]
                self.assertEqual(sequence, [expected[index % len(expected)] for index in range(len(sequence))])
                self.assertEqual(len(set(sequence)), len(expected))
            exposure = scheduler.coverage(self.materialized, selected, arm)
            if arm == "extra-procedural":
                for key in ("fold-sum/prefix", "fold-sum/trace", "gf3/prefix", "gf3/trace"):
                    self.assertEqual(exposure[key]["visits"], 8192)
                    self.assertEqual(exposure[key]["distinct_unvisited"], 0)

    def test_cadence_and_bounded_partial_rounds_are_prefix_consistent(self):
        self.assertEqual([scheduler.auxiliary_stream(m) for m in range(4)],
            [("fold-sum", "prefix"), ("gf3", "prefix"), ("fold-sum", "trace"), ("gf3", "trace")])
        for arm in scheduler.ARMS:
            selected = scheduler.schedule(self.materialized, 1, arm, 9)
            for rounds in (1, 2, 4, 7):
                self.assertEqual(scheduler.schedule(self.materialized, 1, arm, rounds), selected[:rounds * 16])
        for rounds in (0, -1, 4097, 8192, True, 1.0, "1"):
            with self.subTest(rounds=rounds), self.assertRaises(ContractError):
                scheduler.schedule(self.materialized, 0, "extra-final", rounds)
        for seed in (-1, 2, True, 0.0):
            with self.subTest(seed=seed), self.assertRaises(ContractError):
                scheduler.schedule(self.materialized, seed, "extra-final", 1)
        for arm in ("final-only", "procedural", "unknown", None):
            with self.subTest(arm=arm), self.assertRaises(ContractError):
                scheduler.schedule(self.materialized, 0, arm, 1)

    def test_exposure_rejects_heldout_altered_and_control_procedural_rows(self):
        selected = scheduler.schedule(self.materialized, 0, "extra-final", 4)
        for mutation in ("heldout", "changed-answer", "extra-field", "unknown-identity"):
            changed = copy.deepcopy(selected)
            if mutation == "heldout":
                changed[0]["split"] = "test"
            elif mutation == "changed-answer":
                changed[0]["answer"] += "0"
            elif mutation == "extra-field":
                changed[0]["unexpected"] = True
            else:
                changed[0]["example_identity"] = "sha256:" + "0" * 64
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ContractError, "altered or unadmitted"):
                scheduler.coverage(self.materialized, changed, "extra-final")
        treatment = scheduler.schedule(self.materialized, 0, "extra-procedural", 4)
        with self.assertRaisesRegex(ContractError, "cannot enter extra-final"):
            scheduler.coverage(self.materialized, treatment, "extra-final")

    def test_streamed_commitment_matches_the_retained_canonical_encoding(self):
        values = ["identity", {"nested": [1, "é", None]}, []]
        self.assertEqual(scheduler._sequence_identity(json_bytes(v)[:-1] for v in values),
                         identity(json_bytes(values)))
        self.assertEqual(scheduler._sequence_identity(iter(())), identity(json_bytes([])))


if __name__ == "__main__":
    unittest.main()
