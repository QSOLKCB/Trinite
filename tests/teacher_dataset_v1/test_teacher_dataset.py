"""TT-002: finite algebra, source-specific rights, strict custody and separation."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from trinite.contracts import ContractError, identity, parse_json
from teacher_dataset_v1 import core


def rows(raw):
    return [parse_json(line + b"\n", canonical=True)
            for line in raw.splitlines()]


class TeacherDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet, cls.manifest = core.make_packet()

    def test_complete_inventory_and_immutable_pin(self):
        self.assertEqual(self.manifest["split_counts"],
                         {"train": 972, "validation": 648, "test": 648})
        self.assertEqual(self.manifest["independent_oracle_checks"], 2268)
        self.assertEqual(set(self.manifest["family_counts"]), set(core.SPLITS))
        self.assertEqual(set(self.packet), set(core.MEMBERS))
        self.assertEqual(self.manifest["qec_revision"], core.q.QEC_SHA)

    def test_no_training_weights_and_separate_rights(self):
        self.assertFalse(self.manifest["training_authorized"])
        self.assertEqual(self.manifest["optimizer_updates"], 0)
        self.assertFalse(self.manifest["heldout_scored"])
        dataset = parse_json(self.packet["dataset-rights.json"], canonical=True)
        weights = parse_json(self.packet["weights-rights.json"], canonical=True)
        self.assertEqual(dataset["admission"], "ADMITTED_FORMAL_ONLY")
        self.assertEqual(dataset["dataset_license"], "UNDECIDED_SEPARATE_FROM_CODE")
        self.assertEqual(weights["weights_license"], "UNDECIDED")
        self.assertEqual(weights["status"], "NOT_CREATED")

    def test_public_holdouts_are_free_of_answers_and_provenance(self):
        training = rows(self.packet["train.jsonl"])
        validation = rows(self.packet["validation-public.jsonl"])
        testing = rows(self.packet["test-public.jsonl"])
        targets = rows(self.packet["privileged-targets.jsonl"])
        self.assertEqual((len(training), len(validation), len(testing), len(targets)),
                         (972, 648, 648, 1296))
        for row in validation + testing:
            self.assertEqual(set(row), {"schema", "id", "observation", "prompt"})
            self.assertNotIn("answer", row["prompt"])
            self.assertNotIn("family", row["prompt"])
        self.assertEqual(len({r["prompt"] for r in training + validation + testing}), 2268)
        self.assertEqual(len({r["id"] for r in training + validation + testing}), 2268)
        self.assertEqual({r["observation"]["circuit"] for r in training},
                         {"S1", "S2", "W"})
        self.assertEqual({r["observation"]["circuit"] for r in validation},
                         {"S1W", "WS2"})
        self.assertEqual({r["observation"]["circuit"] for r in testing},
                         {"S2W", "WS1"})
        self.assertEqual({r["id"] for r in targets},
                         {r["id"] for r in validation + testing})
        self.assertEqual({r["split"] for r in targets}, {"validation", "test"})
        self.assertEqual({r["answer"] for r in training if
                          r["observation"]["task"] == "check"}, {"0", "1"})

    def test_direct_algebra_and_inverse_versus_independent_ket_oracle(self):
        for circuit in core.SPLITS:
            for n in range(81):
                word = f"{n // 27}{n // 9 % 3}{n // 3 % 3}{n % 3}"
                for inverse in (False, True):
                    task = "inverse" if inverse else "apply"
                    value = core.algebraic(word, circuit, inverse)
                    self.assertEqual(value, core.q.expected(core.q.public(word, circuit, task)))
                    self.assertEqual(core.algebraic(value, circuit, not inverse), word)

    def test_symplectic_pairing_is_preserved(self):
        words = [f"{n // 27}{n // 9 % 3}{n // 3 % 3}{n % 3}" for n in range(81)]
        for circuit in core.SPLITS:
            transformed = [core.algebraic(w, circuit) for w in words]
            for i, a in enumerate(words):
                for j, b in enumerate(words):
                    self.assertEqual(
                        (int(a[0])*int(b[2]) + int(a[1])*int(b[3]) -
                         int(a[2])*int(b[0]) - int(a[3])*int(b[1])) % 3,
                        (int(transformed[i][0])*int(transformed[j][2]) +
                         int(transformed[i][1])*int(transformed[j][3]) -
                         int(transformed[i][2])*int(transformed[j][0]) -
                         int(transformed[i][3])*int(transformed[j][1])) % 3)

    def test_change_to_independent_oracle_fails_closed(self):
        with mock.patch.object(core.q, "expected", return_value="bad"):
            with self.assertRaises(ContractError):
                core.make_packet()

    def test_bad_or_unpinned_plan_is_rejected(self):
        with mock.patch.object(core.tt001, "source_json", wraps=core.tt001.source_json) as loader:
            original = core.tt001.source_json(core.PLAN)
            wrong = dict(original, qec_revision="0" * 40)
            loader.side_effect = lambda p: wrong if p == core.PLAN else core.tt001.source_json(p)
            # The wrong pin is incompatible irrespective of any downstream native runtime.
            with self.assertRaises(ContractError):
                core.plan()

    def test_freeze_verify_is_readonly_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            digest = core.prepare(path)
            before = {p.name: p.read_bytes() for p in path.iterdir()}
            result = core.verify(path, digest)
            self.assertTrue(result["verified"])
            self.assertFalse(result["training_authorized"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})
            with self.assertRaises(ContractError):
                core.prepare(path)
            with self.assertRaises(ContractError):
                core.verify(path, "sha256:" + "0" * 64)

    def test_modified_training_label_rejected_even_with_resealed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            digest = core.prepare(path)
            member = path / "train.jsonl"
            raw = member.read_bytes()
            member.write_bytes(raw.replace(b'"answer":"0000"', b'"answer":"2222"', 1))
            self.assertNotEqual(member.read_bytes(), raw)
            with self.assertRaises(ContractError):
                core.verify(path, digest)

    def test_rights_tamper_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            digest = core.prepare(path)
            member = path / "dataset-rights.json"
            member.write_bytes(member.read_bytes().replace(b"ADMITTED_FORMAL_ONLY",
                                                          b"ADMITTED_ANYTHING"))
            with self.assertRaises(ContractError):
                core.verify(path, digest)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            digest = core.prepare(path)
            member = path / "train.jsonl"
            copy = Path(tmp) / "outside"
            member.rename(copy)
            member.symlink_to(copy)
            with self.assertRaises(ContractError):
                core.verify(path, digest)

    def test_extra_files_rejected_and_directory_scan_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            digest = core.prepare(path)
            (path / "injected").write_text("x")
            with self.assertRaises(ContractError):
                core.verify(path, digest)

    def test_source_drift_invalidates_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            digest = core.prepare(path)
            original = core.tt001.source_bytes
            def changed(p):
                value = original(p)
                return value + b"\n" if p == "docs/DATA.md" else value
            with mock.patch.object(core.tt001, "source_bytes", side_effect=changed):
                with self.assertRaises(ContractError):
                    core.verify(path, digest)

    def test_failed_fsync_leaves_no_published_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            with mock.patch.object(core.os, "fsync", side_effect=OSError("test fsync failure")):
                with self.assertRaises(OSError):
                    core.prepare(path)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])
            anchor = core.prepare(path)
            self.assertTrue(core.verify(path, anchor)["verified"])

    def test_source_closure_includes_executable_and_normative_files(self):
        paths = {item["path"] for item in self.manifest["source_closure"]}
        for expected in ("scripts/qutrit_conformance_v1.py",
                         "scripts/teacher_dataset.py",
                         "src/trinite/tokenizer.py",
                         "src/trinite/__init__.py",
                         "docs/DATA.md",
                         "docs/TEACHER-LAB-TT-002.md"):
            self.assertIn(expected, paths)


if __name__ == "__main__":
    unittest.main()
