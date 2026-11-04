# SPDX-License-Identifier: MPL-2.0
"""Bounded, standard-library-only TT-003-DIAG-001 contract tests."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from teacher_dataset_v1 import core as tt002
from teacher_diag_v1 import core


def synthetic(task, expected, before, after, visited, index):
    return {
        "id": f"train-{index:04d}", "family": ("S1", "S2", "W")[index % 3],
        "task": task, "expected": expected, "visited": visited,
        "initial": core.classify(before, expected, task),
        "final": core.classify(after, expected, task),
    }


class TT003DiagnosticContracts(unittest.TestCase):
    def test_original_identity_pins_are_frozen(self):
        p = core.plan()
        self.assertEqual(p["source_commit"],
                         "6f2ec05b407a3bafe7a5ed947fd9caadc4be65f7")
        self.assertEqual(p["teacher_model_identity"], core.PIN["teacher_model_identity"])
        self.assertFalse(p["training_authorized"])
        self.assertFalse(p["heldout_scored"])
        self.assertEqual(p["optimizer_updates"], 0)
        self.assertEqual((p["cap_generated_tokens"], p["max_execution_seconds"]), (5, 1800))

    def test_scalar_vector_and_eos_classes(self):
        cases = [
            ("check", "0", [48, 257], "exact", True),
            ("check", "1", [48, 257], "wrong_answer", False),
            ("apply", "0000", [48, 257], "premature_eos", False),
            ("inverse", "0000", [48, 48, 48, 48, 257], "exact", True),
            ("check", "0", [48, 48, 257], "excess_length", False),
            ("apply", "0000", [48, 48, 48, 48, 48], "missing_eos", False),
            ("apply", "0000", [258, 257], "invalid_symbol", False),
            ("check", "0", [50, 257], "invalid_symbol", False),
            ("apply", "0000", [48, 48, 51, 48, 257], "invalid_symbol", False),
        ]
        for task, expected, tokens, status, exact in cases:
            with self.subTest(status=status, task=task):
                output = core.classify(tokens, expected, task)
                self.assertEqual(output["classification"], status)
                self.assertIs(output["exact"], exact)

    def test_invalid_token_and_answer_rejection(self):
        for value in ([True, 257], [259, 257], [48, 257] * 3, "0", [-1]):
            with self.subTest(value=value), self.assertRaises(ContractError):
                core.classify(value, "0", "check")
        with self.assertRaises(ContractError):
            core.classify([48, 257], "3", "check")
        with self.assertRaises(ContractError):
            core.classify([48, 257], "000", "apply")

    def test_train_cohorts_and_pairwise_changes_without_any_trainers(self):
        records = []
        for index in range(972):
            visited = index < 128
            if index % 3 == 0:
                records.append(synthetic("check", "0", [48, 257], [48, 257], visited, index))
            elif index % 3 == 1:
                records.append(synthetic("apply", "0000",
                                         [48, 257], [48, 48, 48, 48, 257],
                                         visited, index))
            else:
                records.append(synthetic("inverse", "0000",
                                         [48, 257], [48, 48, 48, 48, 257],
                                         visited, index))
        summary = core.analysis_summary(records)
        self.assertEqual(summary["train_rows"], 972)
        self.assertEqual((summary["visited_rows"], summary["unvisited_rows"]),
                         (128, 844))
        self.assertEqual(summary["groups"]["all"]["initial"]["exact_count"], 324)
        self.assertEqual(summary["groups"]["all"]["final"]["exact_count"], 972)
        self.assertEqual(summary["pairwise_changes"]["improved"], 648)
        self.assertEqual(summary["groups"]["kind:scalar_check"]["count"], 324)
        self.assertEqual(summary["groups"]["kind:vector_apply"]["count"], 324)
        self.assertEqual(summary["groups"]["kind:vector_inverse"]["count"], 324)
        records[1]["final"]["exact"] = False
        with self.assertRaises(ContractError):
            core.analysis_summary(records)

    def test_missing_or_repeated_rows_are_not_partial_success(self):
        one = synthetic("check", "0", [48, 257], [48, 257], True, 0)
        with self.assertRaises(ContractError):
            core.analysis_summary([one])
        with self.assertRaises(ContractError):
            core.analysis_summary([one] * 972)

    def test_source_bound_freeze_and_invalidated_document(self):
        packet, _ = tt002.make_packet()
        train_raw = packet["train.jsonl"]
        rows = [parse_json(line + b"\n", canonical=True) for line in train_raw.splitlines()]
        seed = 17
        ordered = sorted(rows, key=lambda r: identity(json_bytes(
            {"seed": seed, "item": r["id"]})))
        visits = [row["id"] for row in ordered[:128]]
        metadata = {"history": [
            {"step": i+1, "examples": visits[i*4:(i+1)*4]}
            for i in range(32)
        ]}
        fake_teacher = {
            "dataset_train_identity": identity(train_raw),
            "plan": {"seed": 17},
            "train_schedule_identity": identity(json_bytes(visits)),
            "source_closure": [{"path": "src/trinite/contracts.py",
                                "identity": identity(core.lab.source_bytes(
                                    "src/trinite/contracts.py"))}],
        }
        p = core.plan()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            dataset = folder / "dataset"
            dataset.mkdir()
            (dataset / "train.jsonl").write_bytes(train_raw)
            run = folder / "original"
            run.mkdir()
            with mock.patch.object(core, "historical_inputs",
                                   return_value=(p, fake_teacher, metadata, {})):
                frozen = folder / "frozen"
                external = core.freeze(frozen, dataset, run)
                verified = core.verify_request(frozen, external, dataset, run)
                self.assertEqual(verified["train_rows"], 972)
                self.assertEqual((verified["visited_rows"], verified["unvisited_rows"]),
                                 (128, 844))
                with self.assertRaises(ContractError):
                    core.freeze(frozen, dataset, run)
                with self.assertRaises(ContractError):
                    core.verify_request(frozen, "sha256:" + "0"*64, dataset, run)
                old = core.lab.source_bytes
                def changed(path):
                    raw = old(path)
                    return raw + b"\n# changed" if path == "docs/TEACHER-LAB-DIAG-001.md" else raw
                with mock.patch.object(core.lab, "source_bytes", side_effect=changed):
                    with self.assertRaises(ContractError):
                        core.verify_request(frozen, external, dataset, run)

    def test_scheduled_rows_match_exact_128_source_bound_visits(self):
        packet, _ = tt002.make_packet()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.jsonl").write_bytes(packet["train.jsonl"])
            rows = [parse_json(b + b"\n", canonical=True)
                    for b in packet["train.jsonl"].splitlines()]
            order = sorted(rows, key=lambda r: identity(json_bytes(
                {"seed": 17, "item": r["id"]})))
            ids = [e["id"] for e in order[:128]]
            request = {"dataset_train_identity": identity(packet["train.jsonl"]),
                       "train_schedule_identity": identity(json_bytes(ids)),
                       "plan": {"seed": 17}}
            meta = {"history": [
                {"step": i+1, "examples": ids[i*4:(i+1)*4]}
                for i in range(32)
            ]}
            actual, visited = core.scheduled_rows(meta, request, root)
            self.assertEqual(len(actual), 972)
            self.assertEqual(len(visited), 128)
            meta["history"][0]["examples"][0] = ids[1]
            with self.assertRaises(ContractError):
                core.scheduled_rows(meta, request, root)


if __name__ == "__main__":
    unittest.main()
