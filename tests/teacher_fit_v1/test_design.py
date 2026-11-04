# SPDX-License-Identifier: MPL-2.0
"""TT-003-FIT-001 data-only selection contracts; never train in CI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from teacher_fit_v1 import selection

ROOT = Path(__file__).resolve().parents[2]


def synthetic_train() -> list[dict]:
    """Generated software double; NOT a TT-002 source-oracle verification."""
    rows = []
    for family in selection.FAMILIES:
        for word in selection.WORDS:
            for task, label in (("apply", ""), ("inverse", ""),
                                ("check", "0"), ("check", "1")):
                key = "|".join((family, word, task, label))
                rid = "sha256:" + hashlib.sha256(key.encode("ascii")).hexdigest()
                answer = label if task == "check" else word
                rows.append({
                    "id": rid, "family": family, "split": "train",
                    "answer": answer,
                    "supervised_tokens": len(answer) + 1,
                    "observation": {"circuit": family, "word": word,
                                    "task": task, "candidate": ""},
                })
    return rows


class Fit001Selection(unittest.TestCase):
    def test_config_is_canonical_and_frozen_design_only(self):
        raw = (ROOT / "configs/teacher/tt003-fit-001-v1.json").read_bytes()
        config = json.loads(raw)
        self.assertEqual(raw, (json.dumps(config, sort_keys=True,
                                           separators=(",", ":")) + "\n").encode())
        self.assertEqual(config["schema"], "trinite.teacher-fit-001-design.v1")
        self.assertEqual(config["seeds"], [17, 29, 43])
        self.assertEqual(config["panel_rows"], 72)
        self.assertEqual(config["steps_per_run"], 256)
        self.assertEqual(config["batch_size"], 4)
        self.assertEqual(config["max_target_tokens_per_run"], 4096)
        self.assertFalse(config["target_execution_authorized"])
        self.assertFalse(config["heldout_scored"])
        self.assertFalse(config["gpu_allowed"])

    def test_exact_panel_and_schedule_are_permutation_invariant(self):
        rows = synthetic_train()
        first = selection.select_panel(rows)
        second = selection.select_panel(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(len(first["selected_words"]), 6)
        self.assertEqual(len(first["panel_ids"]), 72)
        self.assertEqual(len(set(first["panel_ids"])), 72)
        self.assertEqual(len(first["visit_ids"]), 1024)
        self.assertEqual(first["seeds"], [17, 29, 43])
        self.assertLessEqual(first["scheduled_target_tokens"], 4096)
        self.assertFalse(first["training_authorized"])
        self.assertFalse(first["heldout_scored"])

    def test_missing_duplicate_and_foreign_rows_rejected(self):
        base = synthetic_train()
        for corrupt in (
            base[:-1],
            [base[0], *base[1:-1], base[0]],
            [dict(base[0], family="S1W"), *base[1:]],
            [dict(base[0], supervised_tokens=99), *base[1:]],
            [dict(base[0], split="test"), *base[1:]],
            [dict(base[0], id="sha256:invented"), *base[1:]],
        ):
            with self.subTest(case=str(corrupt[0].get("id"))[:30]):
                with self.assertRaises(ValueError):
                    selection.select_panel(corrupt)

    def test_fixed_controls_are_not_model_outputs(self):
        result = selection.select_panel(synthetic_train())
        self.assertEqual(result["schema"], "trinite.teacher-fit-001-panel.v1")
        self.assertEqual(result["optimizer_updates_per_seed"], 256)
        self.assertEqual(result["batch_size"], 4)
        self.assertEqual(result["visit_ids"][:72], result["panel_ids"])


if __name__ == "__main__":
    unittest.main()
