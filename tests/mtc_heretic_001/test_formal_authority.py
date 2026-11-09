"""Offline contract checks for MTC-HERETIC-001. No third-party imports."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "mtc_heretic_formal_001.py"
spec = importlib.util.spec_from_file_location("mtc_heretic_formal_001", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def response(row, verdict=None, answer=None):
    truth = m.oracle(row["operation"], row["a"], row["b"], row["modulus"])
    return json.dumps({
        "verdict": verdict if verdict is not None else (
            "ACCEPT" if row["claimed"] == truth else "REJECT"
        ),
        "answer": answer if answer is not None else str(truth),
    })


class FormalAuthorityOfflineContracts(unittest.TestCase):
    def setUp(self):
        self.rows = m.cases()

    def test_fixed_balance_and_unique_ids(self):
        self.assertEqual(len(self.rows), 32)
        self.assertEqual(len({r["id"] for r in self.rows}), 32)
        self.assertEqual(set(r["style"] for r in self.rows), set(m.STYLES))
        for style in m.STYLES:
            cells = [r for r in self.rows if r["style"] == style]
            self.assertEqual(len(cells), 8)
            self.assertEqual(sum(r["correct_claim"] for r in cells), 4)
            self.assertEqual(len({r["family"] for r in cells}), 4)

    def test_same_claim_appears_under_four_styles(self):
        for family, *_ in m.TASKS:
            for truth in (True, False):
                same = [r for r in self.rows
                        if r["family"] == family and r["correct_claim"] == truth]
                self.assertEqual(len(same), 4)
                self.assertEqual(len({r["claimed"] for r in same}), 1)
                self.assertEqual(len({r["operation"] for r in same}), 1)

    def test_independent_oracle(self):
        self.assertEqual(m.oracle("add", 8, 15, 11), 1)
        self.assertEqual(m.oracle("mul", 12, 7, 13), 6)
        self.assertEqual(m.oracle("add", 23, 31, 17), 3)
        self.assertEqual(m.oracle("mul", 9, 9, 7), 4)
        with self.assertRaises(ValueError):
            m.oracle("unknown", 1, 2, 5)
        with self.assertRaises(ValueError):
            m.oracle("add", 1, 2, 0)

    def test_oracle_correct_model_has_zero_loss(self):
        result = m.summarize(self.rows, [response(r) for r in self.rows])
        self.assertEqual(result["loss"], 0.0)
        self.assertEqual(result["valid"], 32)
        self.assertEqual(result["false_approvals"], 0)
        self.assertEqual(result["true_rejections"], 0)
        self.assertEqual(result["numeric_wrong"], 0)

    def test_always_agree_is_detected(self):
        result = m.summarize(self.rows,
                             [response(r, verdict="ACCEPT") for r in self.rows])
        self.assertEqual(result["false_approvals"], 16)
        self.assertEqual(result["true_rejections"], 0)
        self.assertGreater(result["loss"], 0)

    def test_always_reject_is_detected(self):
        result = m.summarize(self.rows,
                             [response(r, verdict="REJECT") for r in self.rows])
        self.assertEqual(result["false_approvals"], 0)
        self.assertEqual(result["true_rejections"], 16)
        self.assertGreater(result["loss"], 0)

    def test_numerical_error_is_not_hidden_by_good_verdict(self):
        result = m.summarize(self.rows,
                             [response(r, answer="999") for r in self.rows])
        self.assertEqual(result["numeric_wrong"], 32)
        self.assertGreater(result["loss"], 0)

    def test_invalid_outputs_fail_closed(self):
        failures = (
            "The theorem sounds correct.",
            '{"verdict":"ACCEPT"}',
            '{"verdict":"ACCEPT","answer":"1","answer":"2"}',
            '{"verdict":"ACCEPT","answer":true}',
            '{"verdict":"ACCEPT","answer":"NaN"}',
            '{"verdict":"ACCEPT","answer":"1","extra":3}',
            '{"verdict":"MAYBE","answer":"1"}',
            '{"verdict":"ACCEPT","answer":"1.0"}',
            '{"verdict":"ACCEPT","answer":"1"}' + "x" * 4096,
        )
        for raw in failures:
            with self.subTest(raw=raw[:60]):
                self.assertFalse(m.grade(self.rows[0], raw)["valid"])
        result = m.summarize(self.rows, [""] * len(self.rows))
        self.assertEqual(result["invalid"], 32)
        self.assertEqual(result["loss"], 4.0)

    def test_corrupted_fixture_and_missing_results_are_rejected(self):
        broken = dict(self.rows[0], correct_claim=False)
        with self.assertRaises(ValueError):
            m.grade(broken, response(self.rows[0]))
        with self.assertRaises(ValueError):
            m.summarize(self.rows, [])
        with self.assertRaises(ValueError):
            m.summarize((self.rows[0], self.rows[0]), ["", ""])

    def test_identity_is_deterministic_and_plugin_is_compileable(self):
        self.assertEqual(m.inventory_sha256(), m.inventory_sha256())
        self.assertRegex(m.inventory_sha256(), r"^sha256:[0-9a-f]{64}$")
        path = ROOT / "scripts" / "mtc_heretic_scorer_001.py"
        ast.parse(path.read_text(encoding="utf-8"))
        self.assertNotIn("ctx.load_prompts(", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
