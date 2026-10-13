"""Pure-standard-library regression for independent v3 procedural cursors."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from trinite.contracts import ContractError, identity, json_bytes
from trinite.procedural_qec import data
from trinite.procedural_qec_v3 import scheduler


def previous_control(items, seed, steps):
    """Frozen v2's control schedule computed without Torch or model calls."""
    groups = {key: scheduler.ranked(
        [e for e in items if (e["kind"], e["mode"]) == key], seed)
        for key in scheduler.GROUPS}
    rows = []
    for step in range(steps):
        rows.extend(groups["arithmetic", "final"][(4*step+j) %
                    len(groups["arithmetic", "final"])] for j in range(4))
        rows.extend(groups["fold-sum", "final"][(2*step+j) %
                    len(groups["fold-sum", "final"])] for j in range(2))
        rows.append(groups["gf3", "final"][step % len(groups["gf3", "final"])])
        kind = "fold-sum" if step % 2 == 0 else "gf3"
        rows.append(groups[kind, "final"][(step//2) % len(groups[kind, "final"])])
    return rows


class IndependentCursorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.manifest = data.dataset()
        cls.rows = __import__("json").loads(cls.raw)["examples"]

    def test_v3_protocol_was_precommitted_and_is_immutable(self):
        value = (ROOT/"src/trinite/procedural_qec_v3/protocol.json").read_bytes()
        self.assertEqual(identity(value),
                         "sha256:be5e998e5e79d52ecae40bf679c8b0a686c70577c78ac4ff0af76b49d7f27b18")
        self.assertEqual(__import__("json").loads(value)["schema"],
                         "trinite.procedural-qec-protocol.v3")

    def test_even_and_odd_streams_cycle_every_row_exactly(self):
        for n in (3, 4, 5, 6, 54, 1150):
            group = {("gf3", "trace"): list(range(n))}
            cursor = {("gf3", "trace"): 0}
            values = [scheduler.advance(group, cursor, "gf3", "trace") for _ in range(2*n)]
            self.assertEqual(values[:n], list(range(n)))
            self.assertEqual(values[n:], list(range(n)))
            self.assertEqual(cursor[("gf3", "trace")], 2*n)

    def test_control_matches_v2_exactly_at_both_budgets(self):
        for seed in (0, 1):
            for steps in (8, 4096):
                got = scheduler.schedule(self.rows, seed, "final-only", steps)
                expected = previous_control(self.rows, seed, steps)
                self.assertEqual([e["example_identity"] for e in got],
                                 [e["example_identity"] for e in expected])

    def test_correction_uses_all_available_gf3_rows_and_1024_fold_rows(self):
        for seed in (0, 1):
            scheduled = scheduler.schedule(self.rows, seed, "procedural", 4096)
            report = scheduler.coverage(self.rows, scheduled, "procedural")
            for mode in ("prefix", "trace"):
                fold = report["fold-sum/" + mode]
                gf = report["gf3/" + mode]
                self.assertEqual((fold["visits"], fold["available"],
                                  fold["distinct_visited"], fold["distinct_unvisited"]),
                                 (1024, 1150, 1024, 126))
                self.assertEqual((gf["visits"], gf["available"],
                                  gf["distinct_visited"], gf["distinct_unvisited"]),
                                 (1024, 54, 54, 0))
            self.assertEqual(report["arithmetic/final"]["visits"], 16384)
            self.assertEqual(report["fold-sum/final"]["visits"], 8192)
            self.assertEqual(report["gf3/final"]["visits"], 4096)

    def test_old_parity_exclusion_is_rejected(self):
        # Simulates old stride-2 reuse for fold prefix while all other rows
        # remain correctly scheduled; total visit count is unaffected.
        scheduled = scheduler.schedule(self.rows, 0, "procedural", 4096)
        subset = scheduler.ranked([e for e in self.rows if
                e["kind"] == "fold-sum" and e["mode"] == "prefix"], 0)
        corrupted = list(scheduled)
        k = 0
        for pos, e in enumerate(corrupted):
            if e["kind"] == "fold-sum" and e["mode"] == "prefix":
                corrupted[pos] = subset[(2*k) % len(subset)]
                k += 1
        self.assertEqual(k, 1024)
        with self.assertRaisesRegex(ContractError, "excludes reachable"):
            scheduler.coverage(self.rows, corrupted, "procedural")

    def test_invalid_arms_or_synthetic_duplicates_are_rejected(self):
        with self.assertRaises(ContractError):
            scheduler.schedule(self.rows, 0, "unknown", 8)
        with self.assertRaises(ContractError):
            scheduler.schedule(self.rows, 0, "procedural", 8193)
        group = {("gf3", "trace"): [1, 2]}
        with self.assertRaises(ContractError):
            scheduler.advance(group, {}, "gf3", "prefix")
        with self.assertRaises(ContractError):
            scheduler.ranked([self.rows[0], self.rows[0]], 0)


if __name__ == "__main__":
    unittest.main()
