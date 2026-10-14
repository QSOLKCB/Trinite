"""Standard-library boundaries; real source admission has its own required lane.

Run tests/procedural_qec_auxiliary_data explicitly with the original historical
checkout. Root discovery neither skips nor substitutes synthetic evidence for
that distinct admission/mapping/schedule integration suite.
"""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

from trinite.contracts import ContractError, identity, json_bytes
from trinite.procedural_qec_auxiliary import inputs, scheduler

ROOT = Path(__file__).resolve().parents[1]


class AuxiliaryDataBoundaryTests(unittest.TestCase):
    def test_data_modules_import_without_numerical_backend(self):
        program = (
            "import sys;sys.path.insert(0,sys.argv[1]);"
            "from trinite.procedural_qec_auxiliary import inputs,scheduler;"
            "bad=[n for n in sys.modules if n.split('.')[0] in ('torch','numpy')];"
            "print(bad);sys.exit(bool(bad))")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", program,
                                 str(ROOT / "src")], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout, b"[]\n")

    def test_data_modules_cannot_import_a_model_or_numerical_updater(self):
        forbidden = {"torch", "numpy", "model", "training", "experiment", "checkpoint",
                     "comparison_checkpoint", "observation"}
        for name in ("inputs.py", "scheduler.py"):
            path = ROOT / "src/trinite/procedural_qec_auxiliary" / name
            tree = ast.parse(path.read_text())
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name.split(".") for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append((node.module or "").split("."))
            with self.subTest(module=name):
                self.assertFalse(forbidden.intersection(part for names in imported for part in names))

    def test_independently_pinned_audit_and_original_receipt_are_preserved(self):
        audit, raw, receipt = inputs._load_audit(ROOT)
        self.assertEqual(identity(raw), inputs.AUDIT_IDENTITY)
        self.assertEqual(identity(receipt), inputs.COMPARISON_IDENTITY)
        self.assertEqual(audit.PINS["historical"]["commit"], inputs.HISTORICAL_COMMIT)
        self.assertEqual(audit.PINS["historical"]["dataset"], inputs.HISTORICAL_DATASET)
        self.assertEqual(audit.PINS["historical"]["manifest"], inputs.HISTORICAL_MANIFEST)

    def test_normalization_ignores_only_two_top_level_fields(self):
        row = {"generator_revision": "generator", "example_identity": "example",
               "prompt": "P:", "answer": "A", "split": "train",
               "transformation_lineage": {"example_identity": "retain-nested"}}
        actual = inputs.normalized(row)
        self.assertEqual(set(row) - set(actual), {"generator_revision", "example_identity"})
        self.assertEqual(actual["transformation_lineage"], row["transformation_lineage"])
        for value in (None, [], "row"):
            with self.assertRaises(ContractError):
                inputs.normalized(value)

    def test_unadmitted_inputs_and_out_of_bound_cells_reject(self):
        with self.assertRaisesRegex(ContractError, "materialize_inputs"):
            inputs.MaterializedInputs()
        for value in ({"admitted": True}, [], None):
            with self.assertRaisesRegex(ContractError, "actual admitted"):
                scheduler.schedule(value, 0, "extra-final", 1)
        for rounds in (0, -1, 4097, 8192, True, 1.0, "1"):
            with self.subTest(rounds=rounds), self.assertRaises(ContractError):
                scheduler.schedule(None, 0, "extra-final", rounds)
        for seed in (-1, 2, True, 0.0):
            with self.subTest(seed=seed), self.assertRaises(ContractError):
                scheduler.schedule(None, seed, "extra-final", 1)
        for arm in ("final-only", "procedural", "unknown", None):
            with self.subTest(arm=arm), self.assertRaises(ContractError):
                scheduler.schedule(None, 0, arm, 1)

    def test_batch_cursor_handles_odd_even_streams_through_multiple_wraps(self):
        # Synthetic indexing conformance only; these values cannot be admitted
        # by the public source-bound scheduling entry point.
        key = ("fold-sum", "prefix")
        for size in (3, 4, 5, 6):
            group = list(range(size))
            counters = {}
            visited = []
            for _ in range(3):
                visited.extend(scheduler.advance_batch({key: group}, counters, key))
            with self.subTest(size=size):
                self.assertEqual(visited, [group[index % size] for index in range(24)])
                self.assertGreaterEqual(len(visited), 2 * size)
                self.assertEqual(counters, {key: 24})

    def test_batch_cursors_are_independent_and_controls_share_per_kind(self):
        groups = {("fold-sum", "prefix"): list(range(3)),
                  ("gf3", "prefix"): list(range(4)),
                  ("fold-sum", "trace"): list(range(5)),
                  ("gf3", "trace"): list(range(6)),
                  ("fold-sum", "final"): list(range(5)),
                  ("gf3", "final"): list(range(6))}
        for arm in scheduler.ARMS:
            counters, seen = {}, {}
            for macro in range(12):
                kind, mode = scheduler.auxiliary_stream(macro)
                key = (kind, "final") if arm == "extra-final" else (kind, mode)
                before = dict(counters)
                batch = scheduler.advance_batch(groups, counters, key)
                seen.setdefault(key, []).extend(batch)
                self.assertEqual(counters[key], before.get(key, 0) + 8)
                self.assertEqual({k: v for k, v in counters.items() if k != key},
                                 {k: v for k, v in before.items() if k != key})
            self.assertEqual(len(counters), 2 if arm == "extra-final" else 4)
            for key, visited in seen.items():
                group = groups[key]
                self.assertEqual(visited, [group[index % len(group)] for index in range(len(visited))])
                self.assertGreaterEqual(len(visited), 2 * len(group))
                self.assertEqual(counters[key], 48 if arm == "extra-final" else 24)

    def test_batch_cursor_rejects_unknown_empty_and_malformed_streams(self):
        key = ("fold-sum", "prefix")
        cases = [(None, {}, key), ({key: []}, {}, key), ({key: (1, 2)}, {}, key),
                 ({key: [1]}, [], key), ({key: [1]}, {}, ["fold-sum", "prefix"]),
                 ({key: [1]}, {}, ("unknown", "prefix")), ({}, {}, key)]
        for groups, counters, selected in cases:
            with self.subTest(groups=groups, counters=counters, key=selected), self.assertRaises(ContractError):
                scheduler.advance_batch(groups, counters, selected)
        for value in (-1, True, 1.0, "0", 32761, 32768):
            counters = {key: value}
            with self.subTest(cursor=value), self.assertRaises(ContractError):
                scheduler.advance_batch({key: [1, 2, 3]}, counters, key)
            self.assertEqual(counters, {key: value})
        counters = {key: 32760}
        self.assertEqual(len(scheduler.advance_batch({key: [1, 2, 3]}, counters, key)), 8)
        self.assertEqual(counters[key], 32768)

    def test_fixed_cadence_and_streamed_canonical_encoding(self):
        self.assertEqual([scheduler.auxiliary_stream(m) for m in range(4)],
            [("fold-sum", "prefix"), ("gf3", "prefix"), ("fold-sum", "trace"), ("gf3", "trace")])
        self.assertEqual(scheduler.auxiliary_stream(4095), ("gf3", "trace"))
        for value in (-1, 4096, True, "0"):
            with self.assertRaises(ContractError):
                scheduler.auxiliary_stream(value)
        values = ["identity", {"nested": [1, "é", None]}, []]
        self.assertEqual(scheduler._sequence_identity(json_bytes(v)[:-1] for v in values),
                         identity(json_bytes(values)))
        self.assertEqual(scheduler._sequence_identity(iter(())), identity(json_bytes([])))


if __name__ == "__main__":
    unittest.main()
