"""Independent adversarial workload/schedule checks; no target inference."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes
from batch_composition_v1 import spec


def rows(count=17):
    return [
        {"item_id": f"row-{i}", "split": "train", "prompt": chr(65 + i % 26),
         "prompt_identity": identity(chr(65 + i % 26).encode())}
        for i in range(count)
    ]


class SpecificationReviewTests(unittest.TestCase):
    def test_independent_schedule_oracle_and_short_tail(self):
        workload = rows()
        expected = {
            "singleton_min": [[i] for i in range(17)],
            "singleton_32": [[i] for i in range(17)],
            "batch8_32": [list(range(8)), list(range(8, 16)), [16]],
            "partition4_32": [list(range(i, min(i + 4, 17))) for i in range(0, 17, 4)],
            "reverse8_32": [list(range(7, -1, -1)), list(range(15, 7, -1)), [16]],
            "recompose8_32": [[0, 2, 4, 6, 8, 10, 12, 14],
                             [16, 1, 3, 5, 7, 9, 11, 13], [15]],
            "batch8_min": [list(range(8)), list(range(8, 16)), [16]],
            "payload8_32": [list(range(8)), list(range(8, 16)), [16]],
        }
        self.assertEqual(set(expected), set(spec.LAYOUTS))
        for layout, groups in expected.items():
            with self.subTest(layout=layout):
                self.assertEqual(spec.schedule(workload, layout), groups)
        self.assertEqual(workload, rows())

    def test_schedule_duplicates_missing_foreign_and_boolean_indices_rejected(self):
        cases = ([[0, 0, 2]], [[0, 1]], [[0, 1, 3]], [[0, 1, -1]],
                 [[False, 1, 2]], [[], [0, 1, 2]], [(0, 1, 2)],
                 [[i for i in range(9)]])
        for groups in cases:
            with self.subTest(groups=groups), self.assertRaises(ContractError):
                spec.validate_schedule(rows(3), groups)

    def test_workload_label_and_metadata_injection_rejected(self):
        for field in ("answer", "encoding", "text", "family_id", "generated_tokens"):
            workload = rows(2)
            workload[0][field] = "not permitted"
            with self.subTest(field=field), self.assertRaises(ContractError):
                spec.validate_rows(workload)
        for mutation in ("duplicate", "test", "prompt_identity", "context"):
            workload = rows(2)
            if mutation == "duplicate":
                workload[1]["item_id"] = workload[0]["item_id"]
            elif mutation == "test":
                workload[0]["split"] = "test"
            elif mutation == "prompt_identity":
                workload[0]["prompt_identity"] = identity(b"different")
            else:
                workload[0]["prompt"] = "x" * 32
                workload[0]["prompt_identity"] = identity(b"x" * 32)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                spec.validate_rows(workload)

    def test_workload_projection_is_label_independent_and_excludes_test(self):
        original = [{"split": split, "example_identity": f"{split}-{i}",
                     "prompt": f"{i}?", "answer": "0", "text": "hidden 0",
                     "encoding": {"answer": "0"}, "family_id": "hidden"}
                    for split, count in (("train", 36), ("validation", 6), ("test", 6))
                    for i in range(count)]
        changed = copy.deepcopy(original)
        for row in changed:
            row.update(answer="9", text="changed hidden 9",
                       encoding={"answer": "9"}, family_id="changed")
            if row["split"] == "test":
                row["prompt"] = "test prompt must never execute"
        projected = []
        for source in (original, changed):
            with patch.object(spec, "inputs", return_value=(json_bytes({"examples": source}), b"")):
                projected.append(spec.workload())
        self.assertEqual(projected[0], projected[1])
        self.assertEqual(len(projected[0]), 42)
        self.assertEqual([r["split"] for r in projected[0]], ["train"] * 36 + ["validation"] * 6)

    def test_reference_mapping_binds_original_prompt_and_row_order(self):
        workload = spec.workload()
        references = spec.references(workload)
        self.assertEqual(list(references), list(spec.CELLS))
        self.assertTrue(all(len(value) == 42 for value in references.values()))
        for mutation in ("order", "missing", "prompt"):
            changed = copy.deepcopy(workload)
            if mutation == "order":
                changed[0], changed[1] = changed[1], changed[0]
            elif mutation == "missing":
                changed.pop()
            else:
                changed[0]["prompt"] = "changed"
                changed[0]["prompt_identity"] = identity(b"changed")
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                spec.references(changed)

    def test_bounded_regular_file_reader_rejects_symlink_and_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.write_bytes(b"abc")
            self.assertEqual(spec.read(target, 3), b"abc")
            with self.assertRaises(ContractError):
                spec.read(target, 2)
            link = Path(tmp) / "link"
            link.symlink_to(target)
            for path in (link, Path(tmp), Path(tmp) / "missing"):
                with self.subTest(path=path), self.assertRaises(ContractError):
                    spec.read(path, 3)


if __name__ == "__main__":
    unittest.main()
