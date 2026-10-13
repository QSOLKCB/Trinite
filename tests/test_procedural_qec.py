"""No-Torch QEC exact-procedure admission and family invariants."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.curriculum_data import dataset as parent_dataset
from trinite.procedural_qec import data
from trinite.tokenizer import ByteTokenizer


class ProceduralQECDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.manifest = data.dataset()
        cls.items = parse_json(cls.raw, canonical=True)["examples"]

    def test_gf3_exact_scalar_donor_and_independent_modulo(self):
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    self.assertEqual(data.states("gf3", [a, b, c]),
                                     [0, a, (a + b) % 3, (a + b + c) % 3])
        self.assertEqual(data.states("fold-sum", [2, -1, 2]), [0, 2, 1, 3])

    def test_invalid_operands_and_modes_fail_closed(self):
        for kind, values in (("gf3", [3, 0, 1]), ("gf3", [True, 0, 0]),
                             ("gf3", [1, 2]), ("fold-sum", [1]),
                             ("fold-sum", [0, 1, 7])):
            with self.subTest(kind=kind, values=values), self.assertRaises(ContractError):
                data.states(kind, values)
        with self.assertRaises(ContractError):
            data.make_prompt("gf3", "future", [0, 0, 0], "infix")
        with self.assertRaises(ContractError):
            data.make_prompt("gf3", "trace", [0, 0, 0], "unknown")

    def test_byte_boundaries_and_eos_targets(self):
        self.assertGreater(len(self.items), 100)
        tok = ByteTokenizer(64)
        for item in self.items:
            self.assertEqual(item["text"], item["prompt"] + item["answer"])
            encoding = tok.encode(item["text"], answer_start=len(item["prompt"].encode()))
            self.assertEqual(sum(encoding.loss_mask), len(item["answer"].encode()) + 1)
            self.assertEqual(item["example_identity"], identity(json_bytes({
                k: v for k, v in item.items() if k != "example_identity"})))

    def test_parent_family_and_train_only_lineage(self):
        folds = parse_json(parent_dataset("fold")[0], canonical=True)["examples"]
        approved = {e["example_identity"]: e for e in folds
                    if e["split"] == "train" and e["task"] == "sum"}
        expected = {(e["example_identity"], mode) for e in approved.values()
                    for mode in ("final", "prefix", "trace")}
        actual = {(e["parent_example_identity"], e["mode"]) for e in self.items
                  if e["kind"] == "fold-sum"}
        self.assertEqual(actual, expected)
        self.assertTrue(all(e["split"] == "train" for e in self.items))
        for item in self.items:
            if item["kind"] == "fold-sum":
                parent = approved[item["parent_example_identity"]]
                self.assertEqual(item["family_id"], parent["family_id"])
                self.assertEqual(item["carrier"], parent["carrier"])
                self.assertEqual(item["semantic_identity"], parent["semantic_identity"])
                if item["mode"] == "final":
                    self.assertEqual(item["text"], parent["text"])

    def test_no_prompt_or_identity_collisions_and_manifest(self):
        self.assertEqual(len({e["text"] for e in self.items}), len(self.items))
        self.assertEqual(len({e["example_identity"] for e in self.items}), len(self.items))
        manifest = data.audit_bytes(self.raw, self.manifest)
        self.assertEqual(manifest["dataset_identity"], identity(self.raw))
        self.assertEqual(manifest["qec_dependency"]["revision"],
                         "32fdf9c88f4ac1b873c8acc4d3d54c50f87d6e0a")
        self.assertTrue(all(mode in data.MODES for e in self.items
                            for mode in (e["mode"],)))

    def test_tamper_and_reseal_does_not_pass_exact_regeneration(self):
        record = parse_json(self.raw, canonical=True)
        record["examples"][0]["answer"] = "ERR"
        poisoned = json_bytes(record)
        new_manifest = parse_json(self.manifest, canonical=True)
        new_manifest["dataset_identity"] = identity(poisoned)
        with self.assertRaises(ContractError):
            data.audit_bytes(poisoned, json_bytes(new_manifest))


if __name__ == "__main__":
    unittest.main()
