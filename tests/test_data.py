from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from trinite.data import (audit_bytes, audit_directory, build_dataset, parse_prompt,
                          solve, split_families, verify_answer, write_dataset,
                          admission, implementation_receipt, validate_admission)

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_complete_source_admission_evidence(self):
        record = admission()
        validate_admission(record)
        self.assertIn("OpenAI Codex", record["author"])
        self.assertIn("Trent Slade", record["author"])
        self.assertEqual(record["supporting_reference"]["content_identity"],
                         implementation_receipt()["data.py"])
        self.assertEqual(len(record["rights_evidence"]["formal_source"]), 24)
        self.assertEqual(len(record["rights_evidence"]["rendered_payload"]), 48)
        self.assertEqual(record["rights_evidence"]["payload_identity"],
                         identity(json_bytes(record["rights_evidence"]["rendered_payload"])))

    def test_incomplete_or_substituted_admission_evidence_fails(self):
        original = admission()
        mutations = [lambda r: r.pop("author"),
                     lambda r: r.update(author=""),
                     lambda r: r.pop("generation_procedure"),
                     lambda r: r["generation_procedure"].update(acquisition=""),
                     lambda r: r.pop("supporting_reference"),
                     lambda r: r["supporting_reference"].update(content_identity=identity(b"other")),
                     lambda r: r["rights_evidence"]["formal_source"][0].update(modulus=True),
                     lambda r: r["rights_evidence"]["rendered_payload"].__setitem__(0, "copied prose")]
        for mutation in mutations:
            record = deepcopy(original)
            mutation(record)
            with self.assertRaises(ContractError):
                validate_admission(record)
            data, manifest = build_dataset()
            manifest["admission"] = record
            with self.assertRaises(ContractError):
                audit_bytes(json_bytes(data), json_bytes(manifest))

    def test_contradictory_admission_policy_fails_standalone(self):
        original = admission()
        changes = [("outcome", "rejected"), ("outcome", "pending"),
                   ("rights_basis", "copied copyrighted prose"),
                   ("scope", "all future internet text"),
                   ("reviewed_on", "not-a-date"), ("reviewed_on", "2026-02-30"),
                   ("reviewed_on", "20261009"), ("reviewed_on", "2026-10-10"),
                   ("reviewer", "unreviewed"), ("author", "unknown"),
                   ("origin", "downloaded prose"), ("limitations", "no limitations")]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                record = deepcopy(original)
                record[field] = value
                with self.assertRaises(ContractError):
                    validate_admission(record)
        for field in original["generation_procedure"]:
            with self.subTest(procedure=field):
                record = deepcopy(original)
                record["generation_procedure"][field] = "downloaded internet text"
                with self.assertRaises(ContractError):
                    validate_admission(record)
        record = deepcopy(original)
        record["rights_evidence"]["review_basis"] = "all future data is admitted"
        with self.assertRaises(ContractError):
            validate_admission(record)

    def test_independent_solver_exhaustive_small_domain(self):
        for m in range(2, 10):
            for a in range(m):
                for b in range(a, m):
                    spec = {"modulus": m, "left": a, "right": b}
                    self.assertEqual(solve(spec), str((a+b) % m))
                    verify_answer(spec, solve(spec))
                    with self.assertRaises(ContractError):
                        verify_answer(spec, str(((a+b) % m+1) % m))

    def test_frozen_fixture_and_full_replay(self):
        report = audit_directory(ROOT/"fixtures/formal-v1")
        self.assertTrue(report["verified"])
        self.assertEqual(report["counts"], {"train": 36, "validation": 6, "test": 6})
        d, m = build_dataset()
        self.assertEqual((ROOT/"fixtures/formal-v1/dataset.json").read_bytes(), json_bytes(d))
        self.assertEqual((ROOT/"fixtures/formal-v1/manifest.json").read_bytes(), json_bytes(m))

    def test_family_semantic_and_text_isolation(self):
        d, m = build_dataset()
        family_splits = {}
        semantic_splits = {}
        texts = set()
        for e in d["examples"]:
            family_splits.setdefault(e["family_id"], set()).add(e["split"])
            semantic_splits.setdefault(e["semantic_identity"], set()).add(e["split"])
            self.assertNotIn(e["text"], texts)
            texts.add(e["text"])
            self.assertEqual(parse_prompt(e["prompt"], e["carrier"]), e["formal"])
        self.assertEqual(len(family_splits), 8)
        self.assertEqual(len(semantic_splits), 24)
        self.assertTrue(all(len(s) == 1 for s in family_splits.values()))
        self.assertTrue(all(len(s) == 1 for s in semantic_splits.values()))
        families = list(m["family_splits"])
        self.assertEqual(split_families(families, 0), split_families(families[::-1], 0))
        self.assertNotEqual(split_families(families, 0), split_families(families, 1))

    def test_tampering_fails_even_with_refreshed_hashes(self):
        data, manifest = build_dataset()
        changes = [lambda d, m: d["examples"][0].update(answer="incorrect"),
                   lambda d, m: d["examples"][0].update(
                       split="train" if d["examples"][0]["split"] != "train" else "test"),
                   lambda d, m: d["examples"][0]["encoding"]["loss_mask"].__setitem__(0, 1),
                   lambda d, m: m["admission"].update(rights_basis="permissive license"),
                   lambda d, m: m.update(generator_identity=identity(b"different code")),
                   lambda d, m: m["counts"].update(train=37),
                   lambda d, m: d["examples"].append(d["examples"][0])]
        for change in changes:
            d, m = deepcopy(data), deepcopy(manifest)
            change(d, m)
            m["dataset_identity"] = identity(json_bytes(d))
            with self.assertRaises(ContractError):
                audit_bytes(json_bytes(d), json_bytes(m))

    def test_unknown_schema_and_unknown_fields(self):
        d, m = build_dataset()
        for change in (lambda d, m: d.update(schema="other"),
                       lambda d, m: d.update(extra=True),
                       lambda d, m: m.update(extra=True)):
            a, b = deepcopy(d), deepcopy(m)
            change(a, b)
            with self.assertRaises(ContractError):
                audit_bytes(json_bytes(a), json_bytes(b))

    def test_bad_spec_prompt_and_seed(self):
        for spec in ({"modulus": True, "left": 0, "right": 0},
                     {"modulus": 2, "left": 1, "right": 0},
                     {"modulus": 2, "left": 0, "right": 2}):
            with self.assertRaises(ContractError):
                solve(spec)
        for prompt in ("(1+1)%2= copied", "(0+2)%2=", "(1+0)%2="):
            with self.assertRaises(ContractError):
                parse_prompt(prompt, "infix")
        for seed in (-1, True, 2**32):
            with self.assertRaises(ContractError):
                build_dataset(seed=seed)
        with self.assertRaises(ContractError):
            build_dataset(ModelConfig(context_length=2))

    def test_new_directory_only_and_exact_membership(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"data"
            write_dataset(output)
            original = (output/"dataset.json").read_bytes()
            with self.assertRaises(FileExistsError):
                write_dataset(output)
            self.assertEqual((output/"dataset.json").read_bytes(), original)
            (output/"extra").write_text("unexpected")
            with self.assertRaises(ContractError):
                audit_directory(output)
            (output/"extra").unlink()
            (output/"manifest.json").unlink()
            with self.assertRaises(ContractError):
                audit_directory(output)

    def test_bounded_input_and_symlink_rejection(self):
        with self.assertRaises(ContractError):
            audit_bytes(b"x"*(256*1024+1), b"{}\n")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/"data"
            write_dataset(output)
            (output/"dataset.json").rename(root/"outside.json")
            try:
                (output/"dataset.json").symlink_to(root/"outside.json")
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(ContractError):
                audit_directory(output)
            (output/"dataset.json").unlink()
            (output/"dataset.json").write_bytes(b"x"*(256*1024+1))
            with self.assertRaises(ContractError):
                audit_directory(output)


if __name__ == "__main__":
    unittest.main()
