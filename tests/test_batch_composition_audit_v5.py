"""Ordinary isolated-startup checks; no adversarial import/cache execution."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_batch_composition_v5.py"
helper_spec = importlib.util.spec_from_file_location(
    "ordinary_isolated_audit_fixture", ROOT / "tests/test_batch_composition_audit_review.py")
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)


class IsolatedAuditTests(unittest.TestCase):
    def fixture(self):
        fixture = helper.AuditReviewTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_missing_required_startup_flags_fail_before_auditing(self):
        fixture = self.fixture()
        before = helper.snapshot(fixture.artifact)
        for index, flags in enumerate(((), ("-I",), ("-S",),
                                       ("-S", "-E", "-s", "-P"))):
            with self.subTest(flags=flags):
                output = fixture.root / ("unsupported-%d.json" % index)
                result = fixture.invoke(output, mode=(flags, ""), script=SCRIPT)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("requires a fresh interpreter: python -I -S -B", result.stderr)
                self.assertFalse(output.exists())
                self.assertEqual(helper.snapshot(fixture.artifact), before)

    def test_isolated_clean_positive_and_nonconformant_packets(self):
        fixture = self.fixture()
        for conforms in (True, False):
            fixture.report["conforms"] = conforms
            fixture.write_packet()
            for optimize, extra in enumerate(((), ("-O",), ("-OO",))):
                with self.subTest(conforms=conforms, optimize=optimize):
                    output = fixture.root / ("isolated-%s-%d.json" % (conforms, optimize))
                    result = fixture.invoke(
                        output, mode=(("-I", "-S", *extra), ""), script=SCRIPT)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    receipt = json.loads(output.read_bytes())
                    self.assertEqual(receipt["schema"], "trinite.independent-batch-ci-audit.v5")
                    self.assertIs(receipt["strict_recorded_report_validation"]["conforms"], conforms)
                    self.assertEqual(receipt["auditor_identity"], helper.identity(SCRIPT.read_bytes()))
                    self.assertEqual(receipt["approved_review_identity"],
                                     helper.identity(fixture.approved.read_bytes()))
                    self.assertEqual(receipt["interpreter_startup"], {
                        "policy": "fresh trusted interpreter with -I -S before path-loadable imports",
                        "isolated": 1, "no_site": 1, "ignore_environment": 1,
                        "no_user_site": 1, "safe_path": True, "optimize": optimize})
                    expected = json.loads(fixture.approved.read_bytes())["source"]
                    self.assertEqual(receipt["executed_project_sources"], {
                        "trinite": expected["src/trinite/__init__.py"],
                        "trinite.contracts": expected["src/trinite/contracts.py"],
                        "batch_composition_v1": expected["scripts/batch_composition_v1/__init__.py"],
                        "batch_composition_v1.spec": expected["scripts/batch_composition_v1/spec.py"],
                        "batch_composition_v1.validation": expected["scripts/batch_composition_v1/validation.py"]})
                    self.assertIs(receipt["local_inference_executed"], False)

    def test_sys_only_bootstrap_guards_every_filesystem_loadable_import(self):
        tree = ast.parse(SCRIPT.read_text())
        self.assertIsInstance(tree.body[0], ast.Expr)  # Module docstring, not execution.
        self.assertIsInstance(tree.body[1], ast.Import)
        self.assertEqual([(n.name, n.asname) for n in tree.body[1].names], [("sys", None)])
        guard = tree.body[2]
        self.assertIsInstance(guard, ast.If)
        self.assertIsInstance(guard.test, ast.UnaryOp)
        self.assertIsInstance(guard.test.op, ast.Not)
        self.assertIsInstance(guard.test.operand, ast.BoolOp)
        self.assertIsInstance(guard.test.operand.op, ast.And)
        self.assertEqual(ast.dump(guard.test), ast.dump(ast.parse(
            "not (sys.flags.isolated == 1 and sys.flags.no_site == 1 "
            "and sys.flags.ignore_environment == 1 and sys.flags.no_user_site == 1 "
            "and sys.flags.safe_path)", mode="eval").body))
        self.assertEqual(len(guard.body), 2)
        self.assertEqual(ast.unparse(guard.body[0].value.func), "sys.stderr.write")
        self.assertEqual(ast.unparse(guard.body[1]), "raise SystemExit(1)")
        self.assertFalse(guard.orelse)
        self.assertIsInstance(tree.body[3], ast.Import)
        self.assertEqual(tree.body[3].names[0].name, "argparse")
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
        old = ast.parse((ROOT / "scripts/audit_batch_composition_v4.py").read_text())
        old_loader = next(n for n in old.body if isinstance(n, ast.ClassDef)
                          and n.name == "VerifiedSourceImports")
        new_loader = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                          and n.name == "VerifiedSourceImports")
        self.assertEqual(ast.dump(new_loader), ast.dump(old_loader))

    def test_retained_isolated_receipts_bind_unchanged_historical_facts(self):
        folder = ROOT / "fixtures/batch-composition-audit-v5"
        raw = (folder / "commitment.json").read_bytes()
        self.assertEqual(helper.identity(raw),
                         "sha256:ef6c28254dd089356b71722fdb2bbaf10a103c9cb8a1d53226621bad589064c9")
        commitment = json.loads(raw)
        self.assertIs(commitment["independent_adversarial_review_completed"], False)
        self.assertIs(commitment["model_inference_executed"], False)
        self.assertIs(commitment["numerical_producer_unchanged"], True)
        self.assertEqual(commitment["required_startup_flags"], ["-I", "-S"])
        self.assertEqual(helper.identity(SCRIPT.read_bytes()), commitment["auditor_identity"])
        for label in ("approved_review", "historical_auditor"):
            self.assertEqual(helper.identity((ROOT / commitment[label + "_path"]).read_bytes()),
                             commitment[label + "_identity"])
        for version in (2, 3, 4):
            name = ("batch-composition-v2" if version == 2
                    else "batch-composition-audit-v%d" % version)
            self.assertEqual(helper.identity((ROOT / "fixtures" / name / "commitment.json").read_bytes()),
                             commitment["historical_v%d_commitment_identity" % version])
        self.assertEqual(set(p.name for p in folder.iterdir()), set(commitment["files"]) | {"commitment.json"})
        self.assertEqual(set(commitment["matrix"]), {"3.11", "3.12", "3.13"})
        for optimize, version in enumerate(("3.11", "3.12", "3.13")):
            info = commitment["matrix"][version]
            raw = (folder / info["receipt_path"]).read_bytes()
            self.assertEqual(helper.identity(raw), info["receipt_identity"])
            self.assertEqual(helper.identity(raw), commitment["files"][info["receipt_path"]])
            receipt = json.loads(raw)
            old = json.loads((ROOT / "fixtures/batch-composition-audit-v4" / (version + ".json")).read_bytes())
            self.assertEqual(receipt["schema"], "trinite.independent-batch-ci-audit.v5")
            self.assertEqual(receipt["auditor_identity"], commitment["auditor_identity"])
            self.assertEqual(receipt["approved_review_identity"], commitment["approved_review_identity"])
            self.assertEqual(receipt["report_identity"], info["report_identity"])
            self.assertEqual(receipt["request_identity"], info["request_identity"])
            self.assertEqual(receipt["interpreter_startup"], {
                "policy": "fresh trusted interpreter with -I -S before path-loadable imports",
                "isolated": 1, "no_site": 1, "ignore_environment": 1,
                "no_user_site": 1, "safe_path": True, "optimize": optimize})
            self.assertEqual(
                {k: v for k, v in receipt.items()
                 if k not in ("schema", "auditor_identity", "interpreter_startup")},
                {k: v for k, v in old.items() if k not in ("schema", "auditor_identity")})
            self.assertEqual(receipt["same_prefix_vector_comparisons"], 10920)
            self.assertEqual(receipt["violating_coordinates"], 0)
            self.assertIs(receipt["local_inference_executed"], False)


if __name__ == "__main__":
    unittest.main()
