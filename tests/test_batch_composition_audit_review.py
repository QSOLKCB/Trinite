"""Independent v3 transport regressions; synthetic records, never model execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_batch_composition_v3.py"
MODES = (((), ""), (("-O",), ""), ((), "1"), ((), "2"))


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def identity(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def snapshot(root):
    return {p.relative_to(root).as_posix(): identity(p.read_bytes())
            for p in root.rglob("*") if p.is_file()}


class AuditReviewTests(unittest.TestCase):
    """Tiny source-bound fixtures isolate auditor behavior, not numerical validation."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / "artifact"
        self.source = self.root / "source"
        self.artifact.mkdir()
        files = {
            "src/trinite/__init__.py": "",
            "src/trinite/contracts.py": (
                "import json\n"
                "def json_bytes(value):\n"
                "    return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\\n').encode()\n"
                "def parse_json(raw,canonical=False):\n"
                "    value=json.loads(raw)\n"
                "    if canonical and json_bytes(value)!=raw: raise ValueError('not canonical')\n"
                "    return value\n"),
            "scripts/batch_composition_v1/__init__.py": "",
            "scripts/batch_composition_v1/spec.py": "LAYOUTS=('synthetic',)\n",
            "scripts/batch_composition_v1/validation.py": (
                "def validate_report(report,request):\n"
                "    return {'conforms':report['conforms'],'synthetic_validator':True}\n"),
        }
        for name, text in files.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        source_ids = {name: identity((self.source / name).read_bytes()) for name in files}
        self.approved = self.root / "approved.json"
        self.approved.write_bytes(canonical({
            "source": source_ids, "protocol_identity": identity(canonical({}))}))
        self.request = {"schema": "trinite.batch-composition-request.v2",
                        "source": source_ids, "protocol": {}, "environment": {}}
        self.request_raw = canonical(self.request)
        self.report = {
            "schema": "trinite.batch-composition-report.v2", "conforms": True,
            "cells": [{
                "seed": 0, "arm": "synthetic", "layouts": {"synthetic": {}},
                "comparisons": [{
                    "forced": [{"max_absolute_hex": (0.0).hex(),
                                "max_normalized_error_hex": (0.0).hex(),
                                "violating_token_ids": []}],
                    "independent_generation": {}}]}]}
        self.write_packet()

    def write_packet(self):
        report_raw = canonical(self.report)
        request_id, report_id = identity(self.request_raw), identity(report_raw)
        (self.artifact / "batch-composition-request.json").write_bytes(self.request_raw)
        (self.artifact / "batch-composition-report.json").write_bytes(report_raw)
        for action in ("run", "verify"):
            directory = self.artifact / action
            directory.mkdir(exist_ok=True)
            (directory / "invocation.json").write_bytes(canonical({
                "schema": "trinite.batch-composition-invocation.v2", "action": action,
                "request_identity": request_id,
                "expected_identity": report_id if action == "verify" else None,
                "deadline_seconds": 300, "final_name": "batch-composition-report.json"}))
            (directory / "status.json").write_bytes(canonical({
                "schema": "trinite.batch-composition-status.v2",
                "outcome": "completed" if self.report["conforms"] else "completed_nonconformant",
                "exit_code": 0 if self.report["conforms"] else 2,
                "wall_seconds_hex": (1.0).hex()}))
            (directory / "computed-report.json").write_bytes(report_raw)
            (directory / "completed-cells.jsonl").write_bytes(
                b"".join(canonical(cell) for cell in self.report["cells"]))
            (directory / "completed-layouts.jsonl").write_bytes(canonical({
                "seed": 0, "arm": "synthetic", "layout": "synthetic", "result": {}}))
        (self.artifact / "verify/replay-receipt.json").write_bytes(canonical({
            "schema": "trinite.batch-composition-replay-receipt.v2",
            "request_identity": request_id, "expected_identity": report_id,
            "actual_identity": report_id, "exact_match": True,
            "conforms": self.report["conforms"],
            "scope": "same-environment complete inference replay, not optimizer replay"}))

    def invoke(self, output, mode=((), ""), artifact=None, script=SCRIPT):
        flags, optimize = mode
        env = dict(os.environ)
        env.pop("PYTHONOPTIMIZE", None)
        env.pop("PYTHONPATH", None)
        if optimize:
            env["PYTHONOPTIMIZE"] = optimize
        return subprocess.run([
            sys.executable, "-B", *flags, str(script),
            "--artifact-dir", str(artifact or self.artifact),
            "--source-root", str(self.source), "--approved-review", str(self.approved),
            "--output", str(output)], cwd=self.root, env=env,
            text=True, capture_output=True, timeout=15)

    def mutate(self, name, key, value):
        path = self.artifact / name
        record = json.loads(path.read_bytes())
        record[key] = value
        path.write_bytes(canonical(record))

    def replace_synthetic_validator(self, body):
        name = "scripts/batch_composition_v1/validation.py"
        path = self.source / name
        path.write_text(body)
        self.request["source"][name] = identity(path.read_bytes())
        approved = json.loads(self.approved.read_bytes())
        approved["source"] = self.request["source"]
        self.approved.write_bytes(canonical(approved))
        self.request_raw = canonical(self.request)
        self.write_packet()

    def test_optimization_never_disables_rejection(self):
        mutations = (
            ("verify/replay-receipt.json", "exact_match", False),
            ("verify/replay-receipt.json", "exact_match", 1),
            ("verify/replay-receipt.json", "actual_identity", identity(b"wrong")),
            ("verify/replay-receipt.json", "conforms", 1),
            ("verify/status.json", "outcome", "timeout"),
            ("verify/status.json", "exit_code", False),
            ("verify/status.json", "exit_code", 0.0),
            ("run/status.json", "outcome", "failed"),
            ("run/invocation.json", "deadline_seconds", True),
        )
        for mutation in mutations:
            self.write_packet()
            self.mutate(*mutation)
            for mode in MODES:
                with self.subTest(mutation=mutation, mode=mode):
                    output = self.root / "rejected.json"
                    before = snapshot(self.artifact), snapshot(self.source)
                    result = self.invoke(output, mode)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertFalse(output.exists())
                    self.assertEqual((snapshot(self.artifact), snapshot(self.source)), before)

    def test_valid_and_nonconformant_receipts_preserve_provenance_in_every_mode(self):
        for conforms in (True, False):
            self.report["conforms"] = conforms
            self.write_packet()
            for index, mode in enumerate(MODES):
                with self.subTest(conforms=conforms, mode=mode):
                    output = self.root / ("accepted-%s-%d.json" % (conforms, index))
                    before = snapshot(self.artifact), snapshot(self.source)
                    result = self.invoke(output, mode)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    receipt = json.loads(output.read_bytes())
                    self.assertEqual(receipt["schema"], "trinite.independent-batch-ci-audit.v3")
                    self.assertEqual(receipt["auditor_identity"], identity(SCRIPT.read_bytes()))
                    self.assertEqual(receipt["approved_review_identity"], identity(self.approved.read_bytes()))
                    self.assertIs(receipt["strict_recorded_report_validation"]["conforms"], conforms)
                    self.assertIs(receipt["local_inference_executed"], False)
                    self.assertIs(receipt["read_only_inputs_unchanged"], True)
                    self.assertEqual((snapshot(self.artifact), snapshot(self.source)), before)

    def test_output_path_aliases_and_existing_destinations_never_modify_inputs(self):
        nested = self.artifact / "nested"
        nested.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(self.artifact, target_is_directory=True)
        parent_alias = self.root / "parent-alias"
        parent_alias.symlink_to(nested, target_is_directory=True)
        existing = self.root / "existing.json"
        existing.write_bytes(b"preserve")
        linked = self.root / "linked.json"
        linked.symlink_to(self.artifact / "batch-composition-report.json")
        dangling = self.root / "dangling.json"
        dangling.symlink_to(self.artifact / "new.json")
        hardlink = self.root / "hardlink.json"
        os.link(self.artifact / "batch-composition-report.json", hardlink)
        cases = (
            self.artifact, self.artifact / "new.json", nested / "new.json",
            alias / "new.json", parent_alias / "new.json",
            Path("artifact/new.json"), self.root / "artifact/../artifact/new.json",
            existing, linked, dangling, hardlink,
        )
        before = snapshot(self.artifact)
        for output in cases:
            with self.subTest(output=output):
                result = self.invoke(output, (("-O",), ""))
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(snapshot(self.artifact), before)
                self.assertEqual(existing.read_bytes(), b"preserve")
        result = self.invoke(self.artifact / "new.json", artifact=alias)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(snapshot(self.artifact), before)

    def test_status_types_fields_pairings_and_nonfinite_time(self):
        spec = importlib.util.spec_from_file_location("reviewed_auditor", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        good = json.loads((self.artifact / "run/status.json").read_bytes())
        self.assertIs(module.validate_status(good, True), True)
        for key, value in (
            ("exit_code", True), ("exit_code", False), ("exit_code", 0.0),
            ("exit_code", "0"), ("outcome", "timeout"), ("outcome", "unknown"),
            ("wall_seconds_hex", "nan"), ("wall_seconds_hex", "inf"),
            ("wall_seconds_hex", (-1.0).hex()), ("wall_seconds_hex", "0x1p+0"),
            ("extra", "ignored"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(module.AuditError):
                module.validate_status(dict(good, **{key: value}), True)
        with self.assertRaises(module.AuditError):
            module.validate_status(good, 1)
        with self.assertRaises(module.AuditError):
            module.validate_status(good, False)
        for outcome, code in (("timeout", 124), ("failed", 1),
                              ("interrupted", 130), ("controller_failed", 1)):
            self.assertIs(module.validate_status(
                dict(good, outcome=outcome, exit_code=code), True), False)

    def test_changed_auditor_copy_cannot_certify_its_new_bytes(self):
        copy = self.root / "auditor-copy.py"
        copy.write_bytes(SCRIPT.read_bytes())
        self.replace_synthetic_validator(
            "from pathlib import Path\nimport sys\n"
            "def validate_report(report,request):\n"
            "    path=Path(sys.argv[0])\n"
            "    path.write_bytes(path.read_bytes()+b'\\n# injected change\\n')\n"
            "    return {'conforms':report['conforms'],'synthetic_validator':True}\n")
        output = self.root / "not-certified.json"
        before = snapshot(self.artifact)
        result = self.invoke(output, script=copy)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("auditor changed during audit", result.stderr)
        self.assertFalse(output.exists())
        self.assertEqual(snapshot(self.artifact), before)

    def test_output_parent_retargeted_during_validation_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        link = self.root / "changing-link"
        link.symlink_to(outside, target_is_directory=True)
        self.replace_synthetic_validator(
            "from pathlib import Path\n"
            "def validate_report(report,request):\n"
            "    link=Path(%r)\n"
            "    link.unlink()\n"
            "    link.symlink_to(Path(%r),target_is_directory=True)\n"
            "    return {'conforms':report['conforms'],'synthetic_validator':True}\n"
            % (str(link), str(self.artifact)))
        before = snapshot(self.artifact)
        result = self.invoke(link / "not-certified.json")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("outside artifact directory", result.stderr)
        self.assertFalse((outside / "not-certified.json").exists())
        self.assertFalse((self.artifact / "not-certified.json").exists())
        self.assertEqual(snapshot(self.artifact), before)


if __name__ == "__main__":
    unittest.main()
