from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONPATH=str(ROOT/"src"))
        return subprocess.run([sys.executable, "-m", "trinite", *map(str, args)],
                              cwd=ROOT, env=env, capture_output=True, text=True,
                              timeout=15)

    def test_config_and_tokenizer(self):
        result = self.run_cli("validate-config", ROOT/"configs/reference.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["planned_parameters"], 1_247_232)
        encoded = self.run_cli("tokenize", "é=1", "--answer-start", 3, "--pad-to", 8)
        self.assertEqual(encoded.returncode, 0, encoded.stderr)
        self.assertEqual(json.loads(encoded.stdout)["loss_mask"], [0, 0, 0, 0, 1, 1, 0, 0])

    def test_generate_and_audit_from_new_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"output"
            result = self.run_cli("generate-fixture", path, "--seed", 1)
            self.assertEqual(result.returncode, 0, result.stderr)
            audit = self.run_cli("audit-fixture", path)
            self.assertEqual(audit.returncode, 0, audit.stderr)
            self.assertEqual(result.stdout, audit.stdout)
            repeat = self.run_cli("generate-fixture", path)
            self.assertEqual(repeat.returncode, 1)
            self.assertEqual(repeat.stdout, "")
            self.assertNotIn("Traceback", repeat.stderr)

    def test_failure_status_and_no_false_success(self):
        for args in (("validate-config", "missing.json"), ("tokenize", "x"*255),
                     ("audit-fixture", "missing-directory"),
                     ("generate-fixture", "must-not-exist", "--seed", "-1")):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((ROOT/"must-not-exist").exists())


if __name__ == "__main__":
    unittest.main()
