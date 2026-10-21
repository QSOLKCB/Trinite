"""Ordinary discovery invokes only the isolated, no-site evidence checker."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / 'scripts/check_tiny_preflight_evidence.py'


class TinyPreflightEvidenceTests(unittest.TestCase):
    def test_isolated_archive_and_functional_checks(self):
        for optimize in ([], ['-O']):
            result = subprocess.run([sys.executable, '-I', '-S', '-B', *optimize,
                                     str(CHECKER), '--self-test'],
                                    capture_output=True, text=True, timeout=120, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('OK', result.stderr)

    def test_nonisolated_invocation_rejected(self):
        result = subprocess.run([sys.executable, '-S', '-B', str(CHECKER)],
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires python -I -S', result.stderr)

    def test_environment_and_cwd_cannot_inject_modules(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ('json.py', 'sitecustomize.py', 'tiny_output_v1.py', 'needle.py'):
                (Path(temp) / name).write_text("raise RuntimeError('injected module executed')\n")
            result = subprocess.run([sys.executable, '-I', '-S', '-B', str(CHECKER)],
                                    cwd=temp, env={**os.environ, 'PYTHONPATH': temp},
                                    capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
