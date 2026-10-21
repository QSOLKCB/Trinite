"""Ordinary discovery invokes the isolated no-site offline verifier."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TeacherEvidenceTests(unittest.TestCase):
    def test_isolated_archive_and_functional_checks(self):
        for optimize in ([], ['-O']):
            result = subprocess.run([sys.executable, '-I', '-S', '-B', *optimize,
                                     str(ROOT / 'scripts/check_teacher_pilot_evidence.py'), '--self-test'],
                                    capture_output=True, text=True, timeout=120, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('OK', result.stderr)


if __name__ == '__main__':
    unittest.main()
