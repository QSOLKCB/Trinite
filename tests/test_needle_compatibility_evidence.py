"""Discovery runs the retained evidence audit only in a no-site isolated child."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / 'scripts/check_needle_compatibility_evidence.py'


class NeedleCompatibilityEvidenceTests(unittest.TestCase):
    def test_isolated_archive_and_functional_checks(self):
        for optimize in ([], ['-O']):
            result = subprocess.run([sys.executable, '-I', '-S', '-B', *optimize,
                                     str(CHECKER), '--self-test'],
                                    capture_output=True, text=True, timeout=120, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('Ran 8 tests', result.stderr)
            self.assertIn('OK', result.stderr)

    def test_nonisolated_invocation_rejected(self):
        result = subprocess.run([sys.executable, '-S', '-B', str(CHECKER)],
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires python -I -S', result.stderr)


if __name__ == '__main__':
    unittest.main()
