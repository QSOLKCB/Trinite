"""Launch retained-evidence checks across the isolated no-site boundary.

No project imports or numerical work occur in this test process.
"""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / 'scripts/check_cyclic_learning_evidence.py'


class CyclicEvidenceTests(unittest.TestCase):
    def test_isolated_offline_archive_and_functional_regressions(self):
        for optimize in ([], ['-O']):
            result = subprocess.run([sys.executable, '-I', '-S', '-B', *optimize,
                                     str(CHECKER), '--self-test'], capture_output=True,
                                    text=True, timeout=60, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('OK', result.stderr)


if __name__ == '__main__':
    unittest.main()
