"""Ordinary launcher and Git checkout regressions; no hostile code execution."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / 'scripts/check_ryzen_evidence.py'


class RyzenEvidenceBoundaryTests(unittest.TestCase):
    def test_fresh_isolated_no_site_archive_check(self):
        result = subprocess.run([sys.executable, '-I', '-S', '-B', str(CHECKER)],
                                cwd=ROOT, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Ran 2 tests', result.stderr)

    def test_missing_isolation_flags_rejected(self):
        # Ordinary clean processes only, not startup-hook/cache poisoning probes.
        for flags in (['-I', '-B'], ['-S', '-B']):
            with self.subTest(flags=flags):
                result = subprocess.run([sys.executable, *flags, str(CHECKER)],
                                        cwd=ROOT, capture_output=True, text=True, timeout=15)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('requires a fresh interpreter', result.stderr)

    def test_content_addressed_objects_survive_autocrlf_checkout(self):
        folder = ROOT / 'fixtures/batch-composition-ryzen-20261020'
        commitment = json.loads((folder / 'commitment.json').read_bytes())
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            objects = checkout / 'fixtures/batch-composition-ryzen-20261020/objects'
            objects.mkdir(parents=True)
            shutil.copyfile(ROOT / 'fixtures/.gitattributes', checkout / 'fixtures/.gitattributes')
            names = [n for n in commitment['files'] if n.startswith('objects/')]
            for name in names:
                shutil.copyfile(folder / name, objects / Path(name).name)
            def git(*args):
                subprocess.run(['git', '-c', 'core.autocrlf=true', '-c', 'core.safecrlf=false',
                                *args], cwd=checkout, check=True, capture_output=True)
            git('init', '--quiet')
            git('add', 'fixtures')
            # Remove only generated copies, then materialize the actual Git index.
            for path in objects.iterdir():
                path.unlink()
            git('checkout-index', '--all', '--force')
            for name in names:
                actual = (objects / Path(name).name).read_bytes()
                self.assertEqual('sha256:' + hashlib.sha256(actual).hexdigest(),
                                 commitment['files'][name], name)


if __name__ == '__main__':
    unittest.main()
