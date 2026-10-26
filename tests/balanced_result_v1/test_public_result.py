"""Original-byte, admission, all-cell and no-execution preservation controls."""
from pathlib import Path
import shutil
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError
from balanced_result_v1.check import FIXTURE, verify


class PublicResultTests(unittest.TestCase):
    def test_complete_result_and_no_model_execution(self):
        with patch('trinite.model.Decoder.__init__', side_effect=AssertionError('model construction')), \
             patch('trinite.training.run_steps', side_effect=AssertionError('optimizer run')):
            report = verify()
        self.assertEqual(report['original_files'], 43)
        self.assertFalse(report['closed_custody'])
        self.assertFalse(report['optimizer_replayed'])
        for seed in range(3):
            control = report['cells'][f'control-{seed}']
            candidate = report['cells'][f'candidate-{seed}']
            self.assertEqual(control['unique_training']['correct'], 20)
            self.assertEqual(candidate['unique_training']['correct'], 144)
            self.assertEqual(control['primary_in_range']['correct'], 4)
            self.assertEqual(candidate['primary_in_range']['correct'], (8, 0, 0)[seed])
            for cell in (control, candidate):
                self.assertEqual(cell['seen_anchors']['correct'], 20)
                self.assertEqual(cell['primary_in_range']['rows'], 56)
                self.assertEqual(cell['primary_in_range']['families'], 15)
                self.assertEqual(cell['development']['range-length']['correct'], 0)
                self.assertEqual(cell['development']['range-length']['rows'], 138)

    def test_foreign_import_origin_rejected(self):
        import balanced_addition_v1.numerical as numerical
        with patch.object(numerical, '__file__', '/tmp/foreign/numerical.py'):
            with self.assertRaisesRegex(ContractError, 'foreign root'):
                verify()

    def test_foreign_flattened_package_import_rejected(self):
        from balanced_result_v1.check import ROOT
        with tempfile.TemporaryDirectory() as temp:
            shutil.copyfile(ROOT/'scripts/balanced_addition_v1/numerical.py',
                            Path(temp)/'numerical.py')
            code = """
import sys
import balanced_addition_v1
balanced_addition_v1.__path__.insert(0, sys.argv[1])
import balanced_addition_v1.numerical
from balanced_result_v1.check import verify
from trinite.contracts import ContractError
try:
    verify()
except ContractError as error:
    assert 'foreign root' in str(error), str(error)
else:
    raise AssertionError('foreign import accepted')
"""
            environment = dict(os.environ, PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'scripts'))
            result = subprocess.run([sys.executable, '-B', '-c', code, temp],
                                    cwd=ROOT, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_tampered_original_rejected(self):
        for name in ('matrix.json', 'source-bindings.json',
                     'candidate-0/training-4000.json',
                     'candidate-1/development-shared-answer.json',
                     'control-2/description.json', 'control-0/completion.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)/'public'
                shutil.copytree(FIXTURE, root)
                path = root/name
                path.write_bytes(path.read_bytes()+b' ')
                with self.assertRaises(ContractError):
                    verify(root)

    def test_missing_or_extra_cell_rejected(self):
        for extra in (False, True):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)/'public'
                shutil.copytree(FIXTURE, root)
                if extra:
                    (root/'unreviewed-private-file.json').write_text('{}')
                else:
                    shutil.rmtree(root/'candidate-2')
                with self.assertRaises(ContractError):
                    verify(root)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'public'
            shutil.copytree(FIXTURE, root)
            path = root/'candidate-2/report.json'
            path.unlink()
            path.symlink_to(FIXTURE/'candidate-2/report.json')
            with self.assertRaises(ContractError):
                verify(root)


if __name__ == '__main__':
    unittest.main()
