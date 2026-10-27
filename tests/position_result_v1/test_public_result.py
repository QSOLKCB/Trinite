"""Original-byte, all-cell, no-execution and rejection controls."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError
from position_result_v1.check import FIXTURE, ROOT, verify


class PublicResultTests(unittest.TestCase):
    def test_complete_result_and_no_model_execution(self):
        with patch('trinite.model.Decoder.__init__', side_effect=AssertionError('model construction')), \
             patch('trinite.training.run_steps', side_effect=AssertionError('optimizer run')), \
             patch('position_study_v1.numerical.restore', side_effect=AssertionError('checkpoint restore')):
            report = verify()
        self.assertEqual(report['original_files'], 115)
        self.assertEqual(report['prediction_records'], 5232)
        for flag in ('closed_custody', 'checkpoint_tensors_verified', 'inference_replayed',
                     'optimizer_replayed', 'fresh_confirmation', 'learning_adequate',
                     'phase5_ready', 'release_ready'):
            self.assertFalse(report[flag])
        self.assertFalse(report['criteria']['all_descriptive_criteria_met'])
        for seed in range(3):
            for arm, rows, families in (('control', (16,16,22), (4,4,7)),
                                        ('candidate', (17,18,26), (3,4,10))):
                cell = report['cells'][f'{arm}-{seed}']
                self.assertEqual(cell['seen']['correct'], 154)
                self.assertEqual(cell['seen']['all_renderings_correct_families'], 64)
                self.assertEqual(cell['development']['unseen']['correct'], rows[seed])
                self.assertEqual(cell['development']['unseen']['all_renderings_correct_families'], families[seed])
                for s in ('length3', 'length4'):
                    self.assertEqual(cell['development'][s]['correct'], 0)
                    self.assertEqual(cell['development'][s]['rows'], 23)
            self.assertEqual(report['criteria']['per_seed'][str(seed)]['unseen_gain'], seed == 2)

    def test_foreign_import_origin_rejected(self):
        import position_study_v1.numerical as numerical
        with patch.object(numerical, '__file__', '/tmp/foreign/numerical.py'):
            with self.assertRaisesRegex(ContractError, 'foreign root'):
                verify()

    def test_foreign_flattened_import_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            shutil.copyfile(ROOT/'scripts/position_study_v1/numerical.py', Path(temp)/'numerical.py')
            code = """
import sys
import position_study_v1
position_study_v1.__path__.insert(0, sys.argv[1])
import position_study_v1.numerical
from position_result_v1.check import verify
from trinite.contracts import ContractError
try:
    verify()
except ContractError as error:
    assert 'foreign root' in str(error), str(error)
else:
    raise AssertionError('foreign import accepted')
"""
            env = dict(os.environ, PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'scripts'))
            result = subprocess.run([sys.executable, '-B', '-c', code, temp],
                                    cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_tampered_original_rejected(self):
        for name in ('matrix.json', 'source-bindings.json', 'candidate-0/training-0.json',
                     'candidate-1/development-unseen.json', 'control-2/description.json',
                     'control-0/completion.json', 'candidate-2/raw-training-4000.jsonl',
                     'control-1/raw-development-length3.jsonl'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)/'public'
                shutil.copytree(FIXTURE, root)
                path = root/name
                path.write_bytes(path.read_bytes()+b' ')
                with self.assertRaises(ContractError):
                    verify(root)

    def test_missing_or_extra_member_rejected(self):
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

    def test_symlink_and_symlink_ancestor_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'public'
            shutil.copytree(FIXTURE, root)
            link = Path(temp)/'alias'
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ContractError):
                verify(link)
            path = root/'candidate-2/report.json'
            path.unlink()
            path.symlink_to(FIXTURE/'candidate-2/report.json')
            with self.assertRaises(ContractError):
                verify(root)

    def test_source_change_rejected(self):
        import position_result_v1.check as check
        real = check.read
        def altered(path):
            value = real(path)
            return value+b' ' if Path(path) == ROOT/'scripts/position_study_v1/data.py' else value
        with patch.object(check, 'read', side_effect=altered):
            with self.assertRaisesRegex(ContractError, 'producer source differs'):
                verify()


if __name__ == '__main__':
    unittest.main()
