"""Actual pinned CPU CLI and fresh source-bound report transport checks."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from trinite.contracts import identity, parse_json
import check_position_coupling

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / 'scripts/check_position_coupling.py'


class PositionCliTests(unittest.TestCase):
    def run_cli(self, *arguments):
        env = dict(os.environ)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        return subprocess.run([sys.executable, '-B', str(CLI), *map(str, arguments)],
                              cwd=ROOT, env=env, capture_output=True, timeout=60)

    def assert_report(self, raw, before):
        report = parse_json(raw, canonical=True)
        self.assertEqual(report['schema'], 'trinite.position-conformance-report.v1')
        self.assertEqual(report['outcome'], 'passed', report.get('error'))
        self.assertFalse(report['target_training_run'])
        self.assertFalse(report['learning_claim'])
        self.assertFalse(report['closed_custody'])
        self.assertEqual(report['source'], before)
        self.assertEqual(check_position_coupling.sources(), before)
        for name, expected in report['source'].items():
            self.assertEqual(identity((ROOT / name).read_bytes()), expected, name)
        for name in ('scripts/check_position_coupling.py', 'scripts/position_coupling_v1/model.py',
                     'scripts/position_coupling_v1/positions.py', 'tests/position_coupling_v1/test_cli.py',
                     'src/trinite/model.py', 'src/trinite/training.py', 'requirements.lock',
                     'docs/POSITION-COUPLING.md', '.github/workflows/position-coupling-v1.yml'):
            self.assertIn(name, report['source'])
        self.assertEqual([cell['seed'] for cell in report['checks']['cells']], [0, 1, 2])
        self.assertTrue(all(cell['parameter_count'] == 86528 for cell in report['checks']['cells']))
        return report

    def test_stdout_report_passes_and_binds_unchanged_current_sources(self):
        before = check_position_coupling.sources()
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assert_report(result.stdout, before)

    def test_fresh_external_output_passes_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'report.json'
            before = check_position_coupling.sources()
            result = self.run_cli('--output', destination)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(result.stdout, b'')
            raw = destination.read_bytes()
            self.assert_report(raw, before)
            second = self.run_cli('--output', destination)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(destination.read_bytes(), raw)
            self.assertIn(b'FileExistsError', second.stderr)

    def test_existing_external_non_report_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'original.txt'
            destination.write_bytes(b'historical evidence must remain unchanged\n')
            result = self.run_cli('--output', destination)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(destination.read_bytes(), b'historical evidence must remain unchanged\n')

    def test_repository_output_and_symlink_alias_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            # An external alias does not turn the source tree into an allowed sink.
            alias = Path(directory) / 'checkout'
            alias.symlink_to(ROOT, target_is_directory=True)
            basename = 'position-forbidden-' + Path(directory).name + '.json'
            actual = ROOT / basename
            self.assertFalse(actual.exists())
            for path in (actual, alias / basename):
                result = self.run_cli('--output', path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b'outside the repository tree', result.stderr)
                self.assertFalse(actual.exists())
                self.assertEqual(result.stdout, b'')

    def test_research_flag_is_rejected_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'report.json'
            result = self.run_cli('--output', destination, '--research')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'unrecognized arguments', result.stderr)
            self.assertFalse(destination.exists())
