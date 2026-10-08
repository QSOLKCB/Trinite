"""Failure retention, resource margins, checkpoint agreement and Git byte pins."""
from dataclasses import replace
import importlib.util
import itertools
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from trinite.comparison import LANES, WORKLOADS, SEEDS, freeze_request, summarize
from trinite.comparison_checkpoint import snapshot
from trinite.comparison_training import protocol, state_for
from trinite.contracts import ContractError, parse_json
from trinite.model import Decoder
from trinite.qec_source import verify_pin

ROOT = Path(__file__).resolve().parents[2]
REQUEST = 'sha256:' + '0' * 64
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def make_cell(directory, workload, lane, seed, request_identity):
    training = {name: 'same' for name in (
        'initialization_identity', 'dataset_identity', 'manifest_identity',
        'plan_identity', 'schedule_identity')}
    training.update(workload=workload, lane=lane, seed=seed,
        parameter_count=10, target_tokens=20, steps=4, latent_float32_bytes=40,
        training_wall_seconds_hex=(1.).hex())
    return {'outcome': 'completed', 'training': training,
        'evaluation': {'scores': {'test': {'family_macro_accuracy_hex': (.5).hex()}}}}


class ReviewFixTests(unittest.TestCase):
    def summarize_cells(self, root, factory):
        for workload, seed, lane in itertools.product(WORKLOADS, SEEDS, LANES):
            directory = root / f'{workload}-{seed}-{lane}'
            directory.mkdir()
            (directory / 'report.json').write_bytes(b'{}')
        with (patch('trinite.comparison.read_request'),
              patch('trinite.comparison.verified_cell', side_effect=factory)):
            result = summarize(root, root / 'request.json', REQUEST, {})
        self.assertEqual(parse_json((root / 'summary.json').read_bytes(), canonical=True), result)
        self.assertEqual(len(result['cells']), 27)
        return result

    def test_late_pair_mismatches_retain_all_cells_and_no_partial_groups(self):
        for field in ('initialization_identity', 'dataset_identity', 'manifest_identity',
                      'plan_identity', 'parameter_count', 'target_tokens', 'schedule_identity', 'steps'):
            with self.subTest(field=field):
                def mismatch(*args):
                    cell = make_cell(*args)
                    if args[1:4] == ('ququart', 'four-state', 2):
                        cell['training'][field] = 'different'
                    return cell
                with tempfile.TemporaryDirectory() as directory:
                    result = self.summarize_cells(Path(directory), mismatch)
                self.assertEqual(result['outcome'], 'failed')
                self.assertEqual(result['groups'], {})
                self.assertTrue(all(c['outcome'] == 'completed' for c in result['cells']))
                self.assertIn('unmatched comparison ' + field, result['aggregate_errors'][0])

    def test_latent_byte_ratios_cover_equal_lower_and_above_margin(self):
        def resources(*args):
            cell = make_cell(*args)
            # Synthetic values test the margin independently of today's equal
            # float32 storage. These records do not claim packed inference.
            if args[2] == 'four-state':
                cell['training']['latent_float32_bytes'] = (20, 40, 80)[args[3]]
            return cell
        with tempfile.TemporaryDirectory() as directory:
            result = self.summarize_cells(Path(directory), resources)
        self.assertEqual(result['outcome'], 'completed')
        self.assertEqual(result['aggregate_errors'], [])
        for workload in WORKLOADS:
            for lane in ('dense', 'ternary'):
                group = result['groups'][workload][lane]
                self.assertEqual(group['paired_latent_byte_ratio_hex'], [(1.).hex()] * 3)
                self.assertEqual(group['latent_byte_margin_each_seed'], [True] * 3)
            group = result['groups'][workload]['four-state']
            self.assertEqual(group['paired_latent_byte_ratio_hex'], [v.hex() for v in (.5, 1., 2.)])
            self.assertEqual(group['latent_byte_margin_each_seed'], [True, True, False])

    def test_malformed_late_metrics_retain_failed_summary(self):
        mutations = [('latent_float32_bytes', v) for v in (0, -1, True, '40')]
        mutations += [('training_wall_seconds_hex', v) for v in ('nan', 'inf', '0x0p+0', '-0x1p+0', 'not-a-number')]
        mutations += [('accuracy', v) for v in ('nan', 'inf', (1.1).hex(), (-.1).hex())]
        mutations += [('missing', None)]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                def invalid(*args):
                    cell = make_cell(*args)
                    if args[1:4] == ('ququart', 'four-state', 2):
                        if field == 'accuracy':
                            cell['evaluation']['scores']['test']['family_macro_accuracy_hex'] = value
                        elif field == 'missing':
                            del cell['training']['latent_float32_bytes']
                        else:
                            cell['training'][field] = value
                    return cell
                with tempfile.TemporaryDirectory() as directory:
                    result = self.summarize_cells(Path(directory), invalid)
                self.assertEqual(result['outcome'], 'failed')
                self.assertEqual(result['groups'], {})
                self.assertEqual(len(result['aggregate_errors']), 1)

    def test_snapshot_rejects_state_lane_before_serialization(self):
        state = state_for('formal', 'four-state', 0)
        state.lane = 'dense'
        with patch('trinite.comparison_checkpoint.save') as serialize:
            with self.assertRaisesRegex(ContractError, 'model/plan/lane mismatch'):
                snapshot(state, 'formal', REQUEST)
            serialize.assert_not_called()

    def test_snapshot_rejects_model_plan_mismatch_before_serialization(self):
        state = state_for('formal', 'four-state', 0)
        state.model = Decoder(replace(state.config.model, context_length=32), lane='four-state', seed=0)
        with patch('trinite.comparison_checkpoint.save') as serialize:
            with self.assertRaisesRegex(ContractError, 'model/plan/lane mismatch'):
                snapshot(state, 'formal', REQUEST)
            serialize.assert_not_called()

    def test_runner_retains_success_pair_mismatch_and_worker_timeouts(self):
        spec = importlib.util.spec_from_file_location('review_fix_runner', ROOT / 'scripts/run_comparison.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for mode in ('success', 'pair-mismatch', 'train-timeout', 'evaluate-timeout'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                request = root / 'request.json'
                frozen = freeze_request(request)
                output = root / 'comparison'
                stages = []
                def launch(stage, cell, *args):
                    stages.append(stage)
                    if mode == stage + '-timeout':
                        raise subprocess.TimeoutExpired('comparison-cell', 180)
                    cell.mkdir(exist_ok=True)
                    if stage == 'evaluate':
                        (cell / 'report.json').write_bytes(b'{}')
                    return subprocess.CompletedProcess(['comparison-cell'], 0)
                def verified(*args):
                    cell = make_cell(*args)
                    if mode == 'pair-mismatch' and args[1:4] == ('ququart', 'four-state', 2):
                        cell['training']['initialization_identity'] = 'different'
                    return cell
                argv = ['run_comparison.py', '--request', str(request),
                        '--request-identity', frozen['request_identity'], '--output', str(output)]
                with (patch.object(sys, 'argv', argv),
                      patch.object(runner, 'launch', side_effect=launch),
                      patch('trinite.comparison.verified_cell', side_effect=verified),
                      patch('builtins.print')):
                    exit_code = runner.main()
                result = parse_json((output / 'summary.json').read_bytes(), canonical=True)
                self.assertEqual(len(result['cells']), 27)
                self.assertEqual(exit_code, int(mode != 'success'))
                self.assertEqual(result['outcome'], 'completed' if mode == 'success' else 'failed')
                expected_stages = ['train'] * 27 + ([] if mode == 'train-timeout' else ['evaluate'] * 27)
                self.assertEqual(stages, expected_stages)
                if mode != 'success':
                    self.assertEqual(result['groups'], {})
                if mode.endswith('timeout'):
                    self.assertEqual(len(result['worker_errors']), 27)
                    self.assertEqual(len(list(output.glob('*-' + mode[:-8] + '.log'))), 27)
                    self.assertTrue(all(c['outcome'] == 'failed' and 'exceeded 180 seconds' in c['errors']
                                        for c in result['cells']))
                elif mode == 'pair-mismatch':
                    self.assertIn('unmatched comparison', result['aggregate_errors'][0])

    def test_workflow_timeout_covers_workers_and_retention_headroom(self):
        workflow = (ROOT / '.github/workflows/comparison.yml').read_text()
        timeout_lines = [line for line in workflow.splitlines() if line.strip().startswith('timeout-minutes:')]
        self.assertEqual(len(timeout_lines), 1)
        minutes = int(timeout_lines[0].split(':', 1)[1])
        resources = protocol()['resources']
        worker_seconds = resources['matrix_max_cells'] * 2 * resources['cell_max_seconds']
        self.assertGreaterEqual(minutes * 60, worker_seconds + 18 * 60)
        self.assertIn('if: always()', workflow)

    def test_autocrlf_checkout_preserves_pinned_qec_license_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repo'
            root.mkdir()
            shutil.copyfile(ROOT / '.gitattributes', root / '.gitattributes')
            package = root / 'src/trinite'
            package.mkdir(parents=True)
            pin = parse_json((ROOT / 'src/trinite/qec-pin.json').read_bytes(), canonical=True)
            for name in ('qec-pin.json', *pin['files']):
                target = package / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / 'src/trinite' / name, target)
            def git(*args):
                return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True, text=True)
            git('init', '--quiet')
            git('config', 'core.autocrlf', 'true')
            git('add', '.')
            checkout = Path(directory) / 'checkout'
            checkout.mkdir()
            git('checkout-index', '--all', '--prefix=' + str(checkout) + '/')
            attributes = git('check-attr', 'text', 'eol', '--', 'src/trinite/QEC-LICENSE.txt').stdout
            self.assertIn('src/trinite/QEC-LICENSE.txt: text: set', attributes)
            self.assertIn('src/trinite/QEC-LICENSE.txt: eol: lf', attributes)
            checked = checkout / 'src/trinite'
            self.assertNotIn(b'\r\n', (checked / 'QEC-LICENSE.txt').read_bytes())
            self.assertEqual(verify_pin(checked), verify_pin())
            # A stale existing checkout can retain converted unchanged files.
            # Exercise the documented index refresh without changing pinned bytes.
            license_path = package / 'QEC-LICENSE.txt'
            license_path.write_bytes(license_path.read_bytes().replace(b'\n', b'\r\n'))
            with self.assertRaises(ContractError):
                verify_pin(package)
            git('checkout-index', '--force', '--', 'src/trinite/QEC-LICENSE.txt')
            self.assertEqual(verify_pin(package), verify_pin())


if __name__ == '__main__':
    unittest.main()
