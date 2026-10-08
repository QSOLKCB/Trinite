from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from trinite import convergence as c
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.foundations_checkpoint import snapshot
from trinite.training import model_identity, run_steps

ROOT = Path(__file__).resolve().parents[2]
REQUEST = 'sha256:'+'0'*64
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def fake_cell(root, workload, seed, candidate, request_identity):
    result = {name: 'same' for name in ('dataset_identity', 'manifest_identity', 'initialization_identity',
                                     'schedule_identity')}
    result.update(workload=workload, seed=seed, candidate=candidate, outcome='completed',
                  steps=1024, parameter_count=86528, target_tokens=21000,
                  curves=[{'training_gate_failures': []}])
    return result


class ConvergenceTests(unittest.TestCase):
    def test_training_only_data_view_and_interrupted_exact_replay(self):
        for candidate in c.CANDIDATES:
            with self.subTest(candidate=candidate):
                plan = replace(c.plan_for(0, candidate), steps=4, validation_every=2)
                state = c.state_for('arithmetic', 0, candidate, plan)
                self.assertIs(state.data['train'], state.data['validation'])
                run_steps(state, stop_after=2)
                payload, metadata = snapshot(state, 'arithmetic', REQUEST)
                resumed = c.restore_state(payload, metadata, 'arithmetic', 0, candidate, REQUEST,
                                          (identity(payload), identity(metadata)), plan)
                self.assertIs(resumed.data['train'], resumed.data['validation'])
                run_steps(state); run_steps(resumed)
                self.assertEqual(state.history, resumed.history)
                self.assertEqual(state.validation, resumed.validation)
                self.assertEqual(model_identity(state.model), model_identity(resumed.model))
                self.assertEqual(snapshot(state, 'arithmetic', REQUEST), snapshot(resumed, 'arithmetic', REQUEST))
        with self.assertRaises(ContractError):
            c.state_for('arithmetic', 0, 'low', c.plan_for(0, 'high'))
        for values in (('arithmetic', True, 'low'), ('unknown', 0, 'low'), ('fold', 0, 'custom')):
            with self.assertRaises(ContractError):
                c.cell_name(*values)

    def test_request_binds_protocol_source_environment_and_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'request.json'
            frozen = c.freeze_request(path)
            c.read_request(path, frozen['request_identity'])
            original = path.read_bytes()
            for key in ('protocol', 'source', 'environment', 'corpora'):
                record = parse_json(original, canonical=True); record[key] = {}
                raw = json_bytes(record); path.write_bytes(raw)
                with self.assertRaises(ContractError):
                    c.read_request(path, identity(raw))
            path.write_bytes(original)
            with self.assertRaises(ContractError):
                c.read_request(path, REQUEST)

    def test_actual_short_cell_evidence_and_mutated_copies(self):
        plan = replace(c.plan_for(0, 'reference'), steps=4, validation_every=2)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); request = base/'request.json'
            frozen = c.freeze_request(request); root = base/'cell'
            with patch.object(c, 'MILESTONES', (2, 4)), patch.object(c, 'plan_for', return_value=plan):
                result = c.train_cell(root, 'arithmetic', 0, 'reference', request, frozen['request_identity'])
                self.assertFalse(result['held_out_scored'])
                checked = c.verified_cell(root, 'arithmetic', 0, 'reference', frozen['request_identity'])
                self.assertEqual(checked, result)
                for name in ('report.json', 'predictions-2.json', 'steps.json', 'metadata.json'):
                    path = root/name; original = path.read_bytes(); path.write_bytes(original+b' ')
                    with self.assertRaises(ContractError):
                        c.verified_cell(root, 'arithmetic', 0, 'reference', frozen['request_identity'])
                    path.write_bytes(original)
                with self.assertRaises(ContractError):
                    c.verified_cell(root, 'arithmetic', 0, 'high', frozen['request_identity'])
                with self.assertRaises(ContractError):
                    c.verified_cell(root, 'arithmetic', 0, 'reference', REQUEST)

    def test_success_and_late_mismatch_summary_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(c, 'read_request'), patch.object(c, 'verified_cell', side_effect=fake_cell):
                result = c.summarize(root, root/'request.json', REQUEST, {})
            self.assertEqual(result['outcome'], 'completed')
            self.assertFalse(result['learning_adequate'])
            self.assertTrue(all(g['training_tasks_pass_all_cells'] for g in result['groups'].values()))
            self.assertEqual(parse_json((root/'summary.json').read_bytes(), canonical=True), result)
        def mismatch(*args):
            result = fake_cell(*args)
            if args[1:4] == ('lattice', 2, 'high'):
                result['initialization_identity'] = 'different'
            return result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(c, 'read_request'), patch.object(c, 'verified_cell', side_effect=mismatch):
                result = c.summarize(root, root/'request.json', REQUEST, {})
            self.assertEqual(result['outcome'], 'failed')
            self.assertEqual(len(result['cells']), 27)
            self.assertEqual(result['groups'], {})
            self.assertIn('unmatched convergence', result['aggregate_errors'][0])
            self.assertTrue((root/'summary.json').exists())

    def test_regressions_and_failed_candidates_remain_descriptive(self):
        records = [fake_cell(None, *cell, REQUEST) for cell in c.cells()]
        records[-1]['curves'][-1]['training_gate_failures'] = ['traversal: below gate']
        groups = c.groups_for(records)
        self.assertFalse(groups['high']['training_tasks_pass_all_cells'])
        self.assertEqual(groups['high']['failed_training_cells'], ['lattice-2-high'])
        self.assertIn('none', groups['high']['selection'])

    def test_stability_records_preclip_norms_and_rejects_nonfinite(self):
        history = [dict(loss_hex=(.5).hex(), gradient_norm_hex=(v).hex()) for v in (0., 1., 2.)]
        value = c.stability(history, 0, 3, '1')
        self.assertEqual(value['clipped_updates'], 1)
        self.assertEqual(value['max_gradient_norm_hex'], (2.).hex())
        history[-1]['loss_hex'] = float('nan').hex()
        with self.assertRaises(ContractError):
            c.stability(history, 0, 3, '1')

    def test_timeouts_retain_all_records_and_summary(self):
        spec = importlib.util.spec_from_file_location('convergence_runner', ROOT/'scripts/run_convergence.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); request = root/'request.json'
            frozen = c.freeze_request(request); output = root/'run'
            argv = ['run_convergence.py', '--request', str(request), '--request-identity', frozen['request_identity'],
                    '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(runner, 'launch',
                    side_effect=subprocess.TimeoutExpired('fixed worker', 240)) as launch, patch('builtins.print'):
                code = runner.main()
            result = parse_json((output/'summary.json').read_bytes(), canonical=True)
            self.assertEqual(code, 1); self.assertEqual(launch.call_count, 27)
            self.assertEqual(len(result['cells']), 27); self.assertEqual(len(result['worker_errors']), 27)
            self.assertEqual(result['groups'], {})
            self.assertFalse(result['held_out_scored'])
            self.assertEqual(len(list(output.glob('*.log'))), 27)

    def test_cumulative_update_timeout_and_no_held_out_model_calls(self):
        plan = replace(c.plan_for(0, 'reference'), steps=4, validation_every=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); request = root/'request.json'
            frozen = c.freeze_request(request)
            # Initial clock, before/after first update. Scoring is never reached.
            with patch.object(c, 'plan_for', return_value=plan), patch.object(c, 'MILESTONES', (2, 4)), \
                 patch.object(c.time, 'perf_counter', side_effect=(0., 0., 121.)), \
                 patch.object(c, 'score') as score:
                with self.assertRaisesRegex(ContractError, 'cumulative update'):
                    c.train_cell(root/'cell', 'arithmetic', 0, 'reference', request, frozen['request_identity'])
                score.assert_not_called()
            self.assertTrue((root/'cell/convergence-failure.json').exists())

    def test_workflow_covers_worker_and_retention_envelope(self):
        text = (ROOT/'.github/workflows/convergence.yml').read_text()
        lines = [line for line in text.splitlines() if line.strip().startswith('timeout-minutes:')]
        self.assertEqual(len(lines), 1)
        minutes = int(lines[0].split(':')[1])
        budget = c.protocol()['resources']
        self.assertGreaterEqual(minutes*60, budget['max_cells']*budget['worker_seconds']+24*60)
        self.assertIn('if: always()', text)


if __name__ == '__main__':
    unittest.main()
