from dataclasses import replace
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite.model import Decoder, CaptureSpec
from trinite.quantizer import quantize_four
from trinite.inspection import inventory
from trinite.training import RunConfig, model_identity, run_steps
from trinite.comparison import (state_for, protocol, sources, greedy, freeze_request, read_request,
                                geometry_measure, train_cell, evaluate_cell, verified_cell, summarize)
from trinite.comparison_checkpoint import snapshot, restore

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
REQUEST = 'sha256:'+'0'*64
PLAN = replace(RunConfig.from_dict(protocol()['training']), steps=4, validation_every=2)


class FourStateTests(unittest.TestCase):
    def test_exact_scale_ties_zero_saturation_and_surrogate(self):
        latent = torch.tensor([[-3., -2., -1., 0., 1., 2., 3., 4.]], requires_grad=True)
        q = quantize_four(latent)
        self.assertEqual(float(q.scale), 1.)
        self.assertEqual(q.codes.tolist(), [[-3, -1, -1, 1, 1, 1, 3, 3]])
        self.assertFalse(q.scale.requires_grad)
        q.weight.backward(torch.arange(1., 9.).reshape(1, 8))
        self.assertEqual(latent.grad.tolist(), [[0, 2, 3, 4, 5, 6, 0, 0]])
        zero = quantize_four(torch.zeros(1, 2))
        self.assertEqual(zero.codes.tolist(), [[1, 1]])
        self.assertGreater(float(zero.scale), 0)

    def test_independent_full_scalar_forward_and_capture_oracle(self):
        spec = importlib.util.spec_from_file_location('four_oracle', ROOT/'tests/comparison/oracle_four.py')
        oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
        config = ModelConfig(context_length=8, blocks=1, width=4, heads=2, feed_forward_width=8)
        model = Decoder(config, lane='four-state')
        parameters = oracle.hand_parameters()
        with torch.no_grad():
            for name, p in model.named_parameters(): p.copy_(torch.tensor(parameters[name]))
        ids = torch.tensor([[256, 49, 43, 257]])
        expected, captures = oracle.forward(parameters, ids[0].tolist(), [1]*4, 'four-state')
        result = model(ids, torch.ones_like(ids, dtype=torch.bool),
                       capture=CaptureSpec(('embedding', 'block.0', 'final'), (0, 1, 2, 3)))
        torch.testing.assert_close(result.logits[0], torch.tensor(expected), rtol=2e-5, atol=2e-6)
        for name, values in captures.items():
            torch.testing.assert_close(result.captures[name][0], torch.tensor(values), rtol=2e-5, atol=2e-6)

    def test_three_lanes_share_initialization_and_inspection_reports_scope(self):
        models = [Decoder(PLAN.model, lane=lane, seed=1) for lane in ('dense', 'ternary', 'four-state')]
        self.assertEqual(len({model_identity(m) for m in models}), 1)
        report = inventory(models[-1])
        self.assertEqual(report['counts']['ternary_forward_elements'], 0)
        self.assertEqual(report['counts']['four_state_forward_elements'], 2048)
        for layer in report['linear_layers']:
            self.assertEqual(set(layer['codes']), {'-3', '-1', '1', '3'})
            self.assertEqual(sum(layer['codes'].values()), layer['elements'])

    def test_bad_inputs_and_effective_overflow_reject_without_mutation(self):
        bad = (None, torch.zeros(2), torch.zeros(0, 2), torch.zeros(1, 2, dtype=torch.float64),
               torch.tensor([[float('nan')]]), torch.ones(1, 2, device='meta'))
        for value in bad:
            with self.assertRaises(ContractError): quantize_four(value)
        maximum = torch.finfo(torch.float32).max
        latent = torch.tensor([[maximum, maximum, maximum, maximum*.1]])
        with self.assertRaises(ContractError): quantize_four(latent)


class ComparisonTests(unittest.TestCase):
    def test_training_consumes_only_train_and_validation_and_matches_budgets(self):
        for workload in ('formal', 'qutrit', 'ququart'):
            states = [state_for(workload, lane, 0, PLAN) for lane in ('dense', 'ternary', 'four-state')]
            self.assertTrue(all(set(s.data) == {'train', 'validation'} for s in states))
            for state in states: run_steps(state)
            self.assertEqual(len({s.target_tokens for s in states}), 1)
            self.assertEqual(len({identity(json_bytes([h['examples'] for h in s.history])) for s in states}), 1)

    def test_snapshot_resumes_all_three_lanes_bit_exact_on_variable_length_tasks(self):
        for lane in ('dense', 'ternary', 'four-state'):
            direct = state_for('ququart', lane, 0, PLAN); run_steps(direct)
            paused = state_for('ququart', lane, 0, PLAN); run_steps(paused, stop_after=2)
            payload, metadata = snapshot(paused, 'ququart', REQUEST)
            replay = restore(payload, metadata, 'ququart', lane, 0, REQUEST,
                             (identity(payload), identity(metadata)), PLAN)
            run_steps(replay)
            self.assertEqual(model_identity(replay.model), model_identity(direct.model))
            self.assertEqual(snapshot(replay, 'ququart', REQUEST), snapshot(direct, 'ququart', REQUEST))

    def test_snapshot_rejects_identity_context_progress_and_tensor_corruption(self):
        state = state_for('qutrit', 'four-state', 0, PLAN); run_steps(state, stop_after=2)
        payload, metadata = snapshot(state, 'qutrit', REQUEST)
        for field, value in (('lane', 'dense'), ('source', {}), ('target_tokens', 0),
                             ('step', True), ('history', []), ('model_identity', REQUEST)):
            changed = parse_json(metadata); changed[field] = value; raw = json_bytes(changed)
            with self.assertRaises(ContractError):
                restore(payload, raw, 'qutrit', 'four-state', 0, REQUEST, (identity(payload), identity(raw)), PLAN)
        with self.assertRaises(ContractError):
            restore(payload+b'bad', metadata, 'qutrit', 'four-state', 0, REQUEST,
                    (identity(payload), identity(metadata)), PLAN)

    def test_frozen_request_rejects_changed_source_protocol_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'request.json'; report = freeze_request(path)
            read_request(path, report['request_identity'])
            original = parse_json(path.read_bytes())
            for field in ('source', 'protocol', 'environment', 'corpora'):
                changed = dict(original); changed[field] = {}; raw = json_bytes(changed); path.write_bytes(raw)
                with self.assertRaises(ContractError): read_request(path, identity(raw))

    def test_greedy_stops_at_eos_and_does_not_use_teacher_forcing(self):
        from types import SimpleNamespace
        class Fake:
            config = SimpleNamespace(context_length=64)
            def __call__(self, ids, mask):
                token = 49 if ids.shape[1] == 3 else 257
                logits = torch.zeros(1, ids.shape[1], 259); logits[0, -1, token] = 1
                return SimpleNamespace(logits=logits)
        self.assertEqual(greedy(Fake(), 'x=', 2), [49, 257])

    def test_geometry_uses_training_only_one_sided_control_and_preserves_rng(self):
        state = state_for('formal', 'four-state', 0, PLAN)
        before, rng = model_identity(state.model), torch.get_rng_state().clone()
        conditions, captures = geometry_measure(state.model, 'formal')
        self.assertEqual(len(captures), 36)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, model_identity(state.model))
        self.assertEqual(set(conditions), {'native', 'one-sided-hash-permutation', 'rademacher'})
        self.assertNotEqual(conditions['native']['pairs'], conditions['one-sided-hash-permutation']['pairs'])

    def test_worker_dispatch_is_closed_and_paths_are_arguments_without_a_shell(self):
        spec = importlib.util.spec_from_file_location('comparison_runner', ROOT/'scripts/run_comparison.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        with patch.object(runner.subprocess, 'run') as launch:
            with self.assertRaises(ValueError):
                runner.launch('train; bad', '.', 'formal', 'dense', 0, '.', REQUEST, io.StringIO(), {})
            launch.assert_not_called()
            runner.launch('train', 'path;echo bad', 'formal', 'four-state', 0,
                          'request;bad', REQUEST, io.StringIO(), {})
            self.assertFalse(launch.call_args.kwargs['shell'])
            self.assertIn('path;echo bad', launch.call_args.args[0])
            self.assertEqual(launch.call_args.args[0][1:4], ['-m', 'trinite', 'comparison-cell'])

    def test_real_cell_evidence_binds_result_copies_and_retains_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); request = root/'request.json'; frozen = freeze_request(request)
            cell = root/'formal-0-four-state'
            trained = train_cell(cell, 'formal', 'four-state', 0, request, frozen['request_identity'])
            evaluated = evaluate_cell(cell, 'formal', 'four-state', 0, request, frozen['request_identity'])
            checked = verified_cell(cell, 'formal', 'four-state', 0, frozen['request_identity'])
            self.assertEqual(checked['training'], trained)
            self.assertEqual(checked['evaluation'], evaluated)
            self.assertTrue(checked['verification']['integrity_verified'])
            raw = parse_json((cell/'evaluation.json').read_bytes()); raw['scores']['test']['correct'] = 999
            (cell/'evaluation.json').write_bytes(json_bytes(raw))
            with self.assertRaises(ContractError):
                verified_cell(cell, 'formal', 'four-state', 0, frozen['request_identity'])
            summary_root = root/'summary'; summary_root.mkdir()
            result = summarize(summary_root, request, frozen['request_identity'], {})
            self.assertEqual(len(result['cells']), 27)
            self.assertEqual(result['outcome'], 'failed')
            self.assertEqual(result['groups'], {})

    def test_aggregation_requires_every_paired_cell_and_matching_initialization(self):
        def cell(directory, workload, lane, seed, request_identity):
            training = {k: 'same' for k in ('initialization_identity', 'dataset_identity',
                'manifest_identity', 'plan_identity', 'schedule_identity')}
            training.update(workload=workload, lane=lane, seed=seed, parameter_count=10,
                            target_tokens=20, steps=4, training_wall_seconds_hex=(1.).hex())
            evaluation = {'scores': {'test': {'family_macro_accuracy_hex':
                (.5 if lane == 'dense' else .4).hex()}}}
            return {'outcome': 'completed', 'training': training, 'evaluation': evaluation}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); request = root/'request.json'; frozen = freeze_request(request)
            for w in ('formal', 'qutrit', 'ququart'):
                for seed in (0, 1, 2):
                    for lane in ('dense', 'ternary', 'four-state'):
                        p = root/f'{w}-{seed}-{lane}'; p.mkdir(); (p/'report.json').write_text('{}')
            with patch('trinite.comparison.verified_cell', side_effect=cell):
                result = summarize(root, request, frozen['request_identity'], {})
            self.assertEqual(result['outcome'], 'completed')
            self.assertEqual(result['groups']['formal']['ternary']['quality_margin_each_seed'], [False]*3)
            (root/'summary.json').unlink()
            def mismatch(*args):
                result = cell(*args)
                if args[2] == 'four-state': result['training']['initialization_identity'] = 'different'
                return result
            with patch('trinite.comparison.verified_cell', side_effect=mismatch):
                with self.assertRaises(ContractError): summarize(root, request, frozen['request_identity'], {})


if __name__ == '__main__': unittest.main()
