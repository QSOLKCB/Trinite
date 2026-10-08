"""Numeric controls, safe checkpoint rejection, exact replay and observer isolation."""
import copy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
import random
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load, save

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.cli import main
from trinite.checkpoint import (checkpoint_bytes, load_checkpoint_bytes, read_checkpoint,
                                rng_snapshot)
from trinite.experiment import train_run
from trinite.inspection import tensor_identity
from trinite.training import (RunConfig, admitted_data, batch, create_state, evaluate,
                              run_steps, target_loss)

ROOT = Path(__file__).resolve().parents[2]
DATA = (ROOT/'fixtures/formal-v1/dataset.json').read_bytes()
MANIFEST = (ROOT/'fixtures/formal-v1/manifest.json').read_bytes()
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def seed_rng():
    random.seed(123); np.random.seed(123); torch.manual_seed(123)


def rng_identity():
    meta, tensors = rng_snapshot()
    return meta, {n: tensor_identity(t) for n, t in tensors.items()}


class TrainingTests(unittest.TestCase):
    def setUp(self):
        seed_rng()
        self.plan = RunConfig(steps=6, validation_every=3)

    def state(self, lane='ternary'):
        return create_state(self.plan, lane, DATA, MANIFEST)

    def test_frozen_plan_and_resource_rejections(self):
        self.assertEqual(RunConfig.load(ROOT/'configs/tiny-training.json'), RunConfig())
        self.assertEqual(RunConfig.from_dict(self.plan.to_dict()), self.plan)
        for field, value in [('steps', True), ('steps', 257), ('batch_size', 9),
                             ('learning_rate', 'nan'), ('beta2', '1'), ('accumulation', 2),
                             ('schedule', 'adaptive'), ('max_seconds', 601), ('max_target_tokens', 1)]:
            with self.subTest(field=field), self.assertRaises(ContractError):
                replace(self.plan, **{field: value})
        extra = self.plan.to_dict(); extra['hidden'] = 1
        with self.assertRaises(ContractError): RunConfig.from_dict(extra)

    def test_optimizer_contract_rejects_mutated_algorithm_and_settings_before_updates(self):
        for key, value in [('decoupled_weight_decay', False), ('lr', .1), ('foreach', True), ('unexpected', 1)]:
            state = self.state(); state.optimizer.param_groups[0][key] = value
            with self.assertRaises(ContractError): run_steps(state)
            self.assertEqual(state.step, 0)
            with self.assertRaises(ContractError): checkpoint_bytes(state)

    def test_seeded_initialization_matches_lanes_without_global_rng_draws(self):
        before = rng_identity()
        a, b = self.state('dense'), self.state('ternary')
        self.assertEqual(a.initialization_identity, b.initialization_identity)
        self.assertEqual(before, rng_identity())
        self.assertEqual([e['identity'] for e in a.data['train']],
                         [e['identity'] for e in b.data['train']])
        self.assertNotEqual([e['identity'] for e in a.data['train']],
                           [e['identity'] for e in admitted_data(DATA, MANIFEST,
                                                                replace(self.plan, seed=1))['train']])

    def test_shifted_answer_eos_loss_matches_independent_target_selection(self):
        state = self.state('dense'); examples = state.data['train'][:4]
        ids, mask, scored = batch(examples)
        logits = state.model(ids[:, :-1], mask[:, :-1]).logits
        selected = []
        for row in range(len(examples)):
            for col in range(1, ids.shape[1]):
                if scored[row, col]:
                    selected.append(-torch.log_softmax(logits[row, col-1], dim=-1)[ids[row, col]])
        actual, count = target_loss(state.model, examples)
        self.assertEqual(count, 8)
        torch.testing.assert_close(actual, torch.stack(selected).mean(), rtol=1e-6, atol=1e-6)
        self.assertEqual(set(state.data), {'train', 'validation'})
        forbidden = {e['example_identity'] for e in parse_json(DATA)['examples'] if e['split'] == 'test'}
        run_steps(state)
        self.assertFalse(forbidden & {n for h in state.history for n in h['examples']})

    def test_both_lanes_learn_visible_fixture_with_matched_budgets(self):
        runs = []
        for lane in ('dense', 'ternary'):
            state = create_state(RunConfig(), lane, DATA, MANIFEST)
            run_steps(state)
            final = float.fromhex(evaluate(state.model, state.data['train'], state.config.batch_size))
            self.assertLess(final, .7*float.fromhex(state.initial_train_loss))
            self.assertEqual(state.target_tokens, 480)
            self.assertEqual([v['step'] for v in state.validation], [20, 40, 60])
            runs.append(state)
        self.assertEqual(runs[0].initialization_identity, runs[1].initialization_identity)
        self.assertEqual([h['examples'] for h in runs[0].history], [h['examples'] for h in runs[1].history])

    def test_interrupted_replay_preserves_all_checkpoint_bytes_and_rng(self):
        for lane in ('dense', 'ternary'):
            seed_rng(); uninterrupted = self.state(lane); run_steps(uninterrupted)
            expected = checkpoint_bytes(uninterrupted)
            self.assertEqual(expected, checkpoint_bytes(uninterrupted))
            seed_rng(); paused = self.state(lane); run_steps(paused, stop_after=2)
            parent = checkpoint_bytes(paused)
            random.random(); np.random.random(); torch.rand(3)
            resumed = load_checkpoint_bytes(*parent, DATA, MANIFEST, self.plan, lane)
            run_steps(resumed)
            self.assertEqual(expected, checkpoint_bytes(resumed))
            self.assertIs(resumed.model.output_weight, resumed.model.token_embedding)

    def test_zero_step_resume_and_wrong_plan_lane_inputs(self):
        state = self.state(); payload, meta = checkpoint_bytes(state)
        restored = load_checkpoint_bytes(payload, meta, DATA, MANIFEST, self.plan, 'ternary')
        self.assertEqual((payload, meta), checkpoint_bytes(restored))
        for data, manifest, plan, lane in [(DATA+b' ', MANIFEST, self.plan, 'ternary'),
                (DATA, MANIFEST+b' ', self.plan, 'ternary'),
                (DATA, MANIFEST, replace(self.plan, seed=1), 'ternary'),
                (DATA, MANIFEST, self.plan, 'dense')]:
            with self.assertRaises(ContractError):
                load_checkpoint_bytes(payload, meta, data, manifest, plan, lane)

    def test_writer_rejects_nonfinite_rng_cache(self):
        state = self.state(); current = random.getstate()
        random.setstate((current[0], current[1], float('nan')))
        try:
            with self.assertRaises(ContractError): checkpoint_bytes(state)
        finally:
            random.setstate(current)

    def test_metadata_rejection_never_restores_rng(self):
        state = self.state(); run_steps(state, stop_after=2)
        payload, raw = checkpoint_bytes(state)
        original = parse_json(raw)
        variants = []
        for field, value in [('schema', 'unknown'), ('step', True), ('cursor', 0),
                             ('target_tokens', 1), ('model_identity', identity(b'wrong'))]:
            m = copy.deepcopy(original); m[field] = value; variants.append(m)
        for field in ('source', 'environment'):
            m = copy.deepcopy(original); m[field]['unexpected'] = 'value'; variants.append(m)
        m = copy.deepcopy(original); m['rng']['python_state'][0] = -1; variants.append(m)
        m = copy.deepcopy(original); m['history'][0]['examples'].reverse(); variants.append(m)
        m = copy.deepcopy(original); m['tensors']['rng.numpy_keys']['shape'][0] = True; variants.append(m)
        for m in variants:
            before = rng_identity()
            with self.assertRaises(ContractError):
                load_checkpoint_bytes(payload, json_bytes(m), DATA, MANIFEST, self.plan, 'ternary')
            self.assertEqual(before, rng_identity())
        with self.assertRaises(ContractError):
            load_checkpoint_bytes(payload, raw.replace(b'"schema":', b'"schema":"duplicate","schema":'),
                                  DATA, MANIFEST, self.plan, 'ternary')

    def test_tensor_rejection_checks_semantics_beyond_matching_hashes(self):
        state = self.state(); run_steps(state, stop_after=2)
        original_payload, raw = checkpoint_bytes(state); original_meta = parse_json(raw)
        names = load(original_payload)
        key = next(n for n in names if n.startswith('model.'))
        variants = [(key, torch.zeros(1)), (key, names[key].to(torch.float64)),
                    (key, torch.full_like(names[key], float('nan'))),
                    ('rng.numpy_keys', torch.full((624,), -1, dtype=torch.int64)),
                    ('rng.torch_cpu', torch.zeros(1, dtype=torch.uint8))]
        step_key = next(n for n in names if n.startswith('optimizer.') and n.endswith('.step'))
        var_key = next(n for n in names if n.endswith('.exp_avg_sq'))
        variants.extend([(step_key, torch.tensor(99.)), (var_key, -torch.ones_like(names[var_key]))])
        for name, value in variants:
            tensors = {n: t.clone() for n, t in names.items()}; tensors[name] = value
            payload = save(tensors); m = copy.deepcopy(original_meta)
            m['tensor_file_identity'] = identity(payload)
            m['tensors'][name] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'identity': tensor_identity(value)}
            before = rng_identity()
            with self.assertRaises(ContractError):
                load_checkpoint_bytes(payload, json_bytes(m), DATA, MANIFEST, self.plan, 'ternary')
            self.assertEqual(before, rng_identity())
        with self.assertRaises(ContractError):
            load_checkpoint_bytes(original_payload[:-1], raw, DATA, MANIFEST, self.plan, 'ternary')
        with self.assertRaises(ContractError):
            load_checkpoint_bytes(b'\xff'*8, raw, DATA, MANIFEST, self.plan, 'ternary')

    def test_checkpoint_directory_rejects_links_and_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); payload, meta = checkpoint_bytes(self.state())
            (root/'tensors.safetensors').write_bytes(payload); (root/'metadata.json').write_bytes(meta)
            self.assertEqual(read_checkpoint(root), (payload, meta))
            (root/'extra').write_bytes(b'')
            with self.assertRaises(ContractError): read_checkpoint(root)
            (root/'extra').unlink(); (root/'metadata.json').rename(root/'meta')
            (root/'metadata.json').symlink_to(root/'meta')
            with self.assertRaises(ContractError): read_checkpoint(root)

    def test_deadline_failure_has_no_update(self):
        state = self.state()
        with patch('trinite.training.time.monotonic', side_effect=[0., 1000.]):
            with self.assertRaises(ContractError): run_steps(state)
        self.assertEqual(state.step, 0)

    def test_failed_run_retains_valid_completed_step_checkpoint(self):
        def timeout(state, **kwargs):
            run_steps(state, stop_after=2)
            raise ContractError('injected deadline after two completed steps')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'failed'
            with patch('trinite.experiment.run_steps', timeout):
                report = train_run(root, ROOT/'fixtures/formal-v1', self.plan)
            self.assertEqual(report['compute_outcome'], 'failed')
            self.assertFalse(report['observer_evidence_eligible'])
            self.assertTrue(report['verification']['integrity_verified'])
            state = load_checkpoint_bytes(*read_checkpoint(root/'checkpoint'), DATA, MANIFEST, self.plan, 'ternary')
            self.assertEqual(state.step, 2)
            run_steps(state)
            expected = self.state(); run_steps(expected)
            self.assertEqual(checkpoint_bytes(state), checkpoint_bytes(expected))

    def test_cli_replays_checkpoint_across_fresh_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = root/'plan.json'
            config.write_bytes(json_bytes(self.plan.to_dict()))
            for lane in ('dense', 'ternary'):
                for name, options in [('full', []), ('paused', ['--stop-after', '2']),
                                      ('resumed', ['--resume', str(root/f'{lane}-paused/checkpoint')])]:
                    command = [sys.executable, '-m', 'trinite', 'train', str(root/f'{lane}-{name}'),
                               '--config', str(config), '--dataset', str(ROOT/'fixtures/formal-v1'),
                               '--lane', lane] + options
                    result = subprocess.run(command, env=dict(os.environ, PYTHONPATH=str(ROOT/'src')),
                                            capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr.decode())
                    self.assertTrue(parse_json(result.stdout)['observer_evidence_eligible'])
                for member in ('steps.json', 'result.json', 'checkpoint/metadata.json', 'checkpoint/tensors.safetensors'):
                    self.assertEqual((root/f'{lane}-full'/member).read_bytes(),
                                     (root/f'{lane}-resumed'/member).read_bytes(), member)

    def test_mandatory_pin_failure_returns_nonzero_after_preserving_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = root/'plan.json'; config.write_bytes(json_bytes(self.plan.to_dict()))
            stdout = io.StringIO()
            with (patch('trinite.observation.PIN_IDENTITY', identity(b'unsupported')),
                  redirect_stdout(stdout), redirect_stderr(io.StringIO())):
                status = main(['train', str(root/'failed-observer'), '--config', str(config),
                               '--dataset', str(ROOT/'fixtures/formal-v1')])
            self.assertEqual(status, 1)
            report = parse_json(stdout.getvalue().encode())
            self.assertEqual(report['compute_outcome'], 'completed')
            self.assertFalse(report['observer_evidence_eligible'])
            self.assertTrue(report['observation_errors'])

    def test_observer_off_on_and_failed_record_preserve_gradients_updates_rng_bytes(self):
        class BrokenObserver:
            def __init__(self, path): pass
            def record(self, *a, **kw): raise OSError('injected required observation failure')
            def finalize(self): raise OSError('injected verification failure')
        original_step = torch.optim.AdamW.step
        gradient_runs = []
        def traced_step(optimizer, *args, **kw):
            gradients.append([tensor_identity(p.grad) for p in optimizer.param_groups[0]['params']])
            return original_step(optimizer, *args, **kw)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); results, rngs = [], []
            for name, observer, factory in [('off', False, None), ('on', True, None), ('broken', True, BrokenObserver)]:
                seed_rng(); gradients = []
                kwargs = {'observer_factory': factory} if factory else {}
                with patch.object(torch.optim.AdamW, 'step', traced_step):
                    report = train_run(root/name, ROOT/'fixtures/formal-v1', self.plan,
                                       observer=observer, **kwargs)
                results.append(report); gradient_runs.append(gradients); rngs.append(rng_identity())
            self.assertTrue(results[1]['observer_evidence_eligible'])
            self.assertFalse(results[0]['observer_evidence_eligible'])
            self.assertFalse(results[2]['observer_evidence_eligible'])
            self.assertEqual(results[2]['compute_outcome'], 'completed')
            self.assertTrue(results[2]['observation_errors'])
            self.assertEqual(gradient_runs[0], gradient_runs[1]); self.assertEqual(gradient_runs[0], gradient_runs[2])
            self.assertEqual(rngs[0], rngs[1]); self.assertEqual(rngs[0], rngs[2])
            for member in ('initialization.json', 'steps.json', 'result.json', 'inventory.json',
                           'checkpoint/metadata.json', 'checkpoint/tensors.safetensors'):
                self.assertEqual((root/'off'/member).read_bytes(), (root/'on'/member).read_bytes(), member)
                self.assertEqual((root/'off'/member).read_bytes(), (root/'broken'/member).read_bytes(), member)


if __name__ == '__main__': unittest.main()
