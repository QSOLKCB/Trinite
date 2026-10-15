import copy
from fractions import Fraction
import random
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from pcgrad_v1.kernel import combine
from pcgrad_v1.spec import specification
from pcgrad_v1.toy import create, run, resume, snapshot, values
from trinite.contracts import ContractError, identity, json_bytes, parse_json


def oracle(a, b):
    a, b = [[Fraction(v) for v in g] for g in (a, b)]
    dot = sum(x*y for x, y in zip(a, b))
    out = []
    for g, h in ((a, b), (b, a)):
        norm = sum(x*x for x in h)
        out.append([x-dot*y/norm for x, y in zip(g, h)] if dot < 0 and norm else g)
    return out


class PCGradTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_hand_derived_exact_fixture_and_immutable_inputs(self):
        for case in specification()['cases']:
            a, b = [torch.tensor(case[k], dtype=torch.float32, requires_grad=True) for k in ('a', 'b')]
            originals = [g.clone() for g in (a, b)]
            exact = oracle(case['a'], case['b'])
            self.assertEqual(exact, [[Fraction(v) for v in row] for row in case['projected']])
            for order in ((0, 1), (1, 0)):
                got = combine(a, b, arm='pcgrad', order=order)
                for actual, expected in zip(got.projected, exact):
                    self.assertTrue(torch.equal(actual, torch.tensor([float(v) for v in expected], dtype=torch.float32)))
                self.assertTrue(torch.equal(got.gradient, got.projected[0]+got.projected[1]))
            for original, actual in zip(originals, (a, b)):
                self.assertTrue(torch.equal(original, actual))
                self.assertIsNone(actual.grad)

    def test_random_fraction_oracle(self):
        rng = random.Random(173)
        for _ in range(100):
            a, b = [[rng.randrange(-8, 9) for _ in range(5)] for _ in range(2)]
            result = combine(torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32), arm='pcgrad')
            for actual, expected in zip(result.projected, oracle(a, b)):
                self.assertTrue(torch.equal(actual, torch.tensor([float(v) for v in expected], dtype=torch.float32)))

    def test_zero_subnormal_extreme_and_actual_rounding(self):
        for scale in (2.**-149, 2.**125):
            a = torch.tensor([scale, 0.], dtype=torch.float32)
            result = combine(a, -a, arm='pcgrad')
            self.assertTrue(torch.equal(result.gradient, torch.zeros(2)))
            self.assertEqual(result.diagnostics['applied'], [True, True])
        zero = combine(torch.zeros(2), torch.zeros(2), arm='pcgrad')
        self.assertIsNone(zero.diagnostics['cosine_hex'])
        self.assertEqual(zero.diagnostics['applied'], [False, False])
        # Rational orthogonality is not asserted for the rounded vector.
        a, b = torch.tensor([1., 2.]), torch.tensor([-3., 1.])
        r = combine(a, b, arm='pcgrad')
        self.assertEqual(r.diagnostics['rounded_residual_dots_hex'][0], torch.dot(r.projected[0].double(), b.double()).item().hex())
        self.assertNotEqual(float.fromhex(r.diagnostics['rounded_residual_dots_hex'][0]), 0.)

    def test_float64_cancellation_is_retained_not_repaired(self):
        a, b = torch.tensor([-27., -7.]), torch.tensor([49., 0.])
        result = combine(a, b, arm='pcgrad')
        self.assertEqual(oracle([-27, -7], [49, 0])[0][0], 0)
        self.assertNotEqual(result.projected[0][0].item(), 0.)
        self.assertLessEqual(abs(result.projected[0][0].item()), 1e-12)
        self.assertLess(float.fromhex(result.diagnostics['rounded_residual_dots_hex'][0]), 0.)

    def test_sum_control_and_overflow_rejection(self):
        a, b = torch.tensor([1., 2.]), torch.tensor([-3., 1.])
        got = combine(a, b, arm='sum')
        self.assertTrue(torch.equal(got.gradient, a+b))
        self.assertEqual(got.diagnostics['applied'], [False, False])
        huge = torch.full((2,), torch.finfo(torch.float32).max)
        for arm in ('sum', 'pcgrad'):
            with self.assertRaises(ContractError): combine(huge, huge, arm=arm)

    def test_input_rejections(self):
        a = torch.ones(2)
        for bad in (None, [1, 2], torch.ones(2, dtype=torch.float64), torch.ones(1, 2),
                    torch.empty(0), torch.ones(65537), torch.tensor([float('nan'), 0.]),
                    torch.tensor([float('inf'), 1.]), torch.ones(3)):
            with self.assertRaises(ContractError): combine(a, bad, arm='pcgrad')
        for order in ((True, False), (0, 0), [0, 1], (0,), (0, 2)):
            with self.assertRaises(ContractError): combine(a, a, arm='pcgrad', order=order)
        with self.assertRaises(ContractError): combine(a, a, arm='unknown')

    def test_replay_every_interruption_complete_state(self):
        for arm in ('sum', 'pcgrad'):
            for seed in (0, 1):
                expected = snapshot(run(create(arm, seed)))
                for stop in range(9):
                    prefix = snapshot(run(create(arm, seed), stop_after=stop))
                    restored = resume(prefix, expected_identity=identity(prefix), arm=arm, seed=seed)
                    self.assertEqual(snapshot(run(restored)), expected)

    def test_same_snapshot_targets_and_optimizer_accounting(self):
        cells = [run(create(arm, 0)) for arm in ('sum', 'pcgrad')]
        self.assertEqual(cells[0].history[0]['before_hex'], cells[1].history[0]['before_hex'])
        self.assertEqual(cells[0].history[0]['original_gradients_hex'], cells[1].history[0]['original_gradients_hex'])
        for state in cells:
            for h in state.history:
                w = torch.tensor([float.fromhex(v) for v in h['before_hex']], requires_grad=True)
                for i, name in enumerate(('final', 'auxiliary')):
                    d = specification()['toy_inputs'][name]
                    loss = ((torch.tensor(d['x'], dtype=torch.float32) @ w-torch.tensor(d['y']))**2).mean()
                    self.assertEqual(values(torch.autograd.grad(loss, w)[0]), h['original_gradients_hex'][i])
                self.assertEqual((h['backwards'], h['optimizer_calls'], h['objective_targets']), (2, 1, [2, 2]))
            self.assertEqual(float(state.optimizer.state[state.model.weight]['step']), 8.)
        self.assertEqual([h['diagnostics']['order'] for h in cells[0].history], [h['diagnostics']['order'] for h in cells[1].history])

    def test_rng_and_input_bytes_untouched(self):
        before = torch.get_rng_state().clone()
        state = create('pcgrad', 1)
        run(state, stop_after=3)
        raw = snapshot(state); retained = raw[:]
        resume(raw, expected_identity=identity(raw), arm='pcgrad', seed=1)
        self.assertEqual(raw, retained)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_checkpoint_tamper_and_wrong_context(self):
        raw = snapshot(run(create('pcgrad', 0), stop_after=3))
        meta = parse_json(raw, canonical=True)
        changes = {'model_hex': ['nan', '0x0.0p+0'], 'rng_hex': '00', 'step': True,
                   'source': {}, 'history': [], 'adamw_hex': {}, 'protocol_identity': 'bad',
                   'environment': {}, 'seed': 1, 'arm': 'sum', 'optimizer_config': {}}
        for key, value in changes.items():
            changed = copy.deepcopy(meta); changed[key] = value
            corrupted = json_bytes(changed)
            with self.assertRaises(ContractError): resume(corrupted, expected_identity=identity(corrupted), arm='pcgrad', seed=0)
        for arm, seed in (('sum', 0), ('pcgrad', 1)):
            with self.assertRaises(ContractError): resume(raw, expected_identity=identity(raw), arm=arm, seed=seed)
        for bad in (b'', b'{}', b' '*262145, raw+b' '):
            with self.assertRaises(ContractError): resume(bad, expected_identity=identity(bad), arm='pcgrad', seed=0)
        with self.assertRaises(ContractError): resume(raw, expected_identity='sha256:'+'0'*64, arm='pcgrad', seed=0)

    def test_scientific_gate_and_bounded_horizon(self):
        with patch('pcgrad_v1.toy.require_environment', side_effect=AssertionError('allocated')):
            with self.assertRaises(ContractError): create('pcgrad', 0, profile='research')
            with self.assertRaises(ContractError): resume(b'', expected_identity='', arm='pcgrad', seed=0, profile='research')
        for seed in (True, -1, 2):
            with self.assertRaises(ContractError): create('pcgrad', seed)
        state = create('pcgrad', 0)
        for stop in (True, -1, 9, 8192):
            with self.assertRaises(ContractError): run(state, stop_after=stop)
        state.optimizer.param_groups[0]['lr'] = .1
        with self.assertRaises(ContractError): run(state)


if __name__ == '__main__': unittest.main()
