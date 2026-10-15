import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from scale_refit import quantizer as candidate
from scale_refit.diagnostic import canonical, identity, measure_suite, report, suite
from trinite.contracts import ContractError
from trinite.model import Decoder
from trinite.quantizer import quantize


def rational_sse(weight, actual):
    return sum((Fraction(float(w))-Fraction(float(a)))**2
               for w, a in zip(weight.reshape(-1), actual.reshape(-1)))


class ScaleRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_additive_aggregate_preserves_historical_order(self):
        from check_research_cpu import load_suites as historical
        from check_scale_refit_cpu import load_suites as combined
        def ids(suite):
            for item in suite:
                if isinstance(item, unittest.TestSuite):
                    yield from ids(item)
                else:
                    yield item.id()
        old = list(ids(historical()))
        added = list(ids(unittest.TestLoader().discover(str(ROOT/'tests/scale_refit'))))
        actual = list(ids(combined()))
        self.assertEqual(len(old), 261)
        self.assertEqual(actual, old + added)
        self.assertEqual(len(actual), len(set(actual)))

    def test_analytic_boundary(self):
        w = torch.tensor([[-2., -1., -.5, 0., .5, 1., 2.]])
        r = candidate.refit(w)
        self.assertEqual(r.baseline_scale.item(), 1.)
        self.assertEqual(r.quantized.scale.item(), 1.5)
        self.assertEqual(r.quantized.codes.tolist(), [[-1, -1, 0, 0, 0, 1, 1]])
        self.assertEqual(Fraction(r.baseline_sse_numerator, 2**298), Fraction(5, 2))
        self.assertEqual(Fraction(r.selected_sse_numerator, 2**298), Fraction(3, 2))

    def test_exhaustive_small_rational_fixed_code_oracle(self):
        for values in itertools.product((-2., -1., 0., 1., 2.), repeat=4):
            w = torch.tensor([values])
            base, r = quantize(w), candidate.refit(w)
            q = base.codes.reshape(-1).tolist()
            support = sum(i*i for i in q)
            actual = rational_sse(w, r.quantized.weight)
            self.assertEqual(actual, Fraction(r.selected_sse_numerator, 2**298))
            self.assertLessEqual(actual, rational_sse(w, base.weight))
            if support:
                # Independent Fraction derivative solution, not a float reduction.
                optimum = sum(Fraction(v)*i for v, i in zip(values, q))/support
                ideal = sum((Fraction(v)-optimum*i)**2 for v, i in zip(values, q))
                self.assertGreaterEqual(actual, ideal)
                # For these exact integer inputs, the chosen scalar must be the
                # rounded real optimum (or retained original on an exact SSE tie).
                rounded = torch.tensor(float(optimum), dtype=torch.float32)
                oracle_effective = base.codes.float()*rounded
                self.assertEqual(actual, rational_sse(w, oracle_effective))

    def test_independent_exhaustive_code_oracle_is_only_lower_bound(self):
        values = (Fraction(-2), Fraction(-1, 2), Fraction(0), Fraction(1))
        all_errors = []
        for q in itertools.product((-1, 0, 1), repeat=4):
            norm = sum(i*i for i in q)
            alpha = max(Fraction(0), sum(v*i for v, i in zip(values, q))/norm) if norm else Fraction(0)
            all_errors.append(sum((v-alpha*i)**2 for v, i in zip(values, q)))
        w = torch.tensor([[float(v) for v in values]])
        selected = candidate.refit(w)
        self.assertGreaterEqual(rational_sse(w, selected.quantized.weight), min(all_errors))
        self.assertTrue(torch.equal(selected.quantized.codes, quantize(w).codes))

    def test_exact_sse_subnormal_extreme_and_cancellation(self):
        w = torch.tensor([[torch.finfo(torch.float32).max, 2.**-149, -2.**-149]])
        actual = torch.tensor([[0., 0., 2.**-149]])
        self.assertEqual(Fraction(candidate.exact_sse_numerator(w, actual), 2**298), rational_sse(w, actual))
        # Float64 summation loses this subnormal contribution; integer SSE keeps it.
        zero = torch.zeros_like(w)
        self.assertNotEqual(candidate.exact_sse_numerator(w, actual), candidate.exact_sse_numerator(w, zero))

    def test_zero_support_constant_and_equal_error_fallback(self):
        for w in (torch.zeros((2, 3)), torch.full((2, 3), 2.**-149), torch.full((2, 3), 3.)):
            base, r = quantize(w), candidate.refit(w)
            self.assertTrue(torch.equal(base.weight, r.quantized.weight))
            self.assertEqual(base.scale.item(), r.quantized.scale.item())
            self.assertIn(r.decision, ('zero_support', 'no_strict_improvement'))

    def test_uncertifiable_or_worse_candidate_retains_baseline(self):
        w = torch.tensor([[-2., -1., -.5, 0., .5, 1., 2.]])
        base = quantize(w)
        for scale, decision in ((float('inf'), 'nonfinite_candidate'),
                                (float('nan'), 'nonfinite_candidate'),
                                (100., 'no_strict_improvement')):
            with patch.object(candidate, '_candidate_scale', return_value=torch.tensor(scale)):
                result = candidate.refit(w)
            self.assertEqual(result.decision, decision)
            self.assertTrue(torch.equal(result.quantized.weight, base.weight))
            self.assertEqual(result.quantized.scale.item(), base.scale.item())
            self.assertEqual(result.selected_sse_numerator, result.baseline_sse_numerator)

    def test_fixed_mask_and_detached_scale(self):
        w = torch.tensor([[-2., -1., -.5, 0., .5, 1., 2.]], requires_grad=True)
        before = w.detach().clone()
        q = candidate.refit(w).quantized
        q.weight.sum().backward()
        self.assertEqual(w.grad.tolist(), [[0., 0., 1., 1., 1., 0., 0.]])
        self.assertFalse(q.scale.requires_grad)
        self.assertTrue(torch.equal(before, w.detach()))

    def test_all_frozen_families_actual_products_and_codes(self):
        for name, w in suite():
            with self.subTest(name=name):
                base, r = quantize(w), candidate.refit(w)
                self.assertTrue(torch.equal(base.codes, r.quantized.codes))
                self.assertLessEqual(r.selected_sse_numerator, r.baseline_sse_numerator)
                self.assertEqual(Fraction(r.selected_sse_numerator, 2**298), rational_sse(w, r.quantized.weight))
                self.assertTrue(torch.equal(r.quantized.weight, r.quantized.codes.float()*r.quantized.scale))

    def test_strided_input_and_deterministic_replay_without_rng_use(self):
        w = torch.tensor([[-2., 1.], [-1., .5], [0., 2.]]).t()
        rng = torch.get_rng_state().clone()
        a, b = candidate.refit(w), candidate.refit(w)
        self.assertTrue(torch.equal(a.quantized.weight, b.quantized.weight))
        self.assertEqual(a.selected_sse_numerator, b.selected_sse_numerator)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_invalid_inputs_and_no_native_lane(self):
        for bad in (None, [], torch.ones(3), torch.ones((0, 2)), torch.ones((1, 65537)),
                    torch.ones((2, 2), dtype=torch.float64), torch.ones((2, 2), dtype=torch.int64),
                    torch.tensor([[float('nan')]]), torch.tensor([[float('inf')]]),
                    torch.ones((2, 2)).to_sparse(), torch.ones((2, 2), device="meta")):
            with self.subTest(value=str(bad)[:30]), self.assertRaises(ContractError):
                candidate.refit(bad)
        with self.assertRaises(ContractError):
            Decoder(lane=candidate.QUANTIZER_ID)
        with self.assertRaises(ContractError):
            candidate.exact_sse_numerator(torch.ones((2, 2)), torch.ones(4))

    def test_output_error_is_separate_and_can_increase(self):
        w = torch.tensor([[-2., -1., -.5, 0., .5, 1., 2.]])
        x = torch.tensor([[0., 1., 0., 0., 0., 0., 0.]])
        base, selected = quantize(w).weight, candidate.refit(w).quantized.weight
        dense = x @ w.t()
        self.assertEqual(candidate.exact_sse_numerator(dense, x @ base.t()), 0)
        self.assertGreater(candidate.exact_sse_numerator(dense, x @ selected.t()), 0)

    def test_receipt_retained_numerics_and_source_bindings(self):
        # Cross-host numerical conformance only; full receipt replay separately
        # requires exact source AND environment, and does not reseal old receipts.
        retained = json.loads((ROOT/'fixtures/scale-refit-v1/report.json').read_bytes())
        current = report()
        self.assertEqual(current['cases'], retained['cases'])
        self.assertEqual(current['source'], retained['source'])
        self.assertEqual(current['aggregate_weight_sse_numerators'], retained['aggregate_weight_sse_numerators'])
        self.assertFalse(current['learning_claim'])
        self.assertFalse(current['native_model_or_checkpoint_lane'])

    def test_cli_actual_recompute_tamper_rejection_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'report.json'
            command = [sys.executable, str(ROOT/'scripts/check_scale_refit.py'), '--output', str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            expected = identity(output.read_bytes())
            before = output.read_bytes()
            verify = command + ['--verify-only', '--expected-identity', expected]
            passed = subprocess.run(verify, capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertEqual(output.read_bytes(), before)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            output.write_bytes(before.replace(b'"learning_claim":false', b'"learning_claim":true'))
            self.assertNotEqual(subprocess.run(verify, capture_output=True).returncode, 0)
            # Even a fresh external hash cannot make tampered numerics a replay.
            verify[-1] = identity(output.read_bytes())
            self.assertNotEqual(subprocess.run(verify, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
