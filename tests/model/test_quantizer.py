import math
import unittest

import torch

from trinite.contracts import ContractError
from trinite.quantizer import quantize

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class QuantizerTests(unittest.TestCase):
    def test_codes_scale_ties_saturation_and_surrogate_boundaries(self):
        # Exactly unit absmean, including half ties and +/-1 boundaries.
        latent = torch.tensor([[-2., -1., -.5, 0., .5, 1., 2.]], requires_grad=True)
        value = quantize(latent)
        self.assertEqual(float(value.scale), 1)
        self.assertEqual(value.codes.tolist(), [[-1, -1, 0, 0, 0, 1, 1]])
        self.assertEqual(value.weight.tolist(), [[-1, -1, 0, 0, 0, 1, 1]])
        self.assertEqual(value.codes.dtype, torch.int8)
        self.assertFalse(value.scale.requires_grad)
        gradient = torch.arange(1., 8.).reshape(1, 7)
        value.weight.backward(gradient)
        self.assertEqual(latent.grad.tolist(), [[0, 0, 3, 4, 5, 0, 0]])

    def test_zero_subnormal_and_extreme_finite_inputs(self):
        for values in ([[0., 0.]], [[1e-40, -1e-40]],
                       [[torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max]]):
            latent = torch.tensor(values, requires_grad=True)
            value = quantize(latent)
            self.assertTrue(bool(torch.isfinite(value.weight).all()))
            self.assertTrue(math.isfinite(float(value.scale)))
            if abs(values[0][0]) < 1e-8:
                self.assertEqual(value.codes.tolist(), [[0, 0]])
                value.weight.sum().backward()
                self.assertEqual(latent.grad.tolist(), [[1, 1]])
            else:
                self.assertEqual(value.weight.tolist(), values)

    def test_independent_scale_and_threshold_oracle(self):
        from oracle import effective
        matrices = [[[0.1, -0.2, .3], [.4, -.5, .6]],
                    [[0., 0., 0., 10.]], [[.125, .25, -.75, -.875]]]
        for matrix in matrices:
            latent = torch.tensor(matrix)
            expected = torch.tensor(effective(latent.tolist(), "ternary"))
            self.assertTrue(torch.equal(quantize(latent).weight, expected))

    def test_noncontiguous_weight_and_no_latent_mutation(self):
        latent = torch.arange(12., dtype=torch.float32).reshape(3, 4).t().requires_grad_()
        before = latent.detach().clone()
        value = quantize(latent)
        value.weight.sum().backward()
        self.assertTrue(torch.equal(before, latent))
        self.assertEqual(latent.grad.shape, latent.shape)

    def test_invalid_shapes_dtypes_layouts_and_nonfinite_values(self):
        invalid = [None, torch.zeros(3), torch.zeros(0, 2), torch.zeros(1, 1, 1),
                   torch.zeros(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.int64),
                   torch.ones(1, 2, device="meta"), torch.zeros(1, 2).to_sparse(),
                   torch.zeros(2000001, 1)]
        invalid.extend(torch.tensor([[n]]) for n in (float("nan"), float("inf"), -float("inf")))
        for value in invalid:
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ContractError):
                    quantize(value)
