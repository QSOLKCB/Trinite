"""Explicit CPU-tensor checks; requires the canonical installed CPU Torch.

Run separately from standard-library NCM contracts:
PYTHONPATH=scripts python -B -m unittest discover -s tests/ncm_reference_cpu -v
No model modules, real checkpoints, or CUDA operations are used.
"""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from ncm_reference_v1 import core, runtime


class ActualLevelDtype(unittest.TestCase):
    def test_pinned_source_uses_integer_enumerate_fill(self):
        core.verify_upstream()
        tree = ast.parse((core.ROOT / 'vendor/models/next_concept.py').read_text())
        assignments = [
            node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'd' for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        calls = [
            node for node in ast.walk(assignments[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch'
            and node.func.attr == 'full'
        ]
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[1], ast.Name)
        self.assertEqual(calls[0].args[1].id, 'i')
        self.assertFalse(calls[0].keywords)

    def test_actual_torch_distinguishes_integer_and_floating_fill(self):
        self.assertEqual(torch.full((2,), 1, device='cpu').dtype, torch.int64)
        self.assertEqual(torch.full((2,), 1.0, device='cpu').dtype, torch.get_default_dtype())

    def test_exact_level_expression_and_wrapper_preserve_int64(self):
        levels = (1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89)
        # Independently written tensor expression corresponding to the inspected
        # integer-fill AST above; do not import or execute the vendored module.
        d = torch.cat([torch.full((l,), i) for i, l in enumerate(levels)]).view(1, 322, 1)
        dT = d.transpose(1, 2)
        expected = dT[:, 0].contiguous()
        cfg = SimpleNamespace(
            model=SimpleNamespace(prefix_len=91, attn_pattern='block_diagonal',
                                  query_level0=True, use_rope=True, l0_autoregressive=False),
            tokenizer=SimpleNamespace(vqvae=SimpleNamespace(
                vector_quantizer_config=SimpleNamespace(
                    levels=SimpleNamespace(manual_levels=list(levels), use_manual_levels=True)))),
        )
        actual = runtime.generator_fixed_buffers(torch, cfg)['lvl_1L']
        self.assertEqual(expected.dtype, torch.int64)
        self.assertEqual(actual.dtype, torch.int64)
        self.assertEqual(tuple(actual.shape), (1, 322))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
