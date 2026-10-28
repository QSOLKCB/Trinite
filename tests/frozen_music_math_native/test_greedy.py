"""Synthetic native greedy mechanics only; no retained music checkpoint inputs."""
import unittest
from types import SimpleNamespace

import torch

from trinite.comparison import greedy
from trinite.tokenizer import BOS, EOS


class SyntheticLogits:
    """A predetermined-logit double, not a trained model or capability result."""
    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = []
        self.config = SimpleNamespace(context_length=256)

    def __call__(self, ids, attention):
        if torch.is_grad_enabled():
            raise AssertionError('generation must disable gradients')
        self.calls.append(ids.tolist()[0])
        if not bool(attention.all()):
            raise AssertionError('synthetic test expects a full prompt mask')
        logits = torch.zeros((1, ids.shape[1], 259), dtype=torch.float32)
        logits[0, -1, self.tokens[len(self.calls)-1]] = 1
        return SimpleNamespace(logits=logits)


class NativeGreedyMechanics(unittest.TestCase):
    def test_prompt_only_raw_eos_counted(self):
        # Deliberately wrong output: this double tests transport, not arithmetic.
        model = SyntheticLogits([ord('9'), EOS])
        self.assertEqual(greedy(model, '2+3=', 16), [ord('9'), EOS])
        self.assertEqual(model.calls, [[BOS, *b'2+3='], [BOS, *b'2+3=9']])

    def test_cap_does_not_append_or_repair_eos(self):
        model = SyntheticLogits([ord('/')] * 16)
        self.assertEqual(greedy(model, '28+46=', 16), [ord('/')] * 16)
        self.assertEqual(len(model.calls), 16)
        self.assertNotIn(EOS, model.calls[-1])

    def test_immediate_eos_is_a_single_forward(self):
        model = SyntheticLogits([EOS])
        self.assertEqual(greedy(model, '87+38=', 16), [EOS])
        self.assertEqual(len(model.calls), 1)


if __name__ == '__main__':
    unittest.main()
