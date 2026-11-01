"""Real pinned CPU operations; no optimizer step and no pilot target outcomes."""
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load, save

import queue_evaluation_v1 as gate
from queue_learning_v1 import codec, numerical, spec
from trinite.comparison_checkpoint import restore_tensors
from trinite.contracts import ContractError
from trinite.model import Decoder
from trinite.training import (TrainingState, create_optimizer, evaluate, model_identity, run_steps)


def public():
    return gate.encode({'schema': 'trinite.queue-observation.v1', 'capacity': 2,
        'observe_ticks': 1, 'horizon': 2, 'rows': [{'tick': 0, 'outage': False,
        'values': [0, 1, 1, 1, 0, 0], 'observed': [True] * 6}]})


def engineering_state():
    config = spec.plan(17)
    model = Decoder(config.model, lane='dense', seed=17)
    enc = codec.supervised(public(), 0)  # Arbitrary exposed engineering label.
    rows = [{'identity': 'exposed-engineering-only', 'ids': enc.input_ids,
             'mask': enc.attention_mask, 'loss': enc.loss_mask}]
    return TrainingState(config, 'dense', model, create_optimizer(model, config),
        {'train': rows, 'validation': rows}, 'engineering-data', 'engineering-binding',
        model_identity(model), evaluate(model, rows, 8))


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numerical.configure()

    def test_actual_safe_tensor_reader_zero_step_and_tampering(self):
        # The data factory is doubled only to avoid materializing pilot labels.
        # Real model, progress, AdamW inventory and safe tensor reader are used.
        original = engineering_state()
        payload, metadata = numerical.snapshot(original, 'engineering-binding')
        expected = gate.digest(payload), gate.digest(metadata)
        with patch.object(numerical, 'state_for', side_effect=lambda *a: engineering_state()):
            restored = numerical.restore(payload, metadata, [], 17, 'engineering-binding', expected)
            self.assertEqual(model_identity(restored.model), model_identity(original.model))
            self.assertEqual(restored.step, 0)
            self.assertEqual(restored.optimizer.state, {})
            with self.assertRaises(ValueError):
                numerical.restore(payload, metadata, [], 17, 'engineering-binding', ('0' * 64, expected[1]))
            tensors = load(payload)
            for mutation in ('extra', 'nan', 'shape', 'dtype'):
                altered = {k: v.clone() for k, v in tensors.items()}
                key = next(iter(altered))
                if mutation == 'extra':
                    altered['unexpected'] = torch.zeros(1)
                elif mutation == 'nan':
                    altered[key].reshape(-1)[0] = float('nan')
                elif mutation == 'shape':
                    altered[key] = altered[key].reshape(-1)[:1]
                else:
                    altered[key] = altered[key].to(torch.float64)
                bad = save(altered)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    numerical.restore(bad, metadata, [], 17, 'engineering-binding', (gate.digest(bad), expected[1]))
        # A closed, non-safetensors payload cannot become a model.
        with self.assertRaises(ValueError):
            restore_tensors(engineering_state(), b'pickle-like-invalid', model_identity(original.model))

    def test_inference_only_passes_public_prompt_to_native_decoder(self):
        sentinel = object()
        captured = []
        def greedy(model, prefix, cap):
            self.assertIs(model, sentinel)
            self.assertEqual(prefix, codec.prompt(public()))
            self.assertEqual(cap, 5)
            captured.append(prefix)
            return [48, 257]
        with patch('trinite.comparison.greedy', side_effect=greedy):
            self.assertEqual(numerical.infer(sentinel, public()), {'status': 'prediction', 'tokens': [48, 257]})
        self.assertEqual(len(captured), 1)
        for forbidden in ('family', 'seed', 'filename', 'sha256', 'provenance', 'answer'):
            self.assertNotIn(forbidden, captured[0])

    def test_native_updater_rejects_token_and_time_budget_before_any_step(self):
        state = engineering_state()
        initial = model_identity(state.model)
        state.target_tokens = state.config.max_target_tokens
        with patch.object(state.optimizer, 'step', side_effect=AssertionError('no model training')):
            with self.assertRaisesRegex(ContractError, 'scored-target budget'):
                run_steps(state, stop_after=1)
        self.assertEqual(model_identity(state.model), initial)
        state.target_tokens = 0
        with patch('trinite.training.time.monotonic', side_effect=[0, 121]):
            with self.assertRaisesRegex(ContractError, 'time budget'):
                run_steps(state, stop_after=1)
        self.assertEqual(state.step, 0)
        self.assertEqual(state.optimizer.state, {})

    def test_feasibility_has_no_optimizer_or_pilot_target_calls(self):
        with (patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('no training')),
              patch.object(gate, 'run', side_effect=AssertionError('no pilot targets'))):
            record = numerical.feasibility_kernel()
        self.assertEqual(record['optimizer_updates'], 0)
        self.assertEqual(record['gradient_calls'], 3)
        self.assertLess(float.fromhex(record['conservative_64_update_estimate_seconds_hex']), 120)


if __name__ == '__main__':
    unittest.main()
