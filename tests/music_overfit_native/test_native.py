"""Tiny synthetic native conformance only; never executes the music experiment."""
from types import SimpleNamespace
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from trinite.contracts import ModelConfig, ContractError, identity, parse_json, json_bytes
from trinite.model import Decoder
from trinite.tokenizer import ByteTokenizer, EOS
from trinite.training import (RunConfig, TrainingState, batch, create_optimizer,
                              evaluate, model_identity, run_steps, target_loss)
from trinite.foundations import score_examples, check_prediction_examples
from music_overfit_v1 import numerical


def fixture():
    # Deliberately not a music source, subset or target-run artifact.
    row = {'example_identity': 'synthetic-contract', 'family_id': 'synthetic',
           'task': 'synthetic', 'split': 'train', 'prompt': 'Q=', 'answer': '7'}
    enc = ByteTokenizer(16).encode('Q=7', answer_start=2)
    carrier = {'identity': row['example_identity'], 'ids': enc.input_ids,
               'mask': enc.attention_mask, 'loss': enc.loss_mask}
    return row, carrier


class NativeConformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def model(self):
        return Decoder(ModelConfig(context_length=128, blocks=1, width=8, heads=2,
                                   feed_forward_width=16), lane='dense', seed=123)

    def test_prefix_and_padding_alignment_real_native(self):
        row, carrier = fixture(); model = self.model()
        ids, mask, scored = batch([carrier]); p = 2; a = 1
        with torch.no_grad():
            full = model(ids[:, :-1], mask[:, :-1]).logits
            for index in (p, p+a):
                prefix = model(ids[:, :index+1], mask[:, :index+1]).logits[0, -1]
                torch.testing.assert_close(full[0, index], prefix, rtol=1e-5, atol=1e-5)
            padded_ids = torch.cat((ids[:, :-1], torch.tensor([[258, 258]])), dim=1)
            padded_mask = torch.cat((mask[:, :-1], torch.tensor([[False, False]])), dim=1)
            padded = model(padded_ids, padded_mask).logits
            torch.testing.assert_close(full, padded[:, :full.shape[1]], rtol=1e-5, atol=1e-5)
            self.assertEqual(torch.count_nonzero(padded[:, -2:]).item(), 0)
            self.assertEqual(scored[:, 1:].sum().item(), 2)
            expected = torch.nn.functional.cross_entropy(full[:, p:p+2].reshape(-1, 259), ids[:, p+1:].reshape(-1))
            actual, tokens = target_loss(model, [carrier])
            torch.testing.assert_close(actual, expected)
            self.assertEqual(tokens, 2)

    def test_native_optimizer_really_updates_synthetic_fixture(self):
        _, carrier = fixture(); model = self.model()
        config = RunConfig(model=model.config, seed=123, steps=8, batch_size=1,
                           learning_rate='0.003', validation_every=8,
                           max_seconds=30, max_target_tokens=16)
        initial = evaluate(model, [carrier], 1); before = model_identity(model)
        state = TrainingState(config, 'dense', model, create_optimizer(model, config),
                              {'train': [carrier], 'validation': [carrier]},
                              'synthetic-data', 'synthetic-admission', before, initial)
        run_steps(state)
        self.assertEqual((state.step, state.cursor, state.target_tokens), (8, 8, 16))
        self.assertNotEqual(model_identity(model), before)
        self.assertLess(float.fromhex(evaluate(model, [carrier], 1)), float.fromhex(initial))
        self.assertEqual(len(state.history), 8)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_scorer_control_exact_eos_required(self):
        row, _ = fixture(); model = self.model()
        for emitted, correct in (([55, EOS], True), ([55], False), ([47]*64, False), ([EOS], False)):
            with patch('trinite.foundations.greedy', return_value=emitted):
                _, predictions = score_examples(model, [row], 'train', 64)
            self.assertEqual(predictions[0]['correct'], correct)
            self.assertEqual(predictions[0]['generated_tokens'], emitted)

    def test_observation_helper_retains_gold_prefix_separately(self):
        row, carrier = fixture(); model = self.model()
        state = SimpleNamespace(model=model, music_subset='synthetic-contract', step=0,
            config=SimpleNamespace(batch_size=1), data={'train':[carrier], 'validation':[carrier]})
        # These adapters explicitly substitute a synthetic contract input only;
        # this is not an admission test or evidence of target music fit.
        with patch.object(numerical, 'validate_rows', return_value=[row]), \
             patch.object(numerical, 'encoded', return_value=[carrier]):
            result = numerical.inspect(state, [row])
        with patch.object(numerical, 'validate_rows', return_value=[row]):
            self.assertEqual(numerical.check_diagnostic(result, [row]), result['raw_score'])
            for field, value in [('rank', 259), ('target_probability_hex', float(2).hex()),
                                 ('logit_position', 0)]:
                changed = deepcopy(result)
                changed['rows'][0]['first_answer'][field] = value
                with self.assertRaises(ContractError):
                    numerical.check_diagnostic(changed, [row])
            changed = deepcopy(result); changed['rows'][0]['first_divergent_position'] = 999
            with self.assertRaises(ContractError):
                numerical.check_diagnostic(changed, [row])
            for impossible in ([47], [EOS, 47]):
                changed = deepcopy(result)
                prediction = changed['rows'][0]['free_generation']
                prediction['generated_tokens'] = impossible
                prediction['correct'] = False
                changed['raw_score'] = check_prediction_examples([row], 'train', 64,
                    json_bytes({'predictions':[prediction]}))
                changed['rows'][0]['first_divergent_position'] = 0
                changed['rows'][0]['termination'] = 'cap'
                with self.assertRaisesRegex(ContractError, 'termination'):
                    numerical.check_diagnostic(changed, [row])
            changed = deepcopy(result); changed['rows'][0]['example_identity'] = 'foreign'
            with self.assertRaises(ContractError):
                numerical.check_diagnostic(changed, [row])
        observation = result['rows'][0]
        self.assertEqual(observation['first_answer']['logit_position'], 2)
        self.assertEqual(observation['gold_prefix_eos']['logit_position'], 3)
        self.assertEqual(len(observation['gold_prefix_eos']['logits_hex']), 259)
        self.assertEqual(observation['first_answer']['argmax'], observation['free_generation']['generated_tokens'][0])
        self.assertEqual([p['target'] for p in observation['positions']], [55, EOS])

    def test_cumulative_deadline_and_failure_artifacts(self):
        state = SimpleNamespace(step=0, config=SimpleNamespace(max_seconds=120))
        retained = {}
        def update(state, stop_after):
            state.step = stop_after
        with patch.object(numerical, 'state_for', return_value=state), \
             patch.object(numerical, 'subset_rows', return_value=[{'synthetic': True}]), \
             patch.object(numerical, 'snapshot', return_value=(b'synthetic-not-tensors', b'{}')), \
             patch.object(numerical, 'inspect', return_value={'synthetic': True}), \
             patch.object(numerical, 'monotonic', side_effect=[0., 70., 100., 151.]), \
             patch('trinite.training.run_steps', side_effect=update):
            with self.assertRaisesRegex(ContractError, 'cumulative'):
                numerical.run_trial(17, 'one', 'synthetic', 'synthetic', retained.__setitem__)
        self.assertEqual(state.step, 2)
        self.assertIn('step-000-diagnostics.json', retained)
        self.assertIn('failure.safetensors', retained)
        self.assertIn('failure-checkpoint.json', retained)
        self.assertIn('failure.json', retained)
        self.assertNotIn('step-008-diagnostics.json', retained)

    def test_prediction_shape_finiteness_and_rank(self):
        vector = torch.zeros(259); vector[55] = 1
        result = numerical._prediction(vector, 55, retain_logits=True)
        self.assertEqual((result['argmax'], result['rank']), (55, 1))
        self.assertEqual(len(result['logits_hex']), 259)
        with self.assertRaises(ContractError):
            numerical._prediction(torch.zeros(258), 55)
        vector[0] = float('nan')
        with self.assertRaises(ContractError):
            numerical._prediction(vector, 55)

    def test_safe_checkpoint_uses_native_reader_without_any_forward(self):
        row, carrier = fixture(); model = self.model()
        config = RunConfig(model=model.config, seed=123, steps=1, batch_size=1,
                           learning_rate='0.003', validation_every=1,
                           max_seconds=30, max_target_tokens=2)
        raw = b'synthetic-only-data'; admission_id = identity(b'synthetic-only-admission')
        request_id = identity(b'synthetic-only-request')
        state = TrainingState(config, 'dense', model, create_optimizer(model, config),
            {'train':[carrier], 'validation':[carrier]}, identity(raw), admission_id,
            model_identity(model), evaluate(model, [carrier], 1))
        state.music_subset = 'one'
        run_steps(state)  # One bounded synthetic update exercises actual AdamW slots.
        payload, meta_raw = numerical.snapshot(state, request_id)
        metadata = parse_json(meta_raw, canonical=True)
        with patch.object(numerical, 'plan_for', return_value=config), \
             patch.object(numerical, 'subset_rows', return_value=[row]), \
             patch.object(numerical, 'encoded', return_value=[carrier]), \
             patch('music_learning_v1.workflow.require_native_admission', return_value=(raw, b'synthetic-receipt')), \
             patch.object(Decoder, 'forward', side_effect=AssertionError('verification must never forward')):
            restored = numerical.verify_checkpoint(payload, metadata, 123, 'one', admission_id, request_id)
            self.assertEqual(model_identity(restored.model), model_identity(state.model))
            self.assertEqual(restored.step, 1)
            with self.assertRaises(ContractError):
                numerical.verify_checkpoint(b'corrupted', metadata, 123, 'one', admission_id, request_id)
            changed = deepcopy(metadata); changed['model_identity'] = identity(b'foreign-model')
            with self.assertRaises(ContractError):
                numerical.verify_checkpoint(payload, changed, 123, 'one', admission_id, request_id)
            from safetensors.torch import load, save
            tensors = load(payload); tensors.pop(next(iter(tensors)))
            with self.assertRaises(ContractError):
                numerical.verify_checkpoint(save(tensors), metadata, 123, 'one', admission_id, request_id)

    def test_nested_predictions_cannot_swap_between_valid_rows(self):
        first, encoded_first = fixture()
        second = {**first, 'example_identity': 'synthetic-second', 'prompt': 'R=', 'answer': '8'}
        value = ByteTokenizer(16).encode('R=8', answer_start=2)
        encoded_second = {'identity': second['example_identity'], 'ids': value.input_ids,
                          'mask': value.attention_mask, 'loss': value.loss_mask}
        rows = [first, second]; train = [encoded_first, encoded_second]
        model = self.model()
        state = SimpleNamespace(model=model, music_subset='synthetic-contract', step=0,
            config=SimpleNamespace(batch_size=2), data={'train':train, 'validation':train})
        with patch.object(numerical, 'validate_rows', return_value=rows), \
             patch.object(numerical, 'encoded', return_value=train):
            result = numerical.inspect(state, rows)
            numerical.check_diagnostic(result, rows)
            changed = deepcopy(result)
            a, b = changed['rows']
            a['free_generation'], b['free_generation'] = b['free_generation'], a['free_generation']
            changed['raw_score'] = check_prediction_examples(rows, 'train', 64,
                json_bytes({'predictions':[v['free_generation'] for v in changed['rows']]}))
            with self.assertRaisesRegex(ContractError, 'ordered row identity'):
                numerical.check_diagnostic(changed, rows)


if __name__ == '__main__':
    unittest.main()
