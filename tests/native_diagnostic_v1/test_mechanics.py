"""Real CPU mechanics and deliberate corruption controls; no target experiment."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from native_diagnostic_v1 import mechanics
from trinite.contracts import ContractError, json_bytes
from trinite.model import Decoder, ModelOutput
from trinite.tokenizer import ByteTokenizer


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class MechanicsTests(unittest.TestCase):
    def model(self):
        return Decoder(mechanics.engineering_plan().model, lane='dense', seed=0)

    def test_complete_preflight_is_deterministic_real_backend(self):
        one, two = mechanics.preflight(), mechanics.preflight()
        self.assertEqual(json_bytes(one), json_bytes(two))
        self.assertIs(mechanics.validate_preflight(one), one)
        self.assertTrue(one['passed'])
        self.assertTrue(one['global_rng_unchanged'])
        self.assertEqual(one['update_resume']['updates'], 3)
        self.assertNotEqual(*one['loss']['answer_lengths'])
        self.assertEqual(one['loss']['scored_tokens'], sum(one['loss']['answer_lengths'])+2)

    def receipt(self):
        # Schema fixture, not evidence of a model run. The real preflight above
        # checks that actual generated receipts satisfy the same contract.
        return {'schema': 'trinite.native-diagnostic-mechanics.v1', 'passed': True,
                'scope': 'three-update engineering controls; no target learning claim',
                'loss_atol_hex': mechanics.LOSS_ATOL.hex(),
                'loss_rtol_hex': mechanics.LOSS_RTOL.hex(),
                'loss': {'examples': 2, 'scored_tokens': 7, 'answer_lengths': [1, 4],
                         'actual_loss_hex': (5.0).hex(), 'oracle_loss_hex': (5.0).hex()},
                'attention': {'same_shape_causal_prefix_exact': True,
                              'masked_suffix_exact': True, 'padded_logits_zero': True,
                              'output_embedding_tied': True},
                'update_resume': {'updates': 3, 'split_after': 1,
                                  'model_and_optimizer_exact': True,
                                  'observation_isolation_exact': True, 'model_updated': True,
                                  'positive_finite_gradient_norms': True,
                                  'completed_gradients_cleared': True},
                'global_rng_unchanged': True}

    def test_receipt_validation_never_executes_model_or_dataset(self):
        with patch.object(mechanics, 'state_for', side_effect=AssertionError('model execution')), \
             patch.object(mechanics, 'dataset', side_effect=AssertionError('dataset execution')), \
             patch.object(mechanics.training, 'require_environment', side_effect=AssertionError('backend')):
            mechanics.validate_preflight(self.receipt())

    def test_receipt_rejects_missing_and_extra_fields_at_every_level(self):
        with self.assertRaises(ContractError):
            mechanics.validate_preflight({'passed': True})
        for section in (None, 'loss', 'attention', 'update_resume'):
            for mutation in ('extra', 'missing'):
                record = self.receipt()
                target = record if section is None else record[section]
                if mutation == 'extra':
                    target['unknown'] = True
                else:
                    del target[next(iter(target))]
                with self.subTest(section=section, mutation=mutation), self.assertRaises(ContractError):
                    mechanics.validate_preflight(record)

    def test_receipt_rejects_changed_controls_and_wrong_types(self):
        cases = [('schema', 'v2'), ('scope', 'learning success'), ('passed', 1),
                 ('global_rng_unchanged', False), ('loss_atol_hex', (1.0).hex()),
                 ('loss_rtol_hex', (1.0).hex())]
        for key, value in cases:
            record = self.receipt(); record[key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                mechanics.validate_preflight(record)
        for section in ('attention', 'update_resume', 'loss'):
            for key, old in self.receipt()[section].items():
                if key.endswith('_hex'):
                    continue
                record = self.receipt()
                record[section][key] = (1 if type(old) is bool else
                                       True if type(old) is int else [True, 4])
                with self.subTest(section=section, key=key), self.assertRaises(ContractError):
                    mechanics.validate_preflight(record)

    def test_receipt_rejects_noncanonical_nonfinite_and_disagreeing_losses(self):
        for key in ('actual_loss_hex', 'oracle_loss_hex'):
            for value in (True, 5, 'nan', 'inf', '-inf', 'garbage', '0x1p+999999',
                          '0x0.0p+0', (-1.0).hex(), '0x1.4p+2', (6.0).hex()):
                record = self.receipt(); record['loss'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ContractError):
                    mechanics.validate_preflight(record)

    def test_loss_corruption_fails_independent_oracle(self):
        original = mechanics.training.target_loss
        def wrong(model, rows):
            loss, count = original(model, rows)
            return loss + .25, count
        with patch.object(mechanics.training, 'target_loss', wrong):
            with self.assertRaisesRegex(ContractError, 'token-mean loss'):
                mechanics.check_loss(self.model())

    def test_per_example_mean_is_not_token_mean(self):
        original = mechanics.training.target_loss
        def wrong(model, rows):
            values = [original(model, [row]) for row in rows]
            return sum(v[0] for v in values)/len(values), sum(v[1] for v in values)
        with patch.object(mechanics.training, 'target_loss', wrong):
            with self.assertRaisesRegex(ContractError, 'token-mean loss'):
                mechanics.check_loss(self.model())

    def test_padding_scored_as_target_is_detected(self):
        original = mechanics.training.batch
        def wrong(rows):
            ids, mask, scored = original(rows)
            return ids, mask, scored | ~mask
        with patch.object(mechanics.training, 'batch', wrong):
            with self.assertRaisesRegex(ContractError, 'scored-token count'):
                mechanics.check_loss(self.model())

    def test_wrong_scored_count_rejected(self):
        original = mechanics.training.target_loss
        def wrong(model, rows):
            loss, count = original(model, rows)
            return loss, count+1
        with patch.object(mechanics.training, 'target_loss', wrong):
            with self.assertRaisesRegex(ContractError, 'scored-token count'):
                mechanics.check_loss(self.model())

    def test_prompt_loss_or_missing_eos_mask_rejected(self):
        original = ByteTokenizer.encode
        for position in (1, -1):
            def wrong(tokenizer, *args, **kwargs):
                result = original(tokenizer, *args, **kwargs)
                loss = list(result.loss_mask)
                loss[position] = 1-loss[position]
                return replace(result, loss_mask=tuple(loss))
            with self.subTest(position=position), patch.object(ByteTokenizer, 'encode', wrong):
                with self.assertRaisesRegex(ContractError, 'mask mismatch'):
                    mechanics.check_loss(self.model())

    def test_held_out_rows_rejected_before_forward(self):
        rows = mechanics._examples()
        rows[0] = {**rows[0], 'split': 'test'}
        with self.assertRaisesRegex(ContractError, 'held-out'):
            mechanics._manual_batch(rows, 64)

    def test_future_leak_is_detected(self):
        model = self.model()
        forward = model.forward
        def leaked(ids, mask, **kwargs):
            result = forward(ids, mask, **kwargs)
            logits = result.logits.clone()
            logits[:, 0, 0] += ids[:, 4].float()
            return ModelOutput(logits, result.captures)
        with patch.object(model, 'forward', leaked):
            with self.assertRaisesRegex(ContractError, 'future tokens'):
                mechanics.check_attention(model)

    def test_padding_leak_is_detected(self):
        model = self.model()
        forward = model.forward
        def leaked(ids, mask, **kwargs):
            result = forward(ids, mask, **kwargs)
            logits = result.logits.clone()
            logits[:, 0, 0] += ids[:, -1].float()
            return ModelOutput(logits, result.captures)
        with patch.object(model, 'forward', leaked):
            with self.assertRaisesRegex(ContractError, 'padding IDs'):
                mechanics.check_attention(model)

    def test_nonzero_padded_output_is_detected(self):
        model = self.model()
        forward = model.forward
        def leaked(ids, mask, **kwargs):
            result = forward(ids, mask, **kwargs)
            logits = result.logits.clone(); logits[:, -1] = 1
            return ModelOutput(logits, result.captures)
        with patch.object(model, 'forward', leaked):
            with self.assertRaisesRegex(ContractError, 'padded output'):
                mechanics.check_attention(model)

    def test_resume_corruption_is_detected(self):
        original = mechanics._resume
        def wrong(*args, **kwargs):
            state = original(*args, **kwargs)
            with torch.no_grad():
                next(state.model.parameters())[0, 0] += 1
            return state
        with patch.object(mechanics, '_resume', wrong):
            with self.assertRaisesRegex(ContractError, 'replay differs'):
                mechanics.check_update_resume()

    def test_no_environment_bypass(self):
        with patch.object(mechanics.training, 'require_environment', side_effect=ContractError('wrong backend')):
            with self.assertRaisesRegex(ContractError, 'wrong backend'):
                mechanics.preflight()


if __name__ == '__main__':
    unittest.main()
