"""Tiny/random numerical checks; never load or score the target checkpoint."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from trinite.contracts import ContractError, ModelConfig, json_bytes
from trinite.model import Decoder
from trinite.tokenizer import ByteTokenizer, EOS
from trinite.training import target_loss
from unseen_evaluation_v1 import measurements as m


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def tiny():
    return Decoder(ModelConfig(context_length=64, blocks=1, width=16, heads=2,
                               feed_forward_width=32), lane='dense', seed=0)


def constant_forward(token=EOS):
    def forward(ids, mask, **kwargs):
        logits = torch.zeros((*ids.shape, 259), dtype=torch.float32)
        logits[:, :, token] = 8
        return SimpleNamespace(logits=logits)
    return forward


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny()
        with patch.object(cls.model, 'forward', side_effect=constant_forward()):
            cls.report = m.measure(cls.model)

    def test_exact_admission_disjointness_and_exposure(self):
        rows = m.examples()
        train = m.train_examples()
        self.assertEqual(len(rows), 128)
        self.assertEqual(len(train), 264)
        for split in ('validation', 'test'):
            group = [r for r in rows if r['split'] == split]
            self.assertEqual(len(group), 64)
            self.assertEqual(len({r['family_id'] for r in group}), 1)
        for field in ('example_identity', 'family_id', 'semantic_identity', 'prompt'):
            self.assertFalse({r[field] for r in rows} & {r[field] for r in train})
        self.assertIn('not globally unseen', self.report['exposure']['research_exposure'])

    def test_constant_control_and_separate_split_task_family_counts(self):
        summary = m.validate_report(self.report)
        self.assertEqual(summary['raw_answer_and_eos_correct'], 0)
        self.assertEqual(summary['training_lookup_misses'], 128)
        self.assertEqual(summary['training_lookup_correct'], 0)
        self.assertEqual(summary['exact_oracle_correct'], 128)
        self.assertEqual(summary['constant_eos_correct'], 0)
        self.assertEqual(summary['raw_premature_eos'], 128)
        self.assertEqual(summary['distinct_generated_outputs'], 1)
        self.assertTrue(self.report['input_sensitivity']['all_conflicting_pairs_identical'])
        for split in ('validation', 'test'):
            group = self.report['by_split'][split]
            self.assertEqual(group['summary']['examples'], 64)
            self.assertEqual(group['summary']['families'], 1)
            self.assertEqual(group['summary']['semantic_items'], 32)
            self.assertEqual(len(group['by_family']), 1)
            self.assertEqual(sum(x['examples'] for x in group['by_task'].values()), 64)
            self.assertGreater(group['summary']['best_constant_ceiling']['correct'], 0)
            self.assertIn('ex-post', group['summary']['best_constant_ceiling']['scope'])

    def test_real_tiny_decoder_observer_neutrality_and_loss(self):
        model = tiny()
        next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
        before = json_bytes(m._state(model))
        report = m.measure(model)
        self.assertEqual(before, json_bytes(m._state(model)))
        m.validate_report(report)
        item = m.examples()[0]
        enc = ByteTokenizer(64).encode(item['text'], answer_start=len(item['prompt']))
        loss, count = target_loss(model, [{'ids': enc.input_ids,
            'mask': enc.attention_mask, 'loss': enc.loss_mask}])
        self.assertEqual(report['rows'][0]['authoritative_loss_hex'], float(loss.detach()).hex())
        self.assertEqual(report['rows'][0]['scored_targets'], count)
        self.assertEqual(report['summary']['forward_calls'], 256+sum(r['output_tokens'] for r in report['rows']))

    def test_raw_generation_gets_only_prompt_once_and_fixed_cap(self):
        prompts = []
        def record(model, prompt, cap):
            prompts.append((prompt, cap))
            return [EOS]
        with patch.object(m, 'greedy', side_effect=record):
            with patch.object(self.model, 'forward', side_effect=constant_forward()):
                m.measure(self.model)
        self.assertEqual(prompts, [(r['prompt'], 5) for r in m.examples()])

    def test_gold_target_offsets_answer_eos_and_probabilities(self):
        offset = 0
        for item, row in zip(m.examples(), self.report['rows']):
            targets = self.report['targets'][offset:offset+row['scored_targets']]
            start = 1 + len(item['prompt'])
            self.assertEqual([t['target_position'] for t in targets], list(range(start, start+len(item['answer'])+1)))
            self.assertEqual([t['gold_token'] for t in targets], [*item['answer'].encode(), EOS])
            for target in targets:
                self.assertAlmostEqual(float.fromhex(target['gold_probability_hex']), math.exp(-float.fromhex(target['loss_hex'])), places=6)
            offset += row['scored_targets']
        self.assertEqual(offset, len(self.report['targets']))

    def test_unfinished_output_and_no_answer_repair(self):
        with patch.object(self.model, 'forward', side_effect=constant_forward(ord('0'))):
            report = m.measure(self.model)
        m.validate_report(report)
        self.assertEqual(report['summary']['raw_unfinished'], 128)
        self.assertEqual(report['summary']['raw_malformed'], 128)
        self.assertEqual(report['summary']['raw_answer_and_eos_correct'], 0)
        self.assertTrue(all(r['generated_tokens'] == [ord('0')]*5 for r in report['rows']))

    def test_all_retained_counts_derive_from_rows_and_targets(self):
        for key in ('raw_answer_correct', 'scored_targets', 'families', 'training_lookup_misses'):
            report = copy.deepcopy(self.report)
            report['summary'][key] += 1
            with self.subTest(key=key), self.assertRaises(ContractError):
                m.validate_report(report)
        report = copy.deepcopy(self.report)
        report['by_split']['validation']['summary']['raw_answer_and_eos_correct'] = 64
        with self.assertRaises(ContractError): m.validate_report(report)

    def test_unknown_missing_duplicate_or_reordered_rows_rejected(self):
        variants = []
        for change in ('missing', 'duplicate', 'order', 'unknown'):
            report = copy.deepcopy(self.report)
            if change == 'missing': report['rows'].pop()
            if change == 'duplicate': report['rows'][1] = copy.deepcopy(report['rows'][0])
            if change == 'order': report['rows'].reverse()
            if change == 'unknown': report['rows'][0]['extra'] = True
            variants.append(report)
        for report in variants:
            with self.assertRaises(ContractError): m.validate_report(report)

    def test_train_or_label_leaking_controls_rejected(self):
        for change in ('train', 'lookup', 'oracle', 'exposure', 'malformed'):
            report = copy.deepcopy(self.report)
            if change == 'train': report['rows'][0]['split'] = 'train'
            if change == 'lookup': report['rows'][0]['non_neural_controls']['training_lookup_tokens'] = report['rows'][0]['expected_tokens']
            if change == 'oracle': report['rows'][0]['non_neural_controls']['exact_oracle_tokens'] = [EOS]
            if change == 'malformed': report['malformed_scope'] = 'full arithmetic syntax'
            if change == 'exposure': report['exposure']['research_exposure'] = 'fresh confirmation'
            with self.subTest(change=change), self.assertRaises(ContractError): m.validate_report(report)

    def test_boolean_integer_alias_and_noncanonical_numbers_rejected(self):
        variants = []
        for field, value in (('gold_rank', True), ('argmax_token', True), ('answer_offset', False), ('is_eos', 0), ('gold_probability_hex', 'nan'), ('loss_hex', '0.0')):
            report = copy.deepcopy(self.report)
            report['targets'][0][field] = value
            variants.append(report)
        report = copy.deepcopy(self.report)
        report['rows'][0]['output_tokens'] = True
        variants.append(report)
        for report in variants:
            with self.assertRaises(ContractError): m.validate_report(report)

    def test_raw_tokens_cannot_silently_terminate_before_cap(self):
        report = copy.deepcopy(self.report)
        report['rows'][0]['generated_tokens'] = [ord('0')]
        with self.assertRaises(ContractError): m.validate_report(report)

    def test_audit_does_not_replay_model_or_generation(self):
        with patch.object(m, 'greedy', side_effect=AssertionError('inference forbidden')):
            with patch.object(m, 'target_loss', side_effect=AssertionError('inference forbidden')):
                m.validate_report(self.report)

    def test_completed_rows_survive_later_generation_failure(self):
        retained = []
        calls = []
        def raw(model, prompt, cap):
            calls.append(prompt)
            if len(calls) > 1:
                raise ContractError('synthetic generation failure')
            return [EOS]
        with patch.object(m, 'greedy', side_effect=raw):
            with patch.object(self.model, 'forward', side_effect=constant_forward()):
                with self.assertRaisesRegex(ContractError, 'synthetic'):
                    m.measure(self.model, lambda row, targets: retained.append((row, targets)))
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0][0], self.report['rows'][0])
        self.assertEqual(len(retained[0][1]), retained[0][0]['scored_targets'])

    def test_progress_callback_cannot_mutate_retained_result(self):
        def mutate(row, targets):
            row.clear()
            targets.clear()
        with patch.object(self.model, 'forward', side_effect=constant_forward()):
            report = m.measure(self.model, mutate)
        self.assertEqual(report, self.report)

    def test_nonfinite_raw_logits_fail_instead_of_argmax_fallback(self):
        calls = []
        base = constant_forward()
        def forward(ids, mask, **kwargs):
            calls.append(True)
            result = base(ids, mask)
            if len(calls) == 3:  # two teacher forwards then first raw generation
                result.logits.fill_(float('nan'))
            return result
        with patch.object(self.model, 'forward', side_effect=forward):
            with self.assertRaisesRegex(ContractError, 'non-finite raw'):
                m.measure(self.model)
        self.assertEqual(len(calls), 3)

    def test_wrong_context_or_lane_rejected_before_forward(self):
        model = Decoder(ModelConfig(context_length=32, blocks=1, width=16, heads=2,
                                    feed_forward_width=32), lane='dense', seed=0)
        with patch.object(model, 'forward', side_effect=AssertionError('must reject first')):
            with self.assertRaises(ContractError): m.measure(model)


if __name__ == '__main__':
    unittest.main()
