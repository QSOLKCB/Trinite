"""Engineering-only synthetic records and fresh tiny models, never target checkpoints.

Hand-constructed exact/constant answers exercise reporting, not model capability.
The unchanged scorer validates their syntax/context before new classifications.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from trinite.contracts import ContractError, ModelConfig, json_bytes, parse_json
from trinite.foundations import majority_answers, metrics, check_prediction_examples
from trinite.model import Decoder
from trinite.tokenizer import EOS
from trinite.training import model_identity
from native_diagnostic_v1.measurements import _state
from addition_development_v1.data import dataset, examples
from addition_development_v1.numerical import view, check_score
from addition_development_v1.protocol import STRATA
from failure_map_v1 import measurements as m


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
FAKE_MODEL = 'sha256:' + 'a' * 64


def exact_answer(item):
    return [*item['answer'].encode('ascii'), EOS]


def prediction(item, split, baseline, emit):
    expected = exact_answer(item)
    generated = list(emit(item))
    constant = [*baseline.encode('ascii'), EOS]
    return {'example_identity': item['example_identity'], 'family_id': item['family_id'],
            'task': item['task'], 'split': split, 'expected_tokens': expected,
            'generated_tokens': generated, 'correct': generated == expected,
            'baseline_tokens': constant, 'baseline_correct': constant == expected}


def historical(arm='control', model=FAKE_MODEL, emit=exact_answer):
    records = {}
    for selector in ('train', *STRATA):
        items, split = view(arm, selector)
        baseline = majority_answers(items)['add']
        predictions = [prediction(item, split, baseline, emit) for item in items if item['split'] == split]
        record = {'model_identity': model, 'summary': metrics(predictions), 'predictions': predictions}
        check_score(record, arm, selector, model)
        records[selector] = json_bytes(record)
    return {s: records[s] for s in STRATA}, records['train']


def current_predictions(arm='control', emit=exact_answer):
    items = parse_json(dataset()[0], canonical=True)['examples']
    anchors = examples(arm)
    baseline = majority_answers(anchors)['add']
    predictions = [prediction(item, 'test', baseline, emit) for item in items]
    for item, predicted in zip(items, predictions):
        check_prediction_examples(anchors + [{**item, 'split': 'test'}], 'test', 5,
                                  json_bytes({'predictions': [predicted]}))
    return predictions


def report(arm='control', emit=exact_answer, prior_emit=None):
    old, train = historical(arm, emit=emit if prior_emit is None else prior_emit)
    result = m._build_report(current_predictions(arm, emit), arm, 0, FAKE_MODEL, old, train)
    return result, old, train


class ClassificationTests(unittest.TestCase):
    def test_disjoint_raw_error_categories(self):
        expected = [49, 51, EOS]
        cases = [([49, 51, EOS], 'exact_correct'), ([49, 49, EOS], 'wrong_value'),
                 ([EOS], 'immediate_eos'), ([49, EOS], 'premature_eos'),
                 ([ord('x'), EOS], 'malformed_output'), ([48, 49, EOS], 'malformed_output'),
                 ([49] * 5, 'missing_eos')]
        for tokens, category in cases:
            with self.subTest(tokens=tokens):
                self.assertEqual(m.classify(tokens, expected), category)

    def test_impossible_generation_trajectories_rejected(self):
        for tokens in ([], [EOS, 49], [49], [49, 50], [49] * 6, [True, EOS], [-1, EOS], [259, EOS]):
            with self.subTest(tokens=tokens), self.assertRaises(ContractError):
                m.classify(tokens, [49, EOS])

    def test_numeric_wrong_value_is_not_eos_failure(self):
        self.assertEqual(m.classify([49, EOS], [50, EOS]), 'wrong_value')
        self.assertEqual(m.classify([50, EOS], [49, 51, EOS]), 'wrong_value')
        self.assertEqual(m.classify([48, EOS], [48, EOS]), 'exact_correct')


class NativeMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.model = Decoder(ModelConfig(context_length=64, blocks=1, width=16, heads=2,
                                        feed_forward_width=32), lane='dense', seed=0)

    def test_real_tiny_decoder_calls_authoritative_scorer_without_training(self):
        from trinite.foundations import score_examples
        old = {}
        mid = model_identity(self.model)
        for selector in ('train', *STRATA):
            items, split = view('control', selector)
            summary, predictions = score_examples(self.model, items, split, 5)
            old[selector] = json_bytes({'model_identity': mid, 'summary': summary, 'predictions': predictions})
        train = old.pop('train')
        before = json_bytes(_state(self.model))
        callbacks = []
        with patch.object(torch.optim, 'AdamW', side_effect=AssertionError('no optimizer allowed')):
            result = m.measure(self.model, 'control', 0, old, train, callbacks.append)
        self.assertEqual(before, json_bytes(_state(self.model)))
        self.assertEqual(len(result['rows']), 338)
        self.assertEqual(callbacks, result['rows'])
        self.assertEqual(result['summary'], m.validate_report(result, 'control', 0, mid, old, train))

    def test_callback_cannot_change_retained_rows_and_rng_mutation_rejected(self):
        mid = model_identity(self.model)
        old, train = historical(model=mid, emit=lambda item: [EOS])
        def forward(ids, mask, **kwargs):
            logits = torch.zeros((*ids.shape, 259))
            logits[:, :, EOS] = 5.0
            return SimpleNamespace(logits=logits)
        with patch.object(self.model, 'forward', side_effect=forward):
            result = m.measure(self.model, 'control', 0, old, train, lambda row: row.clear())
        self.assertTrue(all(result['rows']))
        with patch.object(self.model, 'forward', side_effect=forward), self.assertRaises(ContractError):
            m.measure(self.model, 'control', 0, old, train, lambda row: torch.rand(1))

    def test_wrong_context_dtype_lane_rejected_before_inference(self):
        for config, lane, dtype in [(32, 'dense', torch.float32), (64, 'ternary', torch.float32),
                                    (64, 'dense', torch.float64)]:
            model = Decoder(ModelConfig(context_length=config, blocks=1, width=16, heads=2,
                                        feed_forward_width=32), lane=lane, seed=0).to(dtype=dtype)
            old, train = historical(model=model_identity(model))
            with self.subTest(context=config, lane=lane, dtype=dtype), \
                 patch.object(model, 'forward', side_effect=AssertionError('must reject before inference')), \
                 self.assertRaises(ContractError):
                m.measure(model, 'control', 0, old, train)


    def test_context_budget_never_silently_truncates(self):
        items = m.admitted()
        items[0]['prompt'] = 'A' * 60
        old, train = historical(model=model_identity(self.model))
        with patch.object(m, 'admitted', return_value=items), \
             patch.object(self.model, 'forward', side_effect=AssertionError('no truncated generation')), \
             self.assertRaisesRegex(ContractError, 'truncate'):
            m.measure(self.model, 'control', 0, old, train)

    def test_nonfinite_wrong_dtype_and_wrong_shape_logits_rejected(self):
        old, train = historical(model=model_identity(self.model), emit=lambda item: [EOS])
        def forward_for(kind):
            def forward(ids, mask, **kwargs):
                shape = (*ids.shape, 258 if kind == 'shape' else 259)
                logits = torch.zeros(shape, dtype=torch.float64 if kind == 'dtype' else torch.float32)
                if kind in ('nan', 'inf'):
                    logits[:, :, 0] = float(kind)
                return SimpleNamespace(logits=logits)
            return forward
        for kind in ('nan', 'inf', 'dtype', 'shape'):
            with self.subTest(kind=kind), patch.object(self.model, 'forward', side_effect=forward_for(kind)), \
                 self.assertRaises(ContractError):
                m.measure(self.model, 'control', 0, old, train)


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exact = {arm: report(arm) for arm in ('control', 'candidate')}

    def test_exhaustive_domain_correctness_and_arm_specific_exposure(self):
        for arm, anchor_count in [('control', 20), ('candidate', 40)]:
            result, old, train = self.exact[arm]
            summary = m.validate_report(result, arm, 0, FAKE_MODEL, old, train)
            self.assertEqual(len(result['rows']), 338)
            self.assertEqual({(r['a'], r['b'], r['carrier']) for r in result['rows']},
                             {(a, b, c) for a in range(13) for b in range(13) for c in ('infix', 'fields')})
            self.assertEqual(summary['families'], 91)
            self.assertEqual(summary['raw_correct'], 338)
            self.assertEqual((summary['pass'], summary['fail'], summary['untested']), (338, 0, 0))
            self.assertEqual(summary['seen_anchors']['rows'], anchor_count)
            self.assertEqual(summary['zero_answer_controls']['rows'], 2)
            self.assertEqual(summary['reproducibility']['match'], anchor_count + 298)
            unseen = [r for r in result['rows'] if r['reproducibility'] == 'not_previously_scored']
            self.assertEqual(len(unseen), 20 if arm == 'control' else 0)
            for row in unseen:
                self.assertEqual(row['scientific_exposure'], 'other_arm_training_family_not_seen_in_this_arm')
                self.assertFalse(row['seen_in_this_arm'])
                self.assertIsNone(row['historical_generated_tokens'])
            self.assertFalse(result['fresh_confirmation'])
            self.assertFalse(result['learning_adequate'])

    def test_complete_cross_product_empty_cells_are_untested(self):
        result, _, _ = self.exact['control']
        cells = result['summary']['coverage_cells']
        self.assertEqual(len(cells), 48)
        self.assertEqual(sum(c['rows'] for c in cells), 338)
        self.assertEqual(len({json_bytes(c['criteria']) for c in cells}), 48)
        self.assertTrue(any(not c['rows'] for c in cells))
        for cell in cells:
            self.assertEqual(cell['status'], 'pass' if cell['rows'] else 'untested')
            self.assertEqual(cell['untested_reason'] is not None, cell['rows'] == 0)
        self.assertEqual(sum(c['rows'] for c in cells if c['criteria']['joint_range_digit_length_shift']), 138)
        for row in result['rows']:
            self.assertEqual(row['joint_range_digit_length_shift'], max(row['a'], row['b']) >= 10)

    def test_pair_accounting_excludes_diagonal_order_self_comparisons(self):
        result, _, _ = self.exact['candidate']
        by_id = {r['example_identity']: r for r in result['rows']}
        for group, count in [('carrier', 169), ('operand_order', 156)]:
            pairs = result['summary']['metamorphic'][group]
            self.assertEqual(pairs['pair_count'], count)
            self.assertEqual(pairs['consistent'], count)
            self.assertEqual(pairs['both_correct'], count)
            self.assertEqual(pairs['categories'], {'both_correct': count})
            for pair in pairs['pairs']:
                left, right = by_id[pair['left']], by_id[pair['right']]
                self.assertNotEqual(pair['left'], pair['right'])
                if group == 'carrier':
                    self.assertEqual((left['a'], left['b']), (right['a'], right['b']))
                    self.assertNotEqual(left['carrier'], right['carrier'])
                else:
                    self.assertEqual((left['a'], left['b']), (right['b'], right['a']))
                    self.assertEqual(left['carrier'], right['carrier'])
                    self.assertLess(left['a'], left['b'])

    def test_constant_wrong_consistency_is_never_success(self):
        result, _, _ = report(emit=lambda item: [57, 57, EOS])
        summary = result['summary']
        self.assertEqual((summary['raw_correct'], summary['pass'], summary['fail']), (0, 0, 338))
        self.assertEqual(summary['categories']['wrong_value'], 338)
        self.assertEqual(len(summary['counterexample_identities']), 338)
        for name, count in [('carrier', 169), ('operand_order', 156)]:
            pairs = summary['metamorphic'][name]
            self.assertEqual((pairs['consistent'], pairs['both_correct']), (count, 0))
            self.assertEqual(pairs['categories'], {'both_wrong_same': count})

    def test_replay_mismatch_separate_from_scientific_accuracy(self):
        result, old, train = report(emit=exact_answer, prior_emit=lambda item: [EOS])
        summary = result['summary']
        self.assertEqual(summary['raw_correct'], 338)
        self.assertEqual(summary['reproducibility'], {'mismatch': 318, 'not_previously_scored': 20})
        self.assertEqual((summary['pass'], summary['fail'], summary['untested']), (20, 0, 318))
        self.assertEqual(summary['status'], 'untested')
        self.assertEqual(len(summary['reproducibility_mismatch_identities']), 318)
        self.assertEqual(summary['seen_anchors']['untested'], 20)
        self.assertEqual(summary, m.validate_report(result, 'control', 0, FAKE_MODEL, old, train))
        for row in result['rows']:
            if row['reproducibility'] == 'mismatch':
                self.assertEqual(row['prediction']['generated_tokens'], exact_answer({'answer': row['answer']}))
                self.assertEqual(row['historical_generated_tokens'], [EOS])
                self.assertEqual(row['status'], 'untested')

    def test_order_and_carrier_errors_remain_exact_counterexamples(self):
        def altered(item):
            a, b = item['formal']['a'], item['formal']['b']
            return exact_answer(item) if a <= b and item['carrier'] == 'infix' else [57, 57, EOS]
        result, _, _ = report(emit=altered)
        summary = result['summary']
        self.assertEqual(summary['raw_correct'], 91)
        self.assertEqual(summary['metamorphic']['carrier']['categories'],
                         {'both_wrong_same': 78, 'one_correct': 91})
        self.assertEqual(summary['metamorphic']['operand_order']['categories'],
                         {'both_wrong_same': 78, 'one_correct': 78})
        failed = {r['example_identity'] for r in result['rows'] if not r['prediction']['correct']}
        self.assertEqual(set(summary['counterexample_identities']), failed)
        self.assertTrue(all(r['prompt'] and r['answer'] and r['prediction']['generated_tokens'] for r in result['rows']))

    def test_report_tampering_missing_rows_context_metrics_claims_rejected(self):
        result, old, train = self.exact['control']
        mutations = [lambda r: r['rows'].pop(), lambda r: r['rows'].reverse(),
                     lambda r: r['rows'].append(copy.deepcopy(r['rows'][0])),
                     lambda r: r['rows'][0].update(a=99), lambda r: r['rows'][0].update(status='fail'),
                     lambda r: r['rows'][0].update(reproducibility='not_previously_scored'),
                     lambda r: r['rows'][0].update(seen_in_this_arm=True),
                     lambda r: r['summary'].update(raw_correct=0),
                     lambda r: r['summary']['metamorphic']['carrier'].update(consistent=0),
                     lambda r: r.update(fresh_confirmation=True), lambda r: r.update(checkpoint_step=0)]
        for mutate in mutations:
            changed = copy.deepcopy(result); mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ContractError):
                m.validate_report(changed, 'control', 0, FAKE_MODEL, old, train)
        with self.assertRaises(ContractError):
            m.validate_report(result, 'candidate', 0, FAKE_MODEL, old, train)
        with self.assertRaises(ContractError):
            m.validate_report(result, 'control', 1, FAKE_MODEL, old, train)
        with self.assertRaises(ContractError):
            m.validate_report(result, 'control', 0, 'sha256:' + 'b' * 64, old, train)

    def test_original_source_predictions_checked_not_blindly_compared(self):
        result, old, train = self.exact['control']
        changed = parse_json(train, canonical=True)
        changed['predictions'][0]['correct'] = False
        with self.assertRaises(ContractError):
            m.validate_report(result, 'control', 0, FAKE_MODEL, old, json_bytes(changed))
        with self.assertRaises(ContractError):
            m.validate_report(result, 'control', 0, FAKE_MODEL, {}, train)

    def test_boundary_controls_do_not_claim_model_robustness(self):
        controls = m.boundary_controls()
        self.assertEqual(len(controls['rows']), 9)
        self.assertTrue(all(r['outcome'] == 'rejected_input' and r['model_executed'] is False for r in controls['rows']))
        self.assertIn('not model instruction following', controls['scope'])
        with patch.object(m, 'parse_prompt', return_value={'task': 'add', 'a': 1, 'b': 2}), self.assertRaises(ContractError):
            m.boundary_controls()

    def test_failed_worker_has_complete_untested_map_not_model_failures(self):
        result = m.untested_report('control', 0, FAKE_MODEL, 'timed_out')
        self.assertEqual(result['counts'], {'pass': 0, 'fail': 0, 'untested': 338})
        self.assertEqual(len(result['rows']), 338)
        self.assertTrue(all(r['status'] == 'untested' and r['reason'] == 'timed_out' for r in result['rows']))
        self.assertFalse(result['learning_adequate'])


if __name__ == '__main__':
    unittest.main()
