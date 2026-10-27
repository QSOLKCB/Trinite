"""Pinned CPU import/checker boundary, synthetic predictions only.

No Decoder construction, forward pass, optimizer or target pilot is executed.
This tests the real authoritative prediction checker, not numerical learning.
"""
from copy import deepcopy
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, json_bytes
from trinite.foundations import check_prediction_examples, majority_answers
from trinite.tokenizer import EOS
from music_learning_v1.numerical import GENERATION_CAP, view, check_score
from music_learning_v1.reporting import annotate


def synthetic_record(selector):
    items, native_split = view(selector)
    modes = majority_answers(items)
    predictions = []
    for item in items:
        if item['split'] != native_split:
            continue
        gold = [*item['answer'].encode('ascii'), EOS]
        mode = [*modes[item['task']].encode('ascii'), EOS]
        predictions.append({'example_identity': item['example_identity'],
            'family_id': item['family_id'], 'task': item['task'], 'split': native_split,
            'expected_tokens': gold, 'generated_tokens': gold, 'correct': True,
            'baseline_tokens': mode, 'baseline_correct': mode == gold})
    summary = check_prediction_examples(items, native_split, GENERATION_CAP,
                                       json_bytes({'predictions': predictions}))
    return {'selector': selector, 'model_identity': 'synthetic-not-a-checkpoint',
            'summary': summary, 'predictions': predictions}


class NativeCheckerBoundaryTests(unittest.TestCase):
    def test_actual_shared_checker_and_component_adapter(self):
        with patch('trinite.model.Decoder', side_effect=AssertionError('no model allowed')), \
             patch('trinite.training.run_steps', side_effect=AssertionError('no training allowed')), \
             patch('trinite.training.create_optimizer', side_effect=AssertionError('no optimizer allowed')):
            for selector, count in (('train', 480), ('validation', 240), ('test', 240)):
                record = synthetic_record(selector)
                self.assertEqual(check_score(record, selector, 'synthetic-not-a-checkpoint')['correct'], count)
                description = annotate(selector, record)
                self.assertEqual(sum(r['correct'] for r in description), count)
                self.assertTrue(all(r['components']['correct'] == r['components']['total'] for r in description))

    def test_complete_tokens_failures_and_context_enforced(self):
        record = synthetic_record('test')
        for mutation in (lambda r: r['predictions'].pop(),
                         lambda r: r['predictions'][0].update(correct=False),
                         lambda r: r['predictions'][0].update(generated_tokens=[EOS]),
                         lambda r: r['predictions'][0].update(generated_tokens=[True]),
                         lambda r: r.update(model_identity='foreign')):
            changed = deepcopy(record)
            mutation(changed)
            with self.assertRaises(ContractError):
                check_score(changed, 'test', 'synthetic-not-a-checkpoint')
        missing_eos = deepcopy(record)
        missing_eos['predictions'][0]['generated_tokens'] = missing_eos['predictions'][0]['expected_tokens'][:-1]
        missing_eos['predictions'][0]['correct'] = False
        rows, split = view('test')
        missing_eos['summary'] = check_prediction_examples(rows, split, GENERATION_CAP,
            json_bytes({'predictions': missing_eos['predictions']}))
        self.assertEqual(check_score(missing_eos, 'test', 'synthetic-not-a-checkpoint')['correct'], 239)
        retained = annotate('test', missing_eos)
        self.assertEqual(retained[0]['status'], 'missing_eos')
        self.assertEqual(len(retained), 240)


if __name__ == '__main__':
    unittest.main()
