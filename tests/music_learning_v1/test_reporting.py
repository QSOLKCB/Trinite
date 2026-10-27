"""Standard-library synthetic prediction controls, never model inference."""
from copy import deepcopy
import unittest

from trinite.contracts import ContractError
from trinite.tokenizer import EOS
from music_learning_v1.numerical import examples
from music_learning_v1.reporting import annotate, describe, trivial_baselines


def fake_records():
    rows = examples()
    baseline = trivial_baselines(rows)
    records = {}
    for split in ('train', 'validation', 'test'):
        predictions = []
        for item in rows:
            if item['split'] != split:
                continue
            gold = [*item['answer'].encode(), EOS]
            constant = [*baseline[item['task'], None].encode(), EOS]
            predictions.append({'example_identity': item['example_identity'],
                'family_id': item['family_id'], 'task': item['task'],
                'split': 'train' if split == 'train' else 'test',
                'expected_tokens': gold, 'generated_tokens': gold,
                'correct': True, 'baseline_tokens': constant,
                'baseline_correct': constant == gold})
        records[split] = {'selector': split, 'model_identity': 'synthetic-test-not-a-model',
                          'summary': {}, 'predictions': predictions}
    return records


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = fake_records()

    def test_complete_split_and_task_metrics(self):
        value = describe(self.records)
        for key, n in [('train_fit', 480), ('held_out_validation', 240), ('held_out_test', 240)]:
            self.assertEqual(value[key]['total']['whole_answer_eos_accuracy'], [n, n])
            self.assertEqual(len(value[key]['by_task']), 6)
            self.assertEqual(len(value[key]['by_meter']), 5)
            self.assertEqual(set(value[key]['by_skill_group']), {'signature_group_readout', 'event_structure'})
        self.assertEqual(len(value['train_fit']['by_family_id']), 4)
        self.assertEqual(len(value['held_out_test']['by_family_id']), 2)
        self.assertFalse(value['additional_inference_executed'])
        self.assertFalse(value['learning_adequate'])

    def test_train_only_baseline_ignores_heldout_targets(self):
        rows = examples()
        expected = trivial_baselines(rows)
        for row in rows:
            if row['split'] != 'train':
                row['answer'] = '999'
        self.assertEqual(trivial_baselines(rows), expected)

    def test_wrong_missing_eos_and_special_tokens_retained(self):
        value = deepcopy(self.records['test'])
        for pred in value['predictions'][:3]:
            pred['correct'] = False
        value['predictions'][0]['generated_tokens'] = value['predictions'][0]['expected_tokens'][:-1]
        value['predictions'][1]['generated_tokens'] = [EOS]
        value['predictions'][2]['generated_tokens'] = [256, EOS]
        result = annotate('test', value)
        self.assertEqual(result[0]['status'], 'missing_eos')
        self.assertEqual(result[0]['components']['correct'], result[0]['components']['total'])
        self.assertEqual(result[1]['components']['correct'], 0)
        self.assertEqual(result[2]['components']['correct'], 0)
        self.assertEqual(result[2]['prediction']['generated_tokens'], [256, EOS])
        self.assertEqual(len(result), 240)

    def test_tampering_missing_duplicate_or_foreign_predictions_rejected(self):
        for mutation in (lambda x: x['predictions'].pop(),
            lambda x: x['predictions'].__setitem__(1, deepcopy(x['predictions'][0])),
            lambda x: x['predictions'][0].update(correct=False),
            lambda x: x['predictions'][0].update(family_id='foreign'),
            lambda x: x['predictions'][0].update(generated_tokens=[True]),
            lambda x: x['predictions'][0].update(generated_tokens=[0] * 65),
            lambda x: x['predictions'][0].update(split='validation')):
            value = deepcopy(self.records['test'])
            mutation(value)
            with self.assertRaises(ContractError):
                annotate('test', value)
        value = deepcopy(self.records)
        value['test']['model_identity'] = 'different'
        with self.assertRaises(ContractError):
            describe(value)


if __name__ == '__main__':
    unittest.main()
