"""Synthetic score aggregation only: no target calls, inference or optimizer."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from queue_learning_v1 import evaluation
from queue_learning_v2.reporting import normalize


class ReportingTests(unittest.TestCase):
    def check_case(self, numeric, other):
        entries = [{'method': method, 'id': f'{split}-{i}', 'split': split}
                   for method in evaluation.METHODS for split in ('validation', 'test') for i in range(64)]
        def read(path):
            name = path.name if path.name.endswith('-raw.json') else path.parent.name
            index = int(name.removesuffix('-raw.json').rsplit('-', 1)[1])
            if path.name.endswith('-raw.json'):
                return {'status': 'prediction' if index < numeric else other}
            return ({'status': 'prediction', 'absolute_error': 2, 'exact_match': False}
                    if index < numeric else {'status': 'abstain'})
        with patch.object(evaluation, 'read_json', side_effect=read), \
             patch.object(evaluation.gate, 'run', side_effect=AssertionError('no target execution')):
            raw = evaluation.summarize(entries, Path('/synthetic-commits'), Path('/synthetic-scores'))
            original = deepcopy(raw)
            result = normalize(raw)
        self.assertEqual(raw, original)
        self.assertEqual(raw['schema'], 'trinite.queue-learning-summary.v1')
        self.assertEqual(result['schema'], 'trinite.queue-learning-summary.v2')
        for method in evaluation.METHODS:
            for split in ('validation', 'test'):
                row = result['methods'][method][split]
                self.assertEqual(row['planned'], 64)
                self.assertEqual(row['numeric'], numeric)
                self.assertEqual(row['coverage'], {'numerator': numeric, 'denominator': 64})
                self.assertEqual(row['exact_match'], {'numerator': 0, 'denominator': 64})
                if numeric:
                    self.assertEqual(row['numeric_exact_match'], {'numerator': 0, 'denominator': numeric})
                    self.assertEqual(row['conditional_numeric_mae'], {'numerator': 2 * numeric, 'denominator': numeric})
                else:
                    self.assertEqual(raw['methods'][method][split]['numeric_exact_match'],
                                     {'numerator': 0, 'denominator': 0})
                    self.assertIsNone(row['numeric_exact_match'])
                    self.assertIsNone(row['conditional_numeric_mae'])
                if numeric == 64:
                    self.assertEqual(row['mae'], {'numerator': 128, 'denominator': 64})
                else:
                    self.assertIsNone(row['mae'])

    def test_zero_numeric_failures(self):
        self.check_case(0, 'failure')

    def test_all_malformed(self):
        self.check_case(0, 'malformed')

    def test_all_abstain(self):
        self.check_case(0, 'abstain')

    def test_partial_numeric_coverage(self):
        self.check_case(16, 'malformed')

    def test_full_numeric_coverage(self):
        self.check_case(64, 'abstain')

    def test_recursive_zero_denominators_and_no_mutation(self):
        raw = {'schema': 'trinite.queue-learning-summary.v1', 'nested': [
            {'zero': {'numerator': 0, 'denominator': 0}},
            {'negative_difference': {'numerator': -3, 'denominator': 64}}]}
        before = deepcopy(raw)
        result = normalize(raw)
        self.assertIsNone(result['nested'][0]['zero'])
        self.assertEqual(result['nested'][1], raw['nested'][1])
        self.assertEqual(raw, before)

    def test_wrong_schema_or_invalid_rational_rejected(self):
        with self.assertRaises(ValueError):
            normalize({'schema': 'trinite.queue-learning-summary.v2'})
        for denominator in (-1, True, '0'):
            with self.assertRaises(ValueError):
                normalize({'schema': 'trinite.queue-learning-summary.v1',
                           'ratio': {'numerator': 0, 'denominator': denominator}})


if __name__ == '__main__':
    unittest.main()
