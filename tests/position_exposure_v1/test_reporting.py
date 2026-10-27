"""Synthetic family counts exercise contrasts without constructing any model."""
from copy import deepcopy
import unittest

from position_study_v1.protocol import PROTOCOL as ORIGINAL
from position_exposure_v1 import reporting
from position_exposure_v1.protocol import ARMS, CONTRASTS, SEEDS, PROTOCOL


def description(seen=64, unseen=0, length3=0, length4=0, rows=0):
    def total(n, correct):
        return {'families': n, 'all_renderings_correct_families': correct,
                'correct': rows, 'rows': 1000}
    return {'seen_training_anchors': {'total': total(64, seen)},
            'development': {s: {'total': total(n, c)} for s, n, c in
                            [('unseen', 24, unseen), ('length3', 12, length3), ('length4', 12, length4)]}}


def passing():
    # Every declared contrast gains three families. These are artificial counts,
    # not observations from the target model or an anticipated outcome.
    gains = {'sequence-fixed': 0, 'coupled-fixed': 3,
             'sequence-balanced': 3, 'coupled-balanced': 6}
    return {f'{arm}-{seed}': description(unseen=gains[arm], length3=gains[arm], length4=gains[arm])
            for arm in ARMS for seed in SEEDS}


class ReportingTests(unittest.TestCase):
    def test_original_margins_preserved_and_four_prespecified_directions(self):
        self.assertEqual(PROTOCOL['criteria'], ORIGINAL['criteria'])
        result = reporting.criteria(passing())
        self.assertEqual(set(result['contrasts']), set(CONTRASTS))
        self.assertNotIn('all_descriptive_criteria_met', result)
        for name, record in result['contrasts'].items():
            self.assertEqual((record['left'], record['right']), CONTRASTS[name])
            self.assertTrue(record['all_descriptive_criteria_met'])
            self.assertEqual(set(record['per_seed']), {'0', '1', '2'})
            for counts in record['family_counts'].values():
                self.assertTrue(all(x['difference'] == 3 for x in counts.values()))

    def test_one_bad_seed_is_not_hidden_by_other_seed_gains(self):
        records = passing()
        records['coupled-balanced-2']['development']['unseen']['total']['all_renderings_correct_families'] = 3
        result = reporting.criteria(records)['contrasts']
        for name in ('coupling-balanced', 'offset-coupled'):
            self.assertFalse(result[name]['all_descriptive_criteria_met'])
            self.assertFalse(result[name]['per_seed']['2']['unseen_gain'])
        self.assertTrue(result['coupling-fixed']['all_descriptive_criteria_met'])
        self.assertTrue(result['offset-sequence']['all_descriptive_criteria_met'])

    def test_missing_cell_cannot_pass_its_contrasts(self):
        records = passing(); del records['coupled-balanced-2']
        result = reporting.criteria(records)['contrasts']
        for name in ('coupling-balanced', 'offset-coupled'):
            self.assertFalse(result[name]['all_descriptive_criteria_met'])
            self.assertNotIn('2', result[name]['per_seed'])
        self.assertTrue(result['coupling-fixed']['all_descriptive_criteria_met'])
        self.assertEqual(reporting.criteria({})['contrasts']['coupling-fixed']['per_seed'], {})

    def test_family_metric_not_row_count_drives_decision(self):
        records = {f'{arm}-{seed}': description(rows=1000 if arm.startswith('coupled') else 0)
                   for arm in ARMS for seed in SEEDS}
        self.assertTrue(all(not x['all_descriptive_criteria_met']
                            for x in reporting.criteria(records)['contrasts'].values()))

    def test_unseen_integer_margin_is_three_not_two_families(self):
        records = passing()
        for seed in SEEDS:
            records[f'coupled-fixed-{seed}']['development']['unseen']['total']['all_renderings_correct_families'] = 2
        result = reporting.criteria(records)['contrasts']['coupling-fixed']
        self.assertFalse(result['all_descriptive_criteria_met'])
        self.assertTrue(all(not x['unseen_gain'] for x in result['per_seed'].values()))

    def test_length_floor_and_gain_are_separate(self):
        records = passing()
        for seed in SEEDS:
            records[f'coupled-fixed-{seed}']['development']['length3']['total']['all_renderings_correct_families'] = 2
        result = reporting.criteria(records)['contrasts']['coupling-fixed']
        self.assertTrue(all(x['length3_gain'] for x in result['per_seed'].values()))
        self.assertTrue(all(not x['length3_floor'] for x in result['per_seed'].values()))
        records = passing()
        for seed in SEEDS:
            records[f'sequence-fixed-{seed}']['development']['length4']['total']['all_renderings_correct_families'] = 2
        result = reporting.criteria(records)['contrasts']['coupling-fixed']
        self.assertTrue(all(x['length4_floor'] for x in result['per_seed'].values()))
        self.assertTrue(all(not x['length4_gain'] for x in result['per_seed'].values()))

    def test_seen_floor_and_regression_are_separate(self):
        records = passing()
        for seed in SEEDS:
            records[f'sequence-fixed-{seed}']['seen_training_anchors']['total']['all_renderings_correct_families'] = 62
            records[f'coupled-fixed-{seed}']['seen_training_anchors']['total']['all_renderings_correct_families'] = 61
        result = reporting.criteria(records)['contrasts']['coupling-fixed']
        self.assertTrue(result['all_descriptive_criteria_met'])
        records['coupled-fixed-0']['seen_training_anchors']['total']['all_renderings_correct_families'] = 60
        result = reporting.criteria(records)['contrasts']['coupling-fixed']
        self.assertFalse(result['per_seed']['0']['seen_candidate_floor'])
        self.assertFalse(result['per_seed']['0']['seen_regression_within_allowance'])
        records = passing()
        records['coupled-fixed-0']['seen_training_anchors']['total']['all_renderings_correct_families'] = 62
        result = reporting.criteria(records)['contrasts']['coupling-fixed']
        self.assertTrue(result['per_seed']['0']['seen_candidate_floor'])
        self.assertFalse(result['per_seed']['0']['seen_regression_within_allowance'])

    def test_no_mutation_and_no_omnibus_success_claim(self):
        records = passing(); before = deepcopy(records)
        result = reporting.criteria(records)
        self.assertEqual(records, before)
        self.assertIn('no omnibus gate', result['interpretation'])
        self.assertIn('no', result['interpretation'])

    def test_design_summary_preserves_role_coverage_limit(self):
        result = reporting.design_summary()
        self.assertEqual(set(result['address_role_exposure']), set(ARMS))
        self.assertIn('not gradient support', result['limitations'])
        self.assertIn('mandatory separate limitations', result['coverage_scope'])


if __name__ == '__main__':
    unittest.main()
