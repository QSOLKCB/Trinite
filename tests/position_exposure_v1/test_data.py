"""Static offset/exposure controls; no backend, target model or generation."""
from collections import Counter
import copy
import random
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, json_bytes, parse_json
from position_study_v1 import data as original
from position_exposure_v1 import data
from position_exposure_v1.protocol import ARMS, SEEDS, OFFSETS, PROTOCOL, offset_for


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exposures = data.validate_matching()

    def test_exact_original_admission_bytes_labels_and_rows(self):
        self.assertEqual(data.dataset(), original.dataset())
        self.assertEqual(data.audit_bytes(*data.dataset()), original.audit_bytes(*original.dataset()))
        for arm in ARMS:
            self.assertEqual(data.schedule(arm), original.schedule('control'))
            self.assertEqual(data.examples(arm), original.examples('train'))
        for stratum in ('train', 'unseen', 'length3', 'length4'):
            self.assertEqual(data.examples(stratum), original.examples(stratum))

    def test_original_admission_still_rejects_changed_label(self):
        raw, manifest = data.dataset()
        changed = parse_json(raw, canonical=True)
        changed['examples'][0]['answer'] = 'not arithmetic'
        with self.assertRaises(ContractError):
            data.audit_bytes(json_bytes(changed), manifest)

    def test_twelve_cells_and_cycle_schedule_are_frozen(self):
        self.assertEqual(len(ARMS) * len(SEEDS), 12)
        self.assertEqual(OFFSETS, tuple(range(8)))
        for arm in ARMS:
            for cycle in range(200):
                wanted = cycle % 8 if arm.endswith('balanced') else 0
                self.assertEqual(offset_for(arm, cycle * 160), wanted)
                self.assertEqual(offset_for(arm, cycle * 160 + 159), wanted)
        for arm, cursor in [('foreign', 0), (ARMS[0], -1), (ARMS[0], True), (ARMS[0], 1.0)]:
            with self.assertRaises(ContractError):
                offset_for(arm, cursor)

    def test_each_unique_row_balanced_including_duplicate_diagonals(self):
        multiplicity = Counter(r['example_identity'] for r in original.schedule('control'))
        self.assertEqual(Counter(multiplicity.values()), {1: 148, 2: 6})
        for arm, record in self.exposures.items():
            offsets = range(8) if arm.endswith('balanced') else (0,)
            repeats = 25 if arm.endswith('balanced') else 200
            self.assertEqual(set(record['per_row_offset_visits']), set(multiplicity))
            for key, count in multiplicity.items():
                self.assertEqual(record['per_row_offset_visits'][key],
                                 {str(o): count * repeats for o in offsets})
            self.assertTrue(record['per_row_offset_balance_verified'])

    def test_actual_live_inputs_targets_and_visits_match_all_conditions(self):
        for record in self.exposures.values():
            for key, count in {'cycle_visits': 160, 'cycles': 200, 'scheduled_visits': 32000,
                              'scheduled_model_input_tokens': 307200,
                              'per_example_predictor_input_tokens': 300800,
                              'live_unscored_eos_input_tokens': 6400,
                              'scheduled_target_tokens': 121600}.items():
                self.assertEqual(record[key], count)
            self.assertEqual(sum(record['training_position_row_lookups'].values()), 307200)
            self.assertEqual(sum(x['lookups'] for x in record['training_address_detailed_role_counts']), 307200)
        self.assertEqual(len({r['cycle_identity'] for r in self.exposures.values()}), 1)

    def test_balanced_coarse_coverage_passes_but_fine_length_gaps_remain(self):
        for arm in ('sequence-balanced', 'coupled-balanced'):
            for stratum, report in self.exposures[arm]['evaluation'].items():
                self.assertTrue(report['coarse_roles_covered'])
                self.assertEqual(report['missing_address_coarse_roles'], [])
                if stratum in ('train', 'unseen'):
                    self.assertTrue(report['detailed_roles_covered'])
                else:
                    self.assertFalse(report['detailed_roles_covered'])
                    self.assertTrue(report['missing_address_detailed_roles'])
        expected = [{'address': 3, 'role': 'operand-significance-3'},
                    {'address': 4, 'role': 'answer-slot-3'},
                    {'address': 4, 'role': 'operand-significance-4'},
                    {'address': 5, 'role': 'answer-slot-4'}]
        self.assertEqual(self.exposures['coupled-balanced']['evaluation']['length4']
                         ['missing_address_detailed_roles'], expected)

    def test_fixed_controls_retain_original_lookup_counts_and_missing_roles(self):
        for arm, old in [('sequence-fixed', 'control'), ('coupled-fixed', 'candidate')]:
            record = self.exposures[arm]
            self.assertEqual(record['training_position_row_lookups'],
                             original.exposure(old)['training_position_row_lookups'])
            for stratum in ('length3', 'length4'):
                self.assertFalse(record['evaluation'][stratum]['coarse_roles_covered'])
                self.assertTrue(record['evaluation'][stratum]['missing_address_coarse_roles'])

    def test_evaluation_prefix_does_not_consume_terminal_eos(self):
        for arm, record in self.exposures.items():
            for report in record['evaluation'].values():
                self.assertNotIn('unscored-short-row-eos',
                                 {p['role'] for p in report['required_address_detailed_roles']})
            # Five prior answer digits may be consumed for the sixth/final call.
            final = record['evaluation']['length4']['required_address_detailed_roles']
            last = 15 if arm.startswith('sequence') else 5
            self.assertIn({'address': last, 'role': 'answer-slot-4'}, final)
            self.assertNotIn({'address': last + 1, 'role': 'unscored-short-row-eos'}, final)

    def test_actual_training_eos_is_counted_separately(self):
        for record in self.exposures.values():
            counts = record['training_address_detailed_role_counts']
            self.assertEqual(sum(x['lookups'] for x in counts
                                 if x['role'] == 'unscored-short-row-eos'), 6400)
            self.assertTrue(all(0 <= x['address'] < 64 for x in counts))

    def test_static_design_consumes_no_randomness(self):
        before = random.getstate()
        data.validate_matching()
        self.assertEqual(random.getstate(), before)

    def test_admission_rejects_lost_coarse_coverage_and_changed_budget(self):
        real = data.exposure
        def missing(arm):
            result = copy.deepcopy(real(arm))
            if arm == 'coupled-balanced':
                result['evaluation']['length4']['coarse_roles_covered'] = False
            return result
        with patch.object(data, 'exposure', side_effect=missing):
            with self.assertRaisesRegex(ContractError, 'coverage'):
                data.validate_matching()
        def mismatch(arm):
            result = copy.deepcopy(real(arm)); result['scheduled_target_tokens'] += 1
            return result
        with patch.object(data, 'exposure', side_effect=mismatch):
            with self.assertRaisesRegex(ContractError, 'exposure'):
                data.validate_matching()

    def test_probes_never_enter_training_schedule(self):
        probes = {r['example_identity'] for s in ('unseen', 'length3', 'length4')
                  for r in data.examples(s)}
        for arm in ARMS:
            self.assertFalse(probes & {r['example_identity'] for r in data.schedule(arm)})


if __name__ == '__main__':
    unittest.main()
