"""Finite contract/scoring conformance, never target-model observations."""
import copy
from collections import Counter
import unittest
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from music_contrast_v1.spec import rows as original_rows
from music_learning_v1.numerical import plan_for as original_plan
from music_exposure_v1 import spec, reporting


def predictions(answer=None):
    result = []
    for row in spec.rows():
        gold = [*row['answer'].encode(), 257]
        tokens = gold if answer is None else [*answer.encode(), 257]
        result.append({'example_identity': row['example_identity'], 'family_id': row['family_id'],
            'task': 'attacks', 'split': 'train', 'expected_tokens': gold,
            'generated_tokens': tokens, 'correct': tokens == gold,
            'baseline_tokens': [55, 257], 'baseline_correct': gold == [55, 257]})
    return {'predictions': result}


class ExposureSpecTests(unittest.TestCase):
    def test_exact_original_train_rows_and_order(self):
        self.assertEqual(json_bytes(spec.rows()), json_bytes(original_rows('train')))
        self.assertEqual(len(spec.rows()), 40)
        self.assertEqual({r['split'] for r in spec.rows()}, {'train'})
        with self.assertRaises(TypeError):
            spec.rows('heldout')

    def test_new_admission_is_deterministic_and_train_only(self):
        raw, receipt = spec.dataset()
        self.assertEqual((raw, receipt), spec.dataset())
        self.assertEqual((raw, receipt), spec.require_admission(identity(receipt)))
        self.assertEqual(parse_json(raw, canonical=True)['examples'], spec.rows())
        with self.assertRaises(ContractError):
            spec.require_admission(identity(raw))

    def test_full_exposure_exact_and_balanced_over_rows(self):
        schedule = spec.schedule_receipt()
        self.assertEqual(schedule['scheduled_visits'], 5120)
        self.assertEqual(schedule['target_tokens'], 10240)
        self.assertEqual(Counter(schedule['visits_per_example'].values()), Counter({128: 40}))
        self.assertEqual(schedule['answer_histogram'], {'8': 1280, '7': 2560, '6': 1280})
        self.assertEqual(schedule['live_input_tokens'], 128*sum(len(r['prompt'])+2 for r in spec.rows()))
        rows = spec.rows()
        visits = [rows[i % 40]['example_identity'] for i in range(5120)]
        self.assertEqual(schedule['schedule_identity'], identity(json_bytes(visits)))

    def test_short_reference_preserves_original_partial_cycle(self):
        schedule = spec.schedule_receipt(64)
        self.assertEqual(schedule['scheduled_visits'], 512)
        self.assertEqual(schedule['target_tokens'], 1024)
        self.assertEqual(schedule['live_input_tokens'], 34812)
        self.assertEqual(Counter(schedule['visits_per_example'].values()), Counter({13: 32, 12: 8}))
        self.assertEqual(list(schedule['visits_per_example'].values()), [13]*32+[12]*8)

    def test_zero_and_unfrozen_steps(self):
        schedule = spec.schedule_receipt(0)
        self.assertEqual(schedule['target_tokens'], 0)
        self.assertEqual(schedule['scheduled_visits'], 0)
        self.assertEqual(set(schedule['visits_per_example'].values()), {0})
        for value in (True, -1, 63, 65, 641, '640'):
            with self.assertRaises(ContractError):
                spec.schedule_receipt(value)

    def test_only_step_count_changes_in_native_plan(self):
        for seed in spec.SEEDS:
            old = original_plan(seed).to_dict()
            new = spec.plan_for(seed).to_dict()
            self.assertEqual({k for k in old if old[k] != new[k]}, {'steps'})
            self.assertEqual(new['steps'], 640)
            self.assertEqual(new['max_seconds'], 120)
            self.assertEqual(new['validation_every'], 64)
        for seed in (True, 0, 18, '17'):
            with self.assertRaises(ContractError):
                spec.plan_for(seed)

    def test_protocol_fixed_budget_and_blocked_claims(self):
        self.assertEqual(spec.CHECKPOINTS, (0, 64, 640))
        self.assertEqual(spec.SEEDS, (17, 29))
        self.assertEqual(spec.PROTOCOL['max_decodes'], 3*40*2)
        self.assertEqual(spec.PROTOCOL['max_decode_forwards'], 3*40*2*64)
        self.assertEqual(spec.PROTOCOL['heldout_decodes'], 0)
        self.assertEqual(spec.PROTOCOL['aggregate_update_seconds'], 240)
        self.assertEqual(spec.PROTOCOL['aggregate_worker_seconds'], 1800)
        for key in ('arithmetic_bridge', 'fresh_confirmation', 'learning_adequate', 'phase5_ready', 'release_ready'):
            self.assertIs(spec.PROTOCOL[key], False)


class ReportingTests(unittest.TestCase):
    def test_synthetic_exact_oracle_is_scoring_control_only(self):
        result = reporting.summarize(json_bytes(predictions()))
        self.assertEqual(result['all']['exact_plus_eos'], 40)
        self.assertEqual(result['all']['complete_groups'], 10)
        self.assertEqual(result['all']['both_correct_edges'], 40)
        self.assertEqual(result['unique_raw_outputs'], 3)
        self.assertEqual(result['by_answer']['6'], {'rows': 10, 'exact_plus_eos': 10})
        self.assertEqual(result['by_parent']['a'], {'rows': 20, 'exact_plus_eos': 20})

    def test_constant_seven_is_not_contrast_fit(self):
        result = reporting.summarize(json_bytes(predictions('7')))
        self.assertEqual(result['all']['exact_plus_eos'], 20)
        self.assertEqual(result['all']['complete_groups'], 0)
        self.assertEqual(result['all']['both_correct_edges'], 0)
        self.assertEqual(result['all']['changed_output_edges'], 0)
        self.assertEqual(result['unique_raw_outputs'], 1)
        self.assertEqual(result['raw_output_histogram'], [{'tokens': [55, 257], 'rows': 40}])
        self.assertEqual(result['by_answer']['8']['exact_plus_eos'], 0)
        for key in ('global', 'task_meter', 'header', 'length', 'header_length'):
            self.assertEqual(result['baselines'][key]['exact_plus_eos'], 20)
            self.assertEqual(result['baselines'][key]['complete_groups'], 0)

    def test_missing_duplicate_or_forged_predictions_rejected(self):
        cases = []
        value = predictions(); value['predictions'].pop(); cases.append(value)
        value = predictions(); value['predictions'][1] = copy.deepcopy(value['predictions'][0]); cases.append(value)
        value = predictions(); value['predictions'][0]['correct'] = False; cases.append(value)
        value = predictions(); value['predictions'][0]['split'] = 'test'; cases.append(value)
        value = predictions(); value['predictions'][0]['generated_tokens'] = [257, 257]; cases.append(value)
        for value in cases:
            with self.assertRaises(ContractError):
                reporting.summarize(json_bytes(value))


if __name__ == '__main__':
    unittest.main()
