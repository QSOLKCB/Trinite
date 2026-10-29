"""Synthetic scoring conformance only; no checkpoints or target inference."""
from copy import deepcopy
import unittest

from trinite.contracts import ContractError
from music_contrast_v1.spec import rows
from frozen_music_eval_v1.protocol import (CAP, CONTEXT, PROTOCOL, SEEDS, STEP,
    data_contract, input_rows, score_records)


def oracle_records():
    return [{'seed': seed, 'step': STEP, 'stage': stage,
             'example_identity': row['example_identity'],
             'generated_tokens': [*row['answer'].encode('ascii'), 257]}
            for stage, selector in (('train_replay', 'train'), ('heldout', 'heldout'))
            for seed in SEEDS for row in rows(selector, selector == 'heldout')]


class ProtocolTests(unittest.TestCase):
    def test_regenerated_family_contract_and_input_isolation(self):
        data = data_contract()
        self.assertEqual(data['heldout_structural_parent_count'], 2)
        self.assertEqual(data['heldout_canonical_quartets'], 10)
        self.assertEqual(data['heldout_correlated_sibling_rows'], 80)
        self.assertEqual(data['exact_prompt_overlap'], 0)
        inputs = input_rows()
        self.assertEqual(len(inputs), 160)
        self.assertEqual(sum(r['split'] == 'train' for r in inputs), 40)
        for row in inputs:
            self.assertEqual(set(row), {'example_identity', 'prompt', 'split'})
            self.assertLessEqual(1 + len(row['prompt'].encode('ascii')) + CAP, CONTEXT)
        self.assertEqual(PROTOCOL['max_decode_forwards'], 320 * CAP)
        self.assertEqual(PROTOCOL['training_updates'], 0)

    def test_synthetic_oracle_and_correlated_denominators(self):
        result = score_records(oracle_records())
        self.assertEqual(result['forward_calls'], 640)
        self.assertEqual(result['heldout_structural_parents'], 2)
        for seed in SEEDS:
            heldout = result['by_seed'][str(seed)]['heldout']
            self.assertEqual(heldout['canonical_primary']['complete_groups'], 10)
            self.assertEqual(heldout['canonical_primary']['both_correct_edges'], 40)
            self.assertEqual(heldout['correlated_siblings']['groups'], 20)
            self.assertEqual(heldout['by_parent']['c']['rows'], 60)
            self.assertEqual(heldout['by_task']['attacks']['rows'], 120)
            self.assertEqual(heldout['by_variant']['11']['rows'], 30)
            self.assertEqual(heldout['invariance']['all_three_correct'], 40)
            self.assertEqual(heldout['transformation_pairs']['pitch']['both_correct'], 40)
            self.assertEqual(heldout['canonical_controls']['global']['exact_plus_eos'], 20)
            self.assertEqual(heldout['canonical_controls']['exact_formal_oracle']['exact_plus_eos'], 40)

    def test_constant_heldout_is_not_correct_sensitivity(self):
        records = oracle_records()
        for record in records[80:]:
            record['generated_tokens'] = [55, 257]
        result = score_records(records)['by_seed']['17']['heldout']
        self.assertEqual(result['canonical_primary']['exact_plus_eos'], 20)
        self.assertEqual(result['canonical_primary']['complete_groups'], 0)
        self.assertEqual(result['canonical_primary']['both_correct_edges'], 0)
        self.assertEqual(result['canonical_primary']['changed_output_edges'], 0)
        self.assertEqual(result['invariance']['identical_raw_outputs'], 40)
        self.assertEqual(result['invariance']['all_three_correct'], 20)

    def test_reject_missing_extra_reordering_and_checkpoint_selection(self):
        valid = oracle_records()
        for records in (valid[:-1], valid + [valid[-1]], [valid[1], valid[0]] + valid[2:]):
            with self.assertRaises(ContractError): score_records(records)
        for field, value in (('seed', 29), ('seed', True), ('step', 64), ('stage', 'heldout'), ('extra', 1)):
            records = deepcopy(valid); records[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ContractError):
                score_records(records)

    def test_malformed_eos_and_cap_preserved(self):
        valid = oracle_records()
        for tokens in ([], [55], [257, 55], [True, 257], [-1, 257], [259, 257], [55] * 65):
            records = deepcopy(valid); records[80]['generated_tokens'] = tokens
            with self.subTest(tokens=tokens), self.assertRaises(ContractError): score_records(records)
        records = deepcopy(valid); records[80]['generated_tokens'] = [55] * CAP
        summary = score_records(records)['by_seed']['17']['heldout']
        self.assertEqual(summary['raw']['cap_terminated'], 1)
        self.assertEqual(summary['raw']['exact_plus_eos'], 119)

    def test_training_gate_cannot_be_repaired_by_heldout(self):
        records = oracle_records(); records[0]['generated_tokens'] = [55, 257]
        with self.assertRaises(ContractError): score_records(records)


if __name__ == '__main__': unittest.main()
