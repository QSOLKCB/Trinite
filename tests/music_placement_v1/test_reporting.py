"""Synthetic retained-token scoring only: no training, inference or target outputs."""
from copy import deepcopy
import unittest
from unittest import mock

from trinite.contracts import ContractError
from music_placement_v1 import spec, reporting


def records(stage, tokens=None):
    rows=spec.diagnostic_rows() if stage=='diagnostic' else spec.development_rows()
    return [{'arm':arm,'seed':seed,'step':640,'stage':stage,
             'example_identity':r['example_identity'],
             'generated_tokens':[*r['answer'].encode('ascii'),257] if tokens is None else list(tokens)}
            for arm,seed in spec.CELLS for r in rows]


class ReportingTests(unittest.TestCase):
    def test_oracle_scoring_is_offline_ceiling_not_capability(self):
        result=reporting.matrix_summary(records('diagnostic'),records('development'))
        self.assertEqual(result['decodes'],800)
        self.assertEqual(result['decode_forwards'],1600)
        for stage,n in [('diagnostic',80),('development',120)]:
            for cell in result[stage]['by_cell'].values():
                self.assertEqual(cell['raw']['exact_plus_eos'],n)
                self.assertEqual(cell['canonical_primary']['complete_groups'],10)
                self.assertEqual(cell['canonical_primary']['both_correct_edges'],40)
                self.assertEqual(cell['transformation_pairs']['swap' if stage=='diagnostic' else 'pitch']['both_correct'],40)
                self.assertEqual(cell['canonical_controls']['global']['exact_plus_eos'],20)
                self.assertEqual(cell['canonical_controls']['global']['complete_groups'],0)
        for flag in ('fresh_confirmation','arbitrary_pitch_established','arithmetic_transfer_established','learning_adequate','phase5_ready','release_ready'):
            self.assertIs(result[flag],False)

    def test_duplicate_inputs_separate_from_informative_swap_pairs(self):
        result=reporting.score_records(records('diagnostic'),'diagnostic')
        for cell in result['by_cell'].values():
            self.assertEqual(cell['unique_prompt_surfaces'],70)
            self.assertEqual(cell['rendering_rows'],80)
            strata=cell['transformation_pairs']['swap']['by_input_change']
            self.assertEqual(strata['changed_input'],{'pairs':30,'identical_raw_outputs':30,'both_correct':30})
            self.assertEqual(strata['identical_input'],{'pairs':10,'identical_raw_outputs':10,'both_correct':10})

    def test_bad_training_fit_does_not_suppress_development(self):
        diag=records('diagnostic')
        for r in diag:
            if r['arm']=='swap':r['generated_tokens']=[55,257]
        result=reporting.matrix_summary(diag,records('development'))
        for seed in ('17','29'):
            pair=result['paired_by_seed'][seed]
            self.assertTrue(pair['lost_control_perfect_training_fit'])
            self.assertEqual(pair['diagnostic']['swap_minus_control']['canonical']['exact_plus_eos'],-20)
            self.assertEqual(pair['development']['swap_minus_control']['canonical']['exact_plus_eos'],0)
        self.assertEqual(result['development']['decodes'],480)

    def test_invariance_of_wrong_constants_not_success(self):
        result=reporting.score_records(records('development',[55,257]),'development')
        for cell in result['by_cell'].values():
            self.assertEqual(cell['transformation_pairs']['pitch']['identical_raw_outputs'],40)
            self.assertEqual(cell['transformation_pairs']['pitch']['both_correct'],20)
            self.assertEqual(cell['canonical_primary']['complete_groups'],0)
            self.assertEqual(cell['canonical_primary']['both_correct_edges'],0)
            self.assertEqual(cell['canonical_primary']['changed_output_edges'],0)
            self.assertEqual(cell['canonical_strata']['variant_id']['10']['rows'],10)

    def test_eos_cap_preserved_and_accounted(self):
        result=reporting.score_records(records('diagnostic',[1]*64),'diagnostic')
        self.assertEqual(result['decode_forwards'],320*64)
        for cell in result['by_cell'].values():
            self.assertEqual(cell['raw']['cap_terminated'],80)
            self.assertEqual(cell['raw']['exact_plus_eos'],0)

    def test_missing_extra_order_seed_and_stage_reject(self):
        original=records('development')
        for value in (original[:-1],original+[original[-1]],[original[1],original[0],*original[2:]]):
            with self.assertRaises(ContractError):reporting.score_records(value,'development')
        for key,value in [('seed',True),('arm','best'),('step',639),('stage','diagnostic'),('example_identity','changed')]:
            changed=deepcopy(original);changed[0][key]=value
            with self.assertRaises(ContractError):reporting.score_records(changed,'development')
        for stage in ('test','train',True):
            with self.assertRaises(ContractError):reporting.score_records(original,stage)

    def test_unknown_fields_and_raw_token_repair_reject(self):
        original=records('diagnostic')
        for tokens in ([],[True,257],[55],[257,55],[55,257,257],[259,257],[55]*65):
            changed=deepcopy(original);changed[0]['generated_tokens']=tokens
            with self.assertRaises(ContractError):reporting.score_records(changed,'diagnostic')
        changed=deepcopy(original);changed[0]['correct']=True
        with self.assertRaises(ContractError):reporting.score_records(changed,'diagnostic')


if __name__=='__main__':unittest.main()
