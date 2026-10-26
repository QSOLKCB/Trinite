"""Synthetic retained-score accounting, never a target learning result."""
from copy import deepcopy
from pathlib import Path
import unittest

from trinite.contracts import ContractError, identity, parse_json
from trinite.tokenizer import EOS
from balanced_addition_v1 import data, reporting, request
from balanced_addition_v1.protocol import STRATA
from failure_map_v1.request import sources as historical_sources


def record(selector, mode='correct'):
    predictions = []
    for row in data.examples(selector):
        expected = [*row['answer'].encode('ascii'), EOS]
        generated = expected if mode == 'correct' else [57, 57, EOS]
        predictions.append({'example_identity':row['example_identity'],
            'family_id':row['family_id'], 'expected_tokens':expected,
            'generated_tokens':generated, 'correct':generated == expected})
    return {'predictions': predictions}


def report(arm='candidate', mode='correct'):
    return reporting.describe(arm, record(arm, mode), {s:record(s, mode) for s in STRATA})


class ReportingTests(unittest.TestCase):
    def test_frozen_historical_206_files_preserved_and_extended(self):
        frozen = parse_json((request.ROOT/'fixtures/balanced-addition-v1/historical-sources.json').read_bytes())
        self.assertEqual(len(frozen),206)
        self.assertEqual(frozen,historical_sources())
        current = request.sources()
        self.assertEqual(len(current),216)
        self.assertEqual({p:current[p] for p in frozen},frozen)
        for p in ('data.py','numerical.py','reporting.py','workflow.py','controller.py','request.py','protocol.py','__init__.py'):
            self.assertIn('scripts/balanced_addition_v1/'+p,current)
        self.assertIn('docs/BALANCED-ADDITION.md',current)

    def test_perfect_synthetic_counts_keep_train_dev_separate(self):
        result=report()
        self.assertEqual(result['unique_training']['total']['rows'],144)
        self.assertEqual(result['original_control_seen_anchors']['total']['rows'],20)
        primary=result['primary_in_range_development']
        self.assertEqual(primary['total']['rows'],56)
        self.assertEqual(primary['total']['families'],15)
        self.assertEqual(primary['total']['correct'],56)
        self.assertEqual(primary['total']['all_renderings_correct_families'],15)
        self.assertFalse(result['additional_inference_executed'])

    def test_same_shared_facts_both_arms_and_arm_specific_support(self):
        left,right=report('control'),report('candidate')
        for s in STRATA:
            self.assertEqual([r['example_identity'] for r in left['rows'][s]],
                             [r['example_identity'] for r in right['rows'][s]])
            self.assertFalse(any(r['seen_in_this_arm'] for r in right['rows'][s]))
        self.assertEqual(left['shared_development']['broadened-answer']['by_answer_support']['unseen']['rows'],36)
        self.assertEqual(right['shared_development']['broadened-answer']['by_answer_support']['seen']['rows'],36)
        self.assertEqual(left['shared_development']['range-length']['by_answer_support']['seen']['rows'],20)
        self.assertEqual(right['shared_development']['range-length']['by_answer_support']['seen']['rows'],96)

    def test_exact_pair_counts_exclude_diagonal_order_self_pairs(self):
        p=report()['primary_in_range_development']['metamorphic']
        self.assertEqual(p['carrier']['pair_count'],28)
        self.assertEqual(p['operand_order']['pair_count'],26)
        self.assertEqual(p['carrier']['both_correct'],28)
        r=report()['shared_development']['range-length']['metamorphic']
        self.assertEqual(r['carrier']['pair_count'],69)
        self.assertEqual(r['operand_order']['pair_count'],66)

    def test_consistent_wrong_never_promoted_to_correct(self):
        result=report(mode='wrong')
        p=result['primary_in_range_development']
        self.assertEqual(p['total']['correct'],0)
        self.assertEqual(p['total']['categories']['wrong_value'],56)
        self.assertEqual(p['metamorphic']['carrier']['consistent'],28)
        self.assertEqual(p['metamorphic']['carrier']['both_correct'],0)
        self.assertEqual(p['metamorphic']['carrier']['categories'],{'both_wrong_same':28})

    def test_empty_answer_groups_are_untested(self):
        group=report()['primary_in_range_development']['by_answer']['24']
        self.assertEqual(group['rows'],0)
        self.assertEqual(group['status'],'untested_no_admitted_cases')

    def test_missing_duplicate_and_wrong_strata_rejected(self):
        train=record('candidate');dev={s:record(s) for s in STRATA}
        with self.assertRaises(ContractError):reporting.describe('candidate',train,{})
        for change in ('missing','duplicate'):
            altered=deepcopy(dev)
            if change=='missing':altered['shared-answer']['predictions'].pop()
            else:altered['shared-answer']['predictions'].append(deepcopy(altered['shared-answer']['predictions'][0]))
            with self.assertRaises(ContractError):reporting.describe('candidate',train,altered)

    def test_impossible_greedy_output_rejected_by_reused_classifier(self):
        train=record('candidate');dev={s:record(s) for s in STRATA}
        dev['shared-answer']['predictions'][0]['generated_tokens']=[EOS,49]
        with self.assertRaises(ContractError):reporting.describe('candidate',train,dev)

    def test_scheduled_digit_histograms_match_independent_frozen_counts(self):
        cycle=data.exposure('candidate')['cycle']
        first=[4,68,8,8,12,8,8,12,16,16]
        second=[16,12,8,8,4,4,4,4,4]
        self.assertEqual(cycle['answer_digit_histograms_by_position'],
            {'0':{str(i):n for i,n in enumerate(first)},'1':{str(i):n for i,n in enumerate(second)}})
        self.assertEqual(set(cycle['family_visit_histogram'].values()),{4})
        self.assertEqual(set(data.exposure('control')['cycle']['family_visit_histogram'].values()),{32})
        self.assertEqual(parse_json(data.dataset()[1])['range_support'],{
            'both':{'rows':20,'families':5},'candidate-only':{'rows':76,'families':19},
            'neither':{'rows':42,'families':12}})


if __name__=='__main__':unittest.main()
