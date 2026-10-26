"""Real native numerical diagnostics plus deliberately wrong control models."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from trinite.contracts import ContractError, ModelConfig, json_bytes, parse_json
from trinite.foundations_data import dataset
from trinite.model import Decoder
from trinite.tokenizer import EOS
from trinite.training import model_identity, target_loss
from trinite.tokenizer import ByteTokenizer
from native_diagnostic_v1.measurements import measure, validate_report, sensitivity, _state


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
ALL = parse_json(dataset('arithmetic')[0], canonical=True)['examples']
TRAIN = [row for row in ALL if row['split'] == 'train']


def pair():
    left = TRAIN[0]
    right = next(row for row in TRAIN if row['task']==left['task'] and row['answer']!=left['answer'])
    return [left,right]


def constant_forward(token=EOS, tied=False):
    def forward(ids, mask, **kwargs):
        logits = torch.zeros((*ids.shape,259), dtype=torch.float32)
        if not tied:
            logits[:,:,token] = 8
        return SimpleNamespace(logits=logits)
    return forward


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.model = Decoder(ModelConfig(context_length=32,blocks=1,width=16,heads=2,
                                        feed_forward_width=32),lane='dense',seed=0)

    def test_real_decoder_roundtrip_and_no_mutation(self):
        # Existing gradients and mode are retained, not cleared by the observer.
        next(self.model.parameters()).grad = torch.ones_like(next(self.model.parameters()))
        before = json_bytes(_state(self.model))
        result = measure(self.model,pair(),4)
        self.assertEqual(before,json_bytes(_state(self.model)))
        self.assertEqual(result['summary'],validate_report(result,pair(),4))
        self.assertEqual(result['summary']['forward_calls'],4+sum(r['output_tokens'] for r in result['rows']))
        self.assertEqual(result['summary']['training_lookup_correct'],2)
        self.assertEqual(result['summary']['exact_oracle_correct'],2)
        self.assertEqual(result['model_identity'],model_identity(self.model))

    def test_target_positions_are_answer_and_eos_shifted_once(self):
        selected = [next(row for row in TRAIN if len(row['answer'])==1),
                    next(row for row in TRAIN if len(row['answer'])>=3)]
        result = measure(self.model,selected,4)
        for item,row in zip(selected,result['rows']):
            scored = [r for r in result['targets'] if r['example_identity']==item['example_identity']]
            start = 1+len(item['prompt'].encode('ascii'))
            self.assertEqual([r['target_position'] for r in scored],list(range(start,start+len(item['answer'])+1)))
            self.assertEqual([r['gold_token'] for r in scored],[*item['answer'].encode(),EOS])
            self.assertEqual(sum(r['is_eos'] for r in scored),1)
            enc = ByteTokenizer(32).encode(item['text'],answer_start=len(item['prompt']))
            loss,count = target_loss(self.model,[{'ids':enc.input_ids,'mask':enc.attention_mask,'loss':enc.loss_mask}])
            self.assertEqual(row['authoritative_loss_hex'],float(loss.detach()).hex())
            self.assertEqual(row['scored_targets'],count)
        expected = math.fsum(float.fromhex(r['authoritative_loss_hex'])*r['scored_targets'] for r in result['rows'])/len(result['targets'])
        self.assertEqual(result['summary']['target_mean_loss_hex'],expected.hex())

    def test_constant_output_control_detected_not_called_reasoning(self):
        with patch.object(self.model,'forward',side_effect=constant_forward()):
            result = measure(self.model,pair(),4)
        self.assertTrue(result['input_sensitivity']['all_conflicting_pairs_identical'])
        self.assertEqual(result['summary']['raw_premature_eos'],2)
        self.assertEqual(result['summary']['first_answer_eos_predictions'],2)
        self.assertEqual(result['summary']['raw_answer_and_eos_correct'],0)
        validate_report(result,pair(),4)

    def test_lowest_token_id_tie_rank_and_probability(self):
        with patch.object(self.model,'forward',side_effect=constant_forward(tied=True)):
            result = measure(self.model,[TRAIN[0]],2)
        for target in result['targets']:
            self.assertEqual(target['argmax_token'],0)
            self.assertEqual(target['gold_rank'],target['gold_token']+1)
            self.assertAlmostEqual(float.fromhex(target['gold_probability_hex']),1/259,places=8)
            self.assertAlmostEqual(float.fromhex(target['loss_hex']),math.log(259),places=6)
        validate_report(result,[TRAIN[0]],2)

    def test_answer_bytes_and_eos_are_distinct(self):
        selected = [next(row for row in TRAIN if row['answer']=='0')]
        with patch.object(self.model,'forward',side_effect=constant_forward(ord('0'))):
            result = measure(self.model,selected,1)
        self.assertEqual(result['summary']['raw_answer_correct'],1)
        self.assertEqual(result['summary']['raw_answer_and_eos_correct'],0)
        self.assertEqual(result['summary']['raw_unfinished'],1)
        self.assertEqual(result['rows'][0]['stop_reason'],'token-cap')
        validate_report(result,selected,1)

    def test_rejects_held_out_forgery_duplicates_bounds_before_forward(self):
        heldout = copy.deepcopy(next(row for row in ALL if row['split']=='test'))
        forged = copy.deepcopy(heldout);forged['split']='train'
        changed = copy.deepcopy(TRAIN[0]);changed['answer']='999'
        cases = [[],[heldout],[forged],[changed],[TRAIN[0],TRAIN[0]]]
        with patch.object(self.model,'forward',side_effect=AssertionError('must not call')):
            for examples in cases:
                with self.subTest(examples=examples),self.assertRaises(ContractError):
                    measure(self.model,examples,4)
            for cap in [True,0,33]:
                with self.assertRaises(ContractError):measure(self.model,[TRAIN[0]],cap)
            with self.assertRaises(ContractError):measure(self.model,[TRAIN[0]],4,workload='fold')

    def test_rejects_report_mutation_and_unknown_fields(self):
        result = measure(self.model,pair(),3)
        mutations = [lambda r:r['summary'].__setitem__('raw_answer_correct',999),
            lambda r:r['targets'][0].__setitem__('gold_rank',True),
            lambda r:r['targets'][0].__setitem__('gold_probability_hex','nan'),
            lambda r:r['targets'][0].__setitem__('loss_hex',(-1.).hex()),
            lambda r:r['rows'][0].__setitem__('scored_targets',True),
            lambda r:r['rows'][0]['non_neural_controls'].__setitem__('exact_oracle_tokens',[0]),
            lambda r:r['rows'][0].__setitem__('answer_correct',True),
            lambda r:r['rows'][0].__setitem__('unknown',1),
            lambda r:r['targets'].append(copy.deepcopy(r['targets'][0])),
            lambda r:r['input_sensitivity'].__setitem__('different_answer_pairs',999),
            lambda r:r.__setitem__('split','test')]
        for change in mutations:
            edited=copy.deepcopy(result);change(edited)
            with self.subTest(change=change),self.assertRaises(ContractError):
                validate_report(edited,pair(),3)

    def test_detects_rng_mutation(self):
        original=self.model.forward
        def bad(*args,**kwargs):
            torch.rand(1)
            return original(*args,**kwargs)
        with patch.object(self.model,'forward',side_effect=bad),self.assertRaisesRegex(ContractError,'changed model'):
            measure(self.model,[TRAIN[0]],1)

    def test_existing_prompt_carriers_are_separate_controls(self):
        left=TRAIN[0]
        right=next(row for row in TRAIN if row['semantic_identity']==left['semantic_identity']
                   and row['carrier']!=left['carrier'])
        with patch.object(self.model,'forward',side_effect=constant_forward()):
            result=measure(self.model,[left,right],2)
        controls=result['prompt_carrier_controls']
        self.assertEqual(controls['same_semantics_different_carrier_pairs'],1)
        self.assertEqual(controls['identical_output_pairs'],1)
        self.assertEqual(controls['both_exact_pairs'],0)
        self.assertEqual(result['input_sensitivity']['different_answer_pairs'],0)
        validate_report(result,[left,right],2)

    def test_no_conflicting_answers_is_not_false_evidence(self):
        row={'task':'add','example_identity':'a','expected_tokens':[1],'generated_tokens':[2]}
        result=sensitivity([row])
        self.assertIsNone(result['all_conflicting_pairs_identical'])
        self.assertEqual(result['different_answer_pairs'],0)


if __name__=='__main__':
    unittest.main()
