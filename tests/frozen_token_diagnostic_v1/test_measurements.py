"""Engineering-only toy conformance; never executes returned trained checkpoints."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from trinite.contracts import ContractError, ModelConfig, json_bytes
from trinite.model import Decoder
from trinite.tokenizer import ByteTokenizer, EOS
from trinite.training import model_identity
from native_diagnostic_v1.measurements import _state
from addition_development_v1.data import examples
from frozen_token_diagnostic_v1.measurements import measure, validate_report, CONDITIONING


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def constant_forward(token=None, value=8., dtype=torch.float32):
    def forward(ids, mask, **kwargs):
        logits = torch.zeros((*ids.shape,259), dtype=dtype)
        if token is not None:
            logits[:,:,token] = value
        return SimpleNamespace(logits=logits)
    return forward


class MeasurementsTests(unittest.TestCase):
    def setUp(self):
        self.model = Decoder(ModelConfig(context_length=64, blocks=1, width=16, heads=2,
                                        feed_forward_width=32), lane='dense', seed=0)

    def measured_constant(self, stratum='familiar-answer', token=None, value=8.):
        with patch.object(self.model, 'forward', side_effect=constant_forward(token, value)):
            return measure(self.model, stratum)

    def test_native_model_all_positions_state_and_callback(self):
        next(self.model.parameters()).grad = torch.ones_like(next(self.model.parameters()))
        self.model.eval()
        before = json_bytes(_state(self.model))
        callbacks = []
        result = measure(self.model, 'familiar-answer', callbacks.append)
        self.assertEqual(before, json_bytes(_state(self.model)))
        self.assertEqual(callbacks, result['rows'])
        self.assertEqual(result['summary'], validate_report(result, 'familiar-answer', model_identity(self.model)))
        self.assertEqual(result['conditioning'], CONDITIONING)
        self.assertEqual(result['summary']['scored_targets'],84)
        self.assertEqual(result['summary']['forward_calls'],72)
        self.assertFalse(result['raw_predictions_replayed'])
        for item, row in zip(examples('familiar-answer'), result['rows']):
            start = len(item['prompt']) + 1
            self.assertEqual([t['target_position'] for t in row['targets']], list(range(start,start+len(item['answer'])+1)))
            self.assertEqual([t['gold_token'] for t in row['targets']], [*item['answer'].encode(),EOS])
            self.assertTrue(row['targets'][-1]['is_eos'])
            self.assertFalse(any(t['is_eos'] for t in row['targets'][:-1]))

    def test_all_three_strata_no_selected_subset(self):
        for stratum, n, targets in [('familiar-answer',36,84),('novel-answer',124,310),('range-length',138,414)]:
            with self.subTest(stratum=stratum):
                result = self.measured_constant(stratum)
                self.assertEqual(len(result['rows']),n)
                self.assertEqual(result['summary']['scored_targets'],targets)
                self.assertEqual(sum(v['examples'] for v in result['summary']['by_answer_length_bytes'].values()),n)
                for item,row in zip(examples(stratum),result['rows']):
                    self.assertEqual(row['coverage'],item['coverage'])
                    self.assertEqual(row['answer_length_bytes'],len(item['answer']))
                    self.assertEqual(row['prompt_length_bytes'],len(item['prompt']))
                    self.assertEqual(row['operand_length_bytes'],[len(str(item['formal'][k])) for k in ('a','b')])
                self.assertEqual(result['summary']['positions']['first_answer']['targets'],n)
                self.assertEqual(result['summary']['positions']['answer_eos']['targets'],n)
                self.assertEqual(result['summary']['positions']['later_answer']['targets'], targets-2*n)

    def test_lowest_token_id_ties(self):
        result = self.measured_constant()
        for row in result['rows']:
            for target in row['targets']:
                self.assertEqual(target['argmax_token'],0)
                self.assertEqual(target['gold_rank'],target['gold_token']+1)
                self.assertEqual(target['eos_rank'],EOS+1)
                self.assertAlmostEqual(float.fromhex(target['gold_probability_hex']),1/259,places=8)
                self.assertAlmostEqual(float.fromhex(target['loss_hex']),math.log(259),places=6)

    def test_eos_preference_not_raw_premature_stop(self):
        result = self.measured_constant(token=EOS)
        counts = result['summary']['positions']
        self.assertEqual(counts['first_answer']['eos_top1'],36)
        self.assertEqual(counts['later_answer']['eos_top1'],12)
        self.assertEqual(counts['answer_eos']['gold_top1'],36)
        self.assertEqual(result['summary']['teacher_forced_all_targets_top1'],0)
        self.assertNotIn('raw_premature_eos',result['summary'])
        self.assertNotIn('generated_tokens',result['rows'][0])

    def test_wrong_second_digit_not_hidden_by_first_digit(self):
        result = self.measured_constant(token=ord('1'))
        for item,row in zip(examples('familiar-answer'),result['rows']):
            if item['answer']=='13':
                self.assertTrue(row['targets'][0]['correct'])
                self.assertFalse(row['targets'][1]['correct'])
                self.assertEqual(row['targets'][1]['gold_token'],ord('3'))

    def test_extreme_logits_stable_loss_and_underflow(self):
        result = self.measured_constant(token=EOS,value=1000.)
        target = result['rows'][0]['targets'][0]
        self.assertEqual(float.fromhex(target['gold_probability_hex']),0.)
        self.assertEqual(float.fromhex(target['loss_hex']),1000.)
        self.assertTrue(math.isfinite(float.fromhex(result['summary']['target_mean_loss_hex'])))

    def test_validator_rejects_missing_extra_reordered_rows_and_targets(self):
        result = self.measured_constant()
        for mutate in (lambda r:r['rows'].pop(), lambda r:r['rows'].append(copy.deepcopy(r['rows'][0])),
                       lambda r:r['rows'].reverse(), lambda r:r['rows'][0]['targets'].pop(),
                       lambda r:r['rows'][0]['targets'].reverse()):
            changed=copy.deepcopy(result); mutate(changed)
            with self.assertRaises(ContractError):
                validate_report(changed,'familiar-answer',model_identity(self.model))

    def test_validator_rejects_forged_labels_metrics_and_claims(self):
        result=self.measured_constant()
        mutations = [lambda r:r.update(learning_adequate=True),
            lambda r:r.update(conditioning='raw'),
            lambda r:r['summary'].update(forward_calls=0),
            lambda r:r['rows'][0]['targets'][0].update(gold_rank=1),
            lambda r:r['rows'][0]['targets'][0].update(eos_rank=1),
            lambda r:r['rows'][0]['targets'][0].update(target_position=0),
            lambda r:r['rows'][0]['targets'][0].update(gold_probability_hex=float('nan').hex()),
            lambda r:r['rows'][0]['targets'][0].update(loss_hex=(-1.).hex()),
            lambda r:r['rows'][0].update(authoritative_loss_hex=(0.).hex()),
            lambda r:r['rows'][0].update(answer_length_bytes=99),
            lambda r:r['rows'][0].update(prompt_length_bytes=99),
            lambda r:r['rows'][0].update(operand_length_bytes=[99,99]),
            lambda r:r['rows'][0].update(coverage={})]
        for mutate in mutations:
            changed=copy.deepcopy(result); mutate(changed)
            with self.assertRaises(ContractError):
                validate_report(changed,'familiar-answer',model_identity(self.model))
        with self.assertRaises(ContractError):
            validate_report(result,'familiar-answer','sha256:'+'0'*64)
        with self.assertRaises(ContractError): measure(self.model,'train')

    def test_logits_dtype_and_nonfinite_rejected(self):
        for forward in (constant_forward(dtype=torch.float64), constant_forward(EOS,float('nan')),
                        constant_forward(EOS,float('inf'))):
            with patch.object(self.model,'forward',side_effect=forward), self.assertRaises(ContractError):
                measure(self.model,'familiar-answer')

    def test_callback_copy_and_mutation_guard(self):
        def change_copy(row): row['targets'].clear()
        with patch.object(self.model,'forward',side_effect=constant_forward()):
            report=measure(self.model,'familiar-answer',change_copy)
        self.assertTrue(report['rows'][0]['targets'])
        def change_rng(row): torch.rand(1)
        with patch.object(self.model,'forward',side_effect=constant_forward()), self.assertRaisesRegex(ContractError,'changed model'):
            measure(self.model,'familiar-answer',change_rng)

    def test_full_causal_first_target_matches_prompt_prefix(self):
        item=next(r for r in examples('familiar-answer') if r['answer']=='13')
        enc=ByteTokenizer(64).encode(item['text'],answer_start=len(item['prompt']))
        ids=torch.tensor([enc.input_ids],dtype=torch.int64)
        mask=torch.tensor([enc.attention_mask],dtype=torch.bool)
        position=next(i for i,flag in enumerate(enc.loss_mask) if flag)
        with torch.no_grad():
            full=self.model(ids[:,:-1],mask[:,:-1]).logits[0,position-1]
            prefix=self.model(ids[:,:position],mask[:,:position]).logits[0,-1]
        torch.testing.assert_close(full,prefix,rtol=1e-5,atol=1e-6)
        self.assertEqual(int(full.argmax()),int(prefix.argmax()))


if __name__ == '__main__': unittest.main()
