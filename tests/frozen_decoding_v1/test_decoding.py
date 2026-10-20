import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from trinite.contracts import ContractError, json_bytes
from trinite.comparison import greedy
from frozen_decoding_v1.experiment import generate, score, original, load_model, request


class FakeModel:
    config=SimpleNamespace(context_length=32)
    def __init__(self, sequence):self.sequence=sequence;self.calls=0
    def __call__(self, ids, mask):
        logits=torch.zeros((1,ids.shape[1],259))
        token=self.sequence[min(self.calls,len(self.sequence)-1)];self.calls+=1
        if token is not None:logits[0,-1,token]=10
        return SimpleNamespace(logits=logits)


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_greedy_parity(self):
        for seq in ([257],[256,258],[49,257],[None,None],[255,0]):
            self.assertEqual(generate(FakeModel(seq),'x','greedy')['generated_tokens'],greedy(FakeModel(seq),'x',2))

    def test_syntax_global_digits_and_forced_eos_cost(self):
        for digit in range(48,58):
            model=FakeModel([digit,1]);r=generate(model,'(0+0) mod 2 =','syntax')
            self.assertEqual(r['generated_tokens'],[digit,257]);self.assertEqual(r['forward_calls'],2)
            self.assertEqual(model.calls,2)

    def test_ties_and_no_gold_interface(self):
        r=generate(FakeModel([None]),'arbitrary prompt','syntax')
        self.assertEqual(r['generated_tokens'],[48,257])
        with self.assertRaises(TypeError):generate(FakeModel([48]),'x','syntax',answer='0')

    def test_context_and_early_eos(self):
        self.assertEqual(generate(FakeModel([257]),'x','greedy')['forward_calls'],1)
        self.assertEqual(generate(FakeModel([48]),'x'*30,'syntax')['generated_tokens'],[48,257])
        for mode in ('greedy','syntax'):
            r=generate(FakeModel([48]),'x'*31,mode)
            self.assertEqual(r['generated_tokens'],[48]);self.assertEqual(r['stop_reason'],'context_cap')
            self.assertFalse(score(r['generated_tokens'],'0')['format_valid'])
            self.assertEqual(generate(FakeModel([48]),'x'*32,mode)['forward_calls'],0)

    def test_nonfinite_even_masked_out_rejected(self):
        old=FakeModel.__call__
        for nonfinite in (float('nan'),float('inf'),float('-inf')):
            def bad(self,ids,mask):
                r=old(self,ids,mask);r.logits[0,-1,200]=nonfinite;return r
            with patch.object(FakeModel,'__call__',bad):
                for mode in ('greedy','syntax'):
                    with self.assertRaises(ContractError):generate(FakeModel([48]),'x',mode)

    def test_label_changes_cannot_change_report_generation(self):
        from frozen_decoding_v1.experiment import report
        cell={'seed':0,'arm':'absmean','final_checkpoint':{},'quality':{'train':{'predictions':[{'example_identity':'item','generated_tokens':[48,48]}]}}}
        model=FakeModel([48]);model.parameters=lambda: []
        outputs=[]
        for answer in ('0','9'):
            item={'split':'train','prompt':'fixed prompt','answer':answer,'text':'changed metadata '+answer,'encoding':answer,'example_identity':'item','family_id':'family'}
            with patch('frozen_decoding_v1.experiment.request',return_value={}), patch('frozen_decoding_v1.experiment.original',return_value={'cells':[cell]}), patch('frozen_decoding_v1.experiment.inputs',return_value=(json_bytes({'examples':[item]}),b'')), patch('frozen_decoding_v1.experiment.load_model',return_value=model), patch('frozen_decoding_v1.experiment.model_identity',return_value='fixed'):
                outputs.append(report()['cells'][0]['modes'])
        for mode in ('greedy','syntax'):
            self.assertEqual(outputs[0][mode]['predictions'][0]['generated_tokens'],outputs[1][mode]['predictions'][0]['generated_tokens'])
        self.assertTrue(outputs[0]['syntax']['predictions'][0]['correct'])
        self.assertFalse(outputs[1]['syntax']['predictions'][0]['correct'])

    def test_metrics_partition(self):
        self.assertTrue(score([48,257],'0')['correct'])
        self.assertTrue(score([57,257],'0')['semantic_error'])
        for tokens in ([48],[257],[256,257],[48,48],[]):
            self.assertTrue(score(tokens,'0')['malformed']);self.assertFalse(score(tokens,'0')['correct'])

    def test_pinned_loader_exact_roundtrip_and_reject_tamper(self):
        cells=original()['cells']
        for cell in cells:
            model=load_model(cell)
            for name,p in model.named_parameters():
                self.assertEqual([float(v).hex() for v in p.detach().reshape(-1).tolist()],cell['final_checkpoint']['model_hex'][name])
            self.assertIs(model.output_weight,model.token_embedding)
        for mutation in ('seed','step','nan','missing'):
            cell=copy.deepcopy(cells[0])
            if mutation=='seed':cell['seed']=1
            if mutation=='step':cell['final_checkpoint']['step']=30
            name=next(iter(cell['final_checkpoint']['model_hex']))
            if mutation=='nan':cell['final_checkpoint']['model_hex'][name][0]='nan'
            if mutation=='missing':del cell['final_checkpoint']['model_hex'][name]
            with self.assertRaises(ContractError):load_model(cell)

    def test_request_is_inference_only(self):
        with patch('scale_refit_training_v1.experiment.run',side_effect=AssertionError('no training')):
            r=request();load_model(original()['cells'][0])
        self.assertEqual(len(r['checkpoint_identities']),4)
        self.assertEqual(r['protocol']['training_updates'],0)

if __name__=='__main__':unittest.main()
