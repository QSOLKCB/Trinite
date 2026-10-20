import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from trinite.model import Decoder
from trinite.quantizer import quantize
from trinite.training import run_steps
from scale_refit.quantizer import refit
from scale_refit_training_v1.model import RefitDecoder, RefitLinear, MODEL_ID
from scale_refit_training_v1.experiment import create, run, snapshot, resume, quality, validate, configuration
from scale_refit_training_v1.spec import inputs, specification, sources

class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_matched_inventory_rng_and_baseline_forward_backward(self):
        config=configuration(0).model
        rng=torch.get_rng_state().clone()
        base=Decoder(config,lane='ternary',seed=0)
        control=RefitDecoder(config,arm='absmean',seed=0)
        candidate=RefitDecoder(config,arm='refit',seed=0)
        for model in (control,candidate):
            self.assertEqual(model.inventory()['parameters'],6752)
            self.assertEqual(model.inventory()['scale_parameters'],0)
            self.assertEqual(set(dict(model.named_parameters())),set(dict(base.named_parameters())))
            for name,p in base.named_parameters():self.assertTrue(torch.equal(p,dict(model.named_parameters())[name]))
        x=torch.tensor([[256,49,50]])
        mask=torch.ones_like(x,dtype=torch.bool)
        a,b=base(x,mask).logits,control(x,mask).logits
        self.assertTrue(torch.equal(a,b))
        a.sum().backward();b.sum().backward()
        for name,p in base.named_parameters():self.assertTrue(torch.equal(p.grad,dict(control.named_parameters())[name].grad))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_refit_matrix_actual_forward_and_baseline_mask(self):
        model=RefitDecoder(ModelConfig(context_length=4,blocks=1,width=4,heads=2,feed_forward_width=4),arm='refit')
        linear=model.blocks[0].q
        with torch.no_grad():linear.weight.copy_(torch.tensor([[-2.,-1.,0.,1.]]*4))
        w=linear.weight
        expected=refit(w).quantized
        x=torch.eye(4)
        self.assertTrue(torch.equal(linear(x),expected.weight.t()))
        linear(x).sum().backward()
        scale=quantize(w).scale
        self.assertTrue(torch.equal(w.grad,((w.detach()/scale).abs()<1).float()))
        self.assertFalse(expected.scale.requires_grad)

    def test_bounds_identity_and_native_rejection(self):
        with patch('trinite.model._weight',side_effect=AssertionError('allocation occurred')):
            with self.assertRaises(ContractError):RefitDecoder(ModelConfig(width=512,heads=4),arm='refit')
        with self.assertRaises(ContractError):RefitDecoder(ModelConfig(),arm='learned')
        with self.assertRaises(ContractError):Decoder(lane=MODEL_ID)
        m=RefitDecoder(configuration(0).model,arm='refit')
        m.blocks[0].q.arm='absmean'
        with self.assertRaises(ContractError):m.validate_state()

    def test_native_checkpoint_rejects_versioned_state(self):
        from trinite.checkpoint import checkpoint_bytes
        for arm in ('absmean','refit'):
            with self.assertRaises(ContractError):checkpoint_bytes(create(arm,0))

    def test_split_and_order_matched_no_test_view(self):
        a,b=create('absmean',0),create('refit',0)
        self.assertEqual(a.data,b.data)
        self.assertEqual(set(a.data),{'train','validation'})
        self.assertEqual([len(a.data[x]) for x in ('train','validation')],[36,6])
        examples=json.loads(inputs()[0])['examples']
        families={s:{e['family_id'] for e in examples if e['split']==s} for s in ('train','validation','test')}
        self.assertFalse(families['train']&families['validation'])
        run(a,stop_after=1);run(b,stop_after=1)
        self.assertEqual(a.history[0]['examples'],b.history[0]['examples'])
        self.assertEqual(a.target_tokens,b.target_tokens)
        self.assertEqual(a.target_tokens,8)

    def test_unmodified_native_updater_control_parity(self):
        control=create('absmean',1)
        native=create('absmean',1)
        native.model=Decoder(native.config.model,lane='ternary',seed=1)
        from trinite.training import create_optimizer
        native.optimizer=create_optimizer(native.model,native.config)
        run(control,stop_after=2);run_steps(native,stop_after=2)
        self.assertEqual(control.history,native.history)
        for n,p in native.model.named_parameters():self.assertTrue(torch.equal(p,dict(control.model.named_parameters())[n]))

    def test_full_prefix_resume_and_tampering(self):
        for arm in ('absmean','refit'):
            original=run(create(arm,0),stop_after=2)
            before=snapshot(original)
            restored=resume(before,expected_identity=identity(before),arm=arm,seed=0)
            self.assertEqual(snapshot(restored),before)
            run(original,stop_after=3);run(restored,stop_after=3)
            self.assertEqual(snapshot(original),snapshot(restored))
            altered=json.loads(before);altered['history'][0]['loss_hex']='0x0p+0'
            raw=json_bytes(altered)
            with self.assertRaises(ContractError):resume(raw,expected_identity=identity(raw),arm=arm,seed=0)
            with self.assertRaises(ContractError):resume(before,expected_identity=identity(before),arm=arm,seed=1)
            with self.assertRaises(ContractError):resume(before,expected_identity=identity(before),arm='refit' if arm=='absmean' else 'absmean',seed=0)

    def test_checkpoint_progress_slots_and_source_validation(self):
        state=run(create('refit',0),stop_after=1)
        state.cursor+=1
        with self.assertRaises(ContractError):snapshot(state)
        state.cursor-=1
        p=next(state.model.parameters())
        state.optimizer.state[p]['step'].fill_(0)
        with self.assertRaises(ContractError):snapshot(state)
        self.assertIn('scripts/scale_refit/quantizer.py',sources())
        self.assertIn('src/trinite/training.py',sources())

    def test_quality_uses_prompt_only_and_no_test_labels(self):
        state=create('refit',0)
        seen=[]
        def fake(model,prompt,cap):
            seen.append((prompt,cap));return [48,257]
        with patch('scale_refit_training_v1.experiment.greedy',side_effect=fake):result=quality(state)
        examples=json.loads(inputs()[0])['examples']
        selected=[e for e in examples if e['split'] in ('train','validation')]
        self.assertEqual(len(seen),42)
        self.assertEqual({p for p,c in seen},{e['prompt'] for e in selected})
        self.assertTrue(all(c==2 for p,c in seen))
        self.assertEqual(set(result),{'train','validation'})
        for split in result:
            for row in result[split]['predictions']:
                self.assertEqual(row['correct'],row['generated_tokens']==row['expected_tokens'])

    def test_exclusive_cli_and_request_binding(self):
        import tempfile
        from run_scale_refit_training import main,read
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'request.json'
            self.assertEqual(main(['freeze','--request',str(p)]),0)
            before=p.read_bytes()
            self.assertEqual(main(['freeze','--request',str(p)]),1)
            self.assertEqual(p.read_bytes(),before)
            self.assertEqual(main(['run','--request',str(p),'--request-identity','sha256:'+'0'*64,'--output',str(Path(d)/'out')]),1)
            link=Path(d)/'link';link.symlink_to(p)
            with self.assertRaises(ContractError):read(link,262144)

    def test_cli_io_only_stub_replay_and_tamper(self):
        # This fake checks file/CLI behavior only, never numerical certification.
        import tempfile
        from run_scale_refit_training import main
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'request.json';out=Path(d)/'report.json'
            self.assertEqual(main(['freeze','--request',str(p)]),0)
            args=['--request',str(p),'--request-identity',identity(p.read_bytes()),'--output',str(out)]
            with patch('run_scale_refit_training.report',return_value={'io_test':True}):
                self.assertEqual(main(['run',*args]),0)
                old=out.read_bytes()
                self.assertEqual(main(['verify',*args,'--expected-identity',identity(old)]),0)
                self.assertEqual(out.read_bytes(),old)
                self.assertEqual(main(['run',*args]),1)
                out.write_bytes(json_bytes({'io_test':False}))
                self.assertEqual(main(['verify',*args,'--expected-identity',identity(out.read_bytes())]),1)

if __name__=='__main__':unittest.main()
