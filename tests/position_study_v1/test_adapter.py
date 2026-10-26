"""Tiny engineering prefixes and shared numerical operations only."""
from dataclasses import replace
import unittest
from collections import Counter

import torch
from safetensors.torch import load, save
from trinite.contracts import ContractError, ModelConfig, json_bytes, parse_json
from trinite.model import Decoder, CaptureSpec
from trinite.model import ModelOutput
from trinite.comparison import greedy
from trinite.tokenizer import ByteTokenizer, BOS, EOS, PAD
from trinite.training import batch, target_loss, run_steps, model_identity
from position_coupling_v1.positions import prefix_positions
from position_study_v1.adapter import StudyDecoder, addressing, generation_cap
from position_study_v1 import numerical, controller

controller.configure()
CONFIG = ModelConfig(context_length=64,blocks=1,width=8,heads=2,feed_forward_width=16)


def encoding(prompt, answer, pad_to=None):
    e = ByteTokenizer(64).encode(prompt+answer,answer_start=len(prompt),pad_to=pad_to)
    return {'identity':'synthetic','ids':e.input_ids,'mask':e.attention_mask,'loss':e.loss_mask}


class AdapterTests(unittest.TestCase):
    def test_native_logits_captures_masked_loss_and_all_gradients_exact(self):
        rows = [encoding('1+2=','30'),encoding('47+58=','501')]
        ids,mask,_ = batch(rows)
        native = Decoder(CONFIG,lane='dense',seed=2)
        model = StudyDecoder(CONFIG,mode='sequence',seed=2)
        spec = CaptureSpec(('embedding','block.0','final'),(0,3,5))
        left,right = native(ids,mask,capture=spec),model(ids,mask,capture=spec)
        self.assertTrue(torch.equal(left.logits,right.logits))
        for name in left.captures: self.assertTrue(torch.equal(left.captures[name],right.captures[name]))
        loss,n = target_loss(native,rows); actual,m = target_loss(model,rows)
        self.assertEqual(n,m); self.assertTrue(torch.equal(loss,actual))
        loss.backward();actual.backward()
        for (ln,lp),(rn,rp) in zip(native.named_parameters(),model.named_parameters()):
            self.assertEqual(ln,rn); self.assertTrue(torch.equal(lp.grad,rp.grad),ln)
        self.assertEqual(set(native.state_dict()),set(model.state_dict()))

    def test_every_prefix_agrees_with_teacher_forcing_for_all_widths_modes(self):
        for mode in ('sequence','coupled'):
            model = StudyDecoder(CONFIG,mode=mode)
            for width in range(1,5):
                prompt = '1'.zfill(width)+'+'+'2'.zfill(width)+'='
                out = [ord('x'),PAD,BOS,ord('='),EOS,255][:width+2]
                full = torch.tensor([[BOS,*prompt.encode(),*out]])
                positions = model.positions(full,torch.ones_like(full,dtype=torch.bool))
                for k in range(width+3):
                    prefix = full[:,:1+len(prompt)+k]
                    actual = model.positions(prefix,torch.ones_like(prefix,dtype=torch.bool))
                    self.assertTrue(torch.equal(actual,positions[:,:prefix.shape[1]]))
                    self.assertEqual(actual[0].tolist(),list(prefix_positions(prompt,k,mode=mode)))
                # Shifted canonical loss calls the same actual input prefix.
                self.assertTrue(torch.equal(model.positions(full[:,:-1],torch.ones_like(full[:,:-1],dtype=torch.bool)),positions[:,:-1]))

    def test_malformed_emitted_values_are_not_parsed_or_repaired(self):
        model = StudyDecoder(CONFIG,mode='coupled')
        prompt = [BOS,*b'1+2=']
        rows = torch.tensor([prompt+[PAD,BOS,ord('=')],prompt+[48,49,EOS]])
        mask = torch.ones_like(rows,dtype=torch.bool)
        self.assertTrue(torch.equal(model.positions(rows,mask)[0],model.positions(rows,mask)[1]))
        model(rows,mask)  # malformed output remains a valid bounded model input
        self.assertEqual(generation_cap('1+2='),3)
        with self.assertRaises(ContractError):
            bad = torch.tensor([prompt+[1,2,3,4]])
            model(bad,torch.ones_like(bad,dtype=torch.bool))

    def test_right_padding_and_future_token_isolation_under_descending_positions(self):
        model = StudyDecoder(CONFIG,mode='coupled')
        rows = [encoding('47+58=','501',16),encoding('1+2=','30',16)]
        ids,mask,_ = batch(rows)
        original = model(ids,mask).logits
        changed = ids.clone(); changed[~mask]=255
        self.assertTrue(torch.equal(original,model(changed,mask).logits))
        changed=ids.clone();changed[0,8:11]=ord('x')
        self.assertTrue(torch.equal(original[0,:8],model(changed,mask).logits[0,:8]))
        for i in range(2):
            n=int(mask[i].sum());single=model(ids[i:i+1,:n],mask[i:i+1,:n]).logits
            torch.testing.assert_close(original[i,:n],single[0],atol=2e-6,rtol=1e-5)

    def test_input_prefix_mismatch_and_external_positions_rejected(self):
        model=StudyDecoder(CONFIG,mode='coupled')
        for tokens in ([BOS,*b'1+2'],[0,*b'1+2='],[BOS,*b'1+02='],[BOS,255,*b'+2=']):
            ids=torch.tensor([tokens])
            with self.assertRaises(ContractError): model(ids,torch.ones_like(ids,dtype=torch.bool))
        ids=torch.tensor([[BOS,*b'1+2=']]);mask=torch.ones_like(ids,dtype=torch.bool)
        with self.assertRaises(TypeError): model(ids,mask,position_ids=torch.zeros_like(ids))
        with self.assertRaises(ContractError): StudyDecoder(CONFIG,mode='other')

    def test_actual_greedy_malformed_output_is_bounded_without_forcing_eos(self):
        class Malformed(StudyDecoder):
            def forward(self, ids, mask, **kwargs):
                self.positions(ids,mask)
                logits=torch.zeros((*ids.shape,259),dtype=torch.float32)
                logits[:,:,255]=1
                return ModelOutput(logits,{})
        for mode in ('sequence','coupled'):
            model=Malformed(CONFIG,mode=mode)
            for width in range(1,5):
                prompt='1'.zfill(width)+'+'+'2'.zfill(width)+'='
                out=greedy(model,prompt,generation_cap(prompt))
                self.assertEqual(out,[255]*(width+2));self.assertNotIn(EOS,out)

    def test_frozen_exposure_matches_actual_native_shifted_batches_and_live_mapper(self):
        from position_study_v1.data import schedule, exposure
        for arm,mode in (('control','sequence'),('candidate','coupled')):
            model=StudyDecoder(CONFIG,mode=mode)
            plan=numerical.StudyPlan(model=CONFIG)
            rows=numerical.encoded(schedule(arm),plan)
            counts=Counter();targets=0
            for start in range(0,160,8):
                ids,mask,scored=batch(rows[start:start+8]);targets+=int(scored[:,1:].sum())
                positions=model.positions(ids[:,:-1],mask[:,:-1])
                counts.update(positions[mask[:,:-1]].tolist())
            actual={str(k):v*200 for k,v in counts.items()}
            self.assertEqual(actual,exposure(arm)['training_position_row_lookups'])
            self.assertEqual(targets*200,121600)
            self.assertEqual(sum(actual.values()),307200)

    def test_effective_pairing_rng_and_exact_deterministic_resume(self):
        rng=torch.random.get_rng_state().clone()
        for seed in (0,1,2):
            p=numerical.StudyPlan(seed=seed,model=CONFIG,steps=3,validation_every=2,max_target_tokens=128,max_seconds=30)
            for arm in ('control','candidate'):
                state=numerical.state_for(arm,seed,p)
                counterpart=StudyDecoder(CONFIG,mode='sequence' if arm=='candidate' else 'coupled',seed=seed)
                self.assertEqual(model_identity(state.model),model_identity(counterpart))
                self.assertNotEqual(numerical.behavior_identity(state.model),numerical.behavior_identity(counterpart))
                run_steps(state,stop_after=1)
                payload,metadata=numerical.snapshot(state,arm,seed,'sha256:'+'1'*64)
                resumed=numerical.restore(payload,metadata,arm,seed,'sha256:'+'1'*64,p)
                run_steps(state);run_steps(resumed)
                self.assertEqual(numerical.snapshot(state,arm,seed,'sha256:'+'1'*64),numerical.snapshot(resumed,arm,seed,'sha256:'+'1'*64))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))

    def test_checkpoint_mode_schema_config_and_safe_tensors_reject(self):
        p=numerical.StudyPlan(model=CONFIG,steps=1,validation_every=1,max_target_tokens=64,max_seconds=30)
        state=numerical.state_for('control',0,p);run_steps(state)
        payload,metadata=numerical.snapshot(state,'control',0,'sha256:'+'1'*64)
        for field,value in [('schema','old'),('addressing',addressing('coupled')),('behavior_identity','sha256:'+'0'*64)]:
            r=parse_json(metadata,canonical=True);r[field]=value
            with self.assertRaises(ContractError):numerical.restore(payload,json_bytes(r),'control',0,'sha256:'+'1'*64,p)
        with self.assertRaises(ContractError):numerical.restore(payload,metadata,'candidate',0,'sha256:'+'1'*64,p)
        with self.assertRaises(ContractError):numerical.restore(payload,metadata,'control',0,'sha256:'+'2'*64,p)
        tensors=load(payload);key=next(k for k in tensors if k.startswith('model.'))
        for value in (torch.tensor([float('nan')]),torch.zeros_like(tensors[key],dtype=torch.float64),torch.zeros(1)):
            altered=dict(tensors);altered[key]=value
            with self.assertRaises(ContractError):numerical.restore(save(altered),metadata,'control',0,'sha256:'+'1'*64,p)
        for value in (b'pickle',payload+b'trailing'):
            with self.assertRaises(ContractError):numerical.restore(value,metadata,'control',0,'sha256:'+'1'*64,p)


if __name__=='__main__':unittest.main()
