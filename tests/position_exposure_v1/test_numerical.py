"""Tiny engineering controls only; never the twelve-cell target experiment."""
from collections import Counter
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load, save
from trinite.contracts import ContractError, ModelConfig, json_bytes, parse_json
from trinite.model import CaptureSpec, ModelOutput
from trinite.comparison import greedy
from trinite.tokenizer import ByteTokenizer, BOS, EOS, PAD
from trinite.training import batch, target_loss, evaluate, model_identity
from position_coupling_v1.positions import prefix_positions
from position_study_v1.adapter import StudyDecoder
from position_exposure_v1.adapter import ExposureDecoder, addressing, generation_cap
from position_exposure_v1 import numerical
from position_exposure_v1.protocol import ARMS, MODES, offset_for

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
CONFIG = ModelConfig(context_length=64, blocks=1, width=8, heads=2, feed_forward_width=16)
RID = 'sha256:' + '1'*64


def encoding(prompt, answer, pad_to=None):
    e = ByteTokenizer(64).encode(prompt+answer, answer_start=len(prompt), pad_to=pad_to)
    return {'identity': 'synthetic', 'ids': e.input_ids, 'mask': e.attention_mask, 'loss': e.loss_mask}


def tiny(seed=0, steps=22):
    return numerical.ExposurePlan(seed=seed, model=CONFIG, steps=steps,
        validation_every=10, max_target_tokens=1024, max_seconds=30)


class NumericalTests(unittest.TestCase):
    def test_zero_offset_logits_captures_loss_and_all_gradients_exact(self):
        rows = [encoding('1+2=', '30'), encoding('47+58=', '501')]
        ids, mask, _ = batch(rows)
        for arm in ARMS:
            old = StudyDecoder(CONFIG, mode=MODES[arm], seed=2)
            new = ExposureDecoder(CONFIG, arm=arm, seed=2)
            spec = CaptureSpec(('embedding', 'block.0', 'final'), (0, 3, 5))
            left, right = old(ids, mask, capture=spec), new(ids, mask, capture=spec)
            self.assertTrue(torch.equal(left.logits, right.logits))
            for name in left.captures:
                self.assertTrue(torch.equal(left.captures[name], right.captures[name]))
            loss, n = target_loss(old, rows); actual, m = target_loss(new, rows)
            self.assertEqual(n, m); self.assertTrue(torch.equal(loss, actual))
            loss.backward(); actual.backward()
            for (ln, lp), (rn, rp) in zip(old.named_parameters(), new.named_parameters()):
                self.assertEqual(ln, rn); self.assertTrue(torch.equal(lp.grad, rp.grad), ln)
            self.assertEqual(set(old.state_dict()), set(new.state_dict()))

    def test_explicit_offset_prefixes_all_widths_modes_and_values(self):
        for arm in ('sequence-balanced', 'coupled-balanced'):
            model = ExposureDecoder(CONFIG, arm=arm)
            for offset in (0, 1, 7):
                with model.training_offset(offset):
                    for width in range(1, 5):
                        prompt = '1'.zfill(width)+'+'+'2'.zfill(width)+'='
                        out = [ord('x'), PAD, BOS, ord('='), EOS, 255][:width+2]
                        full = torch.tensor([[BOS, *prompt.encode(), *out]])
                        for k in range(width+3):
                            prefix = full[:, :1+len(prompt)+k]
                            positions = model.positions(prefix, torch.ones_like(prefix, dtype=torch.bool))
                            self.assertEqual(positions[0].tolist(), list(prefix_positions(
                                prompt, k, mode=MODES[arm], offset=offset)))

    def test_no_grad_diagnostics_and_generation_use_zero_offset(self):
        for arm in ('sequence-balanced', 'coupled-balanced'):
            model = ExposureDecoder(CONFIG, arm=arm)
            rows = [encoding('47+58=', '501')]
            expected = evaluate(model, rows, 1)
            prompt = torch.tensor([[BOS, *b'47+58=']])
            mask = torch.ones_like(prompt, dtype=torch.bool)
            zero = model.positions(prompt, mask)
            with model.training_offset(7):
                self.assertFalse(torch.equal(zero, model.positions(prompt, mask)))
                self.assertEqual(expected, evaluate(model, rows, 1))
                with torch.no_grad():
                    self.assertTrue(torch.equal(zero, model.positions(prompt, mask)))
            model.require_idle()

    def test_context_rejects_nested_fixed_and_invalid_and_cleans_exception(self):
        model = ExposureDecoder(CONFIG, arm='coupled-balanced')
        for offset in (-1, 8, True, '1'):
            with self.assertRaises(ContractError):
                with model.training_offset(offset): pass
        with self.assertRaises(RuntimeError):
            with model.training_offset(3):
                with self.assertRaises(ContractError):
                    with model.training_offset(2): pass
                raise RuntimeError('engineering interruption')
        model.require_idle()
        fixed = ExposureDecoder(CONFIG, arm='coupled-fixed')
        with self.assertRaises(ContractError):
            with fixed.training_offset(1): pass

    def test_malformed_values_do_not_select_positions(self):
        model = ExposureDecoder(CONFIG, arm='coupled-balanced')
        prompt = [BOS, *b'1+2=']
        rows = torch.tensor([prompt+[PAD, BOS, ord('=')], prompt+[48, 49, EOS]])
        mask = torch.ones_like(rows, dtype=torch.bool)
        with model.training_offset(7):
            self.assertTrue(torch.equal(model.positions(rows, mask)[0], model.positions(rows, mask)[1]))
            model(rows, mask)
        for tokens in ([BOS,*b'1+2'], [0,*b'1+2='], [BOS,*b'1+02='], [BOS,255,*b'+2=']):
            ids = torch.tensor([tokens])
            with self.assertRaises(ContractError): model(ids, torch.ones_like(ids,dtype=torch.bool))

    def test_padding_and_causal_isolation_under_shifted_descending_positions(self):
        model = ExposureDecoder(CONFIG, arm='coupled-balanced')
        rows = [encoding('47+58=', '501', 16), encoding('1+2=', '30', 16)]
        ids, mask, _ = batch(rows)
        with model.training_offset(7):
            original = model(ids, mask).logits
            changed = ids.clone(); changed[~mask] = 255
            self.assertTrue(torch.equal(original, model(changed,mask).logits))
            changed = ids.clone(); changed[0,8:11] = ord('x')
            self.assertTrue(torch.equal(original[0,:8], model(changed,mask).logits[0,:8]))
            for i in range(2):
                n = int(mask[i].sum())
                single = model(ids[i:i+1,:n],mask[i:i+1,:n]).logits
                torch.testing.assert_close(original[i,:n], single[0], atol=2e-6, rtol=1e-5)
            self.assertTrue(torch.equal(model.positions(ids,mask)[~mask],torch.zeros_like(ids[~mask])))

    def test_actual_greedy_malformed_output_kept_and_zero_addressed(self):
        observed = []
        class Malformed(ExposureDecoder):
            def forward(self, ids, mask, **kwargs):
                observed.append(self.positions(ids,mask).clone())
                logits = torch.zeros((*ids.shape,259),dtype=torch.float32)
                logits[:,:,255] = 1
                return ModelOutput(logits,{})
        for arm in ('sequence-balanced','coupled-balanced'):
            model = Malformed(CONFIG,arm=arm)
            for width in range(1,5):
                prompt = '1'.zfill(width)+'+'+'2'.zfill(width)+'='
                observed.clear()
                with model.training_offset(7):
                    out = greedy(model,prompt,generation_cap(prompt))
                self.assertEqual(out,[255]*(width+2)); self.assertNotIn(EOS,out)
                for k, positions in enumerate(observed):
                    self.assertEqual(positions[0].tolist(),list(prefix_positions(prompt,k,mode=MODES[arm],offset=0)))

    def test_exposure_counts_match_actual_shifted_native_batches(self):
        from position_exposure_v1.data import exposure, schedule
        for arm in ARMS:
            model = ExposureDecoder(CONFIG, arm=arm)
            plan = numerical.ExposurePlan(model=CONFIG)
            rows = numerical.encoded(schedule(arm), plan)
            counts = Counter(); targets = 0
            # Eight cycles contain each balanced offset exactly once. Fixed
            # controls traverse the same rows with zero offsets eight times.
            for cycle in range(8):
                with model.training_offset(offset_for(arm, cycle*160)):
                    for start in range(0,160,8):
                        ids,mask,scored = batch(rows[start:start+8])
                        targets += int(scored[:,1:].sum())
                        positions = model.positions(ids[:,:-1],mask[:,:-1])
                        counts.update(positions[mask[:,:-1]].tolist())
            actual = {str(k):v*25 for k,v in counts.items()}
            expected = exposure(arm)
            self.assertEqual(actual,expected['training_position_row_lookups'])
            self.assertEqual(targets*25,121600)
            self.assertEqual(sum(actual.values()),307200)

    def test_all_four_arms_share_effective_initial_tensors_no_rng_changes(self):
        rng = torch.random.get_rng_state().clone()
        for seed in (0,1,2):
            models = [ExposureDecoder(CONFIG,arm=arm,seed=seed) for arm in ARMS]
            self.assertEqual(len({model_identity(m) for m in models}),1)
            self.assertEqual(len({numerical.behavior_identity(m) for m in models}),4)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))

    def test_native_updater_cycle_calls_and_validation_offsets(self):
        p = tiny()
        state = numerical.state_for('coupled-balanced',0,p)
        calls = []
        native = numerical.native_run_steps
        def traced(state, *, stop_after):
            calls.append((state.step,stop_after,state.model._training_offset))
            return native(state,stop_after=stop_after)
        with patch.object(numerical,'native_run_steps',side_effect=traced):
            numerical.run_steps(state)
        self.assertEqual(calls,[(0,20,0),(20,22,1)])
        self.assertEqual(state.validation[-1]['loss_hex'],evaluate(state.model,state.data['validation'],p.batch_size))
        state.model.require_idle()

    def test_exact_resume_across_offset_boundary_and_same_global_rng(self):
        rng = torch.random.get_rng_state().clone()
        for arm in ('sequence-balanced','coupled-balanced'):
            p = tiny(); state = numerical.state_for(arm,0,p)
            numerical.run_steps(state,stop_after=19)
            payload,metadata = numerical.snapshot(state,arm,0,RID)
            resumed = numerical.restore(payload,metadata,arm,0,RID,p)
            numerical.run_steps(state); numerical.run_steps(resumed)
            self.assertEqual(numerical.snapshot(state,arm,0,RID),numerical.snapshot(resumed,arm,0,RID))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))

    def test_fixed_control_native_update_parity(self):
        from position_study_v1.numerical import StudyPlan, state_for
        from trinite.training import run_steps
        from trinite.comparison_checkpoint import tensor_payload
        old_plan = StudyPlan(model=CONFIG,steps=22,validation_every=10,max_target_tokens=1024,max_seconds=30)
        for arm, historical in (('sequence-fixed','control'),('coupled-fixed','candidate')):
            old = state_for(historical,0,old_plan)
            new = numerical.state_for(arm,0,tiny())
            run_steps(old); numerical.run_steps(new)
            self.assertEqual(tensor_payload(old),tensor_payload(new))
            self.assertEqual(old.history,new.history); self.assertEqual(old.validation,new.validation)

    def test_snapshot_context_policy_cursor_and_tensor_guards(self):
        p = tiny(steps=1); state = numerical.state_for('sequence-balanced',0,p)
        numerical.run_steps(state)
        payload,metadata = numerical.snapshot(state,'sequence-balanced',0,RID)
        for field,value in [('schema','old'),('addressing',addressing('sequence-fixed')),
                            ('behavior_identity','sha256:'+'0'*64)]:
            r = parse_json(metadata,canonical=True); r[field] = value
            with self.assertRaises(ContractError): numerical.restore(payload,json_bytes(r),'sequence-balanced',0,RID,p)
        for field,value in [('cursor',0),('next_offset',7),('next_offset',False),('cycle_index',9)]:
            r = parse_json(metadata,canonical=True); r['offset_progress'][field] = value
            with self.assertRaises(ContractError): numerical.restore(payload,json_bytes(r),'sequence-balanced',0,RID,p)
        with self.assertRaises(ContractError): numerical.restore(payload,metadata,'sequence-fixed',0,RID,p)
        with state.model.training_offset(1):
            with self.assertRaises(ContractError): numerical.snapshot(state,'sequence-balanced',0,RID)
            with self.assertRaises(ContractError): numerical.score(state,'sequence-balanced','train')
        tensors = load(payload); key = next(k for k in tensors if k.startswith('model.'))
        for value in (torch.tensor([float('nan')]),torch.zeros_like(tensors[key],dtype=torch.float64),torch.zeros(1)):
            altered = dict(tensors); altered[key] = value
            with self.assertRaises(ContractError): numerical.restore(save(altered),metadata,'sequence-balanced',0,RID,p)
        for value in (b'pickle',payload+b'trailing'):
            with self.assertRaises(ContractError): numerical.restore(value,metadata,'sequence-balanced',0,RID,p)

    def test_context_cleanup_on_updater_failure_and_invalid_progress(self):
        state = numerical.state_for('sequence-balanced',0,tiny(steps=1))
        with patch.object(numerical,'native_run_steps',side_effect=ContractError('synthetic failure')):
            with self.assertRaises(ContractError): numerical.run_steps(state)
        state.model.require_idle()
        state.cursor = 1
        with self.assertRaises(ContractError): numerical.run_steps(state)


if __name__ == '__main__': unittest.main()
