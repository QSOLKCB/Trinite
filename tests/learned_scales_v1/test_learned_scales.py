"""Independent analytic and adversarial tests; never differentiate rounding."""
import copy
from fractions import Fraction
import math
import random
import struct
import unittest
from unittest.mock import patch
import numpy as np
import torch
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite.model import Decoder
from learned_scales_v1.model import ScaleDecoder
from learned_scales_v1.quantizer import quantize, initial_scale, project_scale_
from learned_scales_v1.spec import specification, sources, ROOT, PROTOCOL_ID
from learned_scales_v1.toy import create, run, resume, snapshot, tensors
from learned_scales_v1.evidence import analytic


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def oracle(weights, scale, incoming):
    # Python scalar division with explicit float32 rounding; Fraction ties-to-even
    # gives code independently of torch.round and independently of production h.
    codes, dw, terms = [], [], []
    for w, g in zip(weights, incoming):
        ratio = f32(w/scale)
        code = min(1, max(-1, round(Fraction(ratio))))
        codes.append(code)
        dw.append(g if -1 < ratio < 1 else 0.)
        h = -1 if ratio <= -1 else 1 if ratio >= 1 else f32(code-ratio)
        terms.append(Fraction(g)*Fraction(h))
    return codes, dw, f32(float(sum(terms))/math.sqrt(len(weights)))


class LearnedScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_retained_source_bound_receipt(self):
        path = ROOT/'fixtures/learned-scales-v1'
        raw = (path/'report.json').read_bytes()
        commitment = parse_json((path/'commitment.json').read_bytes(), canonical=True)
        self.assertEqual(identity(raw), commitment['report_identity'])
        report = parse_json(raw, canonical=True)
        self.assertEqual(report['source'], sources())
        self.assertEqual(report['protocol_identity'], PROTOCOL_ID)
        self.assertFalse(report['learning_claim'])
        self.assertFalse(report['scientific_execution'])
        self.assertFalse(report['closed_custody'])
        self.assertEqual(len(report['cells']), 4)
        for pair in report['paired_initialization']:
            self.assertTrue(pair['initial_model_exact_match'])
            self.assertEqual(pair['learned_inventory']['parameters'], 1340)
            self.assertEqual(pair['frozen_inventory']['trainable_parameters'], 1328)
        for cell in report['cells']:
            checkpoint = cell['final_checkpoint']
            self.assertEqual(identity(json_bytes(checkpoint)), cell['final_checkpoint_identity'])
            self.assertEqual(checkpoint['source'], report['source'])
            self.assertEqual(checkpoint['step'], 4)
            self.assertEqual(len(cell['interruptions']), 4)
            self.assertTrue(all(x['exact_match'] for x in cell['interruptions']))

    def test_hand_fixture(self):
        self.assertEqual(analytic()['independent_expected_scale_gradient'], '9/sqrt(7) rounded float32')

    def test_independent_scalar_oracle(self):
        for scale in (0.25, 1., 2.):
            for row in ([-2.,-1.,-.5,0.,.5,1.,2.], [0.,0.,0.], [-.75,.25,.75,1.25]):
                incoming = [float(i+1) for i in range(len(row))]
                codes, dw, ds = oracle(row, scale, incoming)
                w = torch.tensor([row], dtype=torch.float32, requires_grad=True)
                s = torch.tensor(scale, dtype=torch.float32, requires_grad=True)
                q = quantize(w,s)
                q.weight.backward(torch.tensor([incoming], dtype=torch.float32))
                self.assertEqual(q.codes.tolist(), [codes])
                self.assertEqual(w.grad.tolist(), [dw])
                self.assertEqual(float(s.grad), ds)
                self.assertEqual(q.weight.tolist(), [[f32(c*scale) for c in codes]])

    def test_ties_endpoints_and_neighbors(self):
        points = []
        for x in (-1.,-.5,.5,1.):
            a = torch.tensor(x, dtype=torch.float32)
            points += [float(torch.nextafter(a, torch.tensor(-math.inf))), x,
                       float(torch.nextafter(a, torch.tensor(math.inf)))]
        codes, dw, ds = oracle(points, 1., [1.]*len(points))
        w = torch.tensor([points], dtype=torch.float32, requires_grad=True)
        s = torch.tensor(1., requires_grad=True)
        q = quantize(w,s); q.weight.sum().backward()
        self.assertEqual(q.codes.tolist(), [codes]); self.assertEqual(w.grad.tolist(), [dw])
        self.assertEqual(float(s.grad), ds)
        self.assertEqual(q.codes[0,4].item(), 0); self.assertEqual(q.codes[0,7].item(), 0)
        self.assertEqual(w.grad[0,1].item(), 0); self.assertEqual(w.grad[0,10].item(), 0)

    def test_zero_subnormal_and_extreme_finite(self):
        tiny = float(torch.nextafter(torch.tensor(0.), torch.tensor(1.)))
        maximum = torch.finfo(torch.float32).max
        for row in ([0.,0.], [-tiny,tiny], [-maximum,maximum], [maximum,maximum]):
            w = torch.tensor([row], dtype=torch.float32, requires_grad=True)
            for scale in (1e-8, maximum):
                s = torch.tensor(scale, dtype=torch.float32, requires_grad=True)
                q = quantize(w,s); q.weight.sum().backward()
                self.assertTrue(torch.isfinite(q.weight).all()); self.assertTrue(torch.isfinite(w.grad).all())
                self.assertTrue(torch.isfinite(s.grad).all()); w.grad = None
        self.assertEqual(float(initial_scale(torch.zeros(2,2))), f32(1e-8))
        self.assertEqual(float(initial_scale(torch.full((2,2),maximum))), maximum)

    def test_invalid_dtype_shape_and_nonfinite(self):
        w = torch.ones(2,2); s = torch.tensor(1.)
        for bad in (torch.ones(2), torch.empty(0,2), torch.ones(257,256), w.double(),
                    torch.full((2,2), math.nan), torch.full((2,2), math.inf)):
            with self.assertRaises(ContractError): quantize(bad,s)
        for bad in (torch.tensor(0.),torch.tensor(-1.),torch.tensor(1e-9),torch.tensor(math.nan),
                    torch.tensor(math.inf),torch.tensor([1.]),s.double()):
            with self.assertRaises(ContractError): quantize(w,bad)
        with torch.autocast('cpu'):
            with self.assertRaises(ContractError): quantize(w,s)

    def test_invalid_backward(self):
        for incoming in (math.inf, math.nan, torch.finfo(torch.float32).max):
            w = torch.ones(2,2,requires_grad=True); s = torch.tensor(1.,requires_grad=True)
            with self.assertRaises(ContractError):
                quantize(w,s).weight.backward(torch.full((2,2),incoming))

    def test_projection_preserves_optimizer_moments(self):
        s = torch.nn.Parameter(torch.tensor(1e-8))
        opt = torch.optim.AdamW([s],lr=.003,foreach=False,fused=False)
        s.grad = torch.tensor(1.); opt.step()
        self.assertLess(float(s.detach()),0.)
        before = {k:v.clone() for k,v in opt.state[s].items()}
        project_scale_(s)
        self.assertEqual(float(s.detach()),f32(1e-8))
        for k,v in before.items(): self.assertTrue(torch.equal(v,opt.state[s][k]))
        with self.assertRaises(ContractError): project_scale_(torch.tensor(math.inf))

    def test_parameter_counts_and_identical_initialization(self):
        for blocks in (1,2,3):
            config = ModelConfig(blocks=blocks,width=4,heads=2,feed_forward_width=8,context_length=4)
            dense = Decoder(config,lane='dense',seed=0)
            a = ScaleDecoder(config,arm='learned',seed=0)
            b = ScaleDecoder(config,arm='frozen',seed=0)
            self.assertEqual(tensors(a),tensors(b))
            self.assertEqual(a.inventory()['parameters'],config.parameter_count+6*blocks)
            self.assertEqual(a.inventory()['trainable_parameters']-b.inventory()['trainable_parameters'],6*blocks)
            self.assertIs(a.output_weight,a.token_embedding)
            for name,p in dense.named_parameters(): self.assertTrue(torch.equal(p,dict(a.named_parameters())[name]))
            ids = torch.tensor([[1,2,3]])
            self.assertTrue(torch.equal(a(ids,torch.ones_like(ids,dtype=torch.bool)).logits,
                                        b(ids,torch.ones_like(ids,dtype=torch.bool)).logits))
            with self.assertRaises(ContractError): ScaleDecoder(config,arm='learned',parameter_budget=config.parameter_count)
        with self.assertRaises(ContractError):
            ScaleDecoder(ModelConfig(width=512,heads=2),arm='learned')

    def test_checkpoint_full_replay(self):
        for arm in ('learned','frozen'):
            for seed in (0,1):
                expected = snapshot(run(create(arm,seed)))
                for stop in (0,1,3,4):
                    prefix = snapshot(run(create(arm,seed),stop_after=stop))
                    restored = resume(prefix,expected_identity=identity(prefix),arm=arm,seed=seed)
                    self.assertEqual(snapshot(run(restored)),expected)

    def test_tampered_checkpoint_rejected(self):
        raw = snapshot(run(create('learned',0),stop_after=1))
        base = parse_json(raw,canonical=True)
        changes = [lambda x:x.update(arm='frozen'), lambda x:x.update(seed=1),
                   lambda x:x.update(rng_hex='00'),lambda x:x.update(source={}),
                   lambda x:x['model_hex']['blocks.0.q.scale'].__setitem__(0,float(2).hex()),
                   lambda x:x['adamw_hex']['blocks.0.q.scale']['exp_avg'].__setitem__(0,float(2).hex()),
                   lambda x:x['history'][0].update(loss_hex=float(2).hex())]
        for change in changes:
            candidate = copy.deepcopy(base); change(candidate); bad = json_bytes(candidate)
            with self.assertRaises(ContractError): resume(bad,expected_identity=identity(bad),arm='learned',seed=0)
        with self.assertRaises(ContractError): resume(raw,expected_identity='sha256:'+'0'*64,arm='learned',seed=0)
        with self.assertRaises(ContractError): resume(raw,expected_identity=identity(raw),arm='frozen',seed=0)

    def test_global_rng_unchanged(self):
        python = random.getstate(); numpy = np.random.get_state(); cpu = torch.get_rng_state().clone()
        state = run(create('learned',0),stop_after=1)
        raw = snapshot(state); resume(raw,expected_identity=identity(raw),arm='learned',seed=0)
        self.assertEqual(random.getstate(),python)
        self.assertTrue(np.array_equal(np.random.get_state()[1],numpy[1]))
        self.assertEqual(np.random.get_state()[2:],numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(),cpu))

    def test_scientific_execution_hard_closed(self):
        for profile in ('research','scientific',None,True):
            with patch('learned_scales_v1.toy.require_environment',side_effect=AssertionError('allocated')):
                with self.assertRaises(ContractError): create('learned',0,profile=profile)
        with self.assertRaises(ContractError): run(create('learned',0),stop_after=5)

    def test_frozen_scales_and_learned_slots(self):
        for arm in ('learned','frozen'):
            state = create(arm,0); before = tensors(state.model); run(state)
            after = tensors(state.model)
            changed = [n for n in before if n.endswith('.scale') and before[n]!=after[n]]
            self.assertEqual(len(changed)>0,arm=='learned')
            for n,p in state.model.named_parameters():
                if n.endswith('.scale'):
                    self.assertEqual(p in state.optimizer.state,arm=='learned')
            self.assertTrue(all(h['targets']==6 and h['backwards']==1 for h in state.history))


if __name__ == '__main__':
    unittest.main()
