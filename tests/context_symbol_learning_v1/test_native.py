"""Actual pinned CPU/tokenizer/safe-reader/oracle, zero optimizer steps."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load,save
from context_symbol_learning_v1 import data,numerical,oracle
from procedural_dataset_v1 import cases
from trinite.training import model_identity,target_loss,run_steps
from trinite.contracts import ContractError


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numerical.configure()
        cls.rows=data.training_rows(data.inventory(),lambda r:cases.expected_target('context',r))

    def test_actual_oracle_class_and_mask_conformance(self):
        # Actual provider calls for exposed formal engineering rows, not a fit.
        request=dict(schema=cases.REQUEST_SCHEMAS['context'],contexts=2,symbols=2,meanings=2,
            table=[[0,1],[1,1]],context_mask=[True,True],symbol_mask=[True,False],query=0)
        for symbol,query,answer in [([True,False],0,'0'),([False,True],0,'-1'),([False,True],1,'1')]:
            r=dict(request,symbol_mask=symbol,query=query)
            self.assertEqual(oracle.observe(r),cases.public_observation('context',r))
            self.assertEqual(oracle.target(r),answer)

    def test_actual_source_specific_admission_never_evaluates_heldout(self):
        from context_symbol_learning_v1 import custody
        allowed={cases.canonical(r['request']) for r in data.inventory() if r['split']=='train'}
        actual=oracle.target
        def guarded(request,log=None):
            self.assertIn(cases.canonical(request),allowed)
            return actual(request,log)
        with patch.object(oracle,'target',side_effect=guarded) as target:
            content=custody.prepare_content()
        self.assertEqual(target.call_count,360)
        self.assertEqual(content['audit']['examples'],648)
        self.assertEqual(len(content['train']),360)
        self.assertEqual(content['schedule']['main-17']['targets'],4900)

    def test_actual_count_zero_step_checkpoint_and_resealed_tampering(self):
        with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('NO optimizer steps')):
            state=numerical.state_for(self.rows,17,'engineering-only',True)
            self.assertEqual(sum(p.numel() for p in state.model.parameters()),24768)
            payload,raw=numerical.snapshot(state,'engineering-only')
            restored=numerical.restore(payload,raw,self.rows,17,'engineering-only',
                [__import__('hashlib').sha256(payload).hexdigest(),__import__('hashlib').sha256(raw).hexdigest()],True)
            self.assertEqual(model_identity(restored.model),model_identity(state.model))
            self.assertEqual(restored.step,0);self.assertEqual(dict(restored.optimizer.state),{})
            import queue_evaluation_v1 as gate
            tensors=load(payload)
            for kind in ('extra','dtype','shape','nan'):
                bad={k:v.clone() for k,v in tensors.items()}
                key=next(iter(bad))
                if kind=='extra':bad['unknown']=torch.zeros(1)
                elif kind=='dtype':bad[key]=bad[key].double()
                elif kind=='shape':bad[key]=bad[key].reshape(-1)[:1]
                else:bad[key].reshape(-1)[0]=float('nan')
                altered=save(bad)
                with self.subTest(kind=kind),self.assertRaises(ValueError):
                    numerical.restore(altered,raw,self.rows,17,'engineering-only',[gate.digest(altered),gate.digest(raw)],True)

    def test_target_shift_eos_and_prompt_positions(self):
        # Deliberately make only shifted answer/EOS positions perfect; every
        # prompt prediction is wrong. Loss must ignore those prompt positions.
        row=data.encode(data.observation(data.inventory()[0]),'-1')
        row['identity']='engineering'
        class Model:
            def __call__(self,ids,mask):
                from types import SimpleNamespace
                logits=torch.full((1,ids.shape[1],259),-30.)
                for j in range(ids.shape[1]):
                    expected=row['ids'][j+1]
                    logits[0,j,expected if row['loss'][j+1] else (expected+1)%259]=30.
                return SimpleNamespace(logits=logits)
        loss,tokens=target_loss(Model(),[row])
        self.assertEqual(tokens,3)
        self.assertLess(float(loss),1e-6)
        row['loss'][-1]=0
        self.assertEqual(target_loss(Model(),[row])[1],2)

    def test_budget_rejection_before_native_optimizer_step(self):
        state=numerical.state_for(self.rows,17,'engineering-only',True)
        before=model_identity(state.model)
        state.target_tokens=state.config.max_target_tokens
        with patch.object(state.optimizer,'step',side_effect=AssertionError('NO optimizer steps')):
            with self.assertRaisesRegex(ContractError,'scored-target budget'):
                run_steps(state,stop_after=1)
        self.assertEqual(model_identity(state.model),before)
        self.assertEqual(state.step,0)
        state.target_tokens=0
        with patch('trinite.training.time.monotonic',side_effect=[0.,121.]):
            with self.assertRaisesRegex(ContractError,'time budget'):
                run_steps(state,stop_after=1)

    def test_native_decoder_receives_only_public_carrier(self):
        obs=data.observation(data.inventory()[0])
        with patch('trinite.comparison.greedy',return_value=[48,257]) as greedy:
            numerical.infer(object(),obs)
        self.assertEqual(greedy.call_args.args[1],data.prompt(obs))
        self.assertEqual(greedy.call_args.args[2],3)
        for word in ('seed','family','answer','filename','provenance'):
            self.assertNotIn(word,greedy.call_args.args[1])


if __name__=='__main__':
    unittest.main()
