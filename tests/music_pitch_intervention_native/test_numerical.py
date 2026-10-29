"""Synthetic mechanisms only: no fixed640 target training or native probe run."""
from contextlib import ExitStack
from copy import deepcopy
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load, save
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.training import model_identity, run_steps
from music_pitch_intervention_v1 import numerical as n
from music_pitch_intervention_v1.spec import CELLS, dataset, diagnostic_rows, development_rows, train_rows

REQUEST = identity(b'synthetic-request')


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.register_buffer('marker', torch.tensor([2.0]))
        self.child = torch.nn.Identity()
        self.config = SimpleNamespace(context_length=256)

    def forward(self, value):
        return value * self.weight


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.admission = identity(dataset()[1])

    def fresh(self, arm='control', seed=17):
        # No model is evaluated on study rows. Initial loss is a synthetic fixture.
        with patch('trinite.training.evaluate', return_value=1.0.hex()):
            return n.state_for(arm, seed, self.admission)

    def synthetic_step(self, arm='control'):
        state = self.fresh(arm)
        # Preserve scheduled record identities/counts but use a tiny synthetic
        # token sequence unrelated to the musical prompts for one update only.
        for selector in state.data:
            state.data[selector] = [{**row, 'ids': [256, 65, 55, 257],
                'mask': [1,1,1,1], 'loss': [0,0,1,1]} for row in state.data[selector]]
        run_steps(state, stop_after=1)
        return state

    def test_pair_initialization_independent_optimizer_common_validation(self):
        control, pitch = self.fresh('control'), self.fresh('pitch')
        self.assertEqual(control.initialization_identity, pitch.initialization_identity)
        self.assertEqual(model_identity(control.model), model_identity(pitch.model))
        self.assertIsNot(control.optimizer, pitch.optimizer)
        self.assertFalse(set(map(id, control.model.parameters())) & set(map(id, pitch.model.parameters())))
        self.assertEqual(control.data['validation'], pitch.data['validation'])
        self.assertEqual(len(control.data['validation']), 40)
        self.assertEqual(len(control.data['train']), 80)
        self.assertNotEqual(control.data['train'], pitch.data['train'])

    def test_zero_checkpoint_model_only_no_training_forward_or_optimizer(self):
        state = self.fresh()
        payload, meta = n.snapshot(state, 'control', REQUEST)
        with ExitStack() as stack:
            for target in ('trinite.training.create_optimizer', 'trinite.training.evaluate',
                           'trinite.training.run_steps', 'trinite.training.TrainingState',
                           'trinite.model.Decoder.forward'):
                stack.enter_context(patch(target, side_effect=AssertionError('forbidden')))
            model, record = n.load_checkpoint(payload, meta, 'control',17,self.admission,REQUEST,require_final=False)
        self.assertEqual(model_identity(model), model_identity(state.model))
        self.assertEqual(record['step'], 0)
        with self.assertRaisesRegex(ContractError,'endpoint'):
            n.load_checkpoint(payload,meta,'control',17,self.admission,REQUEST)

    def test_one_synthetic_update_snapshot_and_slot_validation(self):
        state = self.synthetic_step('pitch')
        payload, meta = n.snapshot(state, 'pitch', REQUEST)
        model, record = n.load_checkpoint(payload,meta,'pitch',17,self.admission,REQUEST,require_final=False)
        self.assertEqual(model_identity(model), model_identity(state.model))
        self.assertEqual(record['target_tokens'],16)
        for field in ('step','exp_avg','exp_avg_sq'):
            tensors=load(payload); key=next(k for k in tensors if k.startswith('optimizer.') and k.endswith('.'+field))
            tensors[key].fill_(-1 if field=='exp_avg_sq' else float('nan'))
            with self.subTest(field=field), self.assertRaises(ContractError):
                n.load_checkpoint(save(tensors),meta,'pitch',17,self.admission,REQUEST,require_final=False)

    def synthetic_final(self, arm='pitch', seed=17):
        state=self.fresh(arm,seed);payload,meta=n.snapshot(state,arm,REQUEST)
        record=parse_json(meta,canonical=True)
        rows=n.encoded(train_rows(arm))
        record.update(step=640,cursor=5120,target_tokens=10240,
            history=[{'step':i+1,'loss_hex':1.0.hex(),'gradient_norm_hex':1.0.hex(),
                      'target_tokens':16,'examples':[rows[(i*8+j)%80]['identity'] for j in range(8)]}
                     for i in range(640)],
            validation=[{'step':i,'loss_hex':1.0.hex()} for i in range(64,641,64)])
        tensors=load(payload)
        for name,p in state.model.named_parameters():
            tensors['optimizer.'+name+'.step']=torch.tensor(640.0)
            tensors['optimizer.'+name+'.exp_avg']=torch.zeros_like(p)
            tensors['optimizer.'+name+'.exp_avg_sq']=torch.zeros_like(p)
        return state,save(tensors),json_bytes(record)

    def test_wholly_synthetic_final_packet_safe_structure_not_execution(self):
        state,payload,meta=self.synthetic_final()
        with patch('trinite.training.create_optimizer',side_effect=AssertionError('no optimizer')):
            model,verified=n.load_checkpoint(payload,meta,'pitch',17,self.admission,REQUEST)
        self.assertEqual(verified['step'],640)
        self.assertEqual(model_identity(model),model_identity(state.model))
        # Fabricated structural fixture: zero actual updates, not evidence that
        # the640 declared updates were executed or numerically replayed.

    def test_real_worker_packet_and_safe_audit_with_synthetic_producer(self):
        from music_pitch_intervention_v1 import workflow
        from music_pitch_intervention_v1.reporting import score_cell
        state,payload,meta=self.synthetic_final('control')
        records=[{'arm':'control','seed':17,'step':640,'stage':'diagnostic',
                  'example_identity':r['example_identity'],'generated_tokens':[*r['answer'].encode(),257]}
                 for r in diagnostic_rows()]
        summary=score_cell(records,'control',17,'diagnostic')
        raw=b'synthetic request bytes'
        def fake_train(arm,seed,request_id,admission_id,write,emit):
            write('final.safetensors',payload);write('final-checkpoint.json',meta)
            for record in records:emit(record)
            write('diagnostic-predictions.json',json_bytes(records))
            return {'arm':arm,'seed':seed,'request_identity':request_id,
                    'admission_identity':admission_id,'step':640,'target_tokens':10240,
                    'initialization_identity':state.initialization_identity,'model_identity':model_identity(state.model),
                    'update_seconds_hex':0.0.hex(),'diagnostic_summary':summary,
                    'diagnostic_decodes':80,'diagnostic_forward_calls':160}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'worker.log').write_bytes(b'')
            st=root.stat()
            with patch.object(workflow,'validate',return_value={'admission_identity':self.admission}), \
                    patch.object(n,'train',side_effect=fake_train):
                completion=workflow.train_worker(root,raw,REQUEST,'control',17,(st.st_dev,st.st_ino))
                before=workflow.bounded(root)
                audited=workflow.audit_cell(root,raw,REQUEST,'control',17,completion,load=True)
                self.assertEqual(audited['records'],records)
                self.assertEqual(audited['metadata']['step'],640)
                self.assertEqual(model_identity(audited['model']),model_identity(state.model))
                self.assertEqual(workflow.bounded(root),before)
                (root/'final.safetensors').write_bytes(payload+b'junk')
                with self.assertRaises(ContractError):
                    workflow.audit_cell(root,raw,REQUEST,'control',17,completion)

    def test_checkpoint_progress_context_inventory_and_pickle_rejected(self):
        state=self.synthetic_step();payload,meta=n.snapshot(state,'control',REQUEST)
        original=parse_json(meta,canonical=True)
        for key,value in [('arm','pitch'),('seed',29),('step',True),('cursor',0),
                          ('target_tokens',0),('history',[]),('initial_train_loss_hex','nan'),
                          ('initialization_identity',identity(b'other'))]:
            changed={**original,key:value}
            with self.subTest(key=key),self.assertRaises(ContractError):
                n.load_checkpoint(payload,json_bytes(changed),'control',17,self.admission,REQUEST,require_final=False)
        for broken in (b'not-safetensors',save({}),save({**load(payload),'unexpected':torch.tensor(1.)})):
            with self.assertRaises(ContractError):
                n.load_checkpoint(broken,meta,'control',17,self.admission,REQUEST,require_final=False)
        for arm,seed in [('best',17),('control',True),('control',31)]:
            with self.assertRaises(ContractError): n.state_for(arm,seed,self.admission)

    def test_preserve_training_rejects_callback_state_mutation(self):
        state=self.synthetic_step(); before=n.snapshot(state,'control',REQUEST)
        with n._preserve_training(state):
            cloned=deepcopy(state.model)
            with torch.no_grad(): next(cloned.parameters()).add_(1)
        self.assertEqual(n.snapshot(state,'control',REQUEST),before)
        with self.assertRaisesRegex(ContractError,'diagnostic changed'):
            with n._preserve_training(state): state.history[0]['loss_hex']=2.0.hex()

    def test_train_cumulative_deadline_retains_failure_without_repair(self):
        state=self.fresh(); outputs={}; clock=iter([0.,121.])
        def no_update(state,stop_after): pass
        with patch.object(n,'state_for',return_value=state),patch.object(n,'monotonic',side_effect=lambda:next(clock)), \
                patch('trinite.training.run_steps',side_effect=no_update):
            with self.assertRaisesRegex(ContractError,'deadline'):
                n.train('control',17,REQUEST,self.admission,lambda name,raw:outputs.setdefault(name,raw))
        self.assertIn('failure.json',outputs); self.assertIn('failure.safetensors',outputs)
        self.assertNotIn('diagnostic-predictions.json',outputs)
        failure=parse_json(outputs['failure.json'],canonical=True)
        self.assertEqual(failure['completed_updates'],0)
        self.assertEqual(failure['outcome'],'incomplete')


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.models={cell:FakeModel() for cell in CELLS}
        self.models[CELLS[0]].child.training=False
        self.gold={r['prompt']:[*r['answer'].encode(),257] for r in diagnostic_rows()+development_rows()}
        self.events=[]

    def greedy(self,model,prompt,cap):
        self.assertIs(type(prompt),str);self.assertEqual(cap,64)
        tokens=self.gold[prompt]
        for _ in tokens:model(torch.tensor([1.]))
        return tokens[:]

    def run_fake(self,stage='development',models=None,decode=None,emit=None):
        with ExitStack() as stack:
            for target in ('trinite.training.create_optimizer','trinite.training.run_steps','trinite.training.evaluate'):
                stack.enter_context(patch(target,side_effect=AssertionError('training forbidden')))
            stack.enter_context(patch('trinite.comparison.greedy',side_effect=decode or self.greedy))
            return n.infer_models(self.models if models is None else models,stage,emit or self.events.append)

    def test_all_four_fixed_order_and_guards_no_fit_gate(self):
        modes={k:[m.training for m in model.modules()] for k,model in self.models.items()}
        rng=torch.get_rng_state().clone();py_rng=random.getstate()
        def wrong(model,prompt,cap):
            for _ in range(2):model(torch.tensor([1.]))
            return [48,257]
        result=self.run_fake(decode=wrong)
        self.assertEqual(result['decodes'],480);self.assertEqual(result['forward_calls'],960)
        self.assertEqual([(r['arm'],r['seed']) for r in self.events], [cell for cell in CELLS for _ in range(120)])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(py_rng,random.getstate())
        self.assertEqual(modes,{k:[m.training for m in model.modules()] for k,model in self.models.items()})

    def test_common_diagnostic_exact80(self):
        result=self.run_fake('diagnostic',{CELLS[0]:self.models[CELLS[0]]})
        self.assertEqual(result['decodes'],80)
        self.assertEqual(result['forward_calls'],160)
        self.assertTrue(all(r['stage']=='diagnostic' for r in self.events))

    def test_missing_extra_shared_models_rejected_before_calls(self):
        for models in ({CELLS[0]:self.models[CELLS[0]]}, {**self.models,('extra',17):FakeModel()},
                       {cell:self.models[CELLS[0]] for cell in CELLS}):
            with self.assertRaises(ContractError):self.run_fake(models=models)
        self.assertEqual(self.events,[])

    def test_mutation_of_other_model_from_callback_stops_next_call(self):
        calls=[]
        def emit(record):
            calls.append(record)
            with torch.no_grad():self.models[CELLS[-1]].weight.add_(1)
        with self.assertRaisesRegex(ContractError,'mutated model'):
            self.run_fake(emit=emit)
        self.assertEqual(len(calls),1)

    def test_callback_cannot_rewrite_generated_tokens(self):
        def rewrite(record): record['generated_tokens']=[48,257]
        with self.assertRaisesRegex(ContractError,'changed raw prediction'):
            self.run_fake(emit=rewrite)

    def test_rng_mode_gradient_buffer_mutation_rejected(self):
        for what in ('rng','mode','gradient','buffer'):
            with self.subTest(what=what):
                self.setUp()
                def bad(model,prompt,cap):
                    value=self.greedy(model,prompt,cap)
                    if what=='rng':random.random()
                    if what=='mode':model.child.train(not model.child.training)
                    if what=='gradient':model.weight.grad=torch.ones_like(model.weight)
                    if what=='buffer':model.marker.add_(1)
                    return value
                with self.assertRaises(ContractError):self.run_fake(decode=bad)
                self.assertEqual(self.events,[])

    def test_bad_tokens_and_forward_counts_rejected(self):
        for tokens in ([True,257],[55],[257,55],[55,257]):
            with self.subTest(tokens=tokens):
                self.setUp()
                with self.assertRaises(ContractError):self.run_fake(decode=lambda *args:tokens)
                self.assertEqual(self.events,[])

    def test_callback_error_restores_modes_and_rng(self):
        modes={k:[m.training for m in model.modules()] for k,model in self.models.items()}
        rng=torch.get_rng_state().clone()
        def fail(record): raise RuntimeError('journal failed')
        with self.assertRaisesRegex(RuntimeError,'journal failed'):self.run_fake(emit=fail)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(modes,{k:[m.training for m in model.modules()] for k,model in self.models.items()})

if __name__=='__main__':unittest.main()
