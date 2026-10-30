"""CPU synthetic mechanics, never CUDA or the target640 study."""
from copy import deepcopy
from dataclasses import replace
from unittest import mock
import unittest
import torch
from safetensors.torch import save,load
from trinite.contracts import ContractError,identity,json_bytes,parse_json
from trinite.model import Decoder
from trinite.training import TrainingState,create_optimizer,run_steps,evaluate as native_evaluate
from music_placement_v1.spec import plan_for,dataset,train_rows
from music_learning_v1.numerical import encoded
from music_device_trial_v1.adapter import DeviceDecoder,model_identity,evaluate
from music_device_trial_v2 import numerical as n
from music_device_trial_v1 import numerical as original

REQUEST='sha256:'+'1'*64


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def state(self,adapter=True):
        cfg=plan_for(17);rows=n._probe_examples()
        model=DeviceDecoder(cfg.model,17,'cpu') if adapter else Decoder(cfg.model,lane='dense',seed=17)
        return TrainingState(cfg,'dense',model,create_optimizer(model,cfg),{'train':rows,'validation':rows},
            'sha256:'+'2'*64,'sha256:'+'3'*64,'sha256:'+'4'*64,(evaluate if adapter else native_evaluate)(model,rows,8))

    def test_shared_mathematical_primitives_are_identical_function_objects(self):
        for name in ('update_once','_context','_progress','snapshot','_rng','_restore_rng',
                     '_same_rng','_probe_examples','compare_probes'):
            self.assertIs(getattr(n,name),getattr(original,name),name)
        self.assertEqual(n.SCHEMA,original.SCHEMA)
        self.assertIs(n.DeviceDecoder,original.DeviceDecoder)
        self.assertIs(n.evaluate,original.evaluate)
        self.assertIs(n.TOLERANCES,original.TOLERANCES)

    def test_runtime_boundary_dispatches_v2_without_rebinding_v1(self):
        from music_device_trial_v2 import runtime
        before = original._runtime
        with mock.patch.object(runtime,'require_environment',return_value={'version':2}) as gate:
            self.assertEqual(n._runtime('cpu'),{'version':2})
            gate.assert_called_once_with('cpu')
        self.assertIs(original._runtime,before)
        self.assertIsNot(n._runtime,original._runtime)

    def test_common_cpu_updater_exact_native_two_toy_steps(self):
        trial=self.state();native=self.state(False)
        trial.config=replace(trial.config,steps=2,validation_every=1,max_target_tokens=32)
        native.config=replace(native.config,steps=2,validation_every=1,max_target_tokens=32)
        for step in (1,2):n.update_once(trial);run_steps(native,stop_after=step)
        self.assertEqual(trial.history,native.history);self.assertEqual(trial.validation,native.validation)
        for (name,a),(_,b) in zip(trial.model.named_parameters(),native.model.named_parameters()):
            self.assertTrue(torch.equal(a,b),name)
            for slot in ('step','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(trial.optimizer.state[a][slot],native.optimizer.state[b][slot]))

    def test_synthetic_state_timing_segments_are_separate_and_no_compile(self):
        rows=n._probe_examples()
        with mock.patch.object(n,'_runtime',return_value={}),mock.patch.object(n,'_sync') as sync, mock.patch.object(n,'_context',return_value=b'{}'), mock.patch.object(n,'train_rows',return_value=[None]*80),mock.patch.object(n,'diagnostic_rows',return_value=[None]*80),mock.patch.object(n,'encoded',side_effect=lambda selected:[rows[i%8] for i in range(len(selected))]):
            state,timing=n._state('cpu','sha256:'+'5'*64)
        self.assertEqual(set(timing),{'cpu_initialization','model_transfer','initial_loss','compile','target_warmup'})
        self.assertEqual(timing['compile'],0.);self.assertEqual(timing['target_warmup'],0.)
        self.assertTrue(all(v>=0 for v in timing.values()));self.assertGreaterEqual(sync.call_count,3)
        self.assertEqual(state.step,0)

    def probe_packet(self):
        with mock.patch.object(n,'_runtime',return_value={}),mock.patch.object(n,'_sync'):
            tensors,report=n._probe_one('cpu')
        # Two copies certify comparator behavior only, never a CUDA execution.
        payload=save({backend+'.'+key:value.clone() for backend in n.BACKENDS for key,value in tensors.items()})
        reports={b:dict(report,backend=b) for b in n.BACKENDS}
        result={'schema':'trinite.music-device-probe.v1','by_backend':reports,
            'probe_input_identity':identity(json_bytes(n._probe_examples())),'tolerances':n.TOLERANCES,
            'budget':{'updates':2,'visits':16,'target_tokens':32,'forward_calls':2},'payload_identity':identity(payload)}
        return payload,result

    def test_probe_comparator_safe_structural_not_hardware_claim(self):
        payload,report=self.probe_packet()
        with mock.patch.object(n,'create_optimizer',side_effect=AssertionError('optimizer forbidden')):
            result=n.compare_probes(payload,report)
        self.assertTrue(result['passed']);self.assertFalse(result['optimizer_replayed'])
        self.assertEqual(len(n._probe_examples()[0]['ids'])-1,67)
        self.assertEqual(sum(sum(x['loss']) for x in n._probe_examples()),16)
        tensors=load(payload);tensors['cuda.logits'][0,0,0]+=1
        changed=save(tensors);r=dict(report,payload_identity=identity(changed))
        self.assertFalse(n.compare_probes(changed,r)['passed'])

    def test_probe_rejects_inventory_initial_slots_nonfinite_and_resealed_budget(self):
        payload,report=self.probe_packet()
        tensors=load(payload)
        for name,value in [('cuda.initial.token_embedding',None),('cuda.step.token_embedding',2.),('cuda.exp_avg_sq.token_embedding',-1.),('cuda.logits',float('nan'))]:
            changed={k:v.clone() for k,v in tensors.items()}
            if value is None:changed[name][0,0]+=1
            else:changed[name].fill_(value)
            raw=save(changed);r=dict(report,payload_identity=identity(raw))
            with self.assertRaises(ContractError):n.compare_probes(raw,r)
        r=deepcopy(report);r['budget']['updates']=3
        with self.assertRaises(ContractError):n.compare_probes(payload,r)
        with self.assertRaises(ContractError):n.compare_probes(b'pickle',dict(report,payload_identity=identity(b'pickle')))

    def test_parity_tolerance_uses_cpu_as_reference(self):
        payload,report=self.probe_packet();values=load(payload)
        values['cpu.gradient.token_embedding'].fill_(1.)
        values['cuda.gradient.token_embedding'].fill_(1.0010105)
        payload=save(values);report=dict(report,payload_identity=identity(payload))
        result=n.compare_probes(payload,report)
        self.assertFalse(result['metrics']['gradient.token_embedding']['passed'])

    def test_probe_rejects_numpy_cached_state_and_grad_mode_mutations(self):
        import numpy as np
        for mutation in ('numpy_cache','grad_mode'):
            original=np.random.get_state();grad=torch.is_grad_enabled()
            def fake(backend):
                if mutation=='numpy_cache':
                    old=np.random.get_state();np.random.set_state((old[0],old[1],old[2],1,1.25))
                else:torch.set_grad_enabled(False)
                return {},{}
            with mock.patch.object(n,'_runtime'),mock.patch.object(n,'_probe_one',side_effect=fake),mock.patch.object(torch.cuda,'get_rng_state',return_value=torch.zeros(4,dtype=torch.uint8)),mock.patch.object(torch.cuda,'set_rng_state'):
                with self.assertRaises(ContractError):n.parity_probe()
            self.assertEqual(np.random.get_state()[2:],original[2:]);self.assertEqual(torch.is_grad_enabled(),grad)

    def packet(self):
        state=self.state();n.update_once(state);data,receipt=dataset();state.dataset_identity=identity(data);state.manifest_identity=identity(receipt)
        state.initialization_identity=model_identity(DeviceDecoder(state.config.model,17,'cpu'))
        # Fabricated final structure tests safe reader, not a640-update claim.
        rows=encoded(train_rows('control'));state.step=640;state.cursor=5120;state.target_tokens=10240
        state.history=[{'step':i+1,'loss_hex':(1.).hex(),'gradient_norm_hex':(1.).hex(),'target_tokens':16,
                        'examples':[rows[(i*8+j)%80]['identity'] for j in range(8)]} for i in range(640)]
        state.validation=[{'step':i,'loss_hex':(1.).hex()} for i in range(64,641,64)]
        for slots in state.optimizer.state.values():slots['step'].fill_(640)
        return (*n.snapshot(state,'cpu',REQUEST),state.manifest_identity)

    def test_final_loader_no_optimizer_or_forward(self):
        payload,meta,admission=self.packet()
        with mock.patch.object(n,'_runtime',return_value={}),mock.patch.object(n,'create_optimizer',side_effect=AssertionError('no optimizer')),mock.patch.object(DeviceDecoder,'forward',side_effect=AssertionError('no forward')):
            model=n.load_checkpoint(payload,meta,'cpu',admission,REQUEST)
        self.assertEqual(model_identity(model),parse_json(meta)['model_identity'])

    def test_final_loader_rejects_progress_context_extra_slot_and_pickle(self):
        payload,meta,admission=self.packet()
        with mock.patch.object(n,'_runtime',return_value={}):
            for field,value in [('backend','cuda'),('step',639),('target_tokens',10241),('initialization_identity','sha256:'+'0'*64)]:
                r=parse_json(meta);r[field]=value
                with self.assertRaises(ContractError):n.load_checkpoint(payload,json_bytes(r),'cpu',admission,REQUEST)
            values=load(payload);values['extra']=torch.ones(1)
            with self.assertRaises(ContractError):n.load_checkpoint(save(values),meta,'cpu',admission,REQUEST)
            with self.assertRaises(ContractError):n.load_checkpoint(b'pickle',meta,'cpu',admission,REQUEST)

    def test_training_failure_retains_partial_not_retry(self):
        writes={};raw,receipt=dataset()
        state=self.state();state.manifest_identity=identity(receipt);state.dataset_identity=identity(raw)
        with mock.patch.object(n,'_state',return_value=(state,{'cpu_initialization':0.,'model_transfer':0.,'initial_loss':0.,'compile':0.,'target_warmup':0.})),mock.patch.object(n,'_runtime',return_value={}),mock.patch.object(n,'_sync'),mock.patch.object(n,'update_once',side_effect=RuntimeError('unsupported deterministic operation')) as update:
            with self.assertRaises(RuntimeError):n.train('cpu',REQUEST,identity(receipt),writes.__setitem__,lambda r:None)
        self.assertEqual(update.call_count,1);self.assertIn('failure.json',writes);self.assertIn('failure.safetensors',writes)
        self.assertEqual(parse_json(writes['failure.json'])['completed_updates'],0)

class InferenceGuardTests(unittest.TestCase):
    def model(self,backend):
        model=torch.nn.Module();model.weight=torch.nn.Parameter(torch.ones(1));model.backend=backend;model.config=plan_for(17).model
        return model

    def identity(self,model):
        return identity(json_bytes({'parameters':[[n,[float(x).hex() for x in p.detach().cpu().reshape(-1)]] for n,p in model.named_parameters()]}))

    def greedy(self,model,prompt,cap):
        self.assertIsInstance(prompt,str);self.assertEqual(cap,64)
        for _ in range(2):
            for hook in model._forward_pre_hooks.values():hook(model,())
        return [55,257]

    def run_guarded(self,models,stage,callback):
        # Explicit CPU process doubles; never a CUDA compatibility claim.
        with mock.patch.object(n,'_runtime',return_value={}),mock.patch.object(n,'_sync'),mock.patch.object(n,'model_identity',side_effect=self.identity),mock.patch.object(torch.cuda,'get_rng_state',return_value=torch.zeros(4,dtype=torch.uint8)),mock.patch.object(torch.cuda,'set_rng_state'),mock.patch.object(n,'greedy',side_effect=self.greedy):
            return n.infer_models(models,stage,callback)

    def test_complete80_prompt_only_diagnostics(self):
        result=self.run_guarded({'cpu':self.model('cpu')},'diagnostic',lambda r:None)
        self.assertEqual(result['decodes'],80);self.assertEqual(result['forward_calls'],160)
        self.assertGreaterEqual(float.fromhex(result['greedy_seconds_hex']['cpu']),0.)
        self.assertLessEqual(float.fromhex(result['greedy_seconds_hex']['cpu']),float.fromhex(result['timing_hex']['cpu']))

    def test_callback_model_mode_gradient_rng_and_record_mutations_stop(self):
        for mutation in ('weight','mode','gradient','rng','record'):
            model=self.model('cpu');calls=[]
            def emit(record):
                calls.append(record)
                if mutation=='weight':model.weight.data.add_(1)
                elif mutation=='mode':model.train()
                elif mutation=='gradient':model.weight.grad=torch.ones(1)
                elif mutation=='rng':torch.rand(1)
                else:record['generated_tokens'][0]=56
            with self.assertRaises(ContractError):self.run_guarded({'cpu':model},'diagnostic',emit)
            self.assertEqual(len(calls),1)

    def test_other_endpoint_mutation_stops_before_next_call(self):
        models={b:self.model(b) for b in n.BACKENDS};calls=[]
        def emit(record):calls.append(record);models['cuda'].weight.data.add_(1)
        with self.assertRaises(ContractError):self.run_guarded(models,'development',emit)
        self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
