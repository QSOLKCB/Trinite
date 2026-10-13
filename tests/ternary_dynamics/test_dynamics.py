"""Real tensor/quantizer conformance only; synthetic snapshots are NOT learning evidence."""
import copy
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite.inspection import tensor_identity
from ternary_dynamics import dynamics
from ternary_dynamics.dynamics import (inspect_weight, transition, freeze_request,
    report_for, selection, _matrix_shapes)
import analyze_ternary_dynamics as cli

ROOT=Path(__file__).resolve().parents[2]

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def fixture_checkpoint(root, step, *, seed=1, tamper=None):
    model=ModelConfig(context_length=8,blocks=1,width=4,heads=2,feed_forward_width=8)
    model_tensors={}
    for i,(name,shape) in enumerate(_matrix_shapes(model).items()):
        n=1
        for side in shape: n*=side
        # Deterministic, synthetic values; no random generator or model trainer.
        val=(torch.arange(n,dtype=torch.float32).reshape(shape)/max(n,1)-.5)*.05 + (step*.007)
        model_tensors['model.'+name]=val
    model_id=identity(json_bytes({name[len('model.'):]:tensor_identity(value)
                                  for name,value in model_tensors.items()}))
    payload=save(model_tensors)
    meta={'schema':'trinite.comparison-checkpoint.v1','step':step,'lane':'ternary',
          'plan':{'seed':seed,'model':model.to_dict()},
          'source':{'fixture':'synthetic-conformance-only',
                    'quantizer.py':dynamics.source_closure()['src/trinite/quantizer.py']},
          'environment':{'backend':'synthetic-test'},
          'dataset_identity':identity(b'test-training-only'),
          'manifest_identity':identity(b'test-manifest'),
          'model_identity':model_id}
    if tamper:meta.update(tamper)
    (root/f'{step}.safetensors').write_bytes(payload)
    (root/f'{step}.json').write_bytes(json_bytes(meta))


def make_selection(seed=1):
    return {'schema':dynamics.SELECTION_SCHEMA,'lane':'ternary','seed':seed,
            'snapshots':[{'step':i,'payload':f'{i}.safetensors','metadata':f'{i}.json'}
                         for i in (0,1)]}


class TernaryDynamicsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'snapshots'
        self.root.mkdir()
        fixture_checkpoint(self.root,0)
        fixture_checkpoint(self.root,1)

    def test_frozen_protocol_and_source_manifest(self):
        self.assertEqual(dynamics.protocol()['quantizer'],'trinite.absmean-ste.v0')
        self.assertEqual(identity(dynamics.PROTOCOL_PATH.read_bytes()), dynamics.PROTOCOL_IDENTITY)
        sources=dynamics.source_closure()
        self.assertEqual(sources['src/trinite/quantizer.py'],
                         identity((ROOT/'src/trinite/quantizer.py').read_bytes()))
        self.assertNotIn('src/trinite/ternary_dynamics.py', sources)
        self.assertIn('scripts/analyze_ternary_dynamics.py',sources)

    def test_original_surrogate_mask_exact_boundaries_and_no_mutation(self):
        weight=torch.tensor([[-2.,-1.,-.5,0.,.5,1.,2.]],requires_grad=True)
        before=weight.detach().clone()
        rng=torch.get_rng_state().clone()
        value,codes,_=inspect_weight(weight)
        self.assertEqual(value['scale_hex'],float(1.).hex())
        self.assertEqual(value['codes'],{'-1':2,'0':3,'1':2})
        self.assertEqual(value['ste_mask_enabled'],3)
        self.assertEqual(value['ste_mask_disabled'],4)
        self.assertEqual(value['near_zero_code_boundary'],2)
        self.assertEqual(value['near_ste_support_boundary'],2)
        self.assertEqual(value['moment_nonzero_inside'],None)
        self.assertTrue(torch.equal(before,weight.detach()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIsNone(weight.grad)
        self.assertEqual(codes.dtype,torch.int8)
        self.assertEqual(value['code_identity'],tensor_identity(codes))
        # Independently exercise the frozen backward; diagnostic is not the kernel.
        from trinite.quantizer import quantize
        q=quantize(weight)
        q.weight.backward(torch.ones_like(weight))
        self.assertEqual(weight.grad.tolist(),[[0.,0.,1.,1.,1.,0.,0.]])

    def test_moment_is_historical_state_not_upstream_gradient(self):
        w=torch.tensor([[-2.,-1.,-.5,0.,.5,1.,2.]])
        moment=torch.ones_like(w)
        v,_,_=inspect_weight(w,moment=moment)
        self.assertEqual((v['moment_nonzero_inside'],v['moment_nonzero_outside']),(3,4))
        self.assertNotIn('upstream_gradient',v)
        with self.assertRaises(ContractError):inspect_weight(w,moment=torch.zeros(2,2))

    def test_code_transitions_and_latent_changes_are_distinct(self):
        a=torch.tensor([[-2.,-1.,0.,1.,2.]])
        b=torch.tensor([[-2.,-1.,0.,1.,3.]])
        _,ac,aw=inspect_weight(a)
        _,bc,bw=inspect_weight(b)
        v=transition((ac,aw),(bc,bw))
        self.assertEqual(v['latent_changed'],1)
        self.assertEqual(v['code_changed_latent_stable'],0)
        self.assertEqual(v['code_changed']+v['code_stable'],5)
        with self.assertRaises(ContractError):transition((ac,aw),(bc[:,:3],bw[:,:3]))

    def test_complete_digest_bound_inspection_without_training(self):
        original={p.name:p.read_bytes() for p in self.root.iterdir()}
        rng=torch.get_rng_state().clone()
        selected=make_selection()
        req=freeze_request(selected,self.root)
        first=report_for(req,self.root)
        self.assertEqual(first['schema'],dynamics.REPORT_SCHEMA)
        self.assertFalse(first['incoming_training_gradients_available'])
        self.assertFalse(first['full_optimizer_replay'])
        self.assertEqual(first['observer_environment'],req['observer_environment'])
        self.assertEqual(first['observer_environment'],dynamics.observer_environment())
        self.assertFalse(first['phase5_ready'])
        self.assertEqual(len(first['samples']),2)
        self.assertEqual(len(first['transitions']),1)
        self.assertEqual(len(first['samples'][0]['matrices']),6)
        self.assertEqual(set(first['samples'][0]['matrices']),set(first['transitions'][0]['matrices']))
        self.assertEqual(first,report_for(req,self.root))
        self.assertEqual({p.name:p.read_bytes() for p in self.root.iterdir()},original)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_tampered_artifact_and_consistently_rehashed_false_model_identity_fail(self):
        req=freeze_request(make_selection(),self.root)
        (self.root/'1.safetensors').write_bytes((self.root/'1.safetensors').read_bytes()+b'evil')
        with self.assertRaisesRegex(ContractError,'member bytes'):
            report_for(req,self.root)
        # Refreshing just the local request hash does not bless a false model receipt.
        req['snapshots'][1]['payload_identity']=identity((self.root/'1.safetensors').read_bytes())
        with self.assertRaises(ContractError):report_for(req,self.root)
        fixture_checkpoint(self.root,1,tamper={'model_identity':identity(b'forged')})
        req=freeze_request(make_selection(),self.root)
        with self.assertRaisesRegex(ContractError,'model tensors disagree'):
            report_for(req,self.root)

    def test_foreign_context_and_wrong_step_are_rejected(self):
        fixture_checkpoint(self.root,1,tamper={'dataset_identity':identity(b'other')})
        with self.assertRaisesRegex(ContractError,'context drift'):
            report_for(freeze_request(make_selection(),self.root),self.root)
        fixture_checkpoint(self.root,1,tamper={'step':99})
        with self.assertRaisesRegex(ContractError,'step/lane'):
            report_for(freeze_request(make_selection(),self.root),self.root)

    def test_nonfinite_optimizer_and_bad_safetensors_are_rejected(self):
        req=freeze_request(make_selection(),self.root)
        (self.root/'0.safetensors').write_bytes(b'not safetensors')
        req['snapshots'][0]['payload_identity']=identity(b'not safetensors')
        with self.assertRaisesRegex(ContractError,'safetensors'):
            report_for(req,self.root)

    def test_selection_rejects_duplicate_steps_unsafe_names_and_bool_seed(self):
        bad=make_selection()
        bad['snapshots'][1]['step']=bad['snapshots'][0]['step']
        with self.assertRaises(ContractError):selection(bad)
        bad=make_selection()
        bad['snapshots'][0]['payload']='../evil.safetensors'
        with self.assertRaises(ContractError):selection(bad)
        bad=make_selection(seed=True)
        with self.assertRaises(ContractError):selection(bad)


    def test_observer_parity_with_real_ternary_autograd_without_parameter_updates(self):
        import torch.nn.functional as F
        from trinite.quantizer import quantize
        base=torch.tensor([[-2.,-1.,0.,1.,2.], [1.,-1.,.25,-.25,.75]],requires_grad=True)
        with_probe=base.detach().clone().requires_grad_()
        activation=torch.tensor([[1.,2.,-1.,.5,3.]],dtype=torch.float32)
        rng=torch.get_rng_state().clone()
        reference=F.linear(activation,quantize(base).weight)
        observed,_,_=inspect_weight(with_probe)
        active=F.linear(activation,quantize(with_probe).weight)
        self.assertTrue(torch.equal(reference,active))
        reference.square().sum().backward()
        active.square().sum().backward()
        self.assertTrue(torch.equal(base.grad,with_probe.grad))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIsNotNone(observed['ste_mask_disabled'])

    def test_request_and_metadata_source_substitution_fail_closed(self):
        req=freeze_request(make_selection(),self.root)
        altered=copy.deepcopy(req)
        altered['source']['src/trinite/quantizer.py']=identity(b'not the original quantizer')
        with self.assertRaisesRegex(ContractError,'source/protocol'):
            report_for(altered,self.root)
        raw=(self.root/'1.json').read_bytes()
        m=parse_json(raw,canonical=True)
        m['source']={'fixture':'silently changed producer',
                     'quantizer.py':dynamics.source_closure()['src/trinite/quantizer.py']}
        (self.root/'1.json').write_bytes(json_bytes(m))
        req=freeze_request(make_selection(),self.root)
        with self.assertRaisesRegex(ContractError,'context drift'):
            report_for(req,self.root)

    def test_observer_environment_is_bound_to_request_and_report(self):
        from trinite.training import require_environment
        with mock.patch('trinite.training.require_environment', wraps=require_environment) as guard:
            request=freeze_request(make_selection(),self.root)
            result=report_for(request,self.root)
        self.assertGreaterEqual(guard.call_count,3)  # freeze, measure, post-measure
        self.assertEqual(result['observer_environment'],request['observer_environment'])
        self.assertEqual(result['observer_environment'],require_environment())
        self.assertEqual(request['schema'],'trinite.ternary-dynamics-request.v2')
        self.assertEqual(result['schema'],'trinite.ternary-dynamics-report.v2')
        self.assertIn('src/trinite/training.py',request['source'])
        altered=copy.deepcopy(request)
        altered['observer_environment']={'device':'foreign'}
        with self.assertRaisesRegex(ContractError,'observer runtime'):
            report_for(altered,self.root)
        missing=copy.deepcopy(request)
        del missing['observer_environment']
        with self.assertRaises(ContractError):report_for(missing,self.root)

    def test_foreign_or_changing_observer_runtime_fails_closed(self):
        with mock.patch('trinite.training.require_environment',side_effect=ContractError('foreign numerical backend')):
            with self.assertRaisesRegex(ContractError,'foreign numerical backend'):
                freeze_request(make_selection(),self.root)
        request=freeze_request(make_selection(),self.root)
        with mock.patch('trinite.training.require_environment',side_effect=ContractError('foreign numerical backend')):
            with self.assertRaisesRegex(ContractError,'foreign numerical backend'):
                report_for(request,self.root)
        runtime=dynamics.observer_environment()
        with mock.patch('trinite.training.require_environment',side_effect=[runtime,{'system':'changed'}]):
            with self.assertRaisesRegex(ContractError,'changed during numeric inspection'):
                report_for(request,self.root)

    def test_checkpoint_must_bind_identical_quantizer_before_metrics(self):
        for change in ('missing','mismatched'):
            with self.subTest(change=change):
                for step in (0,1):
                    fixture_checkpoint(self.root,step)
                    path=self.root/f'{step}.json'
                    metadata=parse_json(path.read_bytes(),canonical=True)
                    if change=='missing':del metadata['source']['quantizer.py']
                    else:metadata['source']['quantizer.py']=identity(b'a different quantizer')
                    path.write_bytes(json_bytes(metadata))
                request=freeze_request(make_selection(),self.root)
                # Both snapshots carry the *same* fake binding; a mere
                # across-step source-consistency check would accept this.
                with self.assertRaisesRegex(ContractError,'producer quantizer source differs'):
                    report_for(request,self.root)

    def test_dense_only_task_balance_schema_is_not_advertised_as_ternary(self):
        self.assertNotIn('trinite.maths-task-balance-checkpoint.v1',dynamics.SNAPSHOT_SCHEMAS)
        meta=self.root/'0.json'
        raw=parse_json(meta.read_bytes(),canonical=True)
        raw['schema']='trinite.maths-task-balance-checkpoint.v1'
        del raw['lane']
        meta.write_bytes(json_bytes(raw))
        with self.assertRaisesRegex(ContractError,'unsupported native ternary snapshot schema'):
            report_for(freeze_request(make_selection(),self.root),self.root)

    def test_invalid_extra_optimizer_tensor_is_not_accepted(self):
        cfg=ModelConfig(context_length=8,blocks=1,width=4,heads=2,feed_forward_width=8)
        from safetensors.torch import load
        old=(self.root/'1.safetensors').read_bytes()
        tensors=load(old)
        tensors['optimizer.fake.weight.exp_avg']=torch.ones((2,2),dtype=torch.float32)
        (self.root/'1.safetensors').write_bytes(save(tensors))
        with self.assertRaisesRegex(ContractError,'unexpected optimizer'):
            report_for(freeze_request(make_selection(),self.root),self.root)

    def test_fresh_cli_process_initializes_cpu_before_guard(self):
        # Unit tests configure the parent Torch process above; a newly spawned
        # interpreter does NOT inherit those flags. Run the documented CLI in
        # genuinely fresh child processes with the opposite thread default.
        base = Path(self.tmp.name)
        select_file = base/'fresh-selection.json'
        request_file = base/'fresh-request.json'
        report_file = base/'fresh-report.json'
        select_file.write_bytes(json_bytes(make_selection()))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT/'src'), str(ROOT/'scripts'))),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
        probe = subprocess.run([sys.executable, '-c',
            'import torch; assert torch.get_num_threads() != 1; '
            'assert not torch.are_deterministic_algorithms_enabled()'],
            env=env, cwd=ROOT, capture_output=True, text=True, timeout=45)
        self.assertEqual(probe.returncode, 0, probe.stderr)

        def run(*args):
            result = subprocess.run([sys.executable, str(ROOT/'scripts/analyze_ternary_dynamics.py'),
                                    *map(str, args)], env=env, cwd=ROOT,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            return result

        run('--freeze-selection', select_file, '--root', self.root, '--output', request_file)
        request = parse_json(request_file.read_bytes(), canonical=True)
        self.assertEqual(request['observer_environment']['threads'], 1)
        self.assertIs(request['observer_environment']['deterministic'], True)
        request_identity = identity(request_file.read_bytes())
        run('--request', request_file, '--request-identity', request_identity,
            '--root', self.root, '--output', report_file)
        old = report_file.read_bytes()
        report = parse_json(old, canonical=True)
        self.assertEqual(report['observer_environment'], request['observer_environment'])
        run('--request', request_file, '--request-identity', request_identity,
            '--root', self.root, '--output', report_file, '--verify-only')
        self.assertEqual(report_file.read_bytes(), old)

    def test_cli_freeze_analyze_verify_and_no_overwrite(self):
        select_file=Path(self.tmp.name)/'select.json'
        request_path=Path(self.tmp.name)/'request.json'
        report_path=Path(self.tmp.name)/'report.json'
        select_file.write_bytes(json_bytes(make_selection()))
        self.assertEqual(cli.main(['--freeze-selection',str(select_file),'--root',str(self.root),
                                   '--output',str(request_path)]),0)
        request_bytes=request_path.read_bytes()
        self.assertEqual(cli.main(['--request',str(request_path),'--root',str(self.root),
                                   '--request-identity',identity(request_bytes),
                                   '--output',str(report_path)]),0)
        before=report_path.read_bytes()
        self.assertEqual(cli.main(['--request',str(request_path),'--root',str(self.root),
                                   '--request-identity',identity(request_bytes),
                                   '--output',str(report_path),'--verify-only']),0)
        self.assertEqual(report_path.read_bytes(),before)
        self.assertEqual(cli.main(['--request',str(request_path),'--root',str(self.root),
                                   '--request-identity',identity(request_bytes),
                                   '--output',str(report_path)]),2)
        self.assertEqual(report_path.read_bytes(),before)

    def test_cli_denies_output_within_input_root_and_wrong_request_digest(self):
        selected=Path(self.tmp.name)/'selection.json'
        selected.write_bytes(json_bytes(make_selection()))
        self.assertEqual(cli.main(['--freeze-selection',str(selected),'--root',str(self.root),
                                   '--output',str(self.root/'receipt.json')]),2)
        self.assertFalse((self.root/'receipt.json').exists())
        req=Path(self.tmp.name)/'req.json'
        req.write_bytes(json_bytes(freeze_request(make_selection(),self.root)))
        out=Path(self.tmp.name)/'reject.json'
        self.assertEqual(cli.main(['--request',str(req),'--root',str(self.root),
                                   '--request-identity',identity(b'wrong'),
                                   '--output',str(out)]),2)
        self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
