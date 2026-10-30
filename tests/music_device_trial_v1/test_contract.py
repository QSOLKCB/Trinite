"""Standard-library controller doubles, never evidence of CUDA compatibility."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from music_device_trial_v1 import spec
from music_device_trial_v1 import controller as c, workflow as w, request as q

REAL_LAUNCH=c.launch
REQUEST=json_bytes({'synthetic':'controller-contract-only'})
REQUEST_ID=identity(REQUEST)
ADMISSION='sha256:'+'a'*64
INITIAL='sha256:'+'b'*64
MODEL='sha256:'+'c'*64

def observations(backend,stage,correct=True):
    rows=spec.diagnostic_rows() if stage=='diagnostic' else spec.development_rows()
    return [{'backend':backend,'seed':17,'step':640,'stage':stage,
        'example_identity':r['example_identity'],'generated_tokens':[*((r['answer'] if correct else '9').encode()),257]} for r in rows]

class Model:
    identity=MODEL

class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.request=self.root/'request.json';self.request.write_bytes(REQUEST)
        self.output=self.root/'output';self.passed=True;self.calls=[]
        self.fake=types.ModuleType('music_device_trial_v1.numerical')
        self.fake.parity_probe=lambda:(b'synthetic-probe',{'passed':self.passed})
        self.fake.compare_probes=lambda payload,report:{'passed':report['passed'],'software_double':True}
        self.fake.model_identity=lambda model:model.identity
        self.fake.load_checkpoint=lambda *a,**k:Model()
        self.fake.train=self.train;self.fake.infer_models=self.infer
        self.stack=contextlib.ExitStack()
        self.stack.enter_context(patch.dict(sys.modules,{'music_device_trial_v1.numerical':self.fake}))
        for module in (c,w):self.stack.enter_context(patch.object(module,'validate',return_value={'admission_identity':ADMISSION}))
        self.stack.enter_context(patch.object(c,'launch',side_effect=self.launch))
    def tearDown(self):self.stack.close();self.temp.cleanup()
    def train(self,backend,request_id,admission,write,emit):
        records=observations(backend,'diagnostic')
        metadata={'step':640,'model_identity':MODEL,'initialization_identity':INITIAL}
        write('final.safetensors',b'synthetic-safe-reader-double')
        write('final-checkpoint.json',json_bytes(metadata))
        write('diagnostic-predictions.json',json_bytes(records))
        for r in records:emit(r)
        return {'backend':backend,'seed':17,'request_identity':request_id,'admission_identity':admission,
            'step':640,'target_tokens':10240,'initialization_identity':INITIAL,'model_identity':MODEL,
            'update_seconds_hex':(0.).hex(),'timing_hex':{k:(0.).hex() for k in spec.TIMING_PHASES},
            'diagnostic_greedy_seconds_hex':(0.).hex(),'diagnostic_decodes':80,'diagnostic_forward_calls':160}
    def infer(self,models,stage,emit):
        records=[r for b in spec.BACKENDS for r in observations(b,stage)]
        for r in records:emit(r)
        return w.generation_accounting(records)|{'records':records,'timing_hex':{b:(1.).hex() for b in spec.BACKENDS},'greedy_seconds_hex':{b:(0.).hex() for b in spec.BACKENDS}}
    def launch(self,action,root,request,request_id,expected_directory,*,backend=None,study=None,gate_id=None,probe_id=None):
        self.calls.append((action,backend));(root/'worker.log').write_bytes(b'synthetic controller test only\n')
        if action=='_probe':w.probe_worker(root,REQUEST,request_id,expected_directory)
        elif action=='_train':w.train_worker(root,REQUEST,request_id,backend,expected_directory,probe_id)
        else:w.development_worker(study,root,REQUEST,request_id,gate_id,expected_directory)
        return 0
    def run_packet(self):return c.run_matrix(self.output,self.request,REQUEST_ID)
    def test_complete_packet_and_readonly_audit(self):
        result=self.run_packet();self.assertEqual(result['outcome'],'completed')
        before=w.bounded(self.output)
        receipt=c.audit_matrix(self.output,REQUEST,REQUEST_ID,result['matrix_identity'])
        self.assertTrue(receipt['verified']);self.assertFalse(receipt['inference_replayed'])
        self.assertEqual(w.bounded(self.output),before)
        self.assertEqual(self.calls,[('_probe',None),('_train','cpu'),('_train','cuda'),('_development',None)])
    def test_failed_parity_never_starts_target(self):
        self.passed=False;result=self.run_packet()
        self.assertEqual(result['outcome'],'incomplete');self.assertEqual(self.calls,[('_probe',None)])
        self.assertEqual(c.audit_matrix(self.output,REQUEST,REQUEST_ID,result['matrix_identity'])['outcome'],'incomplete')
    def test_unsupported_probe_is_not_scientific_failure_or_fallback(self):
        with patch.object(c,'launch',return_value=2):result=self.run_packet()
        matrix=parse_json((self.output/'matrix.json').read_bytes(),canonical=True)
        self.assertEqual(matrix['probe']['outcome'],'unsupported')
        self.assertTrue(all(x['outcome']=='not_attempted' for x in matrix['cells'].values()))
        c.audit_matrix(self.output,REQUEST,REQUEST_ID,result['matrix_identity'])
    def test_later_training_failure_blocks_all_development(self):
        launch=self.launch
        def fail(action,*args,**kwargs):
            if kwargs.get('backend')=='cuda':return 1
            return launch(action,*args,**kwargs)
        with patch.object(c,'launch',side_effect=fail):result=self.run_packet()
        self.assertEqual(result['outcome'],'incomplete')
        self.assertNotIn(('_development',None),self.calls)
        c.audit_matrix(self.output,REQUEST,REQUEST_ID,result['matrix_identity'])
    def test_fit_is_not_a_gate(self):
        with patch(__name__+'.observations',side_effect=lambda b,s,correct=True:self.wrong_records(b,s)):
            result=self.run_packet()
        self.assertEqual(result['outcome'],'completed')
    @staticmethod
    def wrong_records(backend,stage):
        rows=spec.diagnostic_rows() if stage=='diagnostic' else spec.development_rows()
        return [{'backend':backend,'seed':17,'step':640,'stage':stage,'example_identity':r['example_identity'],'generated_tokens':[57,257]} for r in rows]
    def test_independent_identity_and_tamper(self):
        result=self.run_packet()
        with self.assertRaises(ValueError):c.audit_matrix(self.output,REQUEST,REQUEST_ID,'sha256:'+'0'*64)
        path=self.output/'cuda/report.json';path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaises(ValueError):c.audit_matrix(self.output,REQUEST,REQUEST_ID,result['matrix_identity'])
    def test_mutation_during_later_admission_prevents_first_decode(self):
        self.run_packet();gate=(self.output/'development-gate.json').read_bytes()
        loaded={b:{'files':w.bounded(self.output/b),'metadata':{'model_identity':MODEL},'model':Model(),'records':observations(b,'diagnostic')} for b in spec.BACKENDS}
        loaded['cpu']['model'].identity='changed'
        fresh=self.root/'fresh';fresh.mkdir();stat=fresh.stat()
        with patch.object(w,'admit_cells',return_value=loaded),patch.object(self.fake,'infer_models') as infer:
            with self.assertRaisesRegex(ValueError,'earlier admitted'):w.development_worker(self.output,fresh,REQUEST,REQUEST_ID,identity(gate),(stat.st_dev,stat.st_ino))
            infer.assert_not_called()
    def test_probe_files_and_report_are_independently_bound(self):
        result=self.run_packet();path=self.output/'probe/report.json';path.write_bytes(json_bytes({'passed':False}))
        with self.assertRaises(ValueError):c.audit_matrix(self.output,REQUEST,REQUEST_ID,result['matrix_identity'])
    def test_unexpected_extra_file_and_artifact_bounds(self):
        self.run_packet();(self.output/'extra').write_bytes(b'x')
        with self.assertRaises(ValueError):c.audit_matrix(self.output,REQUEST,REQUEST_ID,identity((self.output/'matrix.json').read_bytes()))
        p=self.root/'huge';p.mkdir();(p/'worker.log').write_bytes(b'x'*(1048576+1))
        with self.assertRaises(ValueError):w.bounded(p)
    def test_selector_and_authorization_rejection_before_configure(self):
        bad=[['run','--request','x','--output','y'],['preflight','--backend','cuda'],['freeze','--request','x','--authorize-run'],['audit','--request','x','--backend','cpu']]
        with patch.object(c,'configure') as configure,contextlib.redirect_stderr(io.StringIO()):
            for argv in bad:self.assertNotEqual(c.main(argv),0)
            configure.assert_not_called()
    def test_preflight_is_stdlib_only(self):
        with patch.object(c,'configure') as configure,contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(c.main(['preflight']),0)
        configure.assert_not_called();self.assertFalse(json.loads(out.getvalue())['hardware_tested'])

    def test_internal_training_requires_bound_passed_probe(self):
        self.passed=False;self.run_packet();root=self.output/'cpu';root.mkdir();st=root.stat()
        with patch.object(self.fake,'train') as train:
            with self.assertRaises((ValueError,OSError)):
                w.train_worker(root,REQUEST,REQUEST_ID,'cpu',(st.st_dev,st.st_ino),'sha256:'+'0'*64)
            train.assert_not_called()
        probe_id=identity((self.output/'probe/completion.json').read_bytes())
        with patch.object(self.fake,'train') as train:
            with self.assertRaisesRegex(ValueError,'parity gate'):
                w.train_worker(root,REQUEST,REQUEST_ID,'cpu',(st.st_dev,st.st_ino),probe_id)
            train.assert_not_called()
    def test_timing_and_raw_backend_pair_comparison(self):
        self.run_packet();m=parse_json((self.output/'matrix.json').read_bytes(),canonical=True)
        self.assertEqual(set(m['timing_comparison']['by_backend']),{'cpu','cuda'})
        self.assertEqual(m['summary']['raw_backend_pairs']['development']['canonical']['both_correct'],40)
        self.assertEqual(m['summary']['raw_backend_pairs']['development']['canonical']['identical_but_wrong'],0)
        self.assertEqual(set(m['timing_comparison']['overhead_attempt_seconds_hex']),{'probe','development'})
    def test_parity_mutation_during_late_admission_blocks_decode(self):
        self.run_packet();gate=(self.output/'development-gate.json').read_bytes()
        loaded={b:{'files':w.bounded(self.output/b),'metadata':{'model_identity':MODEL},'model':Model(),'records':observations(b,'diagnostic')} for b in spec.BACKENDS}
        def altered(*args,**kwargs):
            (self.output/'probe/report.json').write_bytes(b'changed')
            return loaded
        fresh=self.root/'fresh';fresh.mkdir();st=fresh.stat()
        with patch.object(w,'admit_cells',side_effect=altered),patch.object(self.fake,'infer_models') as infer:
            with self.assertRaisesRegex(ValueError,'parity packet'):
                w.development_worker(self.output,fresh,REQUEST,REQUEST_ID,identity(gate),(st.st_dev,st.st_ino))
            infer.assert_not_called()

    def test_late_gate_or_cell_file_mutation_blocks_decode(self):
        self.run_packet();gate=(self.output/'development-gate.json').read_bytes()
        for relative in ('development-gate.json','cpu/report.json'):
            path=self.output/relative;original=path.read_bytes()
            loaded={b:{'files':w.bounded(self.output/b),'metadata':{'model_identity':MODEL},'model':Model(),'records':observations(b,'diagnostic')} for b in spec.BACKENDS}
            def altered(*args,**kwargs):
                path.write_bytes(original+b' ');return loaded
            fresh=self.root/('fresh-'+path.name);fresh.mkdir();st=fresh.stat()
            with patch.object(w,'admit_cells',side_effect=altered),patch.object(self.fake,'infer_models') as infer:
                with self.assertRaises(ValueError):w.development_worker(self.output,fresh,REQUEST,REQUEST_ID,identity(gate),(st.st_dev,st.st_ino))
                infer.assert_not_called()
            path.write_bytes(original)
    def test_timed_out_probe_retained_without_target_retry(self):
        with patch.object(c,'launch',side_effect=subprocess.TimeoutExpired('synthetic',900)) as launch:
            result=self.run_packet()
        self.assertEqual(launch.call_count,1)
        matrix=parse_json((self.output/'matrix.json').read_bytes(),canonical=True)
        self.assertEqual(matrix['probe']['outcome'],'timed_out')
        c.audit_matrix(self.output,REQUEST,REQUEST_ID,result['matrix_identity'])

    def test_worker_argv_environment_and_log_bound(self):
        class Process:
            def __init__(self,data):self.stdout=io.BytesIO(data);self.killed=False
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def wait(self,timeout=None):return 0
            def kill(self):self.killed=True
        root=self.root/'launch';root.mkdir();st=root.stat();process=Process(b'bounded log')
        with patch.object(c.subprocess,'Popen',return_value=process) as popen,patch.dict(os.environ,{'PYTHONPATH':'hostile'}):
            code=REAL_LAUNCH('_train',root,self.request,REQUEST_ID,(st.st_dev,st.st_ino),backend='cuda',probe_id=INITIAL)
        self.assertEqual(code,0);args,kwargs=popen.call_args
        self.assertEqual(args[0][:3],[sys.executable,'-I','-B'])
        self.assertIn('--probe-identity',args[0]);self.assertFalse(kwargs['shell'])
        self.assertNotIn('PYTHONPATH',kwargs['env']);self.assertEqual(kwargs['env']['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
        self.assertEqual((root/'worker.log').read_bytes(),b'bounded log')
        root2=self.root/'overflow';root2.mkdir();st=root2.stat();process=Process(b'x'*(1048576+1))
        with patch.object(c.subprocess,'Popen',return_value=process):
            with self.assertRaisesRegex(ValueError,'log failure'):
                REAL_LAUNCH('_probe',root2,self.request,REQUEST_ID,(st.st_dev,st.st_ino))
        self.assertTrue(process.killed);self.assertEqual((root2/'worker.log').stat().st_size,1048576)

class SpecTests(unittest.TestCase):
    def test_fixed_counts_and_not_independent_subjects(self):
        self.assertEqual(spec.PROTOCOL['max_decodes'],400)
        self.assertEqual(spec.PROTOCOL['probe_updates'],2)
        self.assertEqual(spec.PROTOCOL['aggregate_updates'],1280)
        self.assertEqual(len({r['prompt'] for r in spec.diagnostic_rows()}),70)
    def test_exact_records_and_all_backends(self):
        d=[r for b in spec.BACKENDS for r in observations(b,'diagnostic')]
        e=[r for b in spec.BACKENDS for r in observations(b,'development')]
        result=spec.matrix_summary(d,e);self.assertEqual(result['decodes'],400)
        self.assertEqual(result['generated_tokens'],800)
        for mutation in (e[:-1],e[::-1]):
            with self.assertRaises(ValueError):spec.matrix_summary(d,mutation)
        e[0]['backend']='cuda'
        with self.assertRaises(ValueError):spec.matrix_summary(d,e)
    def test_extra_fields_caps_and_seed_election(self):
        for mutate in (lambda r:r.update(seed=29),lambda r:r.update(arm='control'),lambda r:r.update(generated_tokens=[54])):
            r=observations('cpu','diagnostic');mutate(r[0])
            with self.assertRaises(ValueError):spec.score_cell(r,'cpu','diagnostic')
    def test_wrong_outputs_are_retained(self):
        rows=observations('cpu','diagnostic',False)
        self.assertEqual(spec.score_cell(rows,'cpu','diagnostic')['scores']['raw']['exact_plus_eos'],0)

if __name__=='__main__':unittest.main()
