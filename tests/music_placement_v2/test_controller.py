"""Small controller failures and integrity barriers, never target model calls."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from music_placement_v2 import controller as c, workflow as w
from music_placement_v2.spec import CELLS, cell_name
from trinite.contracts import json_bytes, identity

RID='sha256:'+'1'*64
CID='sha256:'+'2'*64


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.req=self.root/'request.json';self.req.write_bytes(b'{}\n')

    def test_cli_rejects_training_election_resume_expansion_and_irrelevant_fields(self):
        for flags in (['run','--seed','29'],['run','--arm','swap'],['preflight','--authorize-run'],['freeze','--output','x']):
            with contextlib.redirect_stderr(io.StringIO()),mock.patch.object(c,'configure',side_effect=AssertionError('must reject early')):
                self.assertEqual(c.main(flags),1)
        for flags in (['resume'],['run','--steps','1280'],['run','--best-seed']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):c.main(flags)
        with contextlib.redirect_stderr(io.StringIO()),mock.patch.object(c,'configure',side_effect=AssertionError('must reject early')):
            self.assertEqual(c.main(['run','--request',str(self.req),'--request-identity',RID,'--output',str(self.root/'out')]),1)

    def test_paths_normalize_without_following_aliases(self):
        alias=self.root/'alias';alias.symlink_to(self.root,target_is_directory=True)
        with mock.patch.object(c,'validate',return_value={}),self.assertRaisesRegex(ValueError,'non-link'):c.run_matrix(alias/'out',self.req,RID)
        with mock.patch.object(c,'validate',return_value={}),self.assertRaises(ValueError):c.run_matrix(self.req,self.req,RID)

    def test_failed_cell_suppresses_development_but_attempts_all_four(self):
        calls=[]
        def attempt(root,action,*args,**kwargs):
            calls.append((action,kwargs.get('arm'),kwargs.get('seed')))
            return {'outcome':'failed','completion_identity':None,'attempt_identity':CID,'files':{}}
        with mock.patch.object(c,'require_sealable',return_value=None),mock.patch.object(c,'validate',return_value={}),mock.patch.object(c,'attempt',side_effect=attempt),mock.patch.object(c,'admit_cells',side_effect=AssertionError('no gate')):
            result=c.run_matrix(self.root/'out',self.req,RID)
        self.assertEqual(result['outcome'],'incomplete');self.assertEqual(len(calls),4)
        matrix=json.loads((self.root/'out/matrix.json').read_bytes())
        self.assertIsNone(matrix['summary']);self.assertEqual(matrix['development']['outcome'],'not_attempted')
        self.assertEqual([(a,s) for _,a,s in calls],list(CELLS))

    def test_interruption_marks_remaining_cells_unattempted(self):
        with mock.patch.object(c,'require_sealable',return_value=None),mock.patch.object(c,'validate',return_value={}),mock.patch.object(c,'attempt',return_value={'outcome':'interrupted','completion_identity':None,'attempt_identity':CID,'files':{}}) as run:
            c.run_matrix(self.root/'out',self.req,RID)
        self.assertEqual(run.call_count,1)
        matrix=json.loads((self.root/'out/matrix.json').read_bytes())
        self.assertEqual(sum(r['outcome']=='not_attempted' for r in matrix['cells'].values()),3)

    def test_completed_cells_develop_even_when_fit_is_not_a_gate(self):
        calls=[]
        def attempt(root,action,*args,**kwargs):
            calls.append(action)
            return {'outcome':'completed','completion_identity':CID,'attempt_identity':CID,'files':{}}
        with mock.patch.object(c,'require_sealable',return_value=None),mock.patch.object(c,'validate',return_value={}),mock.patch.object(c,'attempt',side_effect=attempt),mock.patch.object(c,'admit_cells',return_value={}) as admit,mock.patch.object(c,'audit_development',return_value={'summary':{'fit':False}}):
            result=c.run_matrix(self.root/'out',self.req,RID)
        self.assertEqual(result['outcome'],'completed');self.assertEqual(calls,['_train']*4+['_development']);admit.assert_called_once()

    def test_failed_all_four_admission_stops_before_development(self):
        calls=[]
        def attempt(root,action,*args,**kwargs):
            calls.append(action);return {'outcome':'completed','completion_identity':CID,'attempt_identity':CID,'files':{}}
        with mock.patch.object(c,'require_sealable',return_value=None),mock.patch.object(c,'validate',return_value={}),mock.patch.object(c,'attempt',side_effect=attempt),mock.patch.object(c,'admit_cells',side_effect=ValueError('late cell invalid')):
            result=c.run_matrix(self.root/'out',self.req,RID)
        self.assertEqual(result['outcome'],'incomplete');self.assertEqual(calls,['_train']*4)
        self.assertFalse((self.root/'out/development-gate.json').exists())

    def test_escaping_setup_or_receipt_failure_leaves_unsealed_partial_bytes(self):
        real_attempt = c.attempt
        original_write = c.Store.write
        for position in (0, 2):
            for failure in ('setup', 'receipt', 'interrupt'):
                with self.subTest(position=position, failure=failure):
                    out = self.root / ('partial-' + str(position) + '-' + failure)
                    wanted = cell_name(*CELLS[position])
                    calls = []
                    def attempt(root, action, *args, **kwargs):
                        calls.append(root.name)
                        if root.name == wanted and failure == 'setup':
                            root.mkdir()
                            (root/'raw-partial.txt').write_bytes(b'preserve setup evidence')
                            raise OSError('synthetic setup failure')
                        return real_attempt(root, action, *args, **kwargs)
                    def write(store, name, raw):
                        if name == 'attempt.json' and Path(store.root).name == wanted:
                            if failure == 'receipt': raise OSError('synthetic receipt write failure')
                            if failure == 'interrupt': raise KeyboardInterrupt('outside launch interruption')
                        return original_write(store, name, raw)
                    with mock.patch.object(c, 'validate', return_value={}), \
                         mock.patch.object(c, 'attempt', side_effect=attempt), \
                         mock.patch.object(c, 'launch', side_effect=OSError('retained worker failure')), \
                         mock.patch.object(c.Store, 'write', write):
                        with self.assertRaisesRegex(ValueError, 'unsealed partial study') as caught:
                            c.run_matrix(out, self.req, RID)
                    self.assertEqual(calls, [cell_name(*x) for x in CELLS[:position+1]])
                    self.assertIsInstance(caught.exception.__cause__, (OSError, KeyboardInterrupt))
                    self.assertTrue((out/'request.json').is_file())
                    self.assertTrue((out/wanted).is_dir())
                    self.assertFalse((out/'matrix.json').exists())
                    self.assertFalse((out/'development').exists())
                    for earlier in CELLS[:position]:
                        self.assertTrue((out/cell_name(*earlier)/'attempt.json').is_file())
                    if failure == 'setup':
                        self.assertEqual((out/wanted/'raw-partial.txt').read_bytes(), b'preserve setup evidence')
                    before = w.bounded(out)
                    with mock.patch.object(c, 'validate', return_value={}):
                        with self.assertRaises(OSError): c.audit_matrix(out, self.req.read_bytes(), RID, CID)
                    self.assertEqual(w.bounded(out), before)

    def test_escaping_first_attempt_cli_nonzero_without_matrix_identity(self):
        out = self.root/'cli-partial'
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(c, 'configure'), mock.patch.object(c, 'validate', return_value={}), \
             mock.patch.object(c, 'attempt', side_effect=OSError('setup failed')), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = c.main(['run', '--request', str(self.req), '--request-identity', RID,
                           '--output', str(out), '--authorize-run'])
        self.assertEqual(code, 1)
        self.assertNotIn('matrix_identity', stdout.getvalue())
        self.assertIn('unsealed partial study', stderr.getvalue())
        self.assertFalse((out/'matrix.json').exists())

    def test_development_escaping_setup_or_receipt_failure_is_unsealed(self):
        original_attempt, original_write = c.attempt, c.Store.write
        for fault in ('before_setup','receipt'):
            out=self.root/('development-'+fault)
            def attempt(root, action, path, raw, rid, **kwargs):
                if action=='_development':
                    if fault=='before_setup':raise OSError('development setup failed before directory')
                    return original_attempt(root,action,path,raw,rid,**kwargs)
                # Explicit receipt-only doubles: no model, checkpoint or training claims.
                root.mkdir();name=root.name
                (root/'completion.json').write_bytes(b'{}\n')
                completion=identity(b'{}\n')
                receipt={'schema':'trinite.music-placement-attempt.v2','action':'_train','cell':name,
                    'request_identity':rid,'outcome':'completed','worker_exit':0,'completion_identity':completion,
                    'detail':'synthetic receipt double','wall_seconds_hex':(0.).hex()}
                payload=json_bytes(receipt);(root/'attempt.json').write_bytes(payload)
                return {'outcome':'completed','completion_identity':completion,'attempt_identity':identity(payload),'files':w.bounded(root)}
            def write(store,name,raw):
                if name=='attempt.json' and Path(store.root).name=='development':
                    raise OSError('development receipt write failed')
                return original_write(store,name,raw)
            with mock.patch.object(c,'validate',return_value={}), \
                 mock.patch.object(c,'attempt',side_effect=attempt), \
                 mock.patch.object(c,'admit_cells',return_value={}), \
                 mock.patch.object(c,'launch',side_effect=OSError('synthetic worker failure')), \
                 mock.patch.object(c.Store,'write',write):
                with self.assertRaisesRegex(ValueError,'unsealed partial study'):
                    c.run_matrix(out,self.req,RID)
            self.assertFalse((out/'matrix.json').exists())
            self.assertTrue((out/'development-gate.json').exists())
            self.assertEqual((out/'development').exists(),fault=='receipt')
            for cell in CELLS:self.assertTrue((out/cell_name(*cell)/'attempt.json').exists())

    def test_unknown_partial_root_is_not_dropped_or_sealed(self):
        out=self.root/'unknown-root';original_attempt=c.attempt
        def attempt(root,*args,**kwargs):
            result=original_attempt(root,*args,**kwargs)
            if root.name==cell_name(*CELLS[-1]):
                extra=root.parent/'unreturned-partial';extra.mkdir();(extra/'raw').write_bytes(b'preserve')
            return result
        with mock.patch.object(c,'validate',return_value={}), \
             mock.patch.object(c,'launch',side_effect=OSError('caught')), \
             mock.patch.object(c,'attempt',side_effect=attempt):
            with self.assertRaisesRegex(ValueError,'unrecorded root'):
                c.run_matrix(out,self.req,RID)
        self.assertFalse((out/'matrix.json').exists())
        self.assertEqual((out/'unreturned-partial/raw').read_bytes(),b'preserve')

    def test_mutated_root_request_is_retained_without_seal(self):
        out=self.root/'root-request';original_attempt=c.attempt
        def attempt(root,*args,**kwargs):
            result=original_attempt(root,*args,**kwargs)
            if root.name==cell_name(*CELLS[-1]):
                (root.parent/'request.json').write_bytes(b'{"changed":true}\n')
            return result
        with mock.patch.object(c,'validate',return_value={}), \
             mock.patch.object(c,'launch',side_effect=OSError('caught')), \
             mock.patch.object(c,'attempt',side_effect=attempt):
            with self.assertRaisesRegex(ValueError,'root request changed'):
                c.run_matrix(out,self.req,RID)
        self.assertFalse((out/'matrix.json').exists())
        self.assertEqual((out/'request.json').read_bytes(),b'{"changed":true}\n')
        self.assertEqual(self.req.read_bytes(),b'{}\n')

    def test_root_descriptor_closes_for_all_seal_phase_errors(self):
        original_store, original_bounded, original_json = c.Store, c.bounded, c.json_bytes
        for fault in ('seal_error','seal_interrupt','inventory','json'):
            with self.subTest(fault=fault):
                out=self.root/('close-'+fault);stores=[]
                class TrackingStore(original_store):
                    def __init__(self,*args,**kwargs):
                        super().__init__(*args,**kwargs);stores.append(self)
                def bounded(root):
                    if Path(root)==out and fault=='inventory':raise OSError('inventory failed')
                    return original_bounded(root)
                def encode(value):
                    if isinstance(value,dict) and value.get('schema')=='trinite.music-placement-matrix.v2' and fault=='json':
                        raise TypeError('matrix encoding failed')
                    return original_json(value)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(c,'validate',return_value={}))
                    stack.enter_context(mock.patch.object(c,'launch',side_effect=OSError('caught worker failure')))
                    stack.enter_context(mock.patch.object(c,'Store',TrackingStore))
                    stack.enter_context(mock.patch.object(c,'bounded',side_effect=bounded))
                    stack.enter_context(mock.patch.object(c,'json_bytes',side_effect=encode))
                    if fault in ('seal_error','seal_interrupt'):
                        error=OSError('seal failed') if fault=='seal_error' else KeyboardInterrupt('seal interrupted')
                        stack.enter_context(mock.patch.object(c,'require_sealable',side_effect=error))
                    expected=KeyboardInterrupt if fault=='seal_interrupt' else (TypeError if fault=='json' else ValueError if fault=='seal_error' else OSError)
                    with self.assertRaises(expected):c.run_matrix(out,self.req,RID)
                self.assertEqual(stores[0].root,out)
                with self.assertRaises(OSError):os.fstat(stores[0].fd)
                self.assertTrue((out/'request.json').exists())
                self.assertFalse((out/'matrix.json').exists())

    def test_caught_interruption_returns_all_records_and_audits_incomplete(self):
        out = self.root/'caught-interruption'
        with mock.patch.object(c, 'validate', return_value={}), \
             mock.patch.object(c, 'launch', side_effect=KeyboardInterrupt('caught worker interruption')):
            result = c.run_matrix(out, self.req, RID)
        with mock.patch.object(c, 'validate', return_value={}):
            audit = c.audit_matrix(out, self.req.read_bytes(), RID, result['matrix_identity'])
        self.assertTrue(audit['verified'])
        self.assertEqual(audit['outcome'], 'incomplete')
        self.assertEqual(set(json.loads((out/'matrix.json').read_bytes())['cells']),
                         {cell_name(*x) for x in CELLS})

    def test_timeout_attempt_is_retained_not_retried(self):
        with mock.patch.object(c,'launch',side_effect=subprocess.TimeoutExpired('worker',900)) as launch:
            result=c.attempt(self.root/'cell','_train',self.req,b'{}\n',RID,arm='control',seed=17)
        self.assertEqual(result['outcome'],'timed_out');self.assertIsNone(result['completion_identity']);launch.assert_called_once()
        receipt=json.loads((self.root/'cell/attempt.json').read_bytes());self.assertEqual(receipt['outcome'],'timed_out')

    def test_launch_is_fixed_isolated_cwd_and_absolute_request(self):
        out=self.root/'cell';out.mkdir();st=out.stat()
        class Process:
            stdout=io.BytesIO(b'worker output\n')
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def wait(self,timeout):self.timeout=timeout;return 0
            def kill(self):pass
        proc=Process()
        with mock.patch.object(c.subprocess,'Popen',return_value=proc) as popen:
            self.assertEqual(c.launch('_train',out,self.req,RID,(st.st_dev,st.st_ino),arm='control',seed=17),0)
        args,kwargs=popen.call_args;argv=args[0]
        self.assertEqual(argv[:3],[sys.executable,'-I','-B']);self.assertEqual(kwargs['cwd'],c.ROOT);self.assertFalse(kwargs['shell'])
        self.assertEqual(kwargs['env']['PYTHONDONTWRITEBYTECODE'],'1');self.assertNotIn('PYTHONPATH',kwargs['env'])
        self.assertEqual(argv[argv.index('--request')+1],str(self.req));self.assertEqual(proc.timeout,900)

    def test_oversized_worker_log_kills_and_caps_retained_bytes(self):
        out=self.root/'cell';out.mkdir();st=out.stat()
        class Process:
            stdout=io.BytesIO(b'x'*(1048576+1));killed=False
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def wait(self,timeout):return 0
            def kill(self):self.killed=True
        proc=Process()
        with mock.patch.object(c.subprocess,'Popen',return_value=proc),self.assertRaises(ValueError):
            c.launch('_train',out,self.req,RID,(st.st_dev,st.st_ino),arm='control',seed=17)
        self.assertTrue(proc.killed);self.assertEqual((out/'worker.log').stat().st_size,1048576)

    def test_gate_requires_all_cells_and_paired_initialization(self):
        gate=w.gate_value(RID,{cell_name(a,s):CID for a,s in CELLS})
        calls=[]
        def audit(root,raw,rid,arm,seed,cid,**kwargs):
            calls.append((arm,seed));return {'metadata':{'initialization_identity':str(seed)}}
        with mock.patch.object(w,'audit_cell',side_effect=audit):
            self.assertEqual(len(w.admit_cells(self.root,b'',RID,gate)),4)
        self.assertEqual(calls,list(CELLS))
        gate['cells'].pop('swap-29')
        with mock.patch.object(w,'audit_cell',side_effect=AssertionError('no load')),self.assertRaises(ValueError):w.admit_cells(self.root,b'',RID,gate)
        gate=w.gate_value(RID,{cell_name(a,s):CID for a,s in CELLS})
        def bad(*args,**kwargs):return {'metadata':{'initialization_identity':args[3]}}
        with mock.patch.object(w,'audit_cell',side_effect=bad),self.assertRaisesRegex(ValueError,'paired'):w.admit_cells(self.root,b'',RID,gate)

    def test_failed_matrix_audit_is_readonly_and_resealed_claims_rejected(self):
        out=self.root/'study'
        with mock.patch.object(c,'validate',return_value={}),mock.patch.object(c,'launch',side_effect=OSError('spawn unavailable')):
            result=c.run_matrix(out,self.req,RID)
        before=w.bounded(out)
        with mock.patch.object(c,'validate',return_value={}):
            checked=c.audit_matrix(out,self.req.read_bytes(),RID,result['matrix_identity'])
        self.assertEqual(checked['outcome'],'incomplete');self.assertEqual(w.bounded(out),before)
        obj=json.loads((out/'matrix.json').read_bytes());obj['fresh_confirmation']=True
        raw=json_bytes(obj);(out/'matrix.json').write_bytes(raw)
        with mock.patch.object(c,'validate',return_value={}),self.assertRaises(ValueError):c.audit_matrix(out,self.req.read_bytes(),RID,identity(raw))

    def test_old_matrix_schema_is_rejected_even_after_resealing(self):
        out=self.root/'old-schema'
        with mock.patch.object(c,'validate',return_value={}), mock.patch.object(c,'launch',side_effect=OSError('caught')):
            c.run_matrix(out,self.req,RID)
        matrix=json.loads((out/'matrix.json').read_bytes())
        matrix['schema']='trinite.music-placement-matrix.v1'
        raw=json_bytes(matrix);(out/'matrix.json').write_bytes(raw)
        with mock.patch.object(c,'validate',return_value={}):
            with self.assertRaisesRegex(ValueError,'context'):
                c.audit_matrix(out,self.req.read_bytes(),RID,identity(raw))

    def test_complete_synthetic_packets_recompute_and_audit_readonly(self):
        # Numerical operations are explicit doubles; all scoring/link/audit code is real.
        from music_placement_v1 import numerical as n
        from music_placement_v2.spec import diagnostic_rows, development_rows
        from music_placement_v2.reporting import score_cell
        request={'admission_identity':CID}
        def generated(arm,seed,stage):
            selected=diagnostic_rows() if stage=='diagnostic' else development_rows()
            return [{'arm':arm,'seed':seed,'step':640,'stage':stage,'example_identity':r['example_identity'],
                     'generated_tokens':[*r['answer'].encode('ascii'),257]} for r in selected]
        def train(arm,seed,rid,aid,write,emit):
            meta={'model_identity':identity((arm+str(seed)).encode()),'initialization_identity':identity(str(seed).encode()),'step':640}
            write('final.safetensors',b'synthetic safe-reader double');write('final-checkpoint.json',json_bytes(meta))
            rows=generated(arm,seed,'diagnostic')
            for row in rows:emit(row)
            write('diagnostic-predictions.json',json_bytes(rows))
            return {'arm':arm,'seed':seed,'request_identity':rid,'admission_identity':aid,'step':640,'target_tokens':10240,
                'initialization_identity':meta['initialization_identity'],'model_identity':meta['model_identity'],
                'update_seconds_hex':float(0).hex(),'diagnostic_summary':score_cell(rows,arm,seed,'diagnostic'),
                'diagnostic_decodes':80,'diagnostic_forward_calls':160}
        def loader(payload,meta,*args,**kwargs):
            value=json.loads(meta);return SimpleNamespace(identity=value['model_identity']),value
        def inference(models,stage,emit):
            rows=[r for arm,seed in CELLS for r in generated(arm,seed,stage)]
            for row in rows:emit(row)
            return dict(w.generation_accounting(rows),records=rows)
        def launch(action,root,path,rid,expected_directory,**kwargs):
            (root/'worker.log').write_bytes(b'')
            if action=='_train':w.train_worker(root,self.req.read_bytes(),rid,kwargs['arm'],kwargs['seed'],expected_directory)
            else:w.development_worker(kwargs['study'],root,self.req.read_bytes(),rid,kwargs['gate_id'],expected_directory)
            return 0
        with mock.patch.object(c,'validate',return_value=request),mock.patch.object(w,'validate',return_value=request), \
                mock.patch.object(c,'launch',side_effect=launch),mock.patch.object(n,'train',side_effect=train), \
                mock.patch.object(n,'verify_checkpoint',side_effect=loader),mock.patch.object(n,'infer_models',side_effect=inference), \
                mock.patch.object(w,'loaded_model_identity',side_effect=lambda model:model.identity):
            out=self.root/'complete';result=c.run_matrix(out,self.req,RID)
            self.assertEqual(result['outcome'],'completed')
            before=w.bounded(out)
            checked=c.audit_matrix(out,self.req.read_bytes(),RID,result['matrix_identity'])
            self.assertTrue(checked['verified']);self.assertEqual(w.bounded(out),before)
            (out/'development/predictions.jsonl').write_bytes(b'{}\n')
            with self.assertRaises(ValueError):c.audit_matrix(out,self.req.read_bytes(),RID,result['matrix_identity'])

    def test_later_cell_admission_mutation_stops_before_any_development(self):
        from music_placement_v1 import numerical as n
        for mutation in ('file','model','gate','source'):
            with self.subTest(mutation=mutation):
                root=self.root/mutation;root.mkdir()
                output=root/'development';output.mkdir();(output/'worker.log').write_bytes(b'')
                gate=w.gate_value(RID,{cell_name(a,s):CID for a,s in CELLS})
                gate_raw=json_bytes(gate);(root/'development-gate.json').write_bytes(gate_raw)
                models={cell_name(a,s):SimpleNamespace(identity=identity((a+str(s)).encode())) for a,s in CELLS}
                for name in models:
                    (root/name).mkdir();(root/name/'retained').write_bytes(b'original')
                first=cell_name(*CELLS[0])
                def admission(path,raw,rid,arm,seed,expected,**kwargs):
                    name=cell_name(arm,seed)
                    result={'model':models[name],'files':w.bounded(path),'records':[],
                            'metadata':{'initialization_identity':str(seed),'model_identity':models[name].identity}}
                    if (arm,seed)==CELLS[-1]:
                        if mutation=='file':(root/first/'retained').write_bytes(b'mutated')
                        elif mutation=='model':models[first].identity='changed'
                        elif mutation=='gate':(root/'development-gate.json').write_bytes(b'{}\n')
                    return result
                validations=[{},ValueError('source changed')] if mutation=='source' else None
                validate=mock.Mock(side_effect=validations) if validations else mock.Mock(return_value={})
                st=output.stat()
                with mock.patch.object(w,'validate',validate),mock.patch.object(w,'audit_cell',side_effect=admission), \
                        mock.patch.object(w,'loaded_model_identity',side_effect=lambda model:model.identity), \
                        mock.patch.object(n,'infer_models',side_effect=AssertionError('no development call')) as infer:
                    with self.assertRaises(ValueError):
                        w.development_worker(root,output,b'{}\n',RID,identity(gate_raw),(st.st_dev,st.st_ino))
                    infer.assert_not_called()
                self.assertFalse((output/'predictions.jsonl').exists())

    def test_all_retained_artifact_limits_are_enforced(self):
        for name, limit in [('failure.json',16*1024*1024), ('request.json',4*1024*1024),
                            ('final.safetensors',16*1024*1024)]:
            with self.subTest(name=name):
                out=self.root/('bound-'+name);out.mkdir()
                with (out/name).open('wb') as stream:
                    stream.truncate(limit+1)
                with self.assertRaises(ValueError):w.bounded(out)

    def test_failure_journal_remains_bounded(self):
        out=self.root/'out';out.mkdir();(out/'partial.jsonl').write_bytes(b'x'*(1048576+1))
        with self.assertRaises(ValueError):w.bounded(out)

    def test_launcher_missing_flags_before_pathlib(self):
        root=self.root/'launch';root.mkdir();script=root/'run.py'
        shutil.copyfile(c.ROOT/'scripts/run_music_placement_v2.py',script)
        marker=root/'executed';(root/'pathlib.py').write_text('open('+repr(str(marker))+',"w").write("x")\n')
        for flags in ([],['-I'],['-B']):
            result=subprocess.run([sys.executable,*flags,str(script)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('Use python -I -B',result.stderr)
            self.assertFalse(marker.exists());self.assertFalse(any(root.rglob('__pycache__')))

if __name__=='__main__':unittest.main()
