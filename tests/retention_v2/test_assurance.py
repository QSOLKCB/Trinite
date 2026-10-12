from pathlib import Path
import importlib.util
import json
import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import load,save

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
from trinite.contracts import ContractError,identity,json_bytes,parse_json
from trinite.retention_v2 import spec,evidence,experiment
from trinite.training import run_steps

ROOT=Path(__file__).resolve().parents[2]
loader=importlib.util.spec_from_file_location('retention_v2_runner',ROOT/'scripts/run_maths_retention_v2.py')
runner=importlib.util.module_from_spec(loader);loader.loader.exec_module(runner)


def bytes_map(root):
    return {str(p.relative_to(root)):identity(p.read_bytes()) for p in root.rglob('*') if p.is_file()}


class V2AssuranceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=Path(cls.temp.name)
        cls.request=cls.base/'request.json';cls.request_id=experiment.freeze_request(cls.request,'smoke')['request_identity']
        cls.original=cls.base/'original';cls.original.mkdir();(cls.original/'request.json').write_bytes(cls.request.read_bytes())
        evidence.write(cls.original/'worker-errors.json',{})
        for seed,profile in spec.cells():experiment.train_cell(cls.original/spec.cell_name(seed,profile),seed,profile,cls.request,cls.request_id)
        cls.verification=cls.base/'verification.json'
        cls.verification_id=experiment.freeze_verification_request(cls.verification,cls.original,cls.request,cls.request_id)['verification_request_identity']

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def copy_run(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        parent=Path(temp.name);root=parent/'run';shutil.copytree(self.original,root)
        return parent,root

    def verify(self,root,tier='replay',v=None,vid=None):
        return experiment.verify_cell(root/spec.cell_name(0,'pooled'),0,'pooled',self.request,self.request_id,
                                      v or self.verification,vid or self.verification_id,tier=tier)

    def reseal(self,cell):
        shutil.rmtree(cell/'provenance');(cell/'verification.json').unlink()
        _,raw,packs=experiment.read_request(self.request,self.request_id)
        evidence.seal(cell,spec.inputs(raw,packs))

    def rebind(self,parent,run):
        path=parent/'changed-verification.json'
        return path,experiment.freeze_verification_request(path,run,self.request,self.request_id)['verification_request_identity']

    def test_real_cell_inference_and_complete_replay_have_distinct_truthful_receipts(self):
        before=bytes_map(self.original)
        limited=self.verify(self.original,'inference')['verification']
        self.assertEqual(limited['outcome'],'passed');self.assertFalse(limited['checks']['complete_optimizer_replay'])
        self.assertIsNone(limited['replay_environment_identity'])
        full=self.verify(self.original)['verification']
        self.assertTrue(all(full['checks'].values()));self.assertFalse(full['learning_adequacy'])
        self.assertIsNotNone(full['replay_environment_identity']);self.assertEqual(before,bytes_map(self.original))

    def test_external_commitments_reject_root_mutation_before_resealing(self):
        _,run=self.copy_run();(run/spec.cell_name(0,'pooled')/'steps.json').write_bytes(b'{}\n')
        with self.assertRaises(experiment.VerificationFailure) as caught:self.verify(run)
        self.assertIn('external artifact commitment',str(caught.exception))
        self.assertFalse(any(caught.exception.receipt['checks'].values()))

    def test_valid_custody_still_rejects_wrong_root_copy_under_new_commitments(self):
        parent,run=self.copy_run();(run/spec.cell_name(0,'pooled')/'steps.json').write_bytes(b'{}\n')
        v,vid=self.rebind(parent,run)
        with self.assertRaises(experiment.VerificationFailure) as caught:self.verify(run,v=v,vid=vid)
        self.assertIn('root output copy',str(caught.exception))

    def test_coherently_resealed_wrong_predictions_fail_fresh_model_scoring(self):
        parent,run=self.copy_run();cell=run/spec.cell_name(0,'pooled')
        value=parse_json((cell/'predictions.json').read_bytes());row=value['maths-final-test']['predictions'][0]
        row['generated_tokens']=[257] if row['generated_tokens']!=[257] else [0,257]
        (cell/'predictions.json').write_bytes(json_bytes(value));self.reseal(cell);v,vid=self.rebind(parent,run)
        with self.assertRaises(experiment.VerificationFailure) as caught:self.verify(run,v=v,vid=vid)
        self.assertTrue(caught.exception.receipt['checks']['closed_custody'])
        self.assertFalse(caught.exception.receipt['checks']['final_predictions_and_loss'])

    def test_resealed_history_and_finite_moment_are_limited_inference_only_and_fail_replay(self):
        parent,run=self.copy_run();cell=run/spec.cell_name(0,'pooled')
        m=parse_json((cell/'metadata.json').read_bytes())
        for name in ('loss_hex','gradient_norm_hex'):m['history'][0][name]=(float.fromhex(m['history'][0][name])+1).hex()
        m['train_diagnostics'][0]['loss_hex']=(float.fromhex(m['train_diagnostics'][0]['loss_hex'])+1).hex()
        (cell/'metadata.json').write_bytes(json_bytes(m));(cell/'steps.json').write_bytes(json_bytes({'steps':m['history'],'train_diagnostics':m['train_diagnostics']}))
        tensors=load((cell/'tensors.safetensors').read_bytes());slot=next(k for k in tensors if k.endswith('.exp_avg'))
        tensors[slot]=tensors[slot]+.25;(cell/'tensors.safetensors').write_bytes(save(tensors));self.reseal(cell)
        with self.assertRaises(experiment.VerificationFailure):self.verify(run)  # original commitments still reject
        v,vid=self.rebind(parent,run)
        limited=self.verify(run,'inference',v,vid)['verification'];self.assertFalse(limited['checks']['complete_optimizer_replay'])
        before=bytes_map(run)
        with self.assertRaises(experiment.VerificationFailure) as caught:self.verify(run,v=v,vid=vid)
        self.assertIn('optimizer/history replay mismatch',str(caught.exception))
        self.assertTrue(caught.exception.receipt['checks']['final_predictions_and_loss'])
        self.assertFalse(caught.exception.receipt['checks']['complete_optimizer_replay'])
        self.assertEqual(before,bytes_map(run))

    def test_missing_and_failed_cells_remain_in_conservative_summary(self):
        parent,run=self.copy_run();name=spec.cell_name(1,'pooled');shutil.rmtree(run/name)
        (run/'worker-errors.json').write_bytes(json_bytes({name:'injected worker failure'}));v,vid=self.rebind(parent,run)
        result=experiment.summarize(run,self.request,self.request_id,v,vid,tier='replay')
        self.assertEqual(len(result['cells']),4);self.assertEqual(result['outcome'],'failed');self.assertFalse(result['contrasts'])
        self.assertFalse(result['phase5_ready']);self.assertEqual(sum(c['outcome']=='completed' for c in result['cells']),3)

    def test_pairing_mismatch_suppresses_contrasts_without_discarding_cells(self):
        result=experiment.summarize(self.original,self.request,self.request_id,self.verification,self.verification_id,tier='replay')
        records=parse_json(json_bytes(result['cells']));records[1]['report']['initialization_identity']='sha256:'+'0'*64
        changed=experiment.aggregate(records,{},self.request_id,self.verification_id,'replay')
        self.assertEqual(changed['outcome'],'failed');self.assertTrue(changed['pairing_errors'])
        self.assertEqual(len(changed['cells']),4);self.assertFalse(changed['contrasts'])

    def test_actual_read_only_cli_regenerates_full_replay_without_any_output_writes(self):
        before=bytes_map(self.original)
        r=subprocess.run([sys.executable,str(ROOT/'scripts/run_maths_retention_v2.py'),'--verify-only','--tier','replay',
             '--request',str(self.request),'--request-identity',self.request_id,'--output',str(self.original),
             '--verification-request',str(self.verification),'--verification-request-identity',self.verification_id],
             cwd=ROOT,capture_output=True,text=True,timeout=120)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(parse_json(r.stdout.encode())['outcome'],'completed')
        self.assertEqual(before,bytes_map(self.original))

    def test_context_profile_and_environment_mismatch_cannot_make_a_claim(self):
        parent,run=self.copy_run();cell=run/spec.cell_name(0,'pooled')
        m=parse_json((cell/'metadata.json').read_bytes());m['run_profile']='research';(cell/'metadata.json').write_bytes(json_bytes(m))
        self.reseal(cell);v,vid=self.rebind(parent,run)
        with self.assertRaises(experiment.VerificationFailure):self.verify(run,v=v,vid=vid)
        request=parent/'foreign-request.json';r=parse_json(self.request.read_bytes());r['environment']['machine']='foreign';request.write_bytes(json_bytes(r))
        with self.assertRaises(ContractError):experiment.read_request(request,identity(request.read_bytes()))
        with self.assertRaises(ContractError):spec.cell_name(True,'pooled')

    def test_real_observer_filesystem_failure_retains_completed_outputs_and_failure_receipt(self):
        parent,_=self.copy_run();root=parent/'failed-cell';seal=evidence.seal
        def fail_at_observer(root,inputs):
            (root/'provenance').write_bytes(b'not a directory');seal(root,inputs)
        with patch.object(evidence,'seal',side_effect=fail_at_observer):
            with self.assertRaises(OSError):experiment.train_cell(root,0,'pooled',self.request,self.request_id)
        self.assertTrue(all((root/n).exists() for n in evidence.OUTPUTS))
        self.assertFalse(parse_json((root/'failure.json').read_bytes())['phase5_ready'])

    def test_materialized_schedule_and_unique_diagnostics_match_v1_and_packs_are_immutable(self):
        from trinite.maths_retention import state_for as old
        a=spec.state_for(0,'rehearsal','smoke');b=old(0,'rehearsal',spec.plan_for(0,'smoke'))
        self.assertEqual(a.data,b.data);run_steps(a);run_steps(b)
        self.assertEqual(a.history,b.history);self.assertEqual(experiment.tensor_payload(a),experiment.tensor_payload(b))
        p=spec.corpora()
        with self.assertRaises(TypeError):p['maths'].examples[0]['answer']='changed'
        self.assertEqual(len({r['identity'] for r in spec.diagnostic_rows(a)}),2262)

    def test_fixed_worker_boundary_and_controller_keep_timeouts_spawn_errors_and_all_cells(self):
        with patch.object(runner.subprocess,'run') as launch:
            runner.launch('train',self.request,self.request_id,None,None,self.base/'unused','replay',None,(0,'pooled'))
            argv=launch.call_args.args[0];kwargs=launch.call_args.kwargs
            self.assertEqual(argv[:2],[sys.executable,str(ROOT/'scripts/run_maths_retention_v2.py')])
            self.assertIs(kwargs['shell'],False);self.assertEqual(kwargs['timeout'],1200)
            with self.assertRaises(ValueError):runner.launch('train',self.request,self.request_id,None,None,self.base/'unused','replay',None,(True,'pooled'))
        parent,_=self.copy_run();output=parent/'controller';v=parent/'controller-verification.json';calls=[]
        def fake(stage,req,rid,verification,vid,root,tier,log,cell=None):
            calls.append((stage,cell))
            if stage=='train':
                if cell==(0,'pooled'):raise subprocess.TimeoutExpired('fixed worker',1200)
                if cell==(0,'rehearsal'):raise OSError('spawn failed')
                shutil.copytree(self.original/spec.cell_name(*cell),root);return subprocess.CompletedProcess([],0)
            value=experiment.summarize(root,req,rid,verification,vid,tier=tier)
            evidence.write(root/('summary-'+tier+'.json'),value);return subprocess.CompletedProcess([],1)
        with patch.object(runner,'launch',side_effect=fake):self.assertEqual(runner.matrix(self.request,self.request_id,v,output,'replay'),1)
        result=parse_json((output/'summary-replay.json').read_bytes())
        self.assertEqual(len(result['cells']),4);self.assertEqual(len(result['worker_errors']),2)
        self.assertEqual([c for stage,c in calls if stage=='train'],list(spec.cells()))
        self.assertFalse(result['phase5_ready']);self.assertFalse(result['contrasts'])

    def test_summary_worker_timeout_preserves_four_unverified_records(self):
        parent,_=self.copy_run();output=parent/'controller-timeout';v=parent/'controller-timeout-verification.json'
        def fake(stage,*args,**kwargs):
            if stage=='train':return subprocess.CompletedProcess([],1)
            raise subprocess.TimeoutExpired('fixed summary',1200)
        with patch.object(runner,'launch',side_effect=fake):self.assertEqual(runner.matrix(self.request,self.request_id,v,output,'replay'),1)
        value=parse_json((output/'controller-failure.json').read_bytes())
        self.assertEqual(len(value['cells']),4);self.assertFalse(value['learning_adequacy']);self.assertFalse(value['phase5_ready'])

    def test_nonzero_summary_exit_rejects_even_a_completed_summary(self):
        for code in (1, -9):
            with self.subTest(returncode=code):
                parent,_=self.copy_run();output=parent/'controller-exit';v=parent/'exit-verification.json'
                def fake(stage,req,rid,verification,vid,root,tier,log,cell=None):
                    if stage=='train':
                        shutil.copytree(self.original/spec.cell_name(*cell),root)
                        return subprocess.CompletedProcess([],0)
                    evidence.write(root/('summary-'+tier+'.json'),{'outcome':'completed','contrasts':{}})
                    return subprocess.CompletedProcess([],code)
                with patch.object(runner,'launch',side_effect=fake):
                    self.assertEqual(runner.matrix(self.request,self.request_id,v,output,'replay'),1)
                value=parse_json((output/'controller-failure.json').read_bytes())
                self.assertEqual(value['outcome'],'failed');self.assertEqual(len(value['cells']),4)
                self.assertIn('summary worker exited '+str(code),value['error'])
                self.assertTrue(all(c['outcome']=='failed' for c in value['cells']))
                self.assertEqual(parse_json((output/'summary-replay.json').read_bytes())['outcome'],'completed')

    def test_matrix_workers_have_the_aggregate_per_cell_deadline(self):
        per_cell=spec.protocol()['worker_seconds']
        for stage in ('summarize','verify'):
            for tier in ('inference','replay'):
                with self.subTest(stage=stage,tier=tier),patch.object(runner.subprocess,'run') as launch:
                    runner.launch(stage,self.request,self.request_id,self.verification,self.verification_id,
                                  self.original,tier,None)
                    timeout=launch.call_args.kwargs['timeout']
                    self.assertEqual(timeout,len(spec.cells())*per_cell)
                    self.assertGreater(timeout,len(spec.cells())*spec.plan_for(0,'research').max_seconds)

    def test_summary_rechecks_earlier_cell_commitments_and_custody_after_later_replay(self):
        for mutation in ('report','custody'):
            with self.subTest(mutation=mutation):
                _,run=self.copy_run();first=run/spec.cell_name(*spec.cells()[0])
                verify=experiment.verify_cell;calls=[]
                def mutate_after_later_replay(root,seed,profile,*args,**kwargs):
                    value=verify(root,seed,profile,*args,**kwargs);calls.append((seed,profile))
                    if (seed,profile)==spec.cells()[1]:
                        path=(first/'report.json' if mutation=='report' else
                              next((first/'provenance/artifacts/sha256').iterdir()))
                        path.write_bytes(b'{}\n')
                    return value
                with patch.object(experiment,'verify_cell',side_effect=mutate_after_later_replay):
                    result=experiment.summarize(run,self.request,self.request_id,self.verification,
                                                self.verification_id,tier='replay')
                self.assertEqual(calls,list(spec.cells()))
                self.assertEqual(result['outcome'],'failed');self.assertEqual(len(result['cells']),4)
                self.assertFalse(result['contrasts']);self.assertEqual(result['cells'][0]['outcome'],'failed')
                receipt=result['cells'][0]['verification']
                self.assertEqual(receipt['outcome'],'failed');self.assertFalse(receipt['checks']['closed_custody'])
                self.assertTrue(receipt['error']);self.assertFalse(result['learning_adequacy'])

    def test_new_commitment_cannot_bless_a_wrong_root_request_copy(self):
        parent,run=self.copy_run()
        (run/'request.json').write_bytes(b'{}\n')
        v=parent/'wrong-root-verification.json'
        value=parse_json(self.verification.read_bytes())
        value['root_artifacts']['request.json']=identity((run/'request.json').read_bytes())
        v.write_bytes(json_bytes(value))
        with self.assertRaises(ContractError):
            experiment.summarize(run,self.request,self.request_id,v,identity(v.read_bytes()),tier='replay')

    def test_aggregate_cannot_promote_limited_assurance_to_full_replay(self):
        limited=self.verify(self.original,'inference')
        records=[]
        for seed,profile in spec.cells():
            row=dict(limited);row['seed']=seed;row['profile']=profile;records.append(row)
        with self.assertRaises(ContractError):
            experiment.aggregate(records,{},self.request_id,self.verification_id,'replay')

    def test_summary_spawn_failure_retains_four_records_and_cli_overlap_rejects(self):
        parent,_=self.copy_run();output=parent/'summary-spawn';v=parent/'spawn-verification.json'
        def fake(stage,*args,**kwargs):
            if stage=='train':return subprocess.CompletedProcess([],1)
            raise OSError('summary spawn failed')
        with patch.object(runner,'launch',side_effect=fake):
            self.assertEqual(runner.matrix(self.request,self.request_id,v,output,'replay'),1)
        value=parse_json((output/'controller-failure.json').read_bytes())
        self.assertEqual(len(value['cells']),4);self.assertIn('summary spawn',value['error'])
        self.assertEqual(runner.main(['--request',str(self.request),'--request-identity',self.request_id,
             '--output',str(self.base),'--verification-request',str(self.verification),'--verify-only',
             '--verification-request-identity',self.verification_id]),1)
