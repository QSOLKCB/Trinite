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
from trinite.retention_v2 import evidence
from trinite.retention_v2.budget import spec,experiment
from trinite.training import run_steps

ROOT=Path(__file__).resolve().parents[2]
loader=importlib.util.spec_from_file_location('retention_v2_runner',ROOT/'scripts/run_maths_retention_budget.py')
runner=importlib.util.module_from_spec(loader);loader.loader.exec_module(runner)


def bytes_map(root):
    return {str(p.relative_to(root)):identity(p.read_bytes()) for p in root.rglob('*') if p.is_file()}


class BudgetAssuranceTests(unittest.TestCase):
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
        return experiment.verify_cell(root/spec.cell_name(0,'control'),0,'control',self.request,self.request_id,
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
        _,run=self.copy_run();(run/spec.cell_name(0,'control')/'steps.json').write_bytes(b'{}\n')
        with self.assertRaises(experiment.VerificationFailure) as caught:self.verify(run)
        self.assertIn('external artifact commitment',str(caught.exception))
        self.assertFalse(any(caught.exception.receipt['checks'].values()))

    def test_valid_custody_still_rejects_wrong_root_copy_under_new_commitments(self):
        parent,run=self.copy_run();(run/spec.cell_name(0,'control')/'steps.json').write_bytes(b'{}\n')
        v,vid=self.rebind(parent,run)
        with self.assertRaises(experiment.VerificationFailure) as caught:self.verify(run,v=v,vid=vid)
        self.assertIn('root output copy',str(caught.exception))

    def test_coherently_resealed_wrong_predictions_fail_fresh_model_scoring(self):
        parent,run=self.copy_run();cell=run/spec.cell_name(0,'control')
        value=parse_json((cell/'predictions.json').read_bytes());row=value['maths-final-train']['predictions'][0]
        row['generated_tokens']=[257] if row['generated_tokens']!=[257] else [0,257]
        (cell/'predictions.json').write_bytes(json_bytes(value));self.reseal(cell);v,vid=self.rebind(parent,run)
        with self.assertRaises(experiment.VerificationFailure) as caught:self.verify(run,v=v,vid=vid)
        self.assertTrue(caught.exception.receipt['checks']['closed_custody'])
        self.assertFalse(caught.exception.receipt['checks']['final_predictions_and_loss'])

    def test_resealed_history_and_finite_moment_are_limited_inference_only_and_fail_replay(self):
        parent,run=self.copy_run();cell=run/spec.cell_name(0,'control')
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
        parent,run=self.copy_run();name=spec.cell_name(1,'control');shutil.rmtree(run/name)
        (run/'worker-errors.json').write_bytes(json_bytes({name:'injected worker failure'}));v,vid=self.rebind(parent,run)
        result=experiment.summarize(run,self.request,self.request_id,v,vid,tier='replay')
        self.assertEqual(len(result['cells']),4);self.assertEqual(result['outcome'],'failed');self.assertFalse(result['contrasts'])
        self.assertFalse(result['phase5_ready']);self.assertEqual(sum(c['outcome']=='completed' for c in result['cells']),3)

    def test_real_complete_matrix_is_training_only_smoke_never_accepts_research_gate(self):
        before=bytes_map(self.original)
        result=experiment.summarize(self.original,self.request,self.request_id,self.verification,self.verification_id,tier='replay')
        self.assertEqual(result['outcome'],'completed')
        self.assertFalse(result['joint_training_accepted']);self.assertFalse(result['held_out_scored'])
        self.assertEqual(len(result['cells']),4)
        for cell in result['cells']:
            self.assertTrue(all(cell['verification']['checks'].values()))
            self.assertEqual(set(cell['report']['scores']),{'legacy-initial-train','legacy-final-train','maths-initial-train','maths-final-train'})
            for kind in ('legacy','maths'):
                self.assertEqual(cell['report']['exposure'][kind]['example_visits'],cell['report']['steps']*4)
        self.assertEqual(before,bytes_map(self.original))

    def test_control_matches_v1_numerical_bytes_and_candidate_prefix(self):
        from trinite.maths_retention import state_for as old_state
        from trinite.maths_plan import MathsPlan
        from trinite.comparison_checkpoint import tensor_payload
        for seed in (0,1):
            control=spec.state_for(seed,'control','smoke')
            common=control.config.to_dict();common['schema']='trinite.maths-plan.v1'
            old=old_state(seed,'rehearsal',MathsPlan.from_dict(common))
            candidate=spec.state_for(seed,'candidate','smoke')
            self.assertEqual(control.data['train'],candidate.data['train'][:len(control.data['train'])])
            self.assertEqual(control.data,old.data)
            run_steps(control);run_steps(old);run_steps(candidate,stop_after=control.config.steps)
            self.assertEqual(tensor_payload(control),tensor_payload(old))
            self.assertEqual(tensor_payload(control),tensor_payload(candidate))
            self.assertEqual(control.history,candidate.history)
            self.assertEqual(control.validation,candidate.validation)

    def test_read_only_cli_and_fixed_deadlines(self):
        before=bytes_map(self.original)
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_maths_retention_budget.py'),
            '--verify-only','--request',str(self.request),'--request-identity',self.request_id,
            '--verification-request',str(self.verification),'--verification-request-identity',self.verification_id,
            '--output',str(self.original)],capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(parse_json(result.stdout.encode())['outcome'],'completed')
        self.assertEqual(before,bytes_map(self.original))
        self.assertEqual(runner.worker_timeout('train'),2400)
        self.assertEqual(runner.worker_timeout('verify'),9600)
        with self.assertRaises(ValueError):runner.launch('train',None,None,None,None,None,'replay',None,(0,'pooled'))

    def test_fixed_worker_boundary_and_controller_keep_timeouts_spawn_errors_and_all_cells(self):
        with patch.object(runner.subprocess,'run') as launch:
            runner.launch('train',self.request,self.request_id,None,None,self.base/'unused','replay',None,(0,'control'))
            argv=launch.call_args.args[0];kwargs=launch.call_args.kwargs
            self.assertEqual(argv[:2],[sys.executable,str(ROOT/'scripts/run_maths_retention_budget.py')])
            self.assertIs(kwargs['shell'],False);self.assertEqual(kwargs['timeout'],2400)
            with self.assertRaises(ValueError):runner.launch('train',self.request,self.request_id,None,None,self.base/'unused','replay',None,(True,'control'))
        parent,_=self.copy_run();output=parent/'controller';v=parent/'controller-verification.json';calls=[]
        def fake(stage,req,rid,verification,vid,root,tier,log,cell=None):
            calls.append((stage,cell))
            if stage=='train':
                if cell==(0,'control'):raise subprocess.TimeoutExpired('fixed worker',2400)
                if cell==(0,'candidate'):raise OSError('spawn failed')
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
            raise subprocess.TimeoutExpired('fixed summary',9600)
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

    def test_lossless_archive_contains_external_requests_producer_and_all_members(self):
        import zipfile
        from trinite.retention_v2.budget.archive import archive
        parent,run=self.copy_run()
        (run/'verification-request.json').write_bytes(self.verification.read_bytes())
        result=experiment.summarize(run,self.request,self.request_id,self.verification,self.verification_id,tier='replay')
        evidence.write(run/'summary-replay.json',result)
        output=parent/'evidence.zip';before=bytes_map(run)
        receipt=archive(output,run,self.request,self.request_id,self.verification,self.verification_id)
        self.assertEqual(receipt['archive_identity'],identity(output.read_bytes()))
        self.assertEqual(before,bytes_map(run))
        with zipfile.ZipFile(output) as z:
            inv=parse_json(z.read('inventory.json'),canonical=True)
            self.assertEqual(set(z.namelist()),set(inv['members'])|{'inventory.json'})
            for name,item in inv['members'].items():
                raw=z.read(name);self.assertEqual(item,{'identity':identity(raw),'bytes':len(raw)})
            for name,key in before.items():self.assertEqual(identity(z.read('run/'+name)),key)
            self.assertEqual(z.read('external/verification-request.json'),self.verification.read_bytes())
            self.assertIn('source/scripts/run_maths_retention_budget.py',inv['members'])
        with self.assertRaises(FileExistsError):archive(output,run,self.request,self.request_id,self.verification,self.verification_id)

    def test_research_acceptance_requires_every_candidate_seed_task_and_full_replay(self):
        import copy
        reports=[];packs=spec.corpora()
        for seed,profile in spec.cells():
            source=parse_json((self.original/spec.cell_name(seed,profile)/'report.json').read_bytes())
            source['run_profile']='research';source['steps']=spec.plan_for(seed,'research',profile).steps
            for kind,pack in packs.items():
                counts=spec.expected_counts(spec.scoring_examples(pack,'research'),'train')
                tasks={task:{'examples':n,'correct':n,'baseline_correct':0} for task,n in counts.items()}
                score={'examples':sum(counts.values()),'correct':sum(counts.values()),'tasks':tasks}
                source['scores'][kind+'-final-train']=score
                source['scores'][kind+'-initial-train']=copy.deepcopy(score)
                source['exposure'][kind]['example_visits']=source['steps']*4
            source['training_gate_failures']={'legacy':[],'maths':[]}
            receipt=experiment.receipt_for(self.request_id,self.verification_id,{},'replay')
            receipt['checks']={k:True for k in receipt['checks']};receipt['outcome']='passed'
            reports.append({'seed':seed,'profile':profile,'outcome':'completed','report':source,'verification':receipt})
        result=experiment.aggregate(reports,{},self.request_id,self.verification_id,'replay')
        self.assertTrue(result['joint_training_accepted']);self.assertFalse(result['learning_adequacy'])
        for seed in (0,1):
            changed=copy.deepcopy(reports)
            report=next(r['report'] for r in changed if r['seed']==seed and r['profile']=='candidate')
            task=sorted(report['scores']['maths-final-train']['tasks'])[0]
            delta=report['scores']['maths-final-train']['tasks'][task]['correct']
            report['scores']['maths-final-train']['tasks'][task]['correct']=0
            report['scores']['maths-final-train']['correct']-=delta
            report['training_gate_failures']['maths']=[task]
            self.assertFalse(experiment.aggregate(changed,{},self.request_id,self.verification_id,'replay')['joint_training_accepted'])
            report['training_gate_failures']['maths']=[]
            with self.assertRaises(ContractError):experiment.aggregate(changed,{},self.request_id,self.verification_id,'replay')
        limited=copy.deepcopy(reports)
        for r in limited:
            r['verification']['tier']='inference';r['verification']['checks']['complete_optimizer_replay']=False
        self.assertFalse(experiment.aggregate(limited,{},self.request_id,self.verification_id,'inference')['joint_training_accepted'])
