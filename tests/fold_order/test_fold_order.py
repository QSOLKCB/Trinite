from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite import fold_order as f
from trinite.scalar_training import state_for as scalar_state, plan_for as scalar_plan
from trinite.training import run_steps, model_identity

ROOT = Path(__file__).resolve().parents[2]


class FoldOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

    def tiny(self, ordering='parent', steps=2):
        return replace(f.plan_for(0,ordering), steps=steps, validation_every=steps,
            model=ModelConfig(context_length=64,blocks=1,width=16,heads=2,feed_forward_width=32))

    def test_control_matches_scalar_state_and_updates(self):
        plan=self.tiny(); state=f.state_for(0,'parent',plan)
        prior=scalar_state('fold',0,'dense',replace(scalar_plan(0),steps=2,validation_every=2,model=plan.model))
        self.assertEqual(state.data,prior.data)
        self.assertEqual(state.initialization_identity,prior.initialization_identity)
        run_steps(state); run_steps(prior)
        self.assertEqual(state.history,prior.history)
        self.assertEqual(model_identity(state.model),model_identity(prior.model))

    def test_equal_complete_exposure_and_independent_mixed_order(self):
        import hashlib
        from collections import Counter
        from trinite.scalar_training import training_rows
        for seed in (0,1,2):
            parent=f.plan_for(seed,'parent');mixed=f.plan_for(seed,'mixed')
            original=training_rows('fold',parent)[2]
            state=f.state_for(seed,'mixed',replace(mixed,model=self.tiny().model))
            expected=sorted(original,key=lambda r:(
                hashlib.sha256(json_bytes({'seed':seed,'example_identity':r['identity']})).hexdigest(),r['identity']))
            self.assertEqual(state.data['train'],expected)
            self.assertNotEqual(original,expected)
            self.assertIs(state.data['validation'],state.data['train'])
            self.assertEqual(Counter(r['identity'] for r in expected),Counter(r['identity'] for r in original))
            visits=[];targets=[]
            for rows in (original,expected):
                used=[rows[i%len(rows)] for i in range(parent.steps*parent.batch_size)]
                visits.append(Counter(r['identity'] for r in used))
                targets.append(sum(sum(r['loss']) for r in used))
            self.assertEqual(visits[0],visits[1]);self.assertEqual(set(visits[0].values()),{8})
            self.assertEqual(len(visits[0]),3450);self.assertEqual(targets[0],targets[1])
            self.assertLessEqual(targets[0],parent.max_target_tokens)

    def test_checkpoint_replay_and_wrong_exposure_rejection(self):
        plan=self.tiny('mixed',4); req='sha256:'+'a'*64
        full=f.state_for(0,'mixed',plan); run_steps(full)
        prefix=f.state_for(0,'mixed',plan); run_steps(prefix,stop_after=2)
        payload,meta=f.snapshot(prefix,req)
        replay=f.restore(payload,meta,0,'mixed',req,(identity(payload),identity(meta)),plan)
        run_steps(replay)
        self.assertEqual(f.snapshot(full,req),f.snapshot(replay,req))
        with self.assertRaises(ContractError):
            f.restore(payload,meta,0,'parent',req,(identity(payload),identity(meta)),self.tiny('parent',4))
        r=parse_json(meta);r['history'][0]['examples'][0]='sha256:'+'0'*64;bad=json_bytes(r)
        with self.assertRaises(ContractError): f.restore(payload,bad,0,'mixed',req,(identity(payload),identity(bad)),plan)

    def test_fixed_protocol_and_admission_guards(self):
        self.assertEqual(identity((ROOT/'src/trinite/fold-order-protocol.json').read_bytes()),f.PROTOCOL_IDENTITY)
        self.assertEqual(len(f.cells()),6)
        with patch.object(f,'dataset',return_value=(b'{}\n',b'{}\n')):
            with self.assertRaises(ContractError):f.request_core()
        for bad in (True,3,'unknown'):
            with self.assertRaises(ContractError):replace(f.plan_for(0,'parent'),ordering=bad)

    def test_request_is_exclusive_and_source_bound(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'request.json';r=f.freeze_request(p);f.read_request(p,r['request_identity'])
            with self.assertRaises(FileExistsError):f.freeze_request(p)
            with patch.object(f,'sources',return_value={}):
                with self.assertRaises(ContractError):f.read_request(p,r['request_identity'])

    def test_all_candidates_retained_and_never_unlock(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'request.json';req=f.freeze_request(p)['request_identity']
            out=Path(d)/'out';out.mkdir();(out/'request.json').write_bytes(p.read_bytes())
            def cell(root,seed,ordering,request_identity):
                return {'seed':seed,'ordering':ordering,'outcome':'completed',
                    'report':{'training_gate_failures':[],'initialization_identity':'same','exposure':{},'target_tokens':100}}
            with patch.object(f,'verify_cell',side_effect=cell):r=f.summarize(out,p,req,{})
            self.assertEqual(len(r['cells']),6);self.assertEqual(r['outcome'],'completed')
            self.assertIsNone(r['selected_candidate'])
            for name in ('held_out_scored','dense_learning_adequate','format_selection_eligible','phase5_ready','release_ready'):
                self.assertFalse(r[name])
            with patch.object(f,'verify_cell',side_effect=cell):r=f.summarize(out,p,req,{f.cell_name(0,'mixed'):'timeout'},publish=False)
            self.assertEqual(len(r['cells']),6);self.assertEqual(r['outcome'],'failed');self.assertEqual(r['groups'],{})
            forbidden=out/f.cell_name(0,'parent');forbidden.mkdir();(forbidden/'test-predictions.json').write_bytes(b'{}')
            with self.assertRaises(ContractError):f.summarize(out,p,req,{},publish=False)

    def test_pairing_rejects_initialization_visit_and_target_mismatches(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'request.json';req=f.freeze_request(p)['request_identity']
            out=Path(d)/'out';out.mkdir();(out/'request.json').write_bytes(p.read_bytes())
            for field,value in [('initialization_identity','different'),('exposure',{'changed':True}),('target_tokens',101)]:
                def cell(root,seed,ordering,request_identity):
                    report={'training_gate_failures':[],'initialization_identity':'same','exposure':{},'target_tokens':100}
                    if seed==0 and ordering=='mixed':report[field]=value
                    return {'seed':seed,'ordering':ordering,'outcome':'completed','report':report}
                with self.subTest(field=field),patch.object(f,'verify_cell',side_effect=cell):
                    result=f.summarize(out,p,req,{},publish=False)
                    self.assertEqual(result['outcome'],'failed');self.assertEqual(result['groups'],{})
                    self.assertEqual(len(result['pairing_errors']),1)
                    self.assertIsNone(result['selected_candidate']);self.assertFalse(result['phase5_ready'])

    def test_runner_bounded_argv_and_workflow_envelope(self):
        spec=importlib.util.spec_from_file_location('fold_runner',ROOT/'scripts/run_fold_order.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        for stage,cell,seconds in [('train',(1,'mixed'),600),('summarize',None,1200),('verify',None,1200)]:
            with patch.object(runner.subprocess,'run') as process:
                runner.launch(stage,Path('r'), 'identity',Path('o'),None,cell)
                argv=process.call_args.args[0];options=process.call_args.kwargs
                self.assertEqual(argv[:4],[sys.executable,str(ROOT/'scripts/run_fold_order.py'),'--worker',stage])
                self.assertEqual(options['timeout'],seconds);self.assertFalse(options['shell'])
                self.assertEqual(options['env']['OMP_NUM_THREADS'],'1')
        workflow=(ROOT/'.github/workflows/fold-order.yml').read_text()
        minutes=int(next(line for line in workflow.splitlines() if 'timeout-minutes:' in line).split(':')[1])
        self.assertEqual(minutes,f.protocol()['resources']['workflow_minutes'])
        self.assertGreaterEqual(minutes*60,len(f.cells())*600+2*1200+1200)
        self.assertIn('if: always()',workflow)

    def test_verifier_binds_checkpoint_and_regenerates_unique_training_predictions(self):
        # A real tiny training stage closes the actual upstream bundle. Verification
        # uses the same bounded tiny plan solely as a software conformance fixture.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'request.json';req=f.freeze_request(p)['request_identity'];out=Path(d)/'cell'
            tiny=self.tiny('mixed',2)
            original=f.state_for
            with patch.object(f,'state_for',side_effect=lambda seed,ordering,plan=None:original(seed,ordering,tiny)):
                f.train_cell(out,0,'mixed',p,req)
                r=f.verify_cell(out,0,'mixed',req)
                self.assertEqual(r['report']['scores']['examples'],3450)
                report=parse_json((out/'training.json').read_bytes());report['scores']['correct']=3450
                (out/'training.json').write_bytes(json_bytes(report))
                with self.assertRaises(ContractError):f.verify_cell(out,0,'mixed',req)

    def test_runner_rejects_arbitrary_stage_and_selectors(self):
        spec=importlib.util.spec_from_file_location('fold_runner',ROOT/'scripts/run_fold_order.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for stage,cell in [('evaluate',None),('train',(0,3)),('summarize',(0,'parent'))]:
            with self.assertRaises(ValueError):module.launch(stage,'r','i','o',None,cell)

    def test_runner_retains_spawn_errors_and_attempts_every_cell(self):
        spec=importlib.util.spec_from_file_location('fold_runner',ROOT/'scripts/run_fold_order.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        for summary_spawn_fails in (False,True):
            with self.subTest(summary_spawn_fails=summary_spawn_fails), tempfile.TemporaryDirectory() as d:
                root=Path(d);request=root/'request.json';req=f.freeze_request(request)['request_identity']
                output=root/'run';attempts=[]
                failed_cell=f.cells()[0];failed_name=f.cell_name(*failed_cell)
                def run(argv, **kwargs):
                    stage=argv[argv.index('--worker')+1]
                    cell=(int(argv[argv.index('--seed')+1]),argv[argv.index('--ordering')+1]) if stage=='train' else None
                    attempts.append((stage,cell))
                    if cell==failed_cell:
                        raise OSError('training spawn unavailable')
                    if stage=='summarize':
                        if summary_spawn_fails: raise OSError('summary spawn unavailable')
                        errors=parse_json((output/'worker-errors.json').read_bytes(),canonical=True)
                        f.summarize(output,request,req,errors)
                    return subprocess.CompletedProcess(argv,0)
                def verified(root,seed,ordering,request_identity):
                    return {'seed':seed,'ordering':ordering,'outcome':'completed',
                        'report':{'training_gate_failures':[],'initialization_identity':'same','exposure':{},'target_tokens':100}}
                argv=['run_fold_order.py','--request',str(request),'--request-identity',req,'--output',str(output)]
                with patch.object(sys,'argv',argv),patch.object(runner.subprocess,'run',side_effect=run), \
                        patch.object(f,'verify_cell',side_effect=verified),patch('builtins.print'):
                    self.assertEqual(runner.main(),1)
                self.assertEqual(attempts,[('train',cell) for cell in f.cells()]+[('summarize',None)])
                self.assertEqual((output/'request.json').read_bytes(),request.read_bytes())
                self.assertEqual(len(list(output.glob('*.log'))),7)
                errors={failed_name:'OSError: training spawn unavailable'}
                self.assertEqual(parse_json((output/'worker-errors.json').read_bytes(),canonical=True),errors)
                if summary_spawn_fails:
                    result=parse_json((output/'summary-worker-failure.json').read_bytes(),canonical=True)
                    errors['summary']='OSError: summary spawn unavailable'
                    self.assertFalse(result['integrity_verified'])
                else:
                    result=parse_json((output/'summary.json').read_bytes(),canonical=True)
                    self.assertEqual(len(result['cells']),6)
                    self.assertEqual([r['outcome'] for r in result['cells']],['failed']+['completed']*5)
                    self.assertEqual(result['groups'],{})
                    self.assertIsNone(result['selected_candidate'])
                self.assertEqual(result['outcome'],'failed')
                self.assertEqual(result['worker_errors'],errors)
                self.assertFalse(result['held_out_scored'])
                self.assertFalse(result['phase5_ready'])
