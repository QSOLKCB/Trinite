from dataclasses import replace
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from trinite import learning as l
from trinite import learning_training as lt
from trinite.learning_checkpoint import snapshot, restore
from trinite.learning_plan import LearningPlan
from trinite.budget_plan import BudgetPlan
from trinite.foundations_plan import FoundationsPlan
from trinite.foundations_training import state_for as parent_state
from trinite.comparison_checkpoint import tensor_payload
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.observation import BundleObserver
from trinite.training import model_identity, run_steps

ROOT = Path(__file__).resolve().parents[2]
REQUEST = 'sha256:'+'0'*64
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def scores(correct=10):
    value = correct/10
    return {'examples':10, 'correct':correct,
            'tasks':{'task':{'examples':10,'correct':correct,'baseline_correct':0}},
            'families':{'family':value.hex()}, 'family_macro_accuracy_hex':value.hex()}


def fake_training(root, workload, seed, lane, request_identity):
    # Synthetic control records, never an empirical result or retained fixture.
    report = {'workload':workload,'seed':seed,'lane':lane,'outcome':'completed',
        'initialization_identity':str(seed), 'dataset_identity':workload, 'manifest_identity':workload,
        'plan_identity':str(seed), 'parameter_count':86528,'steps':4096,'target_tokens':21000,
        'schedule_identity':str(seed),'training_wall_seconds_hex':(1.).hex(),
        'latent_float32_bytes':346112,'scores':scores(),'training_gate_failures':[]}
    return {'report':report,'report_identity':identity(json_bytes(report)),
            'manifest_file_identity':identity(b'{}'),
            'verification':{'manifest_identity':'sha256:'+'1'*64}}


def fake_evaluation(root, w, s, lane, request_identity, row, decision):
    return {'scores':scores(), 'learning_adequate':True}, {'integrity_verified':True}


class LearningTests(unittest.TestCase):
    def short_plan(self):
        return replace(lt.plan_for(0), steps=4, validation_every=2)

    def test_separate_plan_bounds_and_original_profile_limits(self):
        plan = LearningPlan(); self.assertEqual(LearningPlan.from_dict(plan.to_dict()), plan)
        for key,value in (('steps',4097),('max_seconds',481),('max_target_tokens',524289),
                          ('steps',True),('batch_size',9),('schedule','linear'),
                          ('order','seeded-hash-rank-cycle.v1'),('schema','trinite.budget-training-plan.v1')):
            with self.subTest(key=key), self.assertRaises(ContractError): replace(plan, **{key:value})
        with self.assertRaises(ContractError): FoundationsPlan(steps=4096)
        with self.assertRaises(ContractError): lt.state_for('arithmetic',0,'dense',BudgetPlan())
        for cell in (('fold',True,'dense'),('unknown',0,'dense'),('fold',0,'qubit')):
            with self.assertRaises(ContractError): lt.cell_name(*cell)

    def test_admission_relabel_preserves_numeric_exposure_and_updates_all_lanes(self):
        config = self.short_plan()
        mapping = {r['example_identity']:r['ordering_identity'] for r in parse_json(l.dataset('arithmetic')[0])['examples']}
        for lane in lt.LANES:
            current = lt.state_for('arithmetic',0,lane,config)
            old = parent_state('arithmetic',lane,0,config)
            old.data = {'train':old.data['train'], 'validation':old.data['train']}
            self.assertIs(current.data['train'],current.data['validation'])
            self.assertEqual([(mapping[e['identity']],e['ids'],e['loss']) for e in current.data['train']],
                             [(e['identity'],e['ids'],e['loss']) for e in old.data['train']])
            self.assertEqual(current.initialization_identity,old.initialization_identity)
            run_steps(current);run_steps(old)
            self.assertEqual(tensor_payload(current),tensor_payload(old))
            normalized = [{**h,'examples':[mapping[e] for e in h['examples']]} for h in current.history]
            self.assertEqual(normalized,old.history);self.assertEqual(current.validation,old.validation)

    def test_exact_interrupted_replay_and_context_guards_all_lanes(self):
        config = self.short_plan()
        for lane in lt.LANES:
            state = lt.state_for('arithmetic',0,lane,config);run_steps(state,stop_after=2)
            payload,metadata = snapshot(state,'arithmetic',REQUEST)
            resumed = restore(payload,metadata,'arithmetic',0,lane,REQUEST,
                              (identity(payload),identity(metadata)),config)
            run_steps(state);run_steps(resumed)
            self.assertEqual(snapshot(state,'arithmetic',REQUEST),snapshot(resumed,'arithmetic',REQUEST))
            record = parse_json(metadata)
            for key,value in (('lane','other'),('data_view','validation'),('schema','trinite.budget-checkpoint.v1'),
                              ('source',{}),('dataset_identity',REQUEST),('environment',{})):
                raw = json_bytes({**record,key:value})
                with self.subTest(lane=lane,key=key), self.assertRaises(ContractError):
                    restore(payload,raw,'arithmetic',0,lane,REQUEST,(identity(payload),identity(raw)),config)
            state.lane = 'dense' if lane!='dense' else 'ternary'
            with self.assertRaises(ContractError): snapshot(state,'arithmetic',REQUEST)
        state = lt.state_for('arithmetic',0,'dense',config)
        state.data['validation'] = list(state.data['train'])
        with self.assertRaises(ContractError): snapshot(state,'arithmetic',REQUEST)
        state = lt.state_for('arithmetic',0,'dense',config)
        state.model.config = replace(state.model.config,context_length=32)
        with self.assertRaises(ContractError): snapshot(state,'arithmetic',REQUEST)

    def test_request_requires_current_source_environment_prior_budget_and_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'request.json';frozen=lt.freeze_request(path)
            self.assertEqual(frozen['planned_cells'],27);lt.read_request(path,frozen['request_identity'])
            original = path.read_bytes()
            for key in ('protocol','source','environment','corpora','prior_budget'):
                record=parse_json(original);record[key]={};raw=json_bytes(record);path.write_bytes(raw)
                with self.assertRaises(ContractError): lt.read_request(path,identity(raw))
            path.write_bytes(original)
            with self.assertRaises(FileExistsError): lt.freeze_request(path)

    def test_runner_pin_checks_actual_bytes(self):
        read=Path.read_bytes;runner=ROOT/'scripts/run_learning.py'
        def changed(path): return read(path)+b'\n' if path==runner else read(path)
        with patch.object(Path,'read_bytes',changed),self.assertRaisesRegex(ContractError,'runner source mismatch'):
            lt.sources()

    def test_real_train_bundle_regenerates_model_predictions_and_loss(self):
        config = self.short_plan()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';frozen=lt.freeze_request(request)
            with patch.object(lt,'plan_for',return_value=config),patch.object(l,'plan_for',return_value=config):
                with patch.object(l,'score',wraps=l.score) as scorer:
                    result=l.train_cell(root/'cell','arithmetic',0,'four-state',request,frozen['request_identity'])
                    verified=l.verified_training(root/'cell','arithmetic',0,'four-state',frozen['request_identity'])
                    self.assertEqual(result,verified['report'])
                    self.assertEqual(scorer.call_count,2)
                    self.assertTrue(all(c.args[-1]=='train' for c in scorer.call_args_list))
                for name in ('training.json','training-predictions.json','metadata.json','tensors.safetensors'):
                    path=root/'cell'/name;old=path.read_bytes();path.write_bytes(old+b' ')
                    with self.assertRaises(ContractError):
                        l.verified_training(root/'cell','arithmetic',0,'four-state',frozen['request_identity'])
                    path.write_bytes(old)

    def test_custody_valid_false_predictions_and_admission_are_rejected(self):
        config = self.short_plan()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';frozen=lt.freeze_request(request);cell=root/'cell'
            with patch.object(lt,'plan_for',return_value=config),patch.object(l,'plan_for',return_value=config):
                l.train_cell(cell,'arithmetic',0,'dense',request,frozen['request_identity'])
                path=cell/'training-predictions.json';original=path.read_bytes();predictions=parse_json(original)
                row=next(r for r in predictions['predictions'] if not r['correct'])
                row['generated_tokens']=[1,257] if row['generated_tokens']!=[1,257] else [2,257]
                path.write_bytes(json_bytes(predictions))
                def rebuild(data,manifest):
                    shutil.rmtree(cell/'train-provenance')
                    observer=BundleObserver(cell/'train-provenance')
                    observer.record('learning-train',inputs={'request.json':request.read_bytes(),
                        'dataset.json':data,'dataset-manifest.json':manifest},outputs={n:(cell/n).read_bytes() for n in
                        ('training.json','training-predictions.json','steps.json','metadata.json','tensors.safetensors')})
                    self.assertTrue(observer.finalize()['integrity_verified'])
                rebuild(*l.dataset('arithmetic'))
                with self.assertRaisesRegex(ContractError,'model/prediction/loss'):
                    l.verified_training(cell,'arithmetic',0,'dense',frozen['request_identity'])

                path.write_bytes(original);rebuild(b'{}',b'{}')
                with self.assertRaisesRegex(ContractError,'admission mismatch'):
                    l.verified_training(cell,'arithmetic',0,'dense',frozen['request_identity'])

    def test_real_upstream_logical_identity_and_manifest_file_hash_are_distinct(self):
        config = self.short_plan()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';frozen=lt.freeze_request(request)
            cell=root/'arithmetic-0-dense'
            with patch.object(lt,'plan_for',return_value=config),patch.object(l,'plan_for',return_value=config):
                l.train_cell(cell,'arithmetic',0,'dense',request,frozen['request_identity'])
                verified=l.verified_training(cell,'arithmetic',0,'dense',frozen['request_identity'])
                manifest=parse_json((cell/'train-provenance/manifest.json').read_bytes())
                self.assertEqual(verified['verification']['manifest_identity'],manifest['manifest_identity'])
                self.assertEqual(verified['manifest_file_identity'],identity((cell/'train-provenance/manifest.json').read_bytes()))
                self.assertNotEqual(verified['manifest_file_identity'],manifest['manifest_identity'])
                row={'workload':'arithmetic','seed':0,'lane':'dense','outcome':'completed',**verified}
                # Exercise the selected-model binding against a real verified
                # manifest. Gate eligibility is synthetic here, not evidence.
                decision={'eligible':True}
                with patch.object(l,'decision_record',return_value=(decision,[row])), \
                     patch.object(l,'score',wraps=l.score) as scorer:
                    _,errors=l.evaluate_matrix(root,request,frozen['request_identity'],{})
                    self.assertEqual(errors,{})
                    self.assertEqual(scorer.call_count,1);self.assertEqual(scorer.call_args.args[-1],'test')
                    actual=l.verified_evaluation(cell,'arithmetic',0,'dense',frozen['request_identity'],row,
                                                 (root/'test-decision.json').read_bytes())
                    self.assertEqual(actual[0],parse_json((cell/'evaluation.json').read_bytes()))

    def summarize_fake(self, root, factory=fake_training, worker_errors=None):
        l.write(root,'training-worker-errors.json',{})
        with patch.object(l,'read_request'),patch.object(l,'verified_training',side_effect=factory):
            decision,_=l.decision_record(root,REQUEST,{})
            l.write(root,'test-decision.json',decision)
            with patch.object(l,'verified_evaluation',side_effect=fake_evaluation):
                result=l.summarize(root,root/'request.json',REQUEST,worker_errors or {})
                self.assertEqual(l.summarize(root,root/'request.json',REQUEST,worker_errors or {},publish=False),result)
        self.assertEqual(parse_json((root/'summary.json').read_bytes(),canonical=True),result)
        self.assertEqual(len(result['cells']),27)
        return result

    def test_successful_paired_summary_does_not_automatically_unlock_phase5(self):
        with tempfile.TemporaryDirectory() as directory:
            result=self.summarize_fake(Path(directory))
        self.assertEqual(result['outcome'],'completed');self.assertTrue(result['dense_learning_adequate'])
        self.assertFalse(result['phase5_ready']);self.assertFalse(result['release_ready'])
        for workload in lt.WORKLOADS:
            self.assertEqual(result['groups'][workload]['four-state']['paired_latent_byte_ratio_hex'],[(1.).hex()]*3)

    def test_late_pair_mismatch_retains_all_cells_and_suppresses_all_groups(self):
        def mismatch(*args):
            record=fake_training(*args)
            if args[1:4]==('lattice',2,'four-state'):record['report']['dataset_identity']='different'
            return record
        with tempfile.TemporaryDirectory() as directory:
            result=self.summarize_fake(Path(directory),mismatch)
        self.assertEqual(result['outcome'],'failed');self.assertEqual(result['groups'],{})
        self.assertTrue(result['aggregate_errors'])

    def test_dense_floor_blocks_every_held_out_call(self):
        def low(*args):
            record=fake_training(*args)
            if args[1:4]==('lattice',2,'dense'):record['report']['scores']=scores(8)
            return record
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(l,'read_request',return_value=({},b'{}')),patch.object(l,'verified_training',side_effect=low), \
                 patch.object(l,'score') as scorer,patch.object(l,'restore_report') as restored:
                decision,errors=l.evaluate_matrix(root,root/'request.json',REQUEST,{})
                self.assertFalse(decision['eligible']);self.assertEqual(errors,{})
                scorer.assert_not_called();restored.assert_not_called()
                self.assertTrue((root/'test-decision.json').exists())
            result=self.summarize_fake(self.new_directory(root),low)
            self.assertEqual(result['outcome'],'blocked')

    def new_directory(self,root):
        directory=root/'summary';directory.mkdir();return directory

    def test_one_fresh_matrix_authorization_precedes_all_test_calls(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for cell in lt.cells():
                path=root/lt.cell_name(*cell);(path/'train-provenance').mkdir(parents=True)
                record=fake_training(path,*cell,REQUEST)
                (path/'training.json').write_bytes(json_bytes(record['report']))
                (path/'train-provenance/manifest.json').write_bytes(b'{}')
            def scoring(*args):
                self.assertTrue(parse_json((root/'test-decision.json').read_bytes())['eligible'])
                return scores(),[]
            with patch.object(l,'read_request',return_value=({},b'{}')),patch.object(l,'verified_training',side_effect=fake_training) as verification, \
                 patch.object(l,'score',side_effect=scoring) as scorer,patch.object(l,'retain'), \
                 patch.object(l,'restore_report',return_value=SimpleNamespace(model=object())), \
                 patch.object(l,'model_identity',return_value=REQUEST),patch.object(l,'weight_diagnostics',return_value={}):
                decision,errors=l.evaluate_matrix(root,root/'request.json',REQUEST,{})
                self.assertTrue(decision['eligible']);self.assertEqual(errors,{})
                self.assertEqual(verification.call_count,27);self.assertEqual(scorer.call_count,27)
                self.assertTrue(all(c.args[-1]=='test' for c in scorer.call_args_list))

    def test_missing_worker_or_changed_decision_retains_failed_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);l.write(root,'training-worker-errors.json',{})
            with patch.object(l,'read_request'),patch.object(l,'verified_training',side_effect=fake_training):
                decision,_=l.decision_record(root,REQUEST,{})
                decision['eligible']=False;l.write(root,'test-decision.json',decision)
                with patch.object(l,'verified_evaluation') as held_out:
                    result=l.summarize(root,root/'request.json',REQUEST,{})
                    held_out.assert_not_called()
            self.assertEqual(result['outcome'],'failed');self.assertEqual(len(result['cells']),27)
            self.assertEqual(result['groups'],{})
        with tempfile.TemporaryDirectory() as directory:
            result=self.summarize_fake(Path(directory),worker_errors={'fold-2-ternary':'evaluation timeout'})
            self.assertEqual(result['outcome'],'failed');self.assertEqual(result['groups'],{})

    def test_runner_training_timeouts_still_evaluate_decision_and_summarize(self):
        runner=self.runner()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';frozen=lt.freeze_request(request);output=root/'run';stages=[]
            def launch(stage,request,request_identity,output,log,cell=None):
                stages.append(stage)
                if stage=='train':raise subprocess.TimeoutExpired('worker',600)
                if stage=='evaluate':
                    l.evaluate_matrix(output,request,request_identity,parse_json((output/'training-worker-errors.json').read_bytes()))
                else:
                    l.summarize(output,request,request_identity,parse_json((output/'worker-errors.json').read_bytes()))
                return subprocess.CompletedProcess(['worker'],0)
            argv=['run_learning.py','--request',str(request),'--request-identity',frozen['request_identity'],'--output',str(output)]
            with patch.object(sys,'argv',argv),patch.object(runner,'launch',side_effect=launch),patch('builtins.print'):
                code=runner.main()
            result=parse_json((output/'summary.json').read_bytes());self.assertEqual(code,1)
            self.assertEqual(stages,['train']*27+['evaluate','summarize'])
            self.assertEqual(len(result['cells']),27);self.assertEqual(len(result['worker_errors']),27)
            self.assertEqual(len(list(output.glob('*.log'))),29)

    def test_summary_timeout_retains_prior_output_and_conservative_inventory(self):
        runner=self.runner()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';frozen=lt.freeze_request(request);output=root/'run'
            def launch(stage,request,request_identity,output,log,cell=None):
                if stage=='summarize':(output/'summary.json').write_bytes(b'{"partial":true}\n')
                raise subprocess.TimeoutExpired('worker',600 if stage=='train' else 1200)
            argv=['run_learning.py','--request',str(request),'--request-identity',frozen['request_identity'],'--output',str(output)]
            with patch.object(sys,'argv',argv),patch.object(runner,'launch',side_effect=launch),patch('builtins.print'):
                self.assertEqual(runner.main(),1)
            result=parse_json((output/'summary.json').read_bytes())
            self.assertEqual(len(result['cells']),27);self.assertFalse(result['integrity_verified'])
            self.assertFalse(result['dense_learning_adequate']);self.assertEqual(result['groups'],{})
            self.assertEqual((output/'summary-before-worker-failure.json').read_bytes(),b'{"partial":true}\n')

    def runner(self):
        spec=importlib.util.spec_from_file_location('learning_runner',ROOT/'scripts/run_learning.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);return runner

    def test_bounded_workflow_and_closed_subprocess_boundary(self):
        text=(ROOT/'.github/workflows/learning.yml').read_text();resources=lt.protocol()['resources']
        minutes=int(next(line for line in text.splitlines() if line.strip().startswith('timeout-minutes:')).split(':')[1])
        seconds=27*resources['train_worker_seconds']+resources['evaluation_worker_seconds']+resources['summary_worker_seconds']
        self.assertEqual(minutes,360);self.assertGreaterEqual(minutes*60,seconds+50*60);self.assertIn('if: always()',text)
        runner=self.runner()
        for stage,cell in ((';evil',None),('train',('arithmetic',True,'dense')),('evaluate',('fold',0,'dense'))):
            with self.assertRaises(ValueError):runner.launch(stage,None,None,None,None,cell)
        for stage,cell in (('train',('fold',0,'dense')),('evaluate',None),('summarize',None)):
            with patch.object(runner.subprocess,'run') as launched:
                runner.launch(stage,Path('request'),REQUEST,Path('output'),None,cell)
                self.assertFalse(launched.call_args.kwargs['shell'])
                self.assertEqual(launched.call_args.kwargs['timeout'],600 if stage=='train' else 1200)


if __name__=='__main__':
    unittest.main()
