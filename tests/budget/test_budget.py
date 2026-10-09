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

from trinite import budget as b
from trinite.budget_plan import BudgetPlan
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.convergence import state_for as prior_state
from trinite.foundations_plan import FoundationsPlan
from trinite.foundations_checkpoint import snapshot as native_snapshot
from trinite.observation import BundleObserver
from trinite.training import model_identity, run_steps

ROOT = Path(__file__).resolve().parents[2]
REQUEST = 'sha256:'+'0'*64
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def fake_cell(root, workload, seed, candidate, request_identity):
    return dict(workload=workload,seed=seed,candidate=candidate,outcome='completed',
                dataset_identity=workload,manifest_identity=workload,initialization_identity=str(seed),
                schedule_identity=str(seed),steps=4096,parameter_count=86528,target_tokens=21000,
                curves=[{'training_gate_failures':[]}])


class BudgetTests(unittest.TestCase):
    def test_separate_resource_bounds_and_unchanged_native_limits(self):
        plan = BudgetPlan(); self.assertEqual(BudgetPlan.from_dict(plan.to_dict()),plan)
        for name,value in (('steps',4097),('max_seconds',481),('max_target_tokens',524289),
                           ('validation_every',4097),('steps',True),('batch_size',9),
                           ('schedule','linear'),('learning_rate','nan'),('max_target_tokens',1)):
            with self.subTest(name=name,value=value),self.assertRaises(ContractError):
                replace(plan,**{name:value})
        with self.assertRaises(ContractError): FoundationsPlan(steps=4096)
        for invalid in ({},False,FoundationsPlan()):
            with self.assertRaises(ContractError):b.state_for('arithmetic',0,'extended',invalid)
        state=b.state_for('arithmetic',0,'extended',replace(plan,steps=2,validation_every=2))
        with self.assertRaises(ContractError):native_snapshot(state,'arithmetic',REQUEST)
        for values in (('arithmetic',True,'extended'),('fold',0,'low'),('unknown',0,'extended')):
            with self.assertRaises(ContractError): b.cell_name(*values)

    def test_short_prefix_matches_prior_low_rate_updates(self):
        config = replace(b.plan_for(0,'extended'),steps=4,validation_every=2)
        old_config = replace(FoundationsPlan(),learning_rate='0.001',steps=4,validation_every=2)
        current = b.state_for('arithmetic',0,'extended',config)
        old = prior_state('arithmetic',0,'low',old_config)
        run_steps(current); run_steps(old)
        self.assertEqual(model_identity(current.model),model_identity(old.model))
        self.assertEqual(current.history,old.history); self.assertEqual(current.validation,old.validation)

    def test_exact_interrupted_replay_and_checkpoint_profile(self):
        plan = replace(b.plan_for(0,'extended'),steps=4,validation_every=2)
        state = b.state_for('arithmetic',0,'extended',plan); run_steps(state,stop_after=2)
        payload,metadata = b.snapshot_state(state,'arithmetic','extended',REQUEST)
        resumed = b.restore_state(payload,metadata,'arithmetic',0,'extended',REQUEST,
                                   (identity(payload),identity(metadata)),plan)
        self.assertIs(resumed.data['train'],resumed.data['validation'])
        run_steps(state);run_steps(resumed)
        self.assertEqual(b.snapshot_state(state,'arithmetic','extended',REQUEST),
                         b.snapshot_state(resumed,'arithmetic','extended',REQUEST))
        original=parse_json(metadata,canonical=True)
        for field,value in (('data_view','validation'),('schema','trinite.convergence-checkpoint.v1'),
                            ('lane','ternary'),('protocol_identity',REQUEST),('workload','fold')):
            record={**original,field:value};raw=json_bytes(record)
            with self.subTest(field=field),self.assertRaises(ContractError):
                b.restore_state(payload,raw,'arithmetic',0,'extended',REQUEST,
                                 (identity(payload),identity(raw)),plan)
        state.lane='ternary'
        with self.assertRaises(ContractError): b.snapshot_state(state,'arithmetic','extended',REQUEST)
        state.lane='dense';state.data['validation']=list(state.data['train'])
        with self.assertRaises(ContractError): b.snapshot_state(state,'arithmetic','extended',REQUEST)

    def test_request_is_source_environment_admission_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'request.json';frozen=b.freeze_request(path)
            self.assertEqual(frozen['planned_cells'],9);b.read_request(path,frozen['request_identity'])
            original=path.read_bytes()
            for key in ('protocol','source','environment','corpora'):
                record=parse_json(original,canonical=True);record[key]={};raw=json_bytes(record)
                path.write_bytes(raw)
                with self.assertRaises(ContractError): b.read_request(path,identity(raw))

    def test_runner_changes_invalidate_actual_source_receipt(self):
        read=Path.read_bytes;runner=ROOT/'scripts/run_budget.py'
        def changed(path): return read(path)+b'\n' if path==runner else read(path)
        with patch.object(Path,'read_bytes',changed),self.assertRaisesRegex(ContractError,'runner source mismatch'):
            b.sources()

    def test_real_cell_recomputes_all_milestones_from_retained_models(self):
        plan=replace(b.plan_for(0,'extended'),steps=4,validation_every=2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';frozen=b.freeze_request(request)
            with patch.object(b,'plan_for',return_value=plan),patch.object(b,'MILESTONES',(2,4)):
                with patch.object(b,'score',wraps=b.score) as score:
                    result=b.train_cell(root/'cell','arithmetic',0,'extended',request,frozen['request_identity'])
                    self.assertEqual(score.call_count,2)
                    verified=b.verified_cell(root/'cell','arithmetic',0,'extended',frozen['request_identity'])
                    self.assertEqual(score.call_count,4)
                    self.assertTrue(all(call.args[-1]=='train' for call in score.call_args_list))
                self.assertEqual(result,verified)
                for name in ('report.json','metadata-2.json','tensors-2.safetensors','predictions-2.json'):
                    path=root/'cell'/name;raw=path.read_bytes();path.write_bytes(raw+b' ')
                    with self.assertRaises(ContractError):
                        b.verified_cell(root/'cell','arithmetic',0,'extended',frozen['request_identity'])
                    path.write_bytes(raw)

    def test_closed_but_wrong_admission_is_rejected(self):
        plan=replace(b.plan_for(0,'extended'),steps=2,validation_every=2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';frozen=b.freeze_request(request)
            with patch.object(b,'plan_for',return_value=plan),patch.object(b,'MILESTONES',(2,)):
                b.train_cell(root/'cell','arithmetic',0,'extended',request,frozen['request_identity'])
                shutil.rmtree(root/'cell/provenance')
                outputs={p.name:p.read_bytes() for p in (root/'cell').iterdir() if p.name!='verification.json'}
                observer=BundleObserver(root/'cell/provenance')
                observer.record('training-budget',inputs={'request.json':request.read_bytes(),
                    'dataset.json':b'{}','dataset-manifest.json':b'{}'},outputs=outputs)
                self.assertTrue(observer.finalize()['integrity_verified'])
                with self.assertRaisesRegex(ContractError,'admitted input mismatch'):
                    b.verified_cell(root/'cell','arithmetic',0,'extended',frozen['request_identity'])

    def test_custody_valid_false_model_predictions_are_rejected(self):
        plan=replace(b.plan_for(0,'extended'),steps=2,validation_every=2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';frozen=b.freeze_request(request)
            with patch.object(b,'plan_for',return_value=plan),patch.object(b,'MILESTONES',(2,)):
                b.train_cell(root/'cell','arithmetic',0,'extended',request,frozen['request_identity'])
                path=root/'cell/predictions-2.json';predictions=parse_json(path.read_bytes(),canonical=True)
                # A different incorrect answer has the same score but was not generated by this checkpoint.
                row=next(r for r in predictions['predictions'] if not r['correct'])
                row['generated_tokens']=[1,257] if row['generated_tokens']!=[1,257] else [2,257]
                path.write_bytes(json_bytes(predictions));shutil.rmtree(root/'cell/provenance')
                outputs={p.name:p.read_bytes() for p in (root/'cell').iterdir() if p.name!='verification.json'}
                raw,manifest=b.dataset('arithmetic');observer=BundleObserver(root/'cell/provenance')
                observer.record('training-budget',inputs={'request.json':request.read_bytes(),
                    'dataset.json':raw,'dataset-manifest.json':manifest},outputs=outputs)
                self.assertTrue(observer.finalize()['integrity_verified'])
                with self.assertRaisesRegex(ContractError,'model/prediction/loss mismatch'):
                    b.verified_cell(root/'cell','arithmetic',0,'extended',frozen['request_identity'])

    def test_aggregate_success_does_not_unlock_learning_or_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(b,'read_request'),patch.object(b,'verified_cell',side_effect=fake_cell):
                result=b.summarize(root,root/'request.json',REQUEST,{})
            self.assertEqual(result['outcome'],'completed');self.assertEqual(len(result['cells']),9)
            self.assertTrue(result['groups']['extended']['training_tasks_pass_all_cells'])
            self.assertTrue(all(result[key] is False for key in ('held_out_scored','learning_adequate','phase5_ready','release_ready')))
            self.assertEqual(parse_json((root/'summary.json').read_bytes(),canonical=True),result)

    def test_failed_worker_and_aggregate_retain_all_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(b,'read_request'),patch.object(b,'verified_cell',side_effect=fake_cell):
                result=b.summarize(root,root/'request.json',REQUEST,{'fold-2-extended':'timeout'})
            self.assertEqual(result['outcome'],'failed');self.assertEqual(len(result['cells']),9)
            self.assertEqual(result['groups'],{})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(b,'read_request'),patch.object(b,'verified_cell',side_effect=fake_cell), \
                 patch.object(b,'groups_for',side_effect=ContractError('late control mismatch')):
                result=b.summarize(root,root/'request.json',REQUEST,{})
            self.assertEqual(result['outcome'],'failed');self.assertEqual(len(result['cells']),9)
            self.assertEqual(result['groups'],{});self.assertTrue(result['aggregate_errors'])
            self.assertTrue((root/'summary.json').exists())

    def test_late_dataset_pair_mismatch_retains_summary(self):
        def mismatch(*args):
            record=fake_cell(*args)
            if args[1:4]==('lattice',2,'extended'):record['dataset_identity']='different'
            return record
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(b,'read_request'),patch.object(b,'verified_cell',side_effect=mismatch):
                result=b.summarize(root,root/'request.json',REQUEST,{})
            self.assertEqual(result['outcome'],'failed');self.assertEqual(len(result['cells']),9)
            self.assertEqual(result['groups'],{})
            self.assertIn('unmatched budget dataset_identity',result['aggregate_errors'][0])
            self.assertTrue((root/'summary.json').exists())

    def test_update_timeout_retains_failure_before_scoring(self):
        plan=replace(b.plan_for(0,'extended'),steps=2,validation_every=2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';frozen=b.freeze_request(request)
            with patch.object(b,'plan_for',return_value=plan),patch.object(b,'MILESTONES',(2,)), \
                 patch.object(b.time,'perf_counter',side_effect=(0.,0.,481.)),patch.object(b,'score') as score:
                with self.assertRaisesRegex(ContractError,'cumulative update'):
                    b.train_cell(root/'cell','arithmetic',0,'extended',request,frozen['request_identity'])
                score.assert_not_called()
            self.assertTrue((root/'cell/budget-failure.json').exists())

    def test_runner_timeouts_continue_to_failed_summary(self):
        spec=importlib.util.spec_from_file_location('budget_runner',ROOT/'scripts/run_budget.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';frozen=b.freeze_request(request)
            argv=['run_budget.py','--request',str(request),'--request-identity',frozen['request_identity'],'--output',str(root/'run')]
            with patch.object(sys,'argv',argv),patch.object(runner,'launch',side_effect=subprocess.TimeoutExpired('worker',720)) as launch,patch('builtins.print'):
                code=runner.main()
            result=parse_json((root/'run/summary.json').read_bytes(),canonical=True)
            self.assertEqual(code,1);self.assertEqual(launch.call_count,9)
            self.assertEqual(len(result['cells']),9);self.assertEqual(len(result['worker_errors']),9)
            self.assertEqual(len(list((root/'run').glob('*.log'))),9)
            self.assertEqual(result['groups'],{})

    def test_workflow_envelope_and_safe_worker_boundary(self):
        text=(ROOT/'.github/workflows/budget.yml').read_text()
        minutes=int(next(line for line in text.splitlines() if line.strip().startswith('timeout-minutes:')).split(':')[1])
        resources=b.protocol()['resources']
        self.assertGreaterEqual(minutes*60,resources['max_cells']*resources['worker_seconds']+24*60)
        self.assertIn('if: always()',text)
        spec=importlib.util.spec_from_file_location('budget_boundary',ROOT/'scripts/run_budget.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        with self.assertRaises(ValueError): runner.launch('arithmetic',0,'; evil',None,None,None,None)
        with patch.object(runner.subprocess,'run') as launch:
            runner.launch('arithmetic',0,'extended',Path('request'),REQUEST,Path('output'),None)
            self.assertFalse(launch.call_args.kwargs['shell']);self.assertEqual(launch.call_args.kwargs['timeout'],720)
            self.assertEqual(launch.call_args.args[0][:2],[sys.executable,str(ROOT/'scripts/run_budget.py')])


if __name__=='__main__': unittest.main()
