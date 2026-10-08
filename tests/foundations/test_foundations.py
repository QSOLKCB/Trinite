from dataclasses import replace
import importlib.util
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite.foundations_data import dataset
from trinite.foundations_training import state_for, protocol, sources, LANES, WORKLOADS, SEEDS
from trinite.foundations_plan import FoundationsPlan
from trinite.foundations_checkpoint import snapshot, restore
from trinite.training import run_steps, model_identity, RunConfig
from trinite import foundations as f

ROOT=Path(__file__).resolve().parents[2]
REQUEST='sha256:'+'0'*64
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
PLAN=replace(FoundationsPlan(),model=ModelConfig(context_length=64,blocks=1,width=16,heads=2,feed_forward_width=32),
             steps=4,batch_size=4,validation_every=2)


def synthetic_stage(root,workload,lane,seed,request_identity,stage):
    scores={'examples':10,'correct':10,'families':{'f':(1.).hex()},'family_macro_accuracy_hex':(1.).hex(),
            'tasks':{'task':{'examples':10,'correct':10,'baseline_correct':0}}}
    report={'scores':scores,'model_identity':REQUEST,'training_gate_failures':[],
        **{k:'same' for k in ('initialization_identity','dataset_identity','manifest_identity','plan_identity','schedule_identity')},
        'parameter_count':10,'steps':4,'target_tokens':20,'training_wall_seconds_hex':(1.).hex(),
        'latent_float32_bytes':40}
    return {'report':report,'report_identity':REQUEST,'verification':{'manifest_identity':REQUEST}}


class FoundationsTests(unittest.TestCase):
    def test_real_training_bundle_rechecks_checkpoint_predictions_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';frozen=f.freeze_request(request)
            cell=root/'arithmetic-0-dense'
            report=f.train_cell(cell,'arithmetic','dense',0,request,frozen['request_identity'])
            verified=f.verified_stage(cell,'arithmetic','dense',0,frozen['request_identity'],'train')
            self.assertEqual(verified['report'],report)
            self.assertEqual(report['steps'],1024)
            self.assertFalse((cell/'test-predictions.json').exists())
            original=(cell/'training-predictions.json').read_bytes()
            (cell/'training-predictions.json').write_bytes(original+b' ')
            with self.assertRaisesRegex(ContractError,'differs from verified retained evidence'):
                f.verified_stage(cell,'arithmetic','dense',0,frozen['request_identity'],'train')
            (cell/'training-predictions.json').write_bytes(original)
            artifact=next((cell/'train-provenance/artifacts/sha256').iterdir())
            artifact.write_bytes(artifact.read_bytes()+b'x')
            with self.assertRaises(ContractError):
                f.verified_stage(cell,'arithmetic','dense',0,frozen['request_identity'],'train')

    def test_three_lanes_share_inputs_initialization_and_replay_bit_exact(self):
        for workload in WORKLOADS:
            states=[state_for(workload,lane,0,PLAN) for lane in LANES]
            self.assertTrue(all(set(s.data)=={'train','validation'} for s in states))
            self.assertEqual(len({s.initialization_identity for s in states}),1)
            for direct in states:
                run_steps(direct)
                paused=state_for(workload,direct.lane,0,PLAN);run_steps(paused,stop_after=2)
                payload,metadata=snapshot(paused,workload,REQUEST)
                replay=restore(payload,metadata,workload,direct.lane,0,REQUEST,
                    (identity(payload),identity(metadata)),PLAN);run_steps(replay)
                self.assertEqual(snapshot(direct,workload,REQUEST),snapshot(replay,workload,REQUEST))
            self.assertEqual(len({s.target_tokens for s in states}),1)
            self.assertEqual(len({identity(json_bytes([h['examples'] for h in s.history])) for s in states}),1)
        with self.assertRaises(ContractError):RunConfig(steps=1024)

    def test_snapshot_rejects_lane_model_workload_and_tensor_changes(self):
        state=state_for('arithmetic','four-state',0,PLAN)
        state.lane='dense'
        with self.assertRaises(ContractError):snapshot(state,'arithmetic',REQUEST)
        state.lane='four-state'
        with self.assertRaises(ContractError):snapshot(state,'fold',REQUEST)
        payload,metadata=snapshot(state,'arithmetic',REQUEST)
        with self.assertRaises(ContractError):restore(payload+b'x',metadata,'arithmetic','four-state',0,REQUEST,
            (identity(payload),identity(metadata)),PLAN)
        changed=parse_json(metadata);changed['plan']['model']['context_length']=32;raw=json_bytes(changed)
        with self.assertRaises(ContractError):restore(payload,raw,'arithmetic','four-state',0,REQUEST,
            (identity(payload),identity(raw)),PLAN)

    def test_request_rejects_source_protocol_environment_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'request.json';frozen=f.freeze_request(path)
            f.read_request(path,frozen['request_identity'])
            for field in ('source','protocol','environment','corpora'):
                changed=parse_json(path.read_bytes());changed[field]={};raw=json_bytes(changed)
                other=Path(directory)/field;other.write_bytes(raw)
                with self.assertRaises(ContractError):f.read_request(other,identity(raw))

    def test_scoring_passes_prompt_and_workload_cap_only_and_checks_eos(self):
        state=state_for('arithmetic','dense',0,PLAN)
        seen=[]
        def answer(model,prompt,cap):
            self.assertEqual(cap,5);seen.append(prompt)
            return [ord('0')]  # Correct bytes without EOS must never count.
        with patch.object(f,'greedy',side_effect=answer):scores,rows=f.score(state.model,'arithmetic','train')
        self.assertTrue(seen);self.assertEqual(scores['correct'],0)
        self.assertEqual(f.check_predictions('arithmetic','train',json_bytes({'predictions':rows})),scores)
        altered=rows.copy();altered[0]={**rows[0],'correct':True}
        with self.assertRaises(ContractError):f.check_predictions('arithmetic','train',json_bytes({'predictions':altered}))
        with self.assertRaises(ContractError):f.check_predictions('arithmetic','train',json_bytes({'predictions':rows[:-1]}))

    def test_gates_use_exact_counts_and_require_every_task_and_baseline_margin(self):
        def scores(correct,baseline):
            return {'examples':10,'correct':correct,'families':{},'family_macro_accuracy_hex':'0x0.0p+0',
                    'tasks':{'a':{'examples':10,'correct':correct,'baseline_correct':baseline}}}
        self.assertEqual(f.task_failures(scores(9,0),'train'),[])
        self.assertTrue(f.task_failures(scores(8,0),'train'))
        self.assertEqual(f.task_failures(scores(5,4),'test'),[])
        self.assertTrue(f.task_failures(scores(5,5),'test'))
        with self.assertRaises(ContractError):f.task_failures({**scores(9,0),'tasks':{}},'train')

    def test_dense_floor_blocks_every_lane_and_forged_decision_rejects(self):
        def deficient(*args):
            value=synthetic_stage(*args)
            if args[2]=='dense' and args[1]=='fold' and args[3]==2:
                value['report']['scores']['tasks']['task']['correct']=8
            return value
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(f,'verified_stage',side_effect=deficient),patch.object(f,'read_request'):
                decision=f.decide(root,root/'request.json',REQUEST,{})
                self.assertFalse(decision['eligible'])
                self.assertEqual(len(decision['training_receipts']),27)
                result=f.summarize(root,root/'request.json',REQUEST,{})
                self.assertEqual(result['outcome'],'blocked');self.assertEqual(result['groups'],{})
                self.assertEqual(len(result['cells']),27)
                with patch.object(f,'score') as score:
                    with self.assertRaises(ContractError):f.validate_decision(root,REQUEST,decision['decision_identity'])
                    score.assert_not_called()
                changed=parse_json((root/'test-decision.json').read_bytes());changed['eligible']=True
                raw=json_bytes(changed);(root/'test-decision.json').write_bytes(raw)
                with self.assertRaises(ContractError):f.validate_decision(root,REQUEST,identity(raw))

    def test_late_pair_mismatch_retains_failed_summary_and_no_partial_groups(self):
        def mismatch(*args):
            value=synthetic_stage(*args)
            if args[1:4]==('lattice','four-state',2):value['report']['initialization_identity']='different'
            return value
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(f,'verified_stage',side_effect=mismatch),patch.object(f,'read_request'):
                f.decide(root,root/'request.json',REQUEST,{})
                result=f.summarize(root,root/'request.json',REQUEST,{})
            self.assertEqual(result['outcome'],'failed');self.assertEqual(result['groups'],{})
            self.assertEqual(len(result['cells']),27);self.assertIn('unmatched foundations',result['aggregate_errors'][0])
            self.assertEqual(parse_json((root/'summary.json').read_bytes(),canonical=True),result)

    def test_success_and_late_test_failure_paths_retain_all_cells(self):
        for failure in (False,True):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                def verified(*args):
                    value=synthetic_stage(*args)
                    if args[-1]=='test':
                        if failure and args[1:4]==('lattice','four-state',2):raise ContractError('missing test evidence')
                        value['report'].update(training_identity=REQUEST,decision_identity=identity((root/'test-decision.json').read_bytes()),
                                               test_gate_failures=[],learning_adequate=True)
                    return value
                with patch.object(f,'verified_stage',side_effect=verified),patch.object(f,'read_request'):
                    decision=f.decide(root,root/'request.json',REQUEST,{})
                    self.assertTrue(decision['eligible'])
                    result=f.summarize(root,root/'request.json',REQUEST,{})
                self.assertEqual(result['outcome'],'failed' if failure else 'completed')
                self.assertEqual(bool(result['groups']),not failure)
                self.assertEqual(result['dense_learning_adequate'],not failure)
                self.assertEqual(len(result['cells']),27)

    def test_runner_timeouts_and_gate_stop_never_start_test_workers(self):
        spec=importlib.util.spec_from_file_location('foundations_runner',ROOT/'scripts/run_foundations.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'request.json';frozen=f.freeze_request(request)
            output=root/'out';stages=[]
            def launch(stage,*args):stages.append(stage);raise subprocess.TimeoutExpired('worker',240)
            argv=['run_foundations.py','--request',str(request),'--request-identity',frozen['request_identity'],'--output',str(output)]
            with patch.object(sys,'argv',argv),patch.object(runner,'launch',side_effect=launch),patch('builtins.print'):
                self.assertEqual(runner.main(),1)
            result=parse_json((output/'summary.json').read_bytes(),canonical=True)
            self.assertEqual(stages,['train']*27);self.assertEqual(len(result['cells']),27)
            self.assertEqual(len(result['worker_errors']),27);self.assertEqual(result['outcome'],'failed')
            self.assertEqual(len(list(output.glob('*-train.log'))),27)

    def test_fixed_worker_dispatch_and_workflow_envelope(self):
        spec=importlib.util.spec_from_file_location('foundations_runner',ROOT/'scripts/run_foundations.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        with patch.object(runner.subprocess,'run') as launch:
            with self.assertRaises(ValueError):runner.launch('evil',ROOT,'fold','dense',0,ROOT,REQUEST,REQUEST,None,{})
            launch.assert_not_called()
            runner.launch('train',ROOT,'fold','dense',0,ROOT,REQUEST,REQUEST,None,{})
            self.assertFalse(launch.call_args.kwargs['shell']);self.assertEqual(launch.call_args.kwargs['timeout'],240)
        self.assertEqual(identity((ROOT/'scripts/run_foundations.py').read_bytes()),sources()['scripts/run_foundations.py'])
        workflow=(ROOT/'.github/workflows/foundations.yml').read_text()
        self.assertIn('timeout-minutes: 240',workflow);self.assertIn('if: always()',workflow)
        self.assertGreaterEqual(240*60,27*2*240+24*60)


if __name__=='__main__':unittest.main()
