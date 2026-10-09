from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite import fold_exposure as f
from trinite.scalar_training import state_for as scalar_state, plan_for as scalar_plan
from trinite.training import run_steps, model_identity

ROOT = Path(__file__).resolve().parents[2]


class FoldExposureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

    def tiny(self, repeats=1, steps=2):
        return replace(f.plan_for(0,repeats), steps=steps, validation_every=steps,
            model=ModelConfig(context_length=64,blocks=1,width=16,heads=2,feed_forward_width=32))

    def test_control_matches_scalar_state_and_updates(self):
        plan=self.tiny(); state=f.state_for(0,1,plan)
        prior=scalar_state('fold',0,'dense',replace(scalar_plan(0),steps=2,validation_every=2,model=plan.model))
        self.assertEqual(state.data,prior.data)
        self.assertEqual(state.initialization_identity,prior.initialization_identity)
        run_steps(state); run_steps(prior)
        self.assertEqual(state.history,prior.history)
        self.assertEqual(model_identity(state.model),model_identity(prior.model))

    def test_multiplicity_and_train_only_schedule(self):
        raw,_=f.dataset('fold'); examples=parse_json(raw,canonical=True)['examples']
        train={e['example_identity']:e['task'] for e in examples if e['split']=='train'}
        for repeats in (1,2,4):
            state=f.state_for(0,repeats,self.tiny(repeats))
            counts={task:0 for task in ('count','sum','squares')}
            for row in state.data['train']: counts[train[row['identity']]]+=1
            self.assertEqual(counts,{'count':1150,'sum':1150*repeats,'squares':1150})
            self.assertIs(state.data['validation'],state.data['train'])

    def test_checkpoint_replay_and_wrong_exposure_rejection(self):
        plan=self.tiny(4,4); req='sha256:'+'a'*64
        full=f.state_for(0,4,plan); run_steps(full)
        prefix=f.state_for(0,4,plan); run_steps(prefix,stop_after=2)
        payload,meta=f.snapshot(prefix,req)
        replay=f.restore(payload,meta,0,4,req,(identity(payload),identity(meta)),plan)
        run_steps(replay)
        self.assertEqual(f.snapshot(full,req),f.snapshot(replay,req))
        with self.assertRaises(ContractError):
            f.restore(payload,meta,0,2,req,(identity(payload),identity(meta)),self.tiny(2,4))
        r=parse_json(meta);r['history'][0]['examples'][0]='sha256:'+'0'*64;bad=json_bytes(r)
        with self.assertRaises(ContractError): f.restore(payload,bad,0,4,req,(identity(payload),identity(bad)),plan)

    def test_fixed_protocol_and_admission_guards(self):
        self.assertEqual(identity((ROOT/'src/trinite/fold-exposure-protocol.json').read_bytes()),f.PROTOCOL_IDENTITY)
        self.assertEqual(len(f.cells()),9)
        with patch.object(f,'dataset',return_value=(b'{}\n',b'{}\n')):
            with self.assertRaises(ContractError):f.request_core()
        for bad in (True,3,0):
            with self.assertRaises(ContractError):replace(f.plan_for(0,1),sum_repeats=bad)

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
            def cell(root,seed,repeats,request_identity):
                return {'seed':seed,'sum_repeats':repeats,'outcome':'completed',
                    'report':{'training_gate_failures':[],'initialization_identity':'same'}}
            with patch.object(f,'verify_cell',side_effect=cell):r=f.summarize(out,p,req,{})
            self.assertEqual(len(r['cells']),9);self.assertEqual(r['outcome'],'completed')
            self.assertIsNone(r['selected_candidate'])
            for name in ('held_out_scored','dense_learning_adequate','format_selection_eligible','phase5_ready','release_ready'):
                self.assertFalse(r[name])
            with patch.object(f,'verify_cell',side_effect=cell):r=f.summarize(out,p,req,{f.cell_name(0,2):'timeout'},publish=False)
            self.assertEqual(len(r['cells']),9);self.assertEqual(r['outcome'],'failed');self.assertEqual(r['groups'],{})
            forbidden=out/f.cell_name(0,1);forbidden.mkdir();(forbidden/'test-predictions.json').write_bytes(b'{}')
            with self.assertRaises(ContractError):f.summarize(out,p,req,{},publish=False)

    def test_verifier_binds_checkpoint_and_regenerates_unique_training_predictions(self):
        # A real tiny training stage closes the actual upstream bundle. Verification
        # uses the same bounded tiny plan solely as a software conformance fixture.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'request.json';req=f.freeze_request(p)['request_identity'];out=Path(d)/'cell'
            tiny=self.tiny(2,2)
            original=f.state_for
            with patch.object(f,'state_for',side_effect=lambda seed,repeats,plan=None:original(seed,repeats,tiny)):
                f.train_cell(out,0,2,p,req)
                r=f.verify_cell(out,0,2,req)
                self.assertEqual(r['report']['scores']['examples'],3450)
                report=parse_json((out/'training.json').read_bytes());report['scores']['correct']=3450
                (out/'training.json').write_bytes(json_bytes(report))
                with self.assertRaises(ContractError):f.verify_cell(out,0,2,req)

    def test_runner_rejects_arbitrary_stage_and_selectors(self):
        spec=importlib.util.spec_from_file_location('fold_runner',ROOT/'scripts/run_fold_exposure.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for stage,cell in [('evaluate',None),('train',(0,3)),('summarize',(0,1))]:
            with self.assertRaises(ValueError):module.launch(stage,'r','i','o',None,cell)
