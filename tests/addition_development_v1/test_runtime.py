"""Engineering conformance only: tiny injected plans and synthetic scorer observations.

No fixed4000-update target cell is executed or reported as research evidence.
"""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite.tokenizer import EOS
from trinite.foundations import metrics
from addition_development_v1 import controller, data, numerical, request, workflow
from native_diagnostic_v1.artifacts import Store

controller.configure()


def tiny(seed):
    return numerical.AdditionPlan(seed=seed, steps=4, validation_every=2,
        model=ModelConfig(context_length=32,blocks=1,width=8,heads=2,feed_forward_width=16),
        max_target_tokens=128,max_seconds=30)


def fake_score(state, arm, selector):
    # Explicit constant-answer observations: not a model invocation or capability result.
    rows,split=numerical.view(arm,selector)
    from trinite.foundations import majority_answers
    baseline=majority_answers(rows)['add'];predictions=[]
    for row in rows:
        if row['split']!=split:continue
        gold=[*row['answer'].encode(),EOS];out=[48,EOS];base=[*baseline.encode(),EOS]
        predictions.append({'example_identity':row['example_identity'],'family_id':row['family_id'],
            'task':'add','split':split,'expected_tokens':gold,'generated_tokens':out,
            'correct':out==gold,'baseline_tokens':base,'baseline_correct':base==gold})
    return {'model_identity':numerical.model_identity(state.model),
            'summary':metrics(predictions),'predictions':predictions}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def patches(self):
        stack=ExitStack()
        stack.enter_context(patch.object(numerical,'plan_for',side_effect=tiny))
        stack.enter_context(patch.object(workflow,'MILESTONES',(0,2,4)))
        stack.enter_context(patch.object(workflow,'score',side_effect=fake_score))
        return stack

    def packet(self, arm='control', seed=0):
        raw=json_bytes(request.core());rid=identity(raw)
        out=self.root/f'{arm}-{seed}';out.mkdir()
        stat=out.stat()
        cid=workflow.run(out,raw,rid,arm,seed,(stat.st_dev,stat.st_ino))
        return out,raw,rid,cid

    def test_fixed_production_plan_and_six_cells(self):
        self.assertEqual(len(request.cells()),6)
        for arm,seed in request.cells():
            p=numerical.plan_for(seed)
            self.assertEqual((p.steps,p.batch_size,p.max_target_tokens),(4000,8,76800))
            self.assertEqual(p.model.parameter_count,86528)
        for arm,seed in [('other',0),('control',True),('control',3)]:
            with self.assertRaises(ContractError):request.cell_name(arm,seed)

    def test_plan_rejects_changed_schema_order_and_bounds(self):
        for change in ({'schema':'old'},{'order':'old'},{'steps':4001*2},{'batch_size':9}):
            with self.assertRaises(ContractError):replace(numerical.plan_for(0),**change)

    def test_request_binds_sources_environment_data_and_protocol(self):
        raw=json_bytes(request.core());rid=identity(raw)
        self.assertEqual(request.validate(raw,rid),parse_json(raw,canonical=True))
        bad=parse_json(raw,canonical=True);bad['protocol']['steps']=1
        encoded=json_bytes(bad)
        with self.assertRaises(ContractError):request.validate(encoded,identity(encoded))
        with self.assertRaises(ContractError):request.validate(raw,'sha256:'+'0'*64)
        with patch.object(request,'sources',return_value={}):
            with self.assertRaises(ContractError):request.validate(raw,rid)

    def test_wrong_import_origin_rejected(self):
        import trinite
        with patch.object(trinite,'__file__',str(request.ROOT/'unbound.py')):
            with self.assertRaises(ContractError):request.sources()

    def test_state_has_schedule_visits_and_unique_train_diagnostics(self):
        with self.patches():
            left=numerical.state_for('control',0);right=numerical.state_for('candidate',0)
            self.assertEqual(len(left.data['train']),40)
            self.assertEqual(len(left.data['validation']),20)
            self.assertEqual(len(right.data['validation']),40)
            self.assertEqual(left.initialization_identity,right.initialization_identity)
            heldout={r['example_identity'] for s in data.STRATA for r in data.examples(s)}
            for state in (left,right):
                self.assertFalse(heldout&{r['identity'] for rows in state.data.values() for r in rows})

    def test_shared_updater_and_safe_checkpoint_roundtrip(self):
        with self.patches():
            state=numerical.state_for('candidate',1)
            workflow.run_steps(state,stop_after=2)
            raw,meta=numerical.snapshot(state,'candidate',1,'sha256:'+'1'*64)
            restored=numerical.restore(raw,meta,'candidate',1,'sha256:'+'1'*64)
            self.assertEqual(numerical.model_identity(state.model),numerical.model_identity(restored.model))
            with self.assertRaises(ContractError):numerical.restore(raw,meta,'control',1,'sha256:'+'1'*64)
            with self.assertRaises(ContractError):numerical.restore(raw+b'bad',meta,'candidate',1,'sha256:'+'1'*64)

    def test_real_admission_and_upstream_custody_tiny_cell_audit(self):
        with self.patches():
            out,raw,rid,cid=self.packet()
            before=controller.bound_files(out)
            result=workflow.audit(out,raw,rid,'control',0,cid)
            self.assertTrue(result['verified'])
            self.assertFalse(result['optimizer_replayed'])
            self.assertEqual(before,controller.bound_files(out))
            self.assertEqual(result['report']['development']['familiar-answer']['correct'],0)

    def test_completion_external_identity_and_changed_artifact_reject(self):
        with self.patches():
            out,raw,rid,cid=self.packet()
            with self.assertRaises(ContractError):workflow.audit(out,raw,rid,'control',0,'sha256:'+'0'*64)
            (out/'training-0.json').write_bytes(b'{}\n')
            with self.assertRaises(ContractError):workflow.audit(out,raw,rid,'control',0,cid)

    def test_unknown_or_symlink_packet_member_rejected(self):
        with self.patches():
            out,raw,rid,cid=self.packet()
            (out/'extra.json').write_text('{}')
            with self.assertRaises(ContractError):workflow.audit(out,raw,rid,'control',0,cid)
            (out/'extra.json').unlink();(out/'bad').symlink_to(out/'report.json')
            with self.assertRaises(ContractError):workflow.audit(out,raw,rid,'control',0,cid)

    def test_foreign_score_context_and_optimistic_score_rejected(self):
        with self.patches():
            state=numerical.state_for('control',0)
            r=fake_score(state,'control','familiar-answer');mid=numerical.model_identity(state.model)
            numerical.check_score(r,'control','familiar-answer',mid)
            r['summary']['correct']=36
            with self.assertRaises(ContractError):numerical.check_score(r,'control','familiar-answer',mid)
            r=fake_score(state,'control','familiar-answer')
            with self.assertRaises(ContractError):numerical.check_score(r,'control','familiar-answer','sha256:'+'0'*64)

    def test_output_overlap_and_existing_paths_rejected(self):
        req=self.root/'request.json';req.write_text('{}')
        for out in (request.ROOT/'reports','/',self.root):
            with self.assertRaises(ContractError):controller.protect(out,req)
        out=self.root/'result';out.mkdir()
        with self.patches():
            raw=json_bytes(request.core());req.write_bytes(raw)
            with self.assertRaises(FileExistsError):controller.run_matrix(out,req,identity(raw))

    def test_run_requires_explicit_authorization(self):
        with patch('sys.stderr',new=io.StringIO()):
            self.assertEqual(controller.main(['run','--request',str(self.root/'x'),'--request-identity','sha256:'+'0'*64,'--output',str(self.root/'out')]),1)
        self.assertFalse((self.root/'out').exists())

    def matrix_with_launch(self, launch):
        req=self.root/'request.json';raw=json_bytes(request.core());req.write_bytes(raw)
        out=self.root/'matrix'
        with patch.object(controller,'launch',side_effect=launch):
            result=controller.run_matrix(out,req,identity(raw))
        return out,raw,identity(raw),result

    def test_failed_workers_all_retained_and_audited_without_success(self):
        with self.patches():
            out,raw,rid,result=self.matrix_with_launch(lambda *args:7)
            self.assertEqual(result['outcome'],'incomplete');self.assertEqual(len(result['cells']),6)
            a=controller.audit_matrix(out,raw,rid,result['matrix_identity'])
            self.assertEqual(a['reports'],{});self.assertFalse(a['learning_adequate'])
            name=request.cell_name(*request.cells()[0]);(out/name/'attempt.json').write_text('{}')
            with self.assertRaises(ContractError):controller.audit_matrix(out,raw,rid,result['matrix_identity'])

    def test_timeout_and_interruption_conserve_remaining_cells(self):
        def stop(*args):raise KeyboardInterrupt()
        with self.patches():
            out,raw,rid,result=self.matrix_with_launch(stop)
            values=list(result['cells'].values())
            self.assertEqual(values[0]['outcome'],'interrupted')
            self.assertEqual([r['outcome'] for r in values[1:]],['not_attempted']*5)
            controller.audit_matrix(out,raw,rid,result['matrix_identity'])

    def test_nonzero_worker_cannot_promote_completion(self):
        def incomplete(out,req,rid,arm,seed,expected):
            store=Store(out,expected)
            try:store.write('completion.json',b'{}\n')
            finally:store.close()
            return 9
        with self.patches():
            out,raw,rid,result=self.matrix_with_launch(incomplete)
            self.assertTrue(all(r['completion_identity']is None for r in result['cells'].values()))
            controller.audit_matrix(out,raw,rid,result['matrix_identity'])

    def test_all_six_tiny_cells_complete_and_pair_without_target_run(self):
        def tiny_worker(out,req,rid,arm,seed,expected):
            workflow.run(out,Path(req).read_bytes(),rid,arm,seed,expected)
            return 0
        with self.patches():
            out,raw,rid,result=self.matrix_with_launch(tiny_worker)
            self.assertEqual(result['outcome'],'completed')
            receipt=controller.audit_matrix(out,raw,rid,result['matrix_identity'])
            self.assertEqual(len(receipt['reports']),6)
            self.assertEqual(set(receipt['paired_descriptive_counts']),{'0','1','2'})
            self.assertFalse(receipt['fresh_confirmation'])
            original=workflow.audit
            for field,value in [('initialization_identity','sha256:'+'f'*64),('target_tokens',-1)]:
                def wrong_pair(*args,**kwargs):
                    actual=original(*args,**kwargs)
                    if args[3]=='candidate':actual['report'][field]=value
                    return actual
                with patch.object(workflow,'audit',side_effect=wrong_pair):
                    with self.assertRaisesRegex(ContractError,'paired initialization/exposure'):
                        controller.audit_matrix(out,raw,rid,result['matrix_identity'])
            with self.assertRaises(ContractError):controller.audit_matrix(out,raw,rid,'sha256:'+'0'*64)

    def test_timeout_preserves_all_six_attempts(self):
        def timeout(*args):raise subprocess.TimeoutExpired(['fixed-test-double'],900)
        with self.patches():
            out,raw,rid,result=self.matrix_with_launch(timeout)
            self.assertTrue(all(r['outcome']=='timed_out' for r in result['cells'].values()))
            controller.audit_matrix(out,raw,rid,result['matrix_identity'])

    def test_matrix_root_has_strict_member_bound(self):
        root=self.root/'bounded';root.mkdir()
        for i in range(9):(root/str(i)).write_text('x')
        with self.assertRaises(ContractError):controller.matrix_layout(root)

    def test_matrix_root_link_rejected(self):
        root=self.root/'bounded';root.mkdir();(root/'link').symlink_to(self.root)
        with self.assertRaises(ContractError):controller.matrix_layout(root)

    def test_actual_shared_scorer_on_untrained_tiny_training_rows(self):
        with self.patches():
            state=numerical.state_for('control',0)
            record=numerical.score(state,'control','train')
            numerical.check_score(record,'control','train',numerical.model_identity(state.model))
            self.assertEqual(record['summary']['examples'],20)

    def test_milestone_history_prefix_rejected(self):
        with self.patches():
            out,raw,rid,cid=self.packet()
            original=workflow.restore
            def inconsistent(*args,**kwargs):
                state=original(*args,**kwargs)
                if state.step==2:state.history[0]['loss_hex']='0x0.0p+0'
                return state
            with patch.object(workflow,'restore',side_effect=inconsistent):
                with self.assertRaisesRegex(ContractError,'one retained trajectory'):
                    workflow.audit(out,raw,rid,'control',0,cid)


    def test_corrupt_schedule_history_rejected_by_shared_progress(self):
        with self.patches():
            state=numerical.state_for('control',0);workflow.run_steps(state,stop_after=2)
            payload,raw=numerical.snapshot(state,'control',0,'sha256:'+'1'*64)
            for field,value in [('examples',['sha256:'+'0'*64]*8),('target_tokens',999)]:
                record=parse_json(raw,canonical=True);record['history'][0][field]=value
                with self.assertRaises(ContractError):
                    numerical.restore(payload,json_bytes(record),'control',0,'sha256:'+'1'*64)

    def test_authoritative_store_limits_match_protocol(self):
        root=self.root/'store';root.mkdir();store=Store(root)
        try:
            one=b'x'*controller.PROTOCOL['max_artifact_bytes']
            store.write('one',one)
            with self.assertRaises(ContractError):store.write('oversize',one+b'x')
            store.write('two',one)
            self.assertEqual(store.bytes,controller.PROTOCOL['max_retained_payload_bytes'])
            with self.assertRaises(ContractError):store.write('over-total',b'x')
        finally:store.close()

    def test_fixed_launcher_log_drain_is_bounded(self):
        root=self.root/'worker';root.mkdir();st=root.stat()
        class Process:
            def __init__(self):self.stdout=io.BytesIO(b'x'*100);self.killed=False
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def wait(self,timeout=None):return 0
            def kill(self):self.killed=True
        process=Process()
        with patch.dict(controller.PROTOCOL,{'max_worker_log_bytes':8}),patch.object(controller.subprocess,'Popen',return_value=process) as popen:
            with self.assertRaisesRegex(ContractError,'log exceeds'):
                controller.launch(root,self.root/'request.json','sha256:'+'1'*64,'control',0,(st.st_dev,st.st_ino))
            self.assertTrue(process.killed)
            self.assertEqual((root/'worker.log').stat().st_size,8)
            self.assertFalse(popen.call_args.kwargs['shell'])
            self.assertEqual(popen.call_args.args[0][:3],[sys.executable,'-B',str(request.ROOT/'scripts/run_addition_development.py')])


    def test_preflight_is_standard_library_and_cannot_run_target(self):
        env={**os.environ,'PYTHONPATH':str(request.ROOT/'src')+os.pathsep+str(request.ROOT/'scripts')}
        result=subprocess.run([sys.executable,'-B',str(request.ROOT/'scripts/run_addition_development.py'),'preflight'],env=env,capture_output=True,text=True,check=True)
        self.assertFalse(json.loads(result.stdout)['model_executed'])


if __name__=='__main__':unittest.main()
