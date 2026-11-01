"""Optimizer-facing lifecycle uses software doubles, never scientific training."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import queue_evaluation_v1 as gate
from queue_learning_v1.artifacts import read_json,write_json
from trinite.contracts import json_bytes
from context_symbol_learning_v1 import controller,custody,data,scoring
from procedural_dataset_v1.cases import expected_target


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.args=SimpleNamespace(out=self.root/'run',frozen=self.root/'frozen',freeze_sha='a'*64,
            anchors=self.root/'preparation.json',anchors_sha='b'*64,approval=self.root/'owner.json',run_anchor=self.root/'anchor.json')
        write_json(self.args.approval,{'synthetic_engineering_approval_double':True})
        rows=data.inventory()
        self.content=dict(rows=rows,train=data.training_rows(rows,lambda r:expected_target('context',r)),
            public=[dict(id=r['id'],observation=data.observation(r)) for r in rows])
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(custody,'context',return_value=({'producer_sha':'c'*40},self.content)))
        self.auth=self.stack.enter_context(patch.object(custody,'authorize'))
        self.stack.enter_context(patch.object(controller,'launch',side_effect=self.worker))
        self.events=[];self.fail=None;self.partial=False
        self.oracle=self.stack.enter_context(patch.object(controller.oracle,'target',side_effect=self.target))

    def worker(self,argv,seconds,progress=None,max_updates=640,max_targets=12232):
        out=Path(argv[argv.index('--out')+1]);out.mkdir()
        self.events.append(out.name)
        self.assertGreater(seconds,0);self.assertLessEqual(seconds,900)
        if argv[0]=='_validate-fit':
            write_json(out/'certificate.json',dict(engineering_double=True))
        else:
            self.assertEqual(max_updates,128 if out.name=='fit-17' else 256)
            self.assertEqual(max_targets,2432 if out.name=='fit-17' else 4900)
            write_json(out/'training.json',dict(engineering_double=True))
            if out.name!='fit-17':
                self.assertIn('--fit-certificate-sha',argv)
                self.assertTrue((out.parent/'fit-certificate.json').exists())
                public=controller.packet_content(self.content)
                if out.name==self.fail:
                    public=public[:1] if self.partial else []
                gate.write(out/'predictions.jsonl',b''.join(json_bytes(dict(id=r['id'],raw=dict(status='malformed',tokens=[]))) for r in public))
        return dict(exit_code=-9 if out.name==self.fail else 0,failure='timeout' if out.name==self.fail else None)

    def target(self,r,log=None):
        self.assertEqual(self.events,['fit-17','fit-validation','main-17','main-29'])
        self.assertTrue((self.args.out/'committed-predictions.json').exists())
        self.assertTrue(Path(str(self.args.run_anchor)+'.predictions.json').exists())
        committed=read_json(self.args.out/'committed-predictions.json')
        scoring.check_predictions(committed,complete=True)
        return '0'  # Synthetic software-double outcome, never scientific heldout labels.

    def test_complete_sequence_before_any_heldout_score(self):
        result=controller.execute(self.args)
        self.assertFalse(result['failed'],result)
        self.assertEqual(self.events,['fit-17','fit-validation','main-17','main-29'])
        self.assertEqual(self.oracle.call_count,288)
        self.assertEqual([r['status'] for r in read_json(self.args.out/'attempts.json')],['completed']*3)

    def test_complete_software_double_score_verification_is_read_only(self):
        from context_symbol_learning_v1 import verification,numerical,workers
        from queue_learning_v1.artifacts import inventory
        controller.execute(self.args)
        write_json(self.args.out/'supervisor.json',dict(exit_code=0,failure=None))
        sha=write_json(self.args.run_anchor,dict(schema='trinite.context-symbol-evidence-anchor.v1',
            freeze_sha256=self.args.freeze_sha,files=inventory(self.args.out)))
        before=inventory(self.args.out)
        state=SimpleNamespace(step=128,target_tokens=2432,model=object())
        args=SimpleNamespace(**vars(self.args),run=self.args.out,run_sha=sha)
        with (patch.object(workers,'load_checkpoint',return_value=state),
              patch.object(workers,'validate_fit',return_value=dict(engineering_double=True)),
              patch.object(numerical,'require_fit'),patch.object(controller.oracle,'target',return_value='0')):
            result=verification.verify(args)
        self.assertTrue(result['complete']);self.assertTrue(result['scores_recomputed'])
        self.assertEqual(inventory(self.args.out),before)

    def test_fit_failure_and_independent_fit_failure_stop_both_main_seeds(self):
        self.fail='fit-17'
        self.assertTrue(controller.execute(self.args)['failed'])
        self.assertEqual(self.events,['fit-17'])
        self.oracle.assert_not_called()
        self.assertEqual([r['status'] for r in read_json(self.args.out/'attempts.json')],['timeout','not_attempted','not_attempted'])

    def test_independent_fit_validation_failure_stops(self):
        self.fail='fit-validation'
        self.assertTrue(controller.execute(self.args)['failed'])
        self.assertEqual(self.events,['fit-17','fit-validation'])
        self.oracle.assert_not_called()

    def test_partial_failure_retains_predictions_and_stops_next_seed(self):
        self.fail='main-17';self.partial=True
        self.assertTrue(controller.execute(self.args)['failed'])
        self.assertEqual(self.events,['fit-17','fit-validation','main-17'])
        self.oracle.assert_not_called()
        result=read_json(self.args.out/'predictions.json')
        group=[r for r in result if r['method']=='main-17']
        self.assertEqual(sum(r['raw']['status']=='malformed' for r in group),1)
        self.assertEqual(len(group),288)
        self.assertIsNone(read_json(self.args.out/'run.json')['summary']['methods']['main-17']['test']['exact_accuracy'])

    def test_consumed_freeze_prevents_retry_in_another_output(self):
        self.fail='fit-17'
        controller.execute(self.args)
        self.events.clear();self.args.out=self.root/'retry'
        self.assertTrue(controller.execute(self.args)['failed'])
        self.assertEqual(self.events,[])
        self.oracle.assert_not_called()

    def test_rejected_approval_launches_nothing(self):
        self.auth.side_effect=ValueError('wrong freeze approval')
        self.assertTrue(controller.execute(self.args)['failed'])
        self.assertEqual(self.events,[])
        self.oracle.assert_not_called()

    def test_whole_budget_exhaustion_launches_nothing(self):
        with patch.object(controller.time,'monotonic',side_effect=[0.,2700.,2701.]):
            self.assertTrue(controller.execute(self.args)['failed'])
        self.assertEqual(self.events,[])
        self.oracle.assert_not_called()

    def test_public_packet_has_no_heldout_provenance_labels(self):
        packet=controller.public_packet(self.content,'a'*64)
        self.assertEqual(len(packet['public']),288)
        self.assertEqual(set(packet),{'freeze_sha256','train','public'})
        for row in packet['public']:
            self.assertEqual(set(row),{'id','observation'})
            self.assertEqual(set(row['observation']),{'schema','contexts','symbols','meanings','table','context_mask','symbol_mask','query'})


class ApprovalTests(unittest.TestCase):
    def test_exact_owner_producer_freeze_and_budget_declaration(self):
        from context_symbol_learning_v1.spec import AUTHORIZATION
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'approval.json'
            approved=dict(schema='trinite.context-symbol-owner-approval.v1',approved_by='Trent Slade',
                producer_sha='c'*40,freeze_sha256='a'*64,reviewed_merged_green=True,authorization=AUTHORIZATION,
                supervised_token_ceiling=12232,optimizer_updates_ceiling=640)
            write_json(path,approved)
            custody.authorize({'producer_sha':'c'*40},'a'*64,path)
            for key,value in [('producer_sha','d'*40),('freeze_sha256','e'*64),('reviewed_merged_green',1),
                              ('supervised_token_ceiling',12155),('authorization','old queue approval')]:
                altered=deepcopy(approved);altered[key]=value
                path.unlink();write_json(path,altered)
                with self.subTest(key=key),self.assertRaises(ValueError):
                    custody.authorize({'producer_sha':'c'*40},'a'*64,path)


if __name__=='__main__':
    unittest.main()
