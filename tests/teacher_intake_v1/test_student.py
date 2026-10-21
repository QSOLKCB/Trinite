"""Bounded native one-update and orchestration tests, not the target experiment."""
import copy
import unittest
from unittest.mock import patch
import torch
from trinite.contracts import ContractError, identity, json_bytes
from trinite.training import run_steps, require_environment, model_identity
from teacher_intake_v1 import student
from teacher_intake_v1.intake import seal
import test_intake


class Student(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        require_environment()

    def packet(self,count=32):
        fixture=test_intake.IntakeContracts()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return seal(fixture.frozen,fixture.capture(count))

    def test_underfill_blocks_before_model(self):
        with self.assertRaises(ContractError):student.pools(self.packet(15))

    def test_equal_modulus_visit_schedule_and_isolation(self):
        pools=student.pools(self.packet(23))
        for seed in (0,1):
            a=student.ordered(pools['teacher_selected'],seed)
            b=student.ordered(pools['oracle_only'],seed)
            self.assertEqual([r['formal_spec']['n'] for r in a],
                             [r['formal_spec']['n'] for r in b])
            self.assertEqual([a[i%len(a)]['formal_spec']['n'] for i in range(1024)],
                             [b[i%len(b)]['formal_spec']['n'] for i in range(1024)])
        self.assertEqual(len(pools['heldout']),157)
        self.assertTrue(all(r['formal_spec']['n'] in (11,13) for r in pools['heldout']))
        self.assertTrue(all(r['split']=='train' for r in pools['teacher_selected']+pools['oracle_only']))

    def test_three_single_updates_no_target_run(self):
        pools=student.pools(self.packet())
        a=student.create(pools['teacher_selected'],0)
        b=student.create(pools['oracle_only'],0)
        self.assertEqual(a.initialization_identity,b.initialization_identity)
        for state in (a,b):
            run_steps(state,stop_after=1)
            self.assertEqual(state.target_tokens,12)
            self.assertNotEqual(model_identity(state.model),state.initialization_identity)
            self.assertEqual(state.data['validation'],state.data['train'])
        rows=student.diagnostic_rows()
        self.assertEqual(len({r['answer'] for r in rows}),4)
        self.assertTrue(all(r['source_item']['split']=='train' for r in rows))
        d=student.create(rows,0,True)
        run_steps(d,stop_after=1)
        self.assertEqual(d.target_tokens,8)

    def test_teacher_prose_never_tokenized(self):
        packet=self.packet();pools=student.pools(packet)
        for arm in student.PROTOCOL['arms']:
            state=student.create(pools[arm],0)
            for row in state.data['train']:
                self.assertNotIn('raw_base64',row)
        self.assertEqual(packet['raw_admission'],'QUARANTINED_NOT_TRAINING_DATA')

    def test_failed_diagnostic_gates_all_comparison_and_heldout(self):
        frozen=student.request(self.packet(),require_environment())
        records=[]
        def fake_update(state):
            state.step=256;state.target_tokens=2048;state.cursor=1024
        def fake_score(model,rows):
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['split']=='train' for r in rows))
            return {'records':[],'counts':{'train':{'rows':4,'exact':0}}}
        with patch.object(student,'run_steps',create=True):
            # report imports the authoritative updater locally.
            with patch('trinite.training.run_steps',side_effect=fake_update), \
                 patch.object(student,'score',side_effect=fake_score), \
                 patch.object(student,'create',wraps=student.create) as create:
                result=student.report(expected_request=frozen,on_cell=records.append)
        self.assertEqual(create.call_count,1)
        self.assertEqual(result['outcome'],'blocked_train_only_learnability')
        self.assertEqual(result['cells'],[])
        self.assertEqual(records[0]['phase'],'diagnostic_completed')

    def test_request_tamper_rejected_before_update(self):
        frozen=student.request(self.packet(),require_environment())
        frozen['protocol']=copy.deepcopy(frozen['protocol'])
        frozen['protocol']['steps']=1
        with patch('trinite.training.run_steps') as run:
            with self.assertRaises(ContractError):student.report(expected_request=frozen)
            run.assert_not_called()


    def test_success_sequence_all_updates_before_heldout(self):
        frozen=student.request(self.packet(),require_environment())
        updates=[];scores=[];records=[]
        def fake_update(state):
            updates.append(state)
            state.step=256;state.cursor=1024
            state.target_tokens=2048 if len(updates)==1 else 3072
        def fake_score(model,rows):
            scores.append(rows)
            if len(scores)==1:
                self.assertEqual(len(updates),1)
                self.assertEqual(len(rows),4)
                return {'records':[],'counts':{'train':{'rows':4,'exact':4}}}
            self.assertEqual(len(updates),5)
            self.assertTrue(any(r['split']=='test' for r in rows))
            return {'records':[],'counts':{'train':{'rows':32,'exact':0}}}
        with patch('trinite.training.run_steps',side_effect=fake_update), \
             patch.object(student,'score',side_effect=fake_score):
            result=student.report(expected_request=frozen,on_cell=records.append)
        self.assertEqual(result['outcome'],'completed')
        self.assertEqual(len(result['cells']),4)
        self.assertEqual(result['request_identity'],identity(json_bytes(frozen)))
        self.assertEqual([r['phase'] for r in records],
                         ['diagnostic_completed']+['training_completed']*4+['scoring_completed']*4)
        self.assertEqual(len(scores),9)
        for index in (0,2):
            self.assertEqual(result['cells'][index]['initial_model_identity'],
                             result['cells'][index+1]['initial_model_identity'])
