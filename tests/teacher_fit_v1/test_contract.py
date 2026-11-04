# SPDX-License-Identifier: MPL-2.0
"""Actual TT-002 admission and numerical evidence contracts; no target training."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from teacher_dataset_v1 import core as tt002
from teacher_fit_v1 import custody as io, evidence, metrics, spec, supervisor
from teacher_fit_v1.native import NativeBudget
from teacher_logit_v1 import arithmetic as ar


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.dataset=cls.root/'dataset';cls.anchor=tt002.prepare(cls.dataset)
        cls.model=spec.settings()[1]
        cls.raw=(cls.dataset/'train.jsonl').read_bytes()
        cls.rows,cls.panel,cls.schedule=spec.admit_train(cls.raw,cls.model)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def request(self, seed=17):
        return {'source':{'commit':'a'*40},'seed':seed,'dataset_manifest_identity':self.anchor,
            'schedule_identity':identity(json_bytes(self.schedule['visit_ids'])),
            'environment':{'synthetic':'test-only'},'output_directory':str(self.root/'attempt'),
            'limits':spec.LIMITS,'panel':self.panel,'selection':self.schedule,
            'native_config':spec.native_config(self.model,seed)}

    def test_actual_tt002_admission_and_panel(self):
        self.assertTrue(tt002.verify(self.dataset,self.anchor)['verified'])
        self.assertEqual(len(self.rows),972);self.assertEqual(len(self.panel),72)
        self.assertEqual([sum(metrics.task_name(r)==t for r in self.panel)
                          for t in ('apply','inverse','check_0','check_1')],[18]*4)
        self.assertEqual(self.schedule['visit_ids'][:72],[r['id'] for r in self.panel])
        self.assertLessEqual(self.schedule['scheduled_target_tokens'],4096)

    def test_train_ingress_never_opens_privileged_labels(self):
        original=Path.open
        def restricted(path,*args,**kwargs):
            if path.name in ('privileged-targets.jsonl','validation-public.jsonl','test-public.jsonl'):
                self.fail('execution admission accessed heldout file')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',restricted):
            self.assertEqual(spec.train_packet(self.dataset,self.anchor)[2],self.schedule)

    def test_wrong_identity_oracle_prompt_mask_split_rejected(self):
        for key,value in (('id','sha256:'+'0'*64),('answer','2222'),('prompt','bad'),
                          ('supervised_tokens',4),('split','test'),('complete_context_tokens',999)):
            rows=copy.deepcopy(self.rows);rows[0][key]=value
            with self.subTest(key=key),self.assertRaises((ValueError,ContractError)):
                spec.admit_train(tt002.jsonl(rows),self.model)
        for rows in (self.rows[:-1],self.rows+[self.rows[0]],self.rows[:-1]+[self.rows[0]]):
            with self.assertRaises((ValueError,ContractError)):
                spec.admit_train(tt002.jsonl(rows),self.model)

    def test_independent_oracle_disagreement_is_rejected(self):
        with patch('qutrit_conformance_v1.expected',return_value='2222'),self.assertRaises(ContractError):
            spec.admit_train(self.raw,self.model)

    def test_rights_and_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            for p in self.dataset.iterdir():(d/p.name).write_bytes(p.read_bytes())
            (d/'dataset-rights.json').write_bytes(b'{}\n')
            with self.assertRaises(ContractError):spec.train_packet(d,self.anchor)
        with patch.object(tt002,'rights',return_value=({},{})),self.assertRaises(ContractError):
            spec.train_packet(self.dataset,self.anchor)

    def test_three_seeds_share_entire_visit_inventory(self):
        with patch.object(spec,'source',return_value={'commit':'a'*40,'members':[]}),patch.object(spec,'check_review'):
            requests=[spec.request_for(self.dataset,self.anchor,seed,self.root,{}, {'synthetic':True}) for seed in (17,29,43)]
        schedules=[r['selection']['visit_ids'] for r in requests]
        self.assertEqual(schedules[0],schedules[1]);self.assertEqual(schedules[1],schedules[2])
        self.assertEqual(len(schedules[0]),1024)
        self.assertEqual(len({identity(json_bytes(r)) for r in requests}),3)

    def test_three_separate_approvals_required(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(spec,'source',return_value={'commit':'a'*40,'members':[]}),patch.object(spec,'check_review'):
            root=Path(tmp);(root/'requests').mkdir();grants={};requests=[]
            for seed in (17,29,43):
                req=spec.request_for(self.dataset,self.anchor,seed,root,{}, {'synthetic':True})
                anchor=identity(json_bytes(req));requests.append(req)
                io.record(root/'requests'/f'seed-{seed}.json',req)
                p=root/f'synthetic-test-grant-{seed}.json';io.record(p,spec.approval_expected(req,anchor))
                grants[str(seed)]={'path':str(p),'identity':identity(io.read(p)),'request_identity':anchor}
            p=root/'set.json';io.record(p,grants)
            self.assertEqual(len(spec.approval_set(p,identity(io.read(p)),requests[0],self.dataset)[1]),3)
            (root/'synthetic-test-grant-43.json').unlink()
            with self.assertRaises(FileNotFoundError):spec.approval_set(p,identity(io.read(p)),requests[0],self.dataset)

    def test_request_reordered_schedule_and_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(spec,'source',return_value={'commit':'a'*40,'members':[]}),patch.object(spec,'check_review'):
            p=Path(tmp)/'request.json';r=spec.request_for(self.dataset,self.anchor,17,Path(tmp),{}, {'synthetic':True})
            io.record(p,r);spec.verify_request(p,identity(io.read(p)),self.dataset)
            r['selection']['visit_ids'].reverse();p.write_bytes(json_bytes(r))
            with self.assertRaises(ContractError):spec.verify_request(p,identity(io.read(p)),self.dataset)

    def test_frozen_native_optimizer_budget(self):
        p=spec.native_config(self.model,17)
        for key,value in {'learning_rate':'0.001','beta1':'0.9','beta2':'0.999',
                          'weight_decay':'0.01','gradient_clip':'1','steps':256,
                          'batch_size':4,'max_target_tokens':4096,'max_seconds':600}.items():
            self.assertEqual(p[key],value)
        self.assertFalse(spec.settings()[0]['gpu_allowed'])

    def test_cumulative_native_budget_does_not_reset(self):
        times=iter([0.,250.,300.,550.,600.,701.])
        budget=NativeBudget(600,lambda:next(times));calls=[]
        budget.step(lambda:calls.append(1));budget.step(lambda:calls.append(2))
        with self.assertRaises(ContractError):budget.step(lambda:calls.append(3))
        self.assertEqual(budget.elapsed,601.)
        with self.assertRaises(ContractError):budget.step(lambda:calls.append(4))
        self.assertEqual(calls,[1,2,3])

    def test_budget_records_elapsed_on_interruption(self):
        times=iter([1.,9.]);b=NativeBudget(600,lambda:next(times))
        with self.assertRaises(KeyboardInterrupt):b.step(lambda:(_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertEqual(b.elapsed,8.)

    def test_approval_missing_mismatch_and_boolean_types(self):
        request=self.request();anchor='sha256:'+'1'*64
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'synthetic-test-approval.json'
            with self.assertRaises(FileNotFoundError):spec.approval(path,anchor,request,anchor)
            value=spec.approval_expected(request,anchor)
            io.record(path,value)
            a=identity(io.read(path));self.assertEqual(spec.approval(path,a,request,anchor),value)
            for key,bad in (('seed',29),('training_authorized',1),('source_commit','b'*40)):
                v=dict(value);v[key]=bad;path.write_bytes(json_bytes(v))
                with self.assertRaises(ContractError):spec.approval(path,identity(io.read(path)),request,anchor)

    def test_ci_target_dispatch_rejected_before_admission(self):
        with patch.dict(os.environ,{'CI':'true'}),patch.object(spec,'verify_request') as verify:
            with self.assertRaises(ContractError):supervisor.dispatch(None,None,None,None,None)
            verify.assert_not_called()

    def test_exact_review_and_required_checks(self):
        sha='a'*40
        r={'schema':spec.SCHEMA+'.review','commit':sha,'reviewed':True,
           'review_url':'https://github.com/QSOLKCB/Trinite/pull/999',
           'checks':[{'name':n,'head_sha':sha,'conclusion':'success',
                     'url':'https://github.com/QSOLKCB/Trinite/actions/runs/1'}
                    for n in ('FIT contracts','TT-002 contracts','TT-003 contracts')]}
        spec.check_review(r,sha)
        for bad in (dict(r,commit='b'*40),dict(r,reviewed=False),dict(r,checks=r['checks'][:-1])):
            with self.assertRaises(ContractError):spec.check_review(bad,sha)
        r['checks'][0]['conclusion']='failure'
        with self.assertRaises(ContractError):spec.check_review(r,sha)

    def test_eos_residue_baselines_and_complete_outputs(self):
        base=metrics.baselines(self.panel)
        self.assertEqual(base['0']['groups']['check_0']['exact'],18)
        self.assertEqual(base['1']['groups']['check_1']['exact'],18)
        predictions=[{'id':r['id'],'tokens':[257]} for r in self.panel]
        s=metrics.score(self.panel,predictions)
        self.assertEqual(s['groups']['all']['premature_eos'],72)
        self.assertEqual(sum(p['correct'] for p in s['vector_positions']),0)
        predictions[0]['tokens']=[257,48]
        with self.assertRaises(ContractError):metrics.score(self.panel,predictions)

    def observations(self):
        z=[-20.]*259;z[257]=0.
        return metrics.observe(self.panel,lambda prefix:ar.pack(z))

    def test_all_vectors_metrics_and_aggregates_recomputed(self):
        index,raw,measures,predictions=self.observations()
        self.assertEqual(len(index),252+72)
        self.assertEqual(metrics.recalculate(self.panel,predictions,index,raw),measures)
        summary=metrics.summarize(self.panel,predictions,index,measures)
        self.assertEqual(summary['gold_prefix']['all']['positions'],252)
        self.assertEqual(summary['groups']['all']['exact'],0)
        self.assertEqual(json_bytes(summary),json_bytes(parse_json(json_bytes(summary),canonical=True)))
        for corrupt in (raw[:-4],raw+raw[:4]):
            with self.assertRaises(ContractError):metrics.recalculate(self.panel,predictions,index,corrupt)
        bad=copy.deepcopy(index);bad[0]['prefix_tokens'].append(48)
        with self.assertRaises(ContractError):metrics.recalculate(self.panel,predictions,bad,raw)
        badpred=copy.deepcopy(predictions);badpred[0]=badpred[1]
        with self.assertRaises(ContractError):metrics.recalculate(self.panel,badpred,index,raw)

    def test_greedy_can_never_disagree_with_retained_argmax(self):
        index,raw,measures,predictions=self.observations()
        bad=bytearray(raw);offset=next(r['offset_bytes'] for r in index if r['probe']=='greedy')
        z=[0.]*259;z[48]=2.
        bad[offset:offset+ar.VECTOR.size]=ar.pack(z)
        with self.assertRaises(ContractError):metrics.recalculate(self.panel,predictions,index,bytes(bad))

    def test_partial_numeric_records_recomputed_with_torn_tails_preserved(self):
        index,raw,measures,predictions=self.observations()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            io.write(folder/'index.jsonl',b''.join(json_bytes(v) for v in index[:3])+b'{')
            io.write(folder/'metrics.jsonl',b''.join(json_bytes(v) for v in measures[:2])+b'{')
            io.write(folder/'logits.f32le',raw[:2*ar.VECTOR.size+8])
            before=io.inventory(folder)
            r=evidence.partial_checkpoint(self.request(),folder)
            self.assertEqual(r['metrics_recomputed'],2);self.assertEqual(r['unmatched_vector_bytes'],8)
            self.assertEqual(r['index_tail_bytes'],1);self.assertEqual(io.inventory(folder),before)
            measures[0]['target_rank']+=1
            (folder/'metrics.jsonl').write_bytes(b''.join(json_bytes(v) for v in measures[:2]))
            with self.assertRaises(ContractError):evidence.partial_checkpoint(self.request(),folder)

    def test_numerical_extremes_and_true_full_vocab_odds(self):
        for value in (-3e38,-735.,-730.,0.,730.,735.,3e38):
            z=[0.]*259;z[257]=value;z=ar.unpack(ar.pack(ar.unpack(ar.VECTOR.pack(*z))))
            m=ar.metrics(z,49)
            self.assertTrue(math.isfinite(float.fromhex(m['eos_rest_log_odds_hex'])))
            self.assertAlmostEqual(float.fromhex(m['eos_rest_log_odds_hex']),z[257]-math.log(258),places=10)
            if abs(value) in (730.,735.,3e38):self.assertIsNone(m['eos_rest_probability_check_hex'])
        z=[0.]*259;m=ar.metrics(z,48)
        self.assertEqual(m['target_rank'],49);self.assertEqual(m['best_residue_token'],48)
        self.assertAlmostEqual(float.fromhex(m['target_probability_hex']),1/259)

    def test_corrupt_replaced_missing_extra_linked_evidence(self):
        for corruption in ('replace','remove','extra','symlink','hardlink'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as tmp:
                d=Path(tmp);io.record(d/'a.json',{'a':1});anchor=io.seal(d,{'schema':'synthetic'})
                self.assertTrue(io.verify_seal(d,anchor))
                if corruption=='replace':(d/'a.json').write_bytes(b'{}\n')
                if corruption=='remove':(d/'a.json').unlink()
                if corruption=='extra':io.record(d/'b.json',{})
                if corruption=='symlink':(d/'b.json').symlink_to(d/'a.json')
                if corruption=='hardlink':os.link(d/'a.json',d/'b.json')
                with self.assertRaises(ContractError):io.verify_seal(d,anchor)

    def test_output_claim_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'attempt';io.fresh(p)
            io.record(p/'failure.json',{'preserve':True})
            with self.assertRaises(ContractError):io.fresh(p)
            self.assertTrue(io.load(p/'failure.json')['preserve'])

    def test_read_only_outputs_cannot_modify_protected_evidence(self):
        with self.assertRaises(ContractError):io.outside(self.dataset/'receipt.json',self.dataset)

    def test_archive_external_anchor_and_duplicate_inventory(self):
        from teacher_fit_v1.synthesis import archive_verify
        import hashlib,zipfile,warnings
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'archive.zip'
            with zipfile.ZipFile(p,'x') as z:z.writestr('a.json',b'{}\n')
            anchor=identity(p.read_bytes());members={'a.json':identity(b'{}\n')}
            self.assertTrue(archive_verify(p,anchor,members)['verified'])
            with self.assertRaises(ContractError):archive_verify(p,'sha256:'+'0'*64,members)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with zipfile.ZipFile(p,'a') as z:z.writestr('a.json',b'{}\n')
            with self.assertRaises(ContractError):archive_verify(p,identity(p.read_bytes()),members)

    def test_source_closure_includes_agents_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('AGENTS.md','scripts/AGENTS.md','scripts/teacher_fit_v1/AGENTS.md','tests/AGENTS.md'):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('test instructions\n')
            subprocess.run(['git','init','-q',str(root)],check=True)
            subprocess.run(['git','-C',str(root),'add','.'],check=True)
            subprocess.run(['git','-C',str(root),'-c','user.name=Conformance','-c','user.email=conformance@example.invalid',
                            'commit','-qm','synthetic source'],check=True)
            with patch.object(spec,'ROOT',root):
                receipt=spec.source();self.assertEqual(len(receipt['members']),4)
                (root/'scripts/AGENTS.md').write_text('changed\n')
                with self.assertRaises(ContractError):spec.source()

    def test_reordered_schedule_rejected_by_progress(self):
        request=self.request();ids=self.schedule['visit_ids'][:4]
        rows={r['id']:r for r in self.panel};targets=sum(rows[r]['supervised_tokens'] for r in ids)
        meta={'step':1,'cursor':4,'target_tokens':targets,'history':[{'step':1,'examples':ids,
            'target_tokens':targets,'loss_hex':1.0.hex(),'gradient_norm_hex':1.0.hex()}],
            'validation':[],'initial_train_loss_hex':1.0.hex(),'native_elapsed_hex':1.0.hex()}
        evidence.progress(request,meta,1)
        meta['history'][0]['examples']=list(reversed(ids))
        with self.assertRaises(ContractError):evidence.progress(request,meta,1)

    def test_duplicate_json_keys_and_noncanonical_serialization_rejected(self):
        for raw in (b'{"a":1,"a":1}\n',b'{ "a": 1 }\n'):
            with self.assertRaises(ContractError):parse_json(raw,canonical=True)


if __name__=='__main__':unittest.main()
