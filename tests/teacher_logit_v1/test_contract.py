# SPDX-License-Identifier: MPL-2.0
"""Synthetic contracts; no historical checkpoint is used by this suite."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from collections import Counter
from trinite.contracts import ContractError,identity,json_bytes
from teacher_logit_v1 import arithmetic as ar,core,native


def synthetic():
    rows=[];pred=[]
    for family in ('S1','S2','W'):
        for n in range(81):
            for task,answer in [('check','0'),('check','1'),('apply','0120'),('inverse','0120')]:
                rid=str(len(rows));rows.append({'id':rid,'family':family,'split':'train','answer':answer,
                    'prompt':'synthetic=','observation':{'task':task}})
                pred.append({'id':rid,'initial':{'tokens':[61]*5},'final':{'tokens':[48,48,257] if family=='W' and task=='inverse' else [48,257]}})
    return rows,{r['id'] for r in rows[:128]},pred


class Arithmetic(unittest.TestCase):
    def test_full_distribution_and_ties(self):
        m=ar.metrics([0.0]*259,48)
        self.assertEqual(m['argmax'],0);self.assertEqual(m['target_rank'],49)
        self.assertAlmostEqual(float.fromhex(m['full_eos_probability_hex']),1/259)
        self.assertAlmostEqual(float.fromhex(m['eos_rest_log_odds_hex']),-math.log(258))
        self.assertAlmostEqual(float.fromhex(m['restricted_four_token_eos_probability_hex']),.25)
        self.assertNotEqual(m['full_eos_probability_hex'],m['restricted_four_token_eos_probability_hex'])
    def test_pairwise_and_binary_odds(self):
        z=[-.25]*259;z[257]=2.0;z[49]=1.0
        m=ar.metrics(z,49);p=float.fromhex(m['full_eos_probability_hex']);q=float.fromhex(m['target_probability_hex'])
        self.assertAlmostEqual(math.log(p/q),1.0,places=12)
        self.assertAlmostEqual(float.fromhex(m['eos_rest_log_odds_hex']),math.log(p)-math.log1p(-p),places=12)
        self.assertNotEqual(m['eos_target_margin_hex'],m['eos_rest_log_odds_hex'])
    def test_common_offset(self):
        z=[(i%17)*.125 for i in range(259)];base=ar.metrics(z,50)
        for offset in (-1024.,1024.,10000.):
            m=ar.metrics([x+offset for x in z],50)
            for k in ('full_eos_probability_hex','eos_rest_log_odds_hex','eos_target_margin_hex','eos_best_residue_margin_hex','target_log_probability_hex'):
                self.assertAlmostEqual(float.fromhex(base[k]),float.fromhex(m[k]),places=12)
            self.assertEqual(base['argmax'],m['argmax']);self.assertEqual(base['target_rank'],m['target_rank'])
    def test_large_and_underflow(self):
        for sign in (-1,1):
            z=[sign*10000.0]*259;z[48]+=1
            self.assertTrue(math.isfinite(float.fromhex(ar.metrics(z,48)['target_log_probability_hex'])))
        z=[-10000.0]*259;z[257]=10000.;m=ar.metrics(z,48)
        self.assertEqual(float.fromhex(m['target_probability_hex']),0)
        self.assertEqual(float.fromhex(m['target_log_probability_hex']),-20000.)
        self.assertTrue(math.isfinite(float.fromhex(m['eos_rest_log_odds_hex'])))
    def test_fp32_roundtrip_and_bad_vectors(self):
        z=[i*.125 for i in range(259)];self.assertEqual(ar.unpack(ar.pack(z)),z)
        self.assertEqual(ar.pack([-0.0]+[0.0]*258)[:4],b'\x00\x00\x00\x80')
        for bad in ([0.0]*258,[0.0]*260,[math.nan]+[0.0]*258,[math.inf]+[0.0]*258,[-math.inf]+[0.0]*258):
            with self.assertRaises(ContractError):ar.pack(bad)
        with self.assertRaises(ContractError):ar.pack([.1]*259)
        with self.assertRaises(ContractError):ar.unpack(b'\x00'*1035)
    def test_third_token_and_residue_knowledge(self):
        z=[-4.]*259;z[61]=3.;z[257]=2.;z[49]=1.
        m=ar.metrics(z,49)
        self.assertTrue(m['eos_beats_target_loses_third']);self.assertEqual(m['target_residue_rank'],1)
        self.assertFalse(m['target_is_best_non_eos']);self.assertEqual(m['argmax_category'],'other_vocabulary')


class Inventory(unittest.TestCase):
    def test_complete_inventory_and_prefix_separation(self):
        rows,vis,pred=synthetic();idx=core.inventory(rows,vis,pred)
        self.assertEqual(len(idx),13689);self.assertEqual((len(idx)*ar.VECTOR.size),14181804)
        self.assertEqual(len({x['record_id'] for x in idx}),len(idx))
        for n,r in enumerate(idx):
            self.assertEqual(r['offset_bytes'],n*1036)
            self.assertNotIn(257,r['prefix_tokens'])
            if r['probe']=='greedy-prefix':self.assertIsNone(r['expected_next_token'])
        self.assertEqual(Counter(x['phase'] for x in idx if x['probe']=='gold-prefix'),{'residue':4860,'terminal':1944})
    def test_duplicate_missing_extra_nontrain(self):
        for mode in ('duplicate','missing','extra','heldout','post_eos'):
            rows,vis,pred=synthetic()
            if mode=='duplicate':rows[1]=rows[0];pred[1]=pred[0]
            elif mode=='missing':rows.pop()
            elif mode=='extra':rows.append(rows[0])
            elif mode=='heldout':rows[0]['split']='test'
            else:pred[0]['final']['tokens']=[257,48]
            with self.assertRaises(ContractError):core.inventory(rows,vis,pred)
    def test_summary_recalculation(self):
        idx=core.inventory(*synthetic())[:5];z=[0.0]*259
        measures=[dict(ar.metrics(z,i['positional_reference_target']),record_id=i['record_id']) for i in idx]
        summary=core.summarize(idx,measures)
        self.assertEqual(summary['positions'],5)
        self.assertEqual(summary['groups']['initial/gold-prefix/all']['argmax_counts'],{'0':5})
        with self.assertRaises(ContractError):core.summarize(idx,measures[:-1])
        with self.assertRaises(ContractError):core.summarize([idx[0],idx[0]],measures[:2])


class Custody(unittest.TestCase):
    def test_request_anchor_and_source_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);raw=json_bytes({'version':2});(p/'request.json').write_bytes(raw)
            with patch.object(core,'request_for',return_value={'version':2}):
                self.assertEqual(core.verify_request(p,identity(raw),p,p,p),{'version':2})
                with self.assertRaises(ContractError):core.verify_request(p,'sha256:'+'0'*64,p,p,p)
            with patch.object(core,'request_for',return_value={'version':3}):
                with self.assertRaises(ContractError):core.verify_request(p,identity(raw),p,p,p)
    def test_safe_read_and_partial_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'request.json').symlink_to('/etc/passwd')
            with self.assertRaises(ContractError):core.read(p,'request.json')
            with self.assertRaises(FileNotFoundError):core.read(p,'manifest.json')
        with self.assertRaises(ContractError):core.jsonl(b'{}')
    def test_missing_mutated_checkpoint_and_manifest(self):
        # Use real historical hash gates, synthetic bytes, no neural allocation.
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'request.json').write_bytes(b'{}\n');(p/'manifest.json').write_bytes(b'{}\n')
            with patch.object(core.diag.dataset_spec,'verify'):
                with self.assertRaises(FileNotFoundError):core.diag.historical_inputs(p,p)
                (p/'tensors.safetensors').write_bytes(b'not a checkpoint');(p/'metadata.json').write_bytes(b'{}\n')
                with self.assertRaisesRegex(ContractError,'identity mismatch'):core.diag.historical_inputs(p,p)
            (p/'tensors.safetensors').unlink();(p/'metadata.json').unlink()
            raw=(p/'request.json').read_bytes()
            (p/'predictions.jsonl').write_bytes(b'{}\n');(p/'summary.json').write_bytes(b'{}\n')
            with patch.object(core,'PREVIOUS_REQUEST',identity(raw)),patch.object(core.diag,'build_request',return_value={}):
                with self.assertRaisesRegex(ContractError,'manifest identity'):core.historical(p,p,p)
    def test_observer_no_mutation_and_deadline_double(self):
        idx=core.inventory(*synthetic())[:2];raw=ar.pack([0.0]*259);seen=[]
        with patch.object(native,'check_models',return_value=['a','b']) as checks,patch.object(native,'forward_vector',return_value=(raw,0)):
            native.observe(idx,[object(),object()],math.inf,lambda *x:seen.append(x))
            self.assertEqual(checks.call_count,2);self.assertEqual(len(seen),2)
            with self.assertRaisesRegex(ContractError,'deadline'):native.observe(idx,[object(),object()],0,lambda *x:None)
        with patch.object(native,'check_models',side_effect=[['a','b'],['changed','b']]),patch.object(native,'forward_vector',return_value=(raw,0)):
            with self.assertRaisesRegex(ContractError,'state changed'):native.observe(idx,[object(),object()],math.inf,lambda *x:None)
    def test_greedy_mismatch_double(self):
        idx=[x for x in core.inventory(*synthetic()) if x['probe']=='greedy-prefix'][:1]
        with patch.object(native,'check_models',return_value=['a','b']),patch.object(native,'forward_vector',return_value=(ar.pack([0.0]*259),0)):
            with self.assertRaisesRegex(ContractError,'historical greedy token'):native.observe(idx,[object(),object()],math.inf,lambda *x:None)


class RecordedEvidence(unittest.TestCase):
    def test_regeneration_and_resealed_mutations(self):
        idx=core.inventory(*synthetic())[:2]
        vectors=ar.pack([0.0]*259)*2
        measures=[dict(ar.metrics([0.0]*259,r['positional_reference_target']),record_id=r['record_id']) for r in idx]
        req={'positions':2,'logit_bytes':len(vectors)}
        execution={'schema':core.SCHEMA+'.execution','model_identities_before_after':[
            core.diag.PIN['initial_model_identity'],core.diag.PIN['teacher_model_identity']],
            'environment':{},'gold_positions':6804,'greedy_positions':6885,
            'optimizer_updates':0,'heldout_scored':False,'native_argmax_matches_historical':True}
        for change in ('none','duplicate-index','short-payload','metric','summary','missing-file'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);folder=p/'request';folder.mkdir();out=p/'result';out.mkdir();teacher=p/'teacher';teacher.mkdir()
                (folder/'request.json').write_bytes(json_bytes(req));(teacher/'metadata.json').write_bytes(json_bytes({'environment':{}}))
                payloads={'request.json':json_bytes(req),'index.jsonl':b''.join(json_bytes(x) for x in idx),
                    'logits.f32le':vectors,'metrics.jsonl':b''.join(json_bytes(x) for x in measures),
                    'summary.json':json_bytes(core.summarize(idx,measures)),'execution.json':json_bytes(execution)}
                if change=='duplicate-index':payloads['index.jsonl']=json_bytes(idx[0])*2
                if change=='short-payload':payloads['logits.f32le']=vectors[:-4]
                if change=='metric':payloads['metrics.jsonl']=json_bytes(dict(measures[0],argmax=1))+json_bytes(measures[1])
                if change=='summary':payloads['summary.json']=json_bytes({})
                for n,raw in payloads.items():(out/n).write_bytes(raw)
                manifest={'schema':core.SCHEMA+'.manifest','status':'COMPLETED_READ_ONLY','request_identity':identity(json_bytes(req)),
                    'positions':2,'optimizer_updates':0,'members':[{'name':n,'identity':identity(payloads[n]),'bytes':len(payloads[n])} for n in core.RESULT_FILES[:-1]]}
                raw=json_bytes(manifest);(out/'manifest.json').write_bytes(raw)
                if change=='missing-file':(out/'metrics.jsonl').unlink()
                with patch.object(core,'verify_request',return_value=req),patch.object(core,'historical',return_value=([],set(),[])),patch.object(core,'inventory',return_value=idx):
                    args=(folder,identity(json_bytes(req)),p,teacher,p,out,identity(raw))
                    if change=='none':self.assertTrue(native.verify(*args)['verified'])
                    else:
                        with self.assertRaises((ContractError,FileNotFoundError)):native.verify(*args)
    def test_real_source_snapshot_rejects_modified_file(self):
        args=synthetic()
        prior=json_bytes({'diagnostic_source_closure':[]})
        def git(command,**kw):
            if command[1]=='rev-parse':return 'a'*40+'\n'
            path=command[2].split(':',1)[1]
            return core.lab.source_bytes(path)
        with patch.object(core,'historical',return_value=args),patch.object(core.diag,'read_member',return_value=prior),patch.object(core.subprocess,'check_output',side_effect=git):
            baseline=core.request_for(Path('.'),Path('.'),Path('.'))
            self.assertEqual(baseline['positions'],13689)
        def changed_git(command,**kw):
            result=git(command,**kw)
            if command[1]=='show' and command[2].endswith('arithmetic.py'):return result+b'# substituted\n'
            return result
        with patch.object(core,'historical',return_value=args),patch.object(core.diag,'read_member',return_value=prior),patch.object(core.subprocess,'check_output',side_effect=changed_git):
            with self.assertRaisesRegex(ContractError,'uncommitted source'):core.request_for(Path('.'),Path('.'),Path('.'))


class NativeCanary(unittest.TestCase):
    def test_native_cpu_forward_only(self):
        import torch
        from trinite.contracts import ModelConfig
        from trinite.model import Decoder
        from trinite.training import model_identity
        from teacher_training_v1.numerical import configure
        configure();model=Decoder(ModelConfig.load(Path('configs/teacher/gf3-teacher-90k-v1.json')),lane='dense',seed=17)
        before=model_identity(model)
        with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('optimizer forbidden')),patch.object(torch.Tensor,'backward',side_effect=AssertionError('backward forbidden')):
            raw,token=native.forward_vector(model,[256,65,61])
        self.assertEqual(len(raw),1036);self.assertEqual(model_identity(model),before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(token,ar.metrics(ar.unpack(raw),48)['argmax'])


if __name__=='__main__':unittest.main()
