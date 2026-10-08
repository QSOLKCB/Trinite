import copy
from pathlib import Path
import random
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from trinite.capture import audit_capture, capture_prompt
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.geometry import rademacher
from trinite.geometry_run import freeze_observation, measure_observation, protocol, read_request, inputs
from trinite.inspection import tensor_identity
from trinite.model import Decoder
from trinite.checkpoint import rng_snapshot, read_checkpoint
from trinite.experiment import train_run
from trinite.training import RunConfig
from trinite.observation import verify_observation

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)


def rng_id():
    m, t = rng_snapshot()
    return m, {n: tensor_identity(v) for n,v in t.items()}


class CaptureTests(unittest.TestCase):
    def test_prompt_spans_native_states_and_causal_prefix_parity(self):
        model = Decoder(RunConfig().model, lane='ternary', seed=0)
        before = {n: tensor_identity(p) for n,p in model.named_parameters()}; rng = rng_id()
        record, trajectories = audit_capture(capture_prompt(model, 'é=1'))
        self.assertEqual(record['positions'], [1,2,3,4])
        self.assertEqual(record['tokenizer']['byte_spans'], [None,[0,1],[1,2],[2,3],[3,4],None])
        for index in (1,2,3,4):
            from trinite.model import CaptureSpec
            ids = torch.tensor([record['tokenizer']['input_ids'][:index+1]], dtype=torch.int64)
            mask = torch.ones_like(ids, dtype=torch.bool)
            with torch.no_grad(): value = model(ids, mask, capture=CaptureSpec(('final',),(index,))).captures['final'][0,0]
            torch.testing.assert_close(value, torch.tensor(trajectories['final'][index-1]), rtol=1e-5, atol=2e-6)
        self.assertEqual(before, {n: tensor_identity(p) for n,p in model.named_parameters()})
        self.assertEqual(rng, rng_id())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_capture_receipt_rejects_changed_spans_dtype_selection_shapes_and_bytes(self):
        original = parse_json(capture_prompt(Decoder(RunConfig().model), '1+1='))
        variants=[]
        for key,value in [('dtype','float64'),('positions',[0,1,2,3]),('pooling','mean'),('config_identity',identity(b'wrong'))]:
            r=copy.deepcopy(original);r[key]=value;variants.append(r)
        r=copy.deepcopy(original);r['tokenizer']['byte_spans'][1]=[1,2];variants.append(r)
        for field,value in [('shape',[True,16]),('tensor_identity',identity(b'wrong'))]:
            r=copy.deepcopy(original);r['trajectories']['final'][field]=value;variants.append(r)
        for value in ('nan', '0x1.0000000000001p+0', 0):
            r=copy.deepcopy(original);r['trajectories']['final']['coordinates_hex'][0][0]=value;variants.append(r)
        for r in variants:
            with self.assertRaises(ContractError): audit_capture(json_bytes(r))
        with self.assertRaises(ContractError): capture_prompt(Decoder(RunConfig().model), '')
        with self.assertRaises(ContractError): capture_prompt(Decoder(RunConfig().model), 'x'*31)

    def test_null_control_is_deterministic_and_does_not_consume_rng(self):
        before=rng_id();p=[[0.,0.],[1.,2.],[2.,3.]]
        a=rademacher(p,'item');b=rademacher(p,'item')
        self.assertEqual(a,b);self.assertEqual(before,rng_id())
        self.assertEqual({x for row in a for x in row}, {-1.,1.})


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        for lane in ('dense','ternary'):
            random.seed(0);np.random.seed(0);torch.manual_seed(0)
            r=train_run(cls.root/lane,ROOT/'fixtures/formal-v1',RunConfig(),lane=lane,observer=False)
            if r['compute_outcome']!='completed':raise AssertionError(r)
        cls.options=dict(dataset=ROOT/'fixtures/formal-v1',training_config=ROOT/'configs/tiny-training.json',
                         dense_checkpoint=cls.root/'dense/checkpoint',ternary_checkpoint=cls.root/'ternary/checkpoint')
        cls.frozen=freeze_observation(cls.root/'frozen',**cls.options)
        cls.request=cls.root/'frozen/request.json'
        cls.report=measure_observation(cls.root/'on',request=cls.request,
                                      request_identity=cls.frozen['request_identity'],**cls.options)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_frozen_repository_protocol_and_request_bind_inputs_before_capture(self):
        self.assertEqual(protocol(), parse_json((ROOT/'configs/geometry-observation.json').read_bytes(),canonical=True))
        _,raw=inputs(**self.options)
        read_request(self.request,self.frozen['request_identity'],raw)
        before=self.request.read_bytes()
        with self.assertRaises(ContractError):read_request(self.request,identity(b'changed'),raw)
        for key in ('source','inputs','environment','geometry_pin','protocol'):
            request=parse_json(before);request[key]['changed']='value';raw_request=json_bytes(request)
            path=self.root/('bad-'+key+'.json');path.write_bytes(raw_request)
            with self.assertRaises(ContractError):read_request(path,identity(raw_request),raw)
        with self.assertRaises(FileExistsError):freeze_observation(self.root/'frozen',**self.options)

    def test_checkpoint_election_lane_and_plan_are_frozen(self):
        bad=dict(self.options);bad['dense_checkpoint']=self.options['ternary_checkpoint']
        with self.assertRaises(ContractError):inputs(**bad)
        bad=dict(self.options);bad['training_config']=ROOT/'configs/reference.json'
        with self.assertRaises(ContractError):inputs(**bad)

    def test_complete_native_space_pairing_and_all_controls_retained(self):
        self.assertEqual(self.report['compute_outcome'],'completed',self.report)
        self.assertTrue(self.report['observer_evidence_eligible'],self.report)
        verify_observation(self.root/'on/provenance')
        forbidden={e['example_identity'] for e in parse_json((ROOT/'fixtures/formal-v1/dataset.json').read_bytes())['examples'] if e['split']!='train'}
        for lane in ('dense','ternary'):
            r=parse_json((self.root/'on'/f'{lane}-geometry.json').read_bytes())
            captures=parse_json((self.root/'on'/f'{lane}-captures.json').read_bytes())
            self.assertEqual(len(captures['records']),36)
            self.assertFalse(forbidden & {e['example_identity'] for e in captures['records']})
            for record in captures['records']:
                audit_capture(json_bytes(record['capture']))
                self.assertTrue(record['capture']['prompt'].endswith('='))
            for layer in ('final','block.0'):
                self.assertEqual(set(r['comparisons'][layer]), {'native','reverse-token-order','deterministic-rademacher-vectors'})
                for condition in r['comparisons'][layer].values():
                    self.assertEqual(len(condition['families']),6)
                    for family in condition['families']:
                        self.assertEqual(len(family['pairs']),9)
                        self.assertEqual(sum(p['type']=='same-item/different-carrier' for p in family['pairs']),3)

    def test_geo_null_reuse_001_exact_direct_pair_oracle_and_one_materialization(self):
        from trinite.geometry_run import summarize
        from trinite.geometry import alignment
        captures=parse_json((self.root/'on/dense-captures.json').read_bytes())
        records=[]
        for r in captures['records']:
            _,points=audit_capture(json_bytes(r['capture']))
            records.append({**{k:r[k] for k in ('example_identity','semantic_identity','family_id','carrier')},'points':points})
        with patch('trinite.geometry_run.rademacher',wraps=rademacher) as make_null:
            result=summarize(records,lambda:None)
        self.assertEqual(make_null.call_count,36*2)
        by_id={r['example_identity']:r for r in records}
        before=rng_id()
        for layer,conditions in result.items():
            for family in conditions['deterministic-rademacher-vectors']['families']:
                for pair in family['pairs']:
                    a,b=by_id[pair['left']],by_id[pair['right']]
                    expected=alignment(rademacher(a['points'][layer],a['example_identity']+':'+layer),
                                       rademacher(b['points'][layer],b['example_identity']+':'+layer))
                    self.assertEqual(pair['cosine_hex'],expected.hex())
        self.assertEqual(before,rng_id())

    def test_combined_cpu_runner_preserves_standalone_suite_ids(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('cpu_runner',ROOT/'scripts/check_cpu.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        def ids(suite):
            for case in suite:
                if isinstance(case,unittest.TestSuite):yield from ids(case)
                else:yield case.id()
        expected=[]
        for directory in ('tests/model','tests/training','tests/geometry'):
            expected.extend(ids(unittest.TestLoader().discover(str(ROOT/directory))))
        actual=list(ids(runner.load_suites()))
        self.assertEqual(actual,expected)
        self.assertEqual(len(actual),len(set(actual)))
        self.assertFalse(any('_FailedTest' in name for name in actual))

    def test_replay_off_and_failed_observer_preserve_artifacts_rng_checkpoints(self):
        class Broken:
            def __init__(self,path):raise OSError('injected observer acquisition failure')
        old={lane:read_checkpoint(self.options[lane+'_checkpoint']) for lane in ('dense','ternary')}
        for name,observer,factory in [('off',False,None),('broken',True,Broken)]:
            before=rng_id()
            r=measure_observation(self.root/name,request=self.request,request_identity=self.frozen['request_identity'],
                                  observer=observer,**({'observer_factory':factory} if factory else {}),**self.options)
            self.assertEqual(before,rng_id());self.assertEqual(r['compute_outcome'],'completed',r)
            self.assertFalse(r['observer_evidence_eligible'])
            if observer:self.assertTrue(r['observation_errors'])
            for member in ('result.json','dense-captures.json','ternary-captures.json','dense-geometry.json','ternary-geometry.json'):
                self.assertEqual((self.root/name/member).read_bytes(),(self.root/'on'/member).read_bytes(),member)
        for lane in ('dense','ternary'):self.assertEqual(old[lane],read_checkpoint(self.options[lane+'_checkpoint']))

    def test_failed_capture_preserves_error_and_ineligibility(self):
        with patch('trinite.capture.capture_prompt',side_effect=ContractError('injected capture failure')):
            r=measure_observation(self.root/'failed',request=self.request,
                                 request_identity=self.frozen['request_identity'],**self.options)
        self.assertEqual(r['compute_outcome'],'failed');self.assertTrue(r['errors'])
        self.assertFalse(r['observer_evidence_eligible']);self.assertTrue((self.root/'failed/result.json').exists())

    def test_cli_freeze_and_measure_work_in_independent_processes(self):
        options=[]
        for key,value in self.options.items():options.extend(['--'+key.replace('_','-'),str(value)])
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
        freeze=subprocess.run([sys.executable,'-m','trinite','freeze-observation',str(self.root/'cli-freeze'),*options],
                              env=env,capture_output=True,timeout=30)
        self.assertEqual(freeze.returncode,0,freeze.stderr.decode())
        request=parse_json(freeze.stdout)
        measured=subprocess.run([sys.executable,'-m','trinite','measure-observation',str(self.root/'cli-measured'),*options,
                                 '--request',request['request_file'],'--request-identity',request['request_identity']],
                                env=env,capture_output=True,timeout=30)
        self.assertEqual(measured.returncode,0,measured.stderr.decode())
        self.assertTrue(parse_json(measured.stdout)['observer_evidence_eligible'])
