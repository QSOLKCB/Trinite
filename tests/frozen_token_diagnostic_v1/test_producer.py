"""Toy conformance with real producer custody; never target experiment results.

The injected four-update/tiny model recipe is exclusively an engineering test.
All source requests, admission, safe tensors, producer artifacts and provenance
are actually generated and checked; custody/verifier success is not mocked.
"""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from trinite.model import Decoder
from trinite.training import model_identity
from addition_development_v1 import controller, numerical, request, workflow
from frozen_token_diagnostic_v1 import producer


def tiny(seed):
    return numerical.AdditionPlan(seed=seed,steps=4,validation_every=2,
        model=ModelConfig(context_length=64,blocks=1,width=8,heads=2,feed_forward_width=16),
        max_target_tokens=128,max_seconds=30)


def toy_patches():
    stack = ExitStack()
    stack.enter_context(patch.object(numerical,'plan_for',side_effect=tiny))
    stack.enter_context(patch.object(producer,'plan_for',side_effect=tiny))
    stack.enter_context(patch.object(workflow,'MILESTONES',(0,2,4)))
    stack.enter_context(patch.object(producer,'MILESTONES',(0,2,4)))
    stack.enter_context(patch.object(producer,'STEPS',4))
    return stack


def toy_matrix(root):
    """Use the actual training worker/BundleObserver; tiny engineering only."""
    root = Path(root); root.mkdir()
    raw = json_bytes(request.core()); rid = identity(raw)
    (root/'request.json').write_bytes(raw)
    records = {}; models = {}
    for arm,seed in request.cells():
        name = request.cell_name(arm,seed); child = root/name; child.mkdir()
        stat = child.stat()
        cid = workflow.run(child,raw,rid,arm,seed,(stat.st_dev,stat.st_ino))
        # These explicit controller records wrap the actual produced cell.
        (child/'worker.log').write_bytes(b'engineering toy; not target evidence\n')
        from trinite.contracts import parse_json
        models[name] = parse_json((child/'report.json').read_bytes(),canonical=True)['final_model_identity']
        attempt = {'schema':'trinite.addition-development-attempt.v1','cell':name,
            'request_identity':rid,'outcome':'completed','worker_exit':0,
            'detail':'engineering-only actual producer custody','completion_identity':cid,
            'wall_seconds_hex':(1.0).hex()}
        attempt_raw = json_bytes(attempt); (child/'attempt.json').write_bytes(attempt_raw)
        records[name] = {'outcome':'completed','attempt_identity':identity(attempt_raw),
            'completion_identity':cid,'retained_files':controller.bound_files(child)}
    matrix = {'schema':'trinite.addition-development-matrix.v1','request_identity':rid,
        'cells':records,'outcome':'completed','fresh_confirmation':False,
        'learning_adequate':False,'phase5_ready':False,'release_ready':False}
    encoded = json_bytes(matrix); (root/'matrix.json').write_bytes(encoded)
    return rid,identity(encoded),models


class ProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        controller.configure()
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)/'producer'
        cls.patches = toy_patches(); cls.patches.__enter__()
        try:
            cls.rid,cls.mid,cls.models = toy_matrix(cls.root)
        except BaseException:
            cls.patches.close(); cls.temp.cleanup(); raise

    @classmethod
    def tearDownClass(cls):
        cls.patches.close(); cls.temp.cleanup()

    def test_full_actual_custody_import_never_forwards_or_constructs_optimizer(self):
        before = producer.snapshot_matrix(self.root)
        with patch.object(Decoder,'forward',side_effect=AssertionError('unexpected forward')), \
             patch.object(torch.optim,'AdamW',side_effect=AssertionError('unexpected optimizer')):
            binding = producer.inspect_matrix(self.root,self.rid,self.mid,self.models)
            model,scores = producer.load_cell(self.root,binding,'candidate',1)
        self.assertEqual(model_identity(model),self.models['candidate-1'])
        self.assertEqual(set(scores),set(producer.STRATA))
        for stratum,raw in scores.items():
            self.assertEqual(raw,(self.root/'candidate-1'/f'development-{stratum}.json').read_bytes())
            self.assertEqual(identity(raw),binding['cells']['candidate-1']['raw_score_identities'][stratum])
        producer.unchanged_matrix(self.root,before)

    def test_external_commitments_are_required(self):
        wrong = 'sha256:'+'0'*64
        for rid,mid,models in [(wrong,self.mid,self.models),(self.rid,wrong,self.models),
                              (self.rid,self.mid,{**self.models,'candidate-2':wrong}),
                              (self.rid,self.mid,{})]:
            with self.assertRaises(ContractError):
                producer.inspect_matrix(self.root,rid,mid,models)

    def test_matrix_extra_file_and_symlink_rejected(self):
        extra = self.root/'extra'; extra.write_bytes(b'x')
        try:
            with self.assertRaises(ContractError): producer.snapshot_matrix(self.root)
        finally: extra.unlink()
        link = self.root/'control-0'/'link'; link.symlink_to('request.json')
        try:
            with self.assertRaises(ContractError): producer.snapshot_matrix(self.root)
        finally: link.unlink()

    def test_foreign_environment_request_cannot_be_resealed(self):
        from trinite.contracts import parse_json
        path = self.root/'request.json'; original = path.read_bytes()
        changed = parse_json(original,canonical=True)
        changed['environment'] = {'foreign': True}
        encoded = json_bytes(changed); path.write_bytes(encoded)
        try:
            with self.assertRaisesRegex(ContractError,'source/environment/data/protocol'):
                producer.inspect_matrix(self.root,identity(encoded),self.mid,self.models)
        finally: path.write_bytes(original)

    def test_checkpoint_metadata_and_tensor_guards(self):
        from trinite.contracts import parse_json
        root = self.root/'control-0'
        payload = (root/'tensors-4.safetensors').read_bytes()
        metadata = (root/'metadata-4.json').read_bytes()
        req = parse_json((self.root/'request.json').read_bytes(),canonical=True)
        for field,value in [('step',2),('arm','candidate'),('target_tokens',0),
                            ('initialization_identity','sha256:'+'0'*64)]:
            bad = parse_json(metadata,canonical=True); bad[field] = value
            with self.assertRaises(ContractError):
                producer.checkpoint(payload,json_bytes(bad),req,self.rid,'control',0,4)
        with self.assertRaises(ContractError):
            producer.checkpoint(b'not safetensors',metadata,req,self.rid,'control',0,4)
        bad = parse_json(metadata,canonical=True)
        bad['history'][0]['examples'][0] = 'sha256:'+'0'*64
        with self.assertRaises(ContractError):
            producer.checkpoint(payload,json_bytes(bad),req,self.rid,'control',0,4)
        from safetensors.torch import load, save
        for change in ('missing','extra','shape','nan','optimizer-step','negative-moment'):
            tensors = load(payload)
            if change == 'missing': tensors.pop(next(iter(tensors)))
            elif change == 'extra': tensors['model.unexpected'] = torch.zeros(1)
            elif change == 'shape':
                name = next(n for n in tensors if n.startswith('model.'))
                tensors[name] = torch.zeros(1)
            elif change == 'negative-moment':
                name = next(n for n in tensors if n.endswith('.exp_avg_sq'))
                tensors[name].view(-1)[0] = -1
            elif change == 'nan':
                name = next(n for n in tensors if n.startswith('model.'))
                tensors[name].view(-1)[0] = float('nan')
            else:
                name = next(n for n in tensors if n.endswith('.step'))
                tensors[name].fill_(3)
            with self.assertRaises(ContractError):
                producer.checkpoint(save(tensors),metadata,req,self.rid,'control',0,4)


if __name__ == '__main__': unittest.main()
