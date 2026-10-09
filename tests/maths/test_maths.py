from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import torch

torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite.maths_learning import (state_for, snapshot, restore, freeze_request, read_request, cells, cell_name, protocol)
from trinite.maths_plan import MathsPlan
from trinite.training import run_steps, model_identity


def small_plan():
    return replace(MathsPlan(), model=ModelConfig(context_length=64,blocks=1,width=16,heads=2,feed_forward_width=32), steps=4, validation_every=2)


class MathsLearningTests(unittest.TestCase):
    def test_training_inputs_exclude_every_validation_test_identity(self):
        from trinite.maths_learning import corpora
        allowed = {e['example_identity'] for v in corpora().values() for e in v[2] if e['split']=='train'}
        excluded = {e['example_identity'] for v in corpora().values() for e in v[2] if e['split']!='train'}
        state = state_for(0,'expanded',small_plan())
        selected = {e['identity'] for e in state.data['train']}
        self.assertEqual(selected,allowed); self.assertFalse(selected & excluded)
        self.assertIs(state.data['train'],state.data['validation'])

    def test_paired_initialization_and_explicit_unequal_exposure(self):
        a = state_for(0,'control',small_plan()); b = state_for(0,'expanded',small_plan())
        self.assertEqual(a.initialization_identity,b.initialization_identity)
        self.assertLess(len(a.data['train']),len(b.data['train']))
        self.assertEqual(len(cells()),4)
        with self.assertRaises(ContractError): cell_name(True,'control')
        with self.assertRaises(ContractError): cell_name(0,'unknown')

    def test_safe_snapshot_resume_exact_native_updates(self):
        plan = small_plan(); state = state_for(0,'expanded',plan)
        run_steps(state,stop_after=2); payload, metadata = snapshot(state,0,'expanded','sha256:'+'0'*64)
        resumed = restore(payload,metadata,0,'expanded','sha256:'+'0'*64,plan)
        run_steps(state);run_steps(resumed)
        self.assertEqual(model_identity(state.model),model_identity(resumed.model))
        self.assertEqual(snapshot(state,0,'expanded','sha256:'+'0'*64),snapshot(resumed,0,'expanded','sha256:'+'0'*64))
        with self.assertRaises(ContractError): restore(payload[:-1],metadata,0,'expanded','sha256:'+'0'*64,plan)
        r = parse_json(metadata);r['cursor'] += 1
        with self.assertRaises(ContractError): restore(payload,json_bytes(r),0,'expanded','sha256:'+'0'*64,plan)
        with self.assertRaises(ContractError): restore(payload,metadata,0,'control','sha256:'+'0'*64,plan)

    def test_frozen_request_binds_source_data_environment_and_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'request.json'; receipt=freeze_request(p); read_request(p,receipt['request_identity'])
            r=parse_json(p.read_bytes());r['corpora']['maths']['dataset_identity']='sha256:'+'f'*64
            p.write_bytes(json_bytes(r))
            with self.assertRaises(ContractError):read_request(p,identity(p.read_bytes()))
        self.assertEqual(protocol()['selection'],'final update only; no validation/test tuning, outcome-driven changes or candidate selection')
        with self.assertRaises(ContractError): replace(MathsPlan(),steps=4097)
