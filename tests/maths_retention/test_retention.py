from dataclasses import replace
from collections import Counter
from pathlib import Path
import tempfile
import unittest
import torch

torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite.maths_learning import corpora, state_for as prior_state
from trinite.maths_plan import MathsPlan
from trinite.maths_retention import (cells, cell_name, state_for, exposure, snapshot, restore,
                                     freeze_request, read_request, training_failures, protocol)
from trinite.training import run_steps, model_identity


def small_plan(steps=4):
    return replace(MathsPlan(), model=ModelConfig(context_length=64, blocks=1, width=16,
                   heads=2, feed_forward_width=32), steps=steps, validation_every=max(1, steps//2))


class RetentionTests(unittest.TestCase):
    def test_rehearsal_uses_independent_train_streams_and_wraps(self):
        state = state_for(0, 'rehearsal')
        packs = corpora()
        allowed = {e['example_identity'] for v in packs.values() for e in v[2] if e['split']=='train'}
        excluded = {e['example_identity'] for v in packs.values() for e in v[2] if e['split']!='train'}
        legacy = {e['example_identity'] for e in packs['legacy'][2] if e['split']=='train'}
        streams = [[r for r in state.data['validation'] if (r['identity'] in legacy)==is_legacy]
                   for is_legacy in (True, False)]
        selected = {r['identity'] for r in state.data['train']}
        self.assertEqual(selected, allowed); self.assertFalse(selected & excluded)
        self.assertEqual(len(state.data['train']), 32768)
        self.assertEqual(len({r['identity'] for r in state.data['validation']}), 2262)
        self.assertIsNot(state.data['train'], state.data['validation'])
        for step in range(4096):
            batch = state.data['train'][step*8:step*8+8]
            self.assertEqual([r['identity'] in legacy for r in batch], [True]*4+[False]*4)
            self.assertEqual(batch, [stream[(step*4+j)%len(stream)] for stream in streams for j in range(4)])

    def test_pooled_control_preserves_prior_schedule_initialization_and_diagnostics(self):
        plan = small_plan()
        old = prior_state(0, 'expanded', plan)
        pooled = state_for(0, 'pooled', plan); rehearsal = state_for(0, 'rehearsal', plan)
        self.assertEqual(pooled.data, old.data)
        self.assertEqual(rehearsal.data['validation'], old.data['train'])
        self.assertEqual(pooled.initialization_identity, rehearsal.initialization_identity)
        self.assertEqual(pooled.initial_train_loss, rehearsal.initial_train_loss)
        run_steps(old); run_steps(pooled)
        self.assertEqual(old.history, pooled.history)
        self.assertEqual(model_identity(old.model), model_identity(pooled.model))
        self.assertEqual(len(cells()), 4)
        with self.assertRaises(ContractError): cell_name(True, 'pooled')
        with self.assertRaises(ContractError): state_for(0, 'rehearsal', replace(plan, batch_size=4))

    def test_exposure_counts_actual_targets_and_input_tokens_with_unvisited_rows(self):
        state = state_for(0, 'rehearsal', small_plan()); run_steps(state)
        actual = exposure(state)
        mapping = {e['example_identity']: k for k, v in corpora().items() for e in v[2] if e['split']=='train'}
        visits = Counter(key for h in state.history for key in h['examples'])
        for kind in ('legacy', 'maths'):
            rows = [r for r in state.data['validation'] if mapping[r['identity']]==kind]
            self.assertEqual(actual[kind]['example_visits'], 16)
            self.assertEqual(actual[kind]['scored_target_tokens'], sum(visits[r['identity']]*sum(r['loss']) for r in rows))
            self.assertEqual(actual[kind]['nonpadding_input_tokens'], sum(visits[r['identity']]*(len(r['ids'])-1) for r in rows))
            self.assertEqual(actual[kind]['minimum_visits'], 0)
            self.assertEqual(sum(actual[kind]['visit_histogram'].values()), len(rows))
        self.assertEqual(sum(r['scored_target_tokens'] for r in actual.values()), state.target_tokens)

    def test_checkpoint_resume_is_exact_across_legacy_stream_wrap(self):
        plan = small_plan(68); state = state_for(0, 'rehearsal', plan)
        run_steps(state, stop_after=34)
        payload, metadata = snapshot(state, 0, 'rehearsal', 'sha256:'+'0'*64)
        resumed = restore(payload, metadata, 0, 'rehearsal', 'sha256:'+'0'*64, plan)
        run_steps(state); run_steps(resumed)
        self.assertEqual(snapshot(state,0,'rehearsal','sha256:'+'0'*64), snapshot(resumed,0,'rehearsal','sha256:'+'0'*64))
        with self.assertRaises(ContractError): restore(payload[:-1],metadata,0,'rehearsal','sha256:'+'0'*64,plan)
        with self.assertRaises(ContractError): restore(payload,metadata,0,'pooled','sha256:'+'0'*64,plan)
        r=parse_json(metadata); r['history'][0]['examples'].reverse()
        with self.assertRaises(ContractError): restore(payload,json_bytes(r),0,'rehearsal','sha256:'+'0'*64,plan)

    def test_request_binds_corpus_source_environment_and_ratio(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'request.json'; receipt=freeze_request(path)
            read_request(path,receipt['request_identity']); original=path.read_bytes()
            for key in ('source','environment','corpora','protocol'):
                r=parse_json(original); r[key]={}; path.write_bytes(json_bytes(r))
                with self.assertRaises(ContractError): read_request(path,identity(path.read_bytes()))
        self.assertEqual(protocol()['profiles'], ['pooled','rehearsal'])
        self.assertIn('no validation/test tuning', protocol()['selection'])

    def test_criterion_is_per_task_and_accepts_exact_threshold(self):
        scores={'tasks':{'pass':{'examples':10,'correct':9},'fail':{'examples':10,'correct':8}}}
        for kind in ('legacy','maths'):
            self.assertEqual(training_failures(scores,kind),['fail'])
        with self.assertRaises(ContractError): training_failures(scores,'unknown')
