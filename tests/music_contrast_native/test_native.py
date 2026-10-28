"""Small synthetic native conformance; never the target contrast matrix."""
from copy import deepcopy
from types import SimpleNamespace
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from trinite.contracts import ModelConfig, ContractError, identity, json_bytes, parse_json
from trinite.model import Decoder
from trinite.training import (RunConfig, TrainingState, create_optimizer, evaluate,
                              model_identity, run_steps)
from trinite.tokenizer import ByteTokenizer, EOS
from music_contrast_v1 import numerical

RID = identity(b'synthetic-request')
AID = identity(b'synthetic-admission')
DATA = b'synthetic-dataset-not-target'


def fixture():
    row = {'example_identity': identity(b'synthetic-row'), 'family_id': 'synthetic',
           'task': 'synthetic', 'split': 'train', 'prompt': 'Q=', 'answer': '7'}
    enc = ByteTokenizer(128).encode('Q=7', answer_start=2)
    carrier = {'identity': row['example_identity'], 'ids': enc.input_ids,
               'mask': enc.attention_mask, 'loss': enc.loss_mask}
    config = RunConfig(model=ModelConfig(context_length=128, blocks=1, width=8,
        heads=2, feed_forward_width=16), seed=17, steps=8, batch_size=1,
        learning_rate='0.003', validation_every=8, max_seconds=30, max_target_tokens=16)
    model = Decoder(config.model, lane='dense', seed=17)
    state = TrainingState(config, 'dense', model, create_optimizer(model, config),
        {'train': [carrier], 'validation': [carrier]}, identity(DATA), AID,
        model_identity(model), evaluate(model, [carrier], 1))
    return row, carrier, state


class NativeConformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_prompt_only_native_score_and_unchanged_state(self):
        row, _, state = fixture()
        before = model_identity(state.model)
        with patch.object(numerical, '_view', return_value=([row], 'train')):
            summary, raw = numerical.score(state, 'train')
            self.assertEqual(numerical.check_score(raw, 'train'), summary)
        self.assertEqual(model_identity(state.model), before)
        self.assertTrue(all(p.grad is None for p in state.model.parameters()))
        record = parse_json(raw, canonical=True)['predictions'][0]
        self.assertEqual(record['expected_tokens'], [55, EOS])
        self.assertEqual(record['example_identity'], row['example_identity'])

    def test_exact_eos_and_no_repair_scoring_controls(self):
        row, _, state = fixture()
        for tokens, correct in (([55, EOS], True), ([EOS], False), ([47] * 64, False)):
            with patch.object(numerical, '_view', return_value=([row], 'train')), \
                 patch('trinite.foundations.greedy', return_value=tokens) as greedy:
                summary, raw = numerical.score(state, 'train')
                self.assertEqual(summary['correct'], int(correct))
                self.assertEqual(numerical.check_score(raw, 'train'), summary)
                self.assertEqual(greedy.call_args.args[1:], ('Q=', 64))
                self.assertEqual(parse_json(raw)['predictions'][0]['generated_tokens'], tokens)

    def test_non_native_termination_rejected(self):
        row, _, state = fixture()
        for tokens in ([47], [EOS, 55]):
            with patch.object(numerical, '_view', return_value=([row], 'train')), \
                 patch('trinite.foundations.greedy', return_value=tokens):
                _, raw = numerical.score(state, 'train')
                with self.assertRaisesRegex(ContractError, 'termination'):
                    numerical.check_score(raw, 'train')

    def test_prediction_emit_precedes_later_failure(self):
        row, _, state = fixture()
        other = {**row, 'example_identity': identity(b'other-synthetic-row')}
        retained = []
        def emit(record):
            retained.append(record)
            if len(retained) == 2:
                raise RuntimeError('synthetic journal failure')
        with patch.object(numerical, '_view', return_value=([row, other], 'train')), \
             patch('trinite.foundations.greedy', return_value=[55, EOS]):
            with self.assertRaisesRegex(RuntimeError, 'journal failure'):
                numerical.score(state, 'train', emit)
        self.assertEqual(len(retained), 2)
        self.assertEqual(retained[0]['example_identity'], row['example_identity'])
        self.assertEqual(retained[0]['generated_tokens'], [55, EOS])

    def test_safe_checkpoint_roundtrip_without_forward(self):
        row, carrier, state = fixture()
        run_steps(state, stop_after=2)
        payload, meta = numerical.snapshot(state, RID)
        metadata = parse_json(meta, canonical=True)
        with patch.object(numerical, '_admitted', return_value=DATA), \
             patch.object(numerical, 'plan_for', return_value=state.config), \
             patch.object(numerical, 'rows', return_value=[row]), \
             patch.object(numerical, 'encoded', return_value=[carrier]), \
             patch('trinite.training.require_environment'), \
             patch.object(Decoder, 'forward', side_effect=AssertionError('verification forwarded')):
            restored = numerical.verify_checkpoint(payload, metadata, 17, AID, RID)
            self.assertEqual(restored.step, 2)
            self.assertEqual(model_identity(restored.model), model_identity(state.model))
            changed = deepcopy(metadata); changed['seed'] = 29
            with self.assertRaises(ContractError):
                numerical.verify_checkpoint(payload, changed, 17, AID, RID)
            changed = deepcopy(metadata); changed['target_tokens'] += 1
            with self.assertRaises(ContractError):
                numerical.verify_checkpoint(payload, changed, 17, AID, RID)
            with self.assertRaises(Exception):
                numerical.verify_checkpoint(b'not-safe-tensors', metadata, 17, AID, RID)

    def test_holdout_requires_final_checkpoint(self):
        _, _, state = fixture()
        with patch.object(numerical, 'verify_checkpoint', return_value=state), \
             patch.object(numerical, 'score', side_effect=AssertionError('heldout called')):
            with self.assertRaisesRegex(ContractError, 'final checkpoint'):
                numerical.holdout(17, RID, AID, b'bound elsewhere', {}, lambda *_: None)

    def test_train_only_and_cumulative_time_failure_retained(self):
        state = SimpleNamespace(step=0, config=SimpleNamespace(max_seconds=120))
        retained = {}
        def update(state, stop_after):
            state.step = stop_after
        with patch.object(numerical, 'state_for', return_value=state), \
             patch.object(numerical, 'snapshot', return_value=(b'synthetic-not-tensors', b'{}')), \
             patch.object(numerical, 'score', return_value=({'correct': 0, 'examples': 1}, b'{}')) as scoring, \
             patch.object(numerical, 'monotonic', side_effect=[0., 70., 100., 151.]), \
             patch('trinite.training.run_steps', side_effect=update):
            with self.assertRaisesRegex(ContractError, 'deadline'):
                numerical.train(17, RID, AID, lambda name, data: retained.__setitem__(name, data))
        self.assertEqual(scoring.call_args.args[1], 'train')
        self.assertIn('step-000-predictions.json', retained)
        self.assertIn('failure.safetensors', retained)
        self.assertIn('failure.json', retained)
        self.assertNotIn('step-064-predictions.json', retained)

    def test_synthetic_workflow_custody_and_safe_reader_composition(self):
        from music_contrast_v1 import workflow, reporting
        # One synthetic Q=7 row, a tiny model and synthetic request/admission.
        # This is NOT any target music row, matrix, outcome or admission receipt.
        row, carrier, initial_state = fixture()
        config = RunConfig(model=initial_state.config.model, seed=17, steps=64,
            batch_size=8, learning_rate='0.003', validation_every=64,
            max_seconds=30, max_target_tokens=1024)
        model = initial_state.model
        state = TrainingState(config, 'dense', model, create_optimizer(model, config),
            {'train': [carrier], 'validation': [carrier]}, identity(DATA), AID,
            model_identity(model), initial_state.initial_train_loss)
        artifacts = {}
        summaries = {}
        with patch.object(numerical, '_view', return_value=([row], 'train')), \
             patch('trinite.foundations.greedy', return_value=[55, EOS]):
            for step in (0, 64):
                if step:
                    run_steps(state)
                payload, metadata = numerical.snapshot(state, RID)
                prefix = f'step-{step:03d}'
                artifacts[prefix+'.safetensors'] = payload
                artifacts[prefix+'-checkpoint.json'] = metadata
                summary, raw = numerical.score(state, 'train')
                summaries[step] = summary
                artifacts[prefix+'-predictions.json'] = raw
                artifacts[prefix+'-journal.jsonl'] = b''.join(
                    json_bytes(p) for p in parse_json(raw)['predictions'])
        synthetic_contrasts = {'synthetic_contract_only': True}
        report = {'schema': 'trinite.music-contrast-report.v1', 'seed': 17,
            'phase': 'train', 'request_identity': RID, 'admission_identity': AID,
            'model_identity': model_identity(state.model),
            'verification_scope': workflow.SCOPE, **workflow.CLAIMS,
            'step': 64, 'target_tokens': 1024,
            'initial_score': summaries[0], 'final_score': summaries[64],
            # Even a synthetic one-row success cannot meet the real40row gate.
            'training_fit': False, 'update_seconds_hex': float(0).hex(),
            'initial_contrasts': synthetic_contrasts, 'final_contrasts': synthetic_contrasts}
        artifacts['report.json'] = json_bytes(report)
        request_raw = b'synthetic-controller-request-not-target'
        def seal(root):
            completion = {'schema': 'trinite.music-contrast-completion.v1',
                'request_identity': RID, 'cell': workflow.cell_name(17, 'train'),
                'artifacts': {name: identity((root/name).read_bytes())
                              for name in workflow.names('train')}}
            raw = json_bytes(completion)
            (root/'completion.json').write_bytes(raw)
            return identity(raw)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            for name, raw in artifacts.items():
                (root/name).write_bytes(raw)
            (root/'request.json').write_bytes(request_raw)
            expected = seal(root)
            stack.enter_context(patch.object(workflow, 'validate', return_value={'admission_identity': AID}))
            stack.enter_context(patch.object(numerical, '_admitted', return_value=DATA))
            stack.enter_context(patch.object(numerical, 'plan_for', return_value=config))
            stack.enter_context(patch.object(numerical, 'rows', return_value=[row]))
            stack.enter_context(patch.object(numerical, 'encoded', return_value=[carrier]))
            stack.enter_context(patch.object(numerical, '_view', return_value=([row], 'train')))
            stack.enter_context(patch.object(reporting, 'summarize', return_value=synthetic_contrasts))
            stack.enter_context(patch('trinite.training.require_environment'))
            stack.enter_context(patch.object(Decoder, 'forward', side_effect=AssertionError('audit forwarded')))
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            result = workflow.audit(root, request_raw, RID, 17, 'train', expected)
            self.assertTrue(result['verified'])
            self.assertFalse(result['predictions_replayed'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            # Mutation without rebinding fails custody.
            (root/'step-064-journal.jsonl').write_bytes(b'changed')
            with self.assertRaisesRegex(ContractError, 'bound artifact'):
                workflow.audit(root, request_raw, RID, 17, 'train', expected)
            # Even resealing cannot make a false journal agree with predictions.
            expected = seal(root)
            with self.assertRaisesRegex(ContractError, 'journal differs'):
                workflow.audit(root, request_raw, RID, 17, 'train', expected)
            (root/'step-064-journal.jsonl').write_bytes(artifacts['step-064-journal.jsonl'])
            changed = dict(report, training_fit=True)
            (root/'report.json').write_bytes(json_bytes(changed))
            expected = seal(root)
            with self.assertRaisesRegex(ContractError, 'training fit differs'):
                workflow.audit(root, request_raw, RID, 17, 'train', expected)

    def test_seed_is_strict(self):
        for invalid in (True, '17', 0, 18):
            with self.assertRaises(ContractError):
                numerical._seed(invalid)


if __name__ == '__main__':
    unittest.main()
