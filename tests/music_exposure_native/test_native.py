"""Tiny synthetic numerical conformance, never target music training/inference."""
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from trinite.contracts import ModelConfig, ContractError, identity, json_bytes, parse_json
from trinite.model import Decoder
from trinite.training import (RunConfig, TrainingState, create_optimizer, evaluate,
                              model_identity, run_steps)
from trinite.tokenizer import ByteTokenizer, EOS
from music_exposure_v1 import numerical

RID = identity(b'synthetic-exposure-request')
AID = identity(b'synthetic-exposure-admission')
DATA = b'synthetic-dataset-not-target'


def fixture(steps=8, validation_every=2):
    row = {'example_identity': identity(b'synthetic-row'), 'family_id': 'synthetic',
           'task': 'synthetic', 'split': 'train', 'prompt': 'Q=', 'answer': '7'}
    enc = ByteTokenizer(128).encode('Q=7', answer_start=2)
    carrier = {'identity': row['example_identity'], 'ids': enc.input_ids,
               'mask': enc.attention_mask, 'loss': enc.loss_mask}
    config = RunConfig(model=ModelConfig(context_length=128, blocks=1, width=8,
        heads=2, feed_forward_width=16), seed=17, steps=steps, batch_size=1,
        learning_rate='0.003', validation_every=validation_every,
        max_seconds=120, max_target_tokens=2*steps)
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

    def test_real_prompt_only_native_score_uses_independent_model_copy(self):
        row, _, state = fixture()
        before = numerical.snapshot(state, RID)
        models = []
        from trinite.foundations import greedy
        def record_model(model, prompt, cap):
            models.append(model)
            return greedy(model, prompt, cap)
        with patch.object(numerical, 'rows', return_value=[row]), \
             patch('trinite.foundations.greedy', side_effect=record_model):
            summary, raw = numerical.score(state)
            self.assertEqual(numerical.check_score(raw), summary)
        self.assertEqual(numerical.snapshot(state, RID), before)
        self.assertEqual(len(models), 1)
        self.assertIsNot(models[0], state.model)
        for original, isolated in zip(state.model.parameters(), models[0].parameters()):
            self.assertNotEqual(original.data_ptr(), isolated.data_ptr())
        record = parse_json(raw)['predictions'][0]
        self.assertEqual(record['expected_tokens'], [55, EOS])
        self.assertTrue(all(p.grad is None for p in state.model.parameters()))

    def test_exact_eos_and_no_repair_controls(self):
        row, _, state = fixture()
        for tokens, correct in (([55, EOS], True), ([EOS], False), ([47]*64, False)):
            with self.subTest(tokens=tokens), \
                 patch.object(numerical, 'rows', return_value=[row]), \
                 patch('trinite.foundations.greedy', return_value=tokens) as greedy:
                summary, raw = numerical.score(state)
                self.assertEqual(summary['correct'], int(correct))
                self.assertEqual(numerical.check_score(raw), summary)
                self.assertEqual(greedy.call_args.args[1:], ('Q=', 64))
                self.assertEqual(parse_json(raw)['predictions'][0]['generated_tokens'], tokens)

    def test_non_native_termination_rejected(self):
        row, _, state = fixture()
        for tokens in ([47], [EOS, 55]):
            with self.subTest(tokens=tokens), \
                 patch.object(numerical, 'rows', return_value=[row]), \
                 patch('trinite.foundations.greedy', return_value=tokens):
                with self.assertRaisesRegex(ContractError, 'termination'):
                    numerical.score(state)

    def test_intermediate_probe_next_updates_match_no_probe_exactly(self):
        row, _, probed = fixture()
        _, _, control = fixture()
        run_steps(probed, stop_after=2)
        run_steps(control, stop_after=2)
        before = numerical.snapshot(probed, RID)
        with patch.object(numerical, 'rows', return_value=[row]), \
             patch('trinite.foundations.greedy', return_value=[55, EOS]):
            numerical.score(probed)
        self.assertEqual(numerical.snapshot(probed, RID), before)
        run_steps(probed)
        run_steps(control)
        # Includes model and AdamW slots, all losses, history, validation, cursor.
        self.assertEqual(numerical.snapshot(probed, RID), numerical.snapshot(control, RID))

    def test_rng_and_mixed_module_modes_restored_including_error_paths(self):
        row, _, state = fixture()
        state.model.train()
        state.model.final_norm.eval()
        modes = [(m, m.training) for m in state.model.modules()]
        py = random.getstate(); np_state = np.random.get_state(); th = torch.get_rng_state().clone()
        def noisy_emit(_):
            random.random(); np.random.random(); torch.rand(3)
            state.model.eval()
            raise RuntimeError('synthetic journal failure')
        with patch.object(numerical, 'rows', return_value=[row]), \
             patch('trinite.foundations.greedy', return_value=[55, EOS]):
            with self.assertRaisesRegex(RuntimeError, 'journal failure'):
                numerical.score(state, noisy_emit)
        self.assertEqual(random.getstate(), py)
        after = np.random.get_state()
        self.assertEqual(after[0], np_state[0]); np.testing.assert_array_equal(after[1], np_state[1])
        self.assertEqual(after[2:], np_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), th))
        self.assertEqual([(m, m.training) for m, _ in modes], modes)

    def test_original_parameter_mutation_rejected_not_silently_restored(self):
        row, _, state = fixture()
        before = model_identity(state.model)
        def mutate(_):
            with torch.no_grad():
                next(state.model.parameters()).add_(0.01)
        with patch.object(numerical, 'rows', return_value=[row]), \
             patch('trinite.foundations.greedy', return_value=[55, EOS]):
            with self.assertRaisesRegex(ContractError, 'changed live training'):
                numerical.score(state, mutate)
        self.assertNotEqual(model_identity(state.model), before)

    def test_optimizer_mutation_rejected(self):
        row, _, state = fixture()
        run_steps(state, stop_after=2)
        def mutate(_):
            p = next(state.model.parameters())
            state.optimizer.state[p]['exp_avg'].add_(0.001)
        with patch.object(numerical, 'rows', return_value=[row]), \
             patch('trinite.foundations.greedy', return_value=[55, EOS]):
            with self.assertRaisesRegex(ContractError, 'changed live training'):
                numerical.score(state, mutate)

    def test_data_and_progress_mutation_rejected(self):
        for mutation in ('data', 'history'):
            row, _, state = fixture()
            run_steps(state, stop_after=2)
            def mutate(_):
                if mutation == 'data':
                    state.data['train'][0]['identity'] = identity(b'changed')
                else:
                    state.history[0]['loss_hex'] = float(0).hex()
            with self.subTest(mutation=mutation), \
                 patch.object(numerical, 'rows', return_value=[row]), \
                 patch('trinite.foundations.greedy', return_value=[55, EOS]):
                with self.assertRaises(ContractError):
                    numerical.score(state, mutate)

    def test_parameter_learning_flag_or_identity_mutation_rejected(self):
        for mutation in ('requires_grad', 'parameter_identity'):
            row, _, state = fixture()
            def mutate(_):
                if mutation == 'requires_grad':
                    state.model.final_norm.weight.requires_grad_(False)
                else:
                    state.model.final_norm.weight = torch.nn.Parameter(
                        state.model.final_norm.weight.detach().clone())
            with self.subTest(mutation=mutation), \
                 patch.object(numerical, 'rows', return_value=[row]), \
                 patch('trinite.foundations.greedy', return_value=[55, EOS]):
                with self.assertRaisesRegex(ContractError, 'changed live training'):
                    numerical.score(state, mutate)

    def test_pending_gradients_rejected(self):
        row, _, state = fixture()
        next(state.model.parameters()).grad = torch.zeros_like(next(state.model.parameters()))
        with patch.object(numerical, 'rows', return_value=[row]):
            with self.assertRaisesRegex(ContractError, 'cleared'):
                numerical.score(state)

    def test_safe_checkpoint_roundtrip_without_forward(self):
        row, carrier, state = fixture()
        run_steps(state, stop_after=2)
        payload, metadata_raw = numerical.snapshot(state, RID)
        metadata = parse_json(metadata_raw, canonical=True)
        with patch.object(numerical, '_admitted', return_value=DATA), \
             patch.object(numerical, 'plan_for', return_value=state.config), \
             patch.object(numerical, 'rows', return_value=[row]), \
             patch.object(numerical, 'encoded', return_value=[carrier]), \
             patch('trinite.training.require_environment'), \
             patch.object(Decoder, 'forward', side_effect=AssertionError('verification forwarded')):
            restored = numerical.verify_checkpoint(payload, metadata, 17, AID, RID)
            self.assertEqual(numerical.snapshot(restored, RID), (payload, metadata_raw))
            for field, value in (('seed', 29), ('target_tokens', 5), ('cursor', 3)):
                changed = deepcopy(metadata); changed[field] = value
                with self.subTest(field=field), self.assertRaises(ContractError):
                    numerical.verify_checkpoint(payload, changed, 17, AID, RID)
            with self.assertRaises(ContractError):
                numerical.verify_checkpoint(b'not-safe-tensors', metadata, 17, AID, RID)

    def test_continuous_synthetic_train_three_observations_no_restart(self):
        row, _, state = fixture()
        _, _, control = fixture()
        artifacts, journals = {}, []
        with patch.object(numerical, 'state_for', return_value=state) as initialize, \
             patch.object(numerical, 'CHECKPOINTS', (0, 2, 8)), \
             patch.object(numerical, 'rows', return_value=[row]), \
             patch('trinite.foundations.greedy', return_value=[55, EOS]):
            report = numerical.train(17, RID, AID,
                lambda n, b: artifacts.__setitem__(n, b), lambda p, r: journals.append((p, r)))
        initialize.assert_called_once_with(17, AID)
        self.assertEqual(set(report['observations']), {'0', '2', '8'})
        self.assertFalse(report['training_fit'])  # One synthetic row is not40row fit.
        self.assertEqual(report['step'], 8); self.assertEqual(report['target_tokens'], 16)
        self.assertEqual([p for p, _ in journals], ['step-000', 'step-002', 'step-008'])
        run_steps(control)
        self.assertEqual(numerical.snapshot(state, RID), numerical.snapshot(control, RID))
        for step in (0, 2, 8):
            self.assertIn(f'step-{step:03d}-predictions.json', artifacts)

    def test_cumulative_time_does_not_reset_at_intermediate_observation(self):
        row, _, state = fixture(steps=3, validation_every=1)
        artifacts = {}
        # 60+59+2 =121 seconds; each segment alone is below120.
        with patch.object(numerical, 'state_for', return_value=state), \
             patch.object(numerical, 'CHECKPOINTS', (0, 2, 3)), \
             patch.object(numerical, 'rows', return_value=[row]), \
             patch('trinite.foundations.greedy', return_value=[55, EOS]), \
             patch.object(numerical, 'monotonic', side_effect=[0., 60., 100., 159., 200., 202.]):
            with self.assertRaisesRegex(ContractError, 'deadline'):
                numerical.train(17, RID, AID, lambda n, b: artifacts.__setitem__(n, b))
        self.assertEqual(state.step, 3)  # Late final update cannot earn completion.
        failure = parse_json(artifacts['failure.json'])
        self.assertEqual(failure['outcome'], 'incomplete')
        self.assertEqual(failure['completed_observations'], [0, 2])
        self.assertEqual(failure['missing_observations'], [3])
        self.assertEqual(float.fromhex(failure['update_seconds_hex']), 121.)
        self.assertIn('failure.safetensors', artifacts)
        self.assertNotIn('step-003-predictions.json', artifacts)

    def test_exhausted_budget_prevents_next_update(self):
        row, _, state = fixture(steps=3, validation_every=1)
        artifacts = {}
        with patch.object(numerical, 'state_for', return_value=state), \
             patch.object(numerical, 'CHECKPOINTS', (0, 2, 3)), \
             patch.object(numerical, 'rows', return_value=[row]), \
             patch('trinite.foundations.greedy', return_value=[55, EOS]), \
             patch.object(numerical, 'monotonic', side_effect=[0., 60., 100., 160.]):
            with self.assertRaisesRegex(ContractError, 'deadline'):
                numerical.train(17, RID, AID, lambda n, b: artifacts.__setitem__(n, b))
        self.assertEqual(state.step, 2)
        self.assertEqual(parse_json(artifacts['failure.json'])['missing_observations'], [3])

    def test_initialization_failure_retains_all_missing_observations(self):
        artifacts = {}
        with patch.object(numerical, 'state_for', side_effect=RuntimeError('synthetic initialization')):
            with self.assertRaisesRegex(RuntimeError, 'initialization'):
                numerical.train(17, RID, AID, lambda n, b: artifacts.__setitem__(n, b))
        failure = parse_json(artifacts['failure.json'])
        self.assertIsNone(failure['completed_updates'])
        self.assertEqual(failure['completed_observations'], [])
        self.assertEqual(failure['missing_observations'], [0, 64, 640])

    def test_native_workflow_run_and_readonly_audit_synthetic_composition(self):
        from trinite.foundations_plan import FoundationsPlan
        from music_exposure_v1 import workflow, reporting
        # Forty SHORT artificial Qnn=7 rows and width8, not any music prompt.
        # Actual native640 updates/checkpoints/reader, with mock generation and
        # synthetic admission/request/contrast summaries: mechanics, NOT fit.
        synthetic_rows, carriers = [], []
        for i in range(40):
            prompt = f'Q{i:02d}='
            row = {'example_identity': identity(f'synthetic-{i}'.encode()),
                   'family_id': 'synthetic', 'task': 'synthetic', 'split': 'train',
                   'prompt': prompt, 'answer': '7'}
            enc = ByteTokenizer(128).encode(prompt+'7', answer_start=len(prompt))
            synthetic_rows.append(row)
            carriers.append({'identity': row['example_identity'], 'ids': enc.input_ids,
                             'mask': enc.attention_mask, 'loss': enc.loss_mask})
        config = FoundationsPlan(model=ModelConfig(context_length=128, blocks=1,
            width=8, heads=2, feed_forward_width=16), seed=17, steps=640,
            batch_size=8, learning_rate='0.003', validation_every=64,
            max_seconds=120, max_target_tokens=10240)
        model = Decoder(config.model, lane='dense', seed=17)
        state = TrainingState(config, 'dense', model, create_optimizer(model, config),
            {'train': carriers, 'validation': carriers}, identity(DATA), AID,
            model_identity(model), evaluate(model, carriers, 8))
        request_raw = b'synthetic-exposure-request'
        synthetic_contrasts = {'synthetic_contract_only': True}
        def reseal(root):
            completion = parse_json((root/'completion.json').read_bytes(), canonical=True)
            completion['artifacts'] = {name: identity((root/name).read_bytes())
                                       for name in workflow.names()}
            raw = json_bytes(completion)
            (root/'completion.json').write_bytes(raw)
            return identity(raw)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch.object(workflow, 'validate',
                                             return_value={'admission_identity': AID}))
            stack.enter_context(patch.object(numerical, 'state_for', return_value=state))
            stack.enter_context(patch.object(numerical, '_admitted', return_value=DATA))
            stack.enter_context(patch.object(numerical, 'plan_for', return_value=config))
            stack.enter_context(patch.object(numerical, 'rows', return_value=synthetic_rows))
            stack.enter_context(patch.object(numerical, 'encoded', return_value=carriers))
            stack.enter_context(patch.object(reporting, 'summarize', return_value=synthetic_contrasts))
            stack.enter_context(patch('trinite.foundations.greedy', return_value=[55, EOS]))
            stat = root.stat()
            expected = workflow.run(root, request_raw, RID, 17, (stat.st_dev, stat.st_ino))
            self.assertEqual(state.step, 640)
            self.assertEqual(state.cursor, 5120)
            self.assertEqual(state.target_tokens, 10240)
            # Safe reader and consistency audit must not run a forward at all.
            stack.enter_context(patch.object(Decoder, 'forward',
                                             side_effect=AssertionError('audit forwarded')))
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            result = workflow.audit(root, request_raw, RID, 17, expected)
            self.assertTrue(result['verified'])
            self.assertFalse(result['predictions_replayed'])
            self.assertFalse(result['optimizer_replayed'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            self.assertEqual(set(result['report']['observations']), {'0', '64', '640'})
            # Hash resealing cannot legitimize altered prior validation history.
            metadata_path = root/'step-640-checkpoint.json'
            original = metadata_path.read_bytes()
            metadata = parse_json(original, canonical=True)
            metadata['validation'][0]['loss_hex'] = float(0).hex()
            metadata_path.write_bytes(json_bytes(metadata)); expected = reseal(root)
            with self.assertRaisesRegex(ContractError, 'validation lineage'):
                workflow.audit(root, request_raw, RID, 17, expected)
            metadata_path.write_bytes(original); expected = reseal(root)
            # A resealed final report cannot elect the intermediate checkpoint.
            report_path = root/'report.json'
            original_report = report_path.read_bytes()
            changed = parse_json(original_report, canonical=True); changed['step'] = 64
            report_path.write_bytes(json_bytes(changed)); expected = reseal(root)
            with self.assertRaisesRegex(ContractError, 'final report progress'):
                workflow.audit(root, request_raw, RID, 17, expected)
            report_path.write_bytes(original_report); expected = reseal(root)
            (root/'step-640-journal.jsonl').unlink()
            with self.assertRaisesRegex(ContractError, 'physical inventory'):
                workflow.audit(root, request_raw, RID, 17, expected)

    def test_seed_is_strict_and_no_heldout_entrypoint(self):
        for invalid in (True, '17', 0, 18):
            with self.assertRaises(ContractError):
                numerical._seed(invalid)
        self.assertFalse(hasattr(numerical, 'holdout'))


if __name__ == '__main__':
    unittest.main()
