"""Bounded synthetic native mechanics and explicit doubles, not target evidence."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from trinite.model import Decoder
from trinite.training import model_identity
from cpu_cost_ledger_v1.measure import Meter, load_model, measure, unique_storage_bytes


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def make_model(lane='dense'):
    model = Decoder(ModelConfig(context_length=32, blocks=1, width=8, heads=2,
                                feed_forward_width=16), lane=lane, seed=7)
    model.eval()
    return model


def payload(model):
    return save({'model.'+name: parameter.detach().clone().contiguous()
                 for name, parameter in model.named_parameters()})


def make_plan(model, raw=None, **changes):
    value = {'config': model.config.to_dict(), 'lane': model.lane,
             'model_identity': model_identity(model),
             'checkpoint_identity': identity(payload(model) if raw is None else raw),
             'prompts': ['', 'ab', '\u03bb'], 'repetitions': 2, 'output_tokens': 3,
             'worker_seconds': 30}
    value.update(changes)
    return value


def fixed_forward(token):
    """Explicit software double for stopping/tie/failure tests only."""
    def forward(ids, mask):
        logits = torch.zeros((*ids.shape, 259), dtype=torch.float32)
        if token is not None:
            logits[:, :, token] = 1
        return SimpleNamespace(logits=logits)
    return forward


class NativeMeasurements(unittest.TestCase):
    def test_real_synthetic_models_all_lanes_state_and_prefixes(self):
        for lane in ('dense', 'ternary', 'four-state'):
            with self.subTest(lane=lane):
                model = make_model(lane)
                value = make_plan(model)
                before_rng = torch.get_rng_state().clone()
                before_flags = {n: m.training for n, m in model.named_modules()}
                journal = []
                result = measure(model, value, journal.append)
                self.assertEqual(result['rows'], journal)
                self.assertEqual(len(journal), 6)
                self.assertEqual(model_identity(model), value['model_identity'])
                self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
                self.assertEqual(before_flags, {n: m.training for n, m in model.named_modules()})
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertEqual(result['unique_resident_parameter_storage_bytes'],
                                 model.config.parameter_count*4)
                self.assertIsNone(result['native_kv_payload_bytes'])
                self.assertEqual(result['cache_mode'], 'unsupported')
                self.assertIsNone(result['memory_ceiling_bytes'])
                self.assertIsNone(result['transient_forward_peak_bytes'])
                calls = positions = 0
                for index, row in enumerate(journal):
                    prompt_index, repetition = divmod(index, 2)
                    self.assertEqual((row['prompt_index'], row['repetition']), (prompt_index, repetition))
                    self.assertTrue(row['parity'])
                    self.assertEqual(row['before'], row['measured'])
                    self.assertEqual(row['after'], row['measured'])
                    tokens = row['measured']['tokens']
                    base = [256, *value['prompts'][prompt_index].encode('utf-8')]
                    self.assertEqual(len(row['forward_observations']), len(tokens))
                    for step, observation in enumerate(row['forward_observations']):
                        prefix = base + tokens[:step]
                        self.assertEqual(observation['checkpoint_identity'], value['checkpoint_identity'])
                        self.assertEqual(observation['prefix_identity'], identity(json_bytes(prefix)))
                        self.assertEqual(observation['mask_identity'], identity(json_bytes([[True]*len(prefix)])))
                        self.assertEqual(observation['absolute_positions'], list(range(len(prefix))))
                        self.assertEqual(observation['input_tokens'], len(prefix))
                        self.assertEqual(observation['phase'], 'prompt_forward' if step == 0 else 'decode_full_prefix')
                        for metric in ('wall_ns', 'process_cpu_ns'):
                            self.assertIs(type(observation[metric]), int)
                            self.assertGreaterEqual(observation[metric], 0)
                    self.assertEqual(row['forward_observations'][0]['prefix_identity'],
                                     row['prompt_binding']['initial_prefix_identity'])
                    calls += 3*len(tokens)
                    positions += 3*sum(len(base)+i for i in range(len(tokens)))
                self.assertEqual(result['all_passes_forward_calls'], calls)
                self.assertEqual(result['all_passes_processed_positions'], positions)

    def test_tied_embedding_counted_once(self):
        model = make_model()
        self.assertIs(model.output_weight, model.token_embedding)
        self.assertEqual(unique_storage_bytes(model), model.config.parameter_count*4)
        self.assertGreater(sum(p.numel()*p.element_size()
                               for _, p in model.named_parameters(remove_duplicate=False)),
                           unique_storage_bytes(model))

    def test_double_early_eos_and_lowest_id_ties(self):
        for token, expected, stop in ((257, [257], 'eos'), (None, [0, 0, 0], 'output_cap')):
            model = make_model()
            value = make_plan(model, prompts=[''], repetitions=1)
            with self.subTest(token=token), patch.object(model, 'forward', side_effect=fixed_forward(token)):
                result = measure(model, value)
            row = result['rows'][0]
            self.assertEqual(row['measured']['tokens'], expected)
            self.assertEqual(row['measured']['stop_reason'], stop)
            self.assertEqual(result['all_passes_forward_calls'], 3*len(expected))
            self.assertEqual(result['all_passes_processed_positions'], 3*sum(range(1, len(expected)+1)))

    def test_double_parity_mismatch_is_journaled_then_fails(self):
        model = make_model()
        value = make_plan(model, prompts=[''], repetitions=1, output_tokens=2)
        invoked = 0
        def changing(ids, mask):
            nonlocal invoked
            token = 1 if 2 <= invoked < 4 else 0
            invoked += 1
            return fixed_forward(token)(ids, mask)
        journal = []
        with patch.object(model, 'forward', side_effect=changing):
            with self.assertRaisesRegex(ContractError, 'parity'):
                measure(model, value, journal.append)
        self.assertEqual(len(journal), 1)
        self.assertFalse(journal[0]['parity'])
        self.assertEqual(journal[0]['before']['tokens'], [0, 0])
        self.assertEqual(journal[0]['measured']['tokens'], [1, 1])

    def test_meter_limits_reject_before_another_forward(self):
        model = make_model()
        ids = torch.tensor([[256]], dtype=torch.int64)
        mask = torch.ones_like(ids, dtype=torch.bool)
        meter = Meter(model, identity(b'synthetic'), 1, 1)
        with patch.object(model, 'forward', side_effect=fixed_forward(0)) as forward:
            meter(ids, mask)
            with self.assertRaisesRegex(ContractError, 'budget'):
                meter(ids, mask)
            self.assertEqual(forward.call_count, 1)
        meter = Meter(model, identity(b'synthetic'), 1, 0)
        with patch.object(model, 'forward') as forward:
            with self.assertRaisesRegex(ContractError, 'budget'):
                meter(ids, mask)
            forward.assert_not_called()

    def test_double_nonfinite_logits_rejected(self):
        model = make_model()
        meter = Meter(model, identity(b'synthetic'), 1, 1)
        ids = torch.tensor([[256]], dtype=torch.int64)
        output = SimpleNamespace(logits=torch.full((1, 1, 259), float('nan')))
        with patch.object(model, 'forward', return_value=output):
            with self.assertRaisesRegex(ContractError, 'nonfinite'):
                meter(ids, torch.ones_like(ids, dtype=torch.bool))

    def test_existing_gradient_or_wrong_identity_prevents_forward(self):
        model = make_model()
        value = make_plan(model)
        next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
        with patch.object(model, 'forward') as forward:
            with self.assertRaisesRegex(ContractError, 'gradient'):
                measure(model, value)
            forward.assert_not_called()
        next(model.parameters()).grad = None
        value['model_identity'] = identity(b'wrong model')
        with patch.object(model, 'forward') as forward:
            with self.assertRaisesRegex(ContractError, 'identity'):
                measure(model, value)
            forward.assert_not_called()


class ModelOnlyLoading(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='cpu-ledger-native-')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'synthetic.safetensors'
        self.model = make_model()

    def test_model_only_load_matches_exact_native_parameters(self):
        raw = payload(self.model)
        self.path.write_bytes(raw)
        value = make_plan(self.model, raw)
        loaded, artifact_bytes = load_model(self.path, value)
        self.assertEqual(artifact_bytes, len(raw))
        self.assertEqual(model_identity(loaded), value['model_identity'])
        self.assertFalse(loaded.training)
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertTrue(all(p.grad is None for p in loaded.parameters()))
        self.assertIs(loaded.output_weight, loaded.token_embedding)

    def test_checkpoint_hash_rejected_before_allocation_or_decode(self):
        self.path.write_bytes(payload(self.model))
        value = make_plan(self.model, checkpoint_identity=identity(b'wrong'))
        with (patch('cpu_cost_ledger_v1.measure.Decoder') as constructor,
              patch('cpu_cost_ledger_v1.measure.load') as reader):
            with self.assertRaisesRegex(ContractError, 'checkpoint identity'):
                load_model(self.path, value)
            constructor.assert_not_called()
            reader.assert_not_called()

    def test_extra_optimizer_missing_tensor_shape_dtype_nonfinite(self):
        original = {'model.'+n: p.detach().clone() for n, p in self.model.named_parameters()}
        key = next(iter(original))
        mutations = {
            'optimizer': {**original, 'optimizer.unwanted.step': torch.tensor(1.)},
            'missing': {n: p for n, p in original.items() if n != key},
            'shape': {**original, key: torch.zeros(1)},
            'dtype': {**original, key: original[key].double()},
            'nonfinite': {**original, key: torch.full_like(original[key], float('nan'))},
        }
        for label, tensors in mutations.items():
            with self.subTest(case=label):
                raw = save(tensors)
                self.path.write_bytes(raw)
                with self.assertRaises(ContractError):
                    load_model(self.path, make_plan(self.model, raw))

    def test_model_identity_rejected_after_safe_structure(self):
        raw = payload(self.model)
        self.path.write_bytes(raw)
        value = make_plan(self.model, raw, model_identity=identity(b'other tensor identity'))
        with self.assertRaisesRegex(ContractError, 'model identity'):
            load_model(self.path, value)


if __name__ == '__main__':
    unittest.main()
