"""Standard-library admission and file safety; no model or target inference."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from cpu_cost_ledger_v1.request import LIMITS, PLAN_KEYS, plan, read


def example_plan(**changes):
    value = {'config': ModelConfig(context_length=32, blocks=1, width=8,
                heads=2, feed_forward_width=16).to_dict(),
             'lane': 'dense', 'model_identity': identity(b'synthetic model binding'),
             'checkpoint_identity': identity(b'synthetic checkpoint binding'),
             'prompts': ['', 'ab', '\u03bb'], 'repetitions': 2,
             'output_tokens': 3, 'worker_seconds': 30}
    value.update(changes)
    return value


class PlanContracts(unittest.TestCase):
    def test_all_passes_bound_before_observations(self):
        value = example_plan()
        before = copy.deepcopy(value)
        result = plan(value)
        # UTF-8 lambda is two bytes, not one token; every prompt includes BOS.
        self.assertEqual(result['maximum_forward_calls'], 3 * 2 * 3 * 3)
        self.assertEqual(result['maximum_emitted_tokens_all_passes'], 54)
        self.assertEqual(result['maximum_processed_positions'],
                         3 * 2 * sum(3*n + 3 for n in (1, 3, 3)))
        self.assertEqual(result['predicted_unique_parameter_bytes'],
                         ModelConfig.from_dict(value['config']).parameter_count * 4)
        self.assertEqual(value, before)
        self.assertEqual(set(value), PLAN_KEYS)
        self.assertIsNone(result['limits']['memory_ceiling_bytes'])
        self.assertEqual(result['limits']['workers'], 1)
        self.assertEqual(result['limits']['batch'], 1)

    def test_prompt_identity_is_exact_bytes_and_bos_prefix(self):
        value = example_plan()
        bindings = plan(value)['prompt_bindings']
        for prompt, binding in zip(value['prompts'], bindings):
            encoded = prompt.encode('utf-8')
            self.assertEqual(binding, {
                'utf8_identity': identity(encoded),
                'initial_prefix_identity': identity(json_bytes([256, *encoded])),
                'input_tokens_including_bos': 1 + len(encoded)})

    def test_all_native_lanes_admitted_without_model_allocation(self):
        for lane in ('dense', 'ternary', 'four-state'):
            with self.subTest(lane=lane):
                self.assertEqual(plan(example_plan(lane=lane))['maximum_forward_calls'], 54)

    def test_exact_schema_rejects_extra_and_missing_fields(self):
        for key in PLAN_KEYS:
            with self.subTest(missing=key):
                value = example_plan()
                del value[key]
                with self.assertRaises(ContractError):
                    plan(value)
        with self.assertRaises(ContractError):
            plan(example_plan(cache_mode='enabled'))

    def test_numeric_bounds_and_bool_rejection(self):
        for field, high in (('repetitions', LIMITS['max_repetitions']),
                            ('output_tokens', LIMITS['max_output_tokens']),
                            ('worker_seconds', LIMITS['max_worker_seconds'])):
            for invalid in (0, -1, high+1, True, 1.0, '1', None):
                with self.subTest(field=field, value=invalid):
                    with self.assertRaises(ContractError):
                        plan(example_plan(**{field: invalid}))

    def test_highest_repetition_output_and_deadline_admitted(self):
        value = example_plan(repetitions=3, output_tokens=16, worker_seconds=600)
        self.assertEqual(plan(value)['maximum_forward_calls'], 3*3*3*16)

    def test_bounded_distinct_text_prompts(self):
        for invalid in ([], ['x']*2, ['a', 'b', 'c', 'd', 'e'], (), 'abc',
                        [b'abc'], [None], [3]):
            with self.subTest(prompts=invalid):
                with self.assertRaises(ContractError):
                    plan(example_plan(prompts=invalid))
        self.assertEqual(len(plan(example_plan(prompts=['a', 'b', 'c', 'd']))['prompt_bindings']), 4)

    def test_context_boundary_and_one_past_no_truncation(self):
        # Last of three calls consumes 29-byte prompt + BOS + two emitted IDs.
        value = example_plan(prompts=['a'*29])
        self.assertEqual(plan(value)['maximum_processed_positions'], 6*(30+31+32))
        with self.assertRaises(ContractError):
            plan(example_plan(prompts=['a'*30]))
        config = ModelConfig(context_length=2, blocks=1, width=8, heads=2,
                             feed_forward_width=16).to_dict()
        self.assertEqual(plan(example_plan(config=config, prompts=[''], output_tokens=2))
                         ['maximum_processed_positions'], 6*(1+2))
        with self.assertRaises(ContractError):
            plan(example_plan(config=config, prompts=['a'], output_tokens=2))

    def test_checkpoint_and_model_identities_required(self):
        for field in ('checkpoint_identity', 'model_identity'):
            for invalid in ('', 'sha256:'+'g'*64, 'sha256:'+'0'*63, None, 4):
                with self.subTest(field=field, value=invalid):
                    with self.assertRaises(ContractError):
                        plan(example_plan(**{field: invalid}))

    def test_lane_config_and_parameter_admission(self):
        for lane in ('cuda', 'dense-cached', None, 1):
            with self.subTest(lane=lane):
                with self.assertRaises(ContractError):
                    plan(example_plan(lane=lane))
        config = ModelConfig(blocks=24, width=1024, heads=16,
                             feed_forward_width=4096).to_dict()
        with self.assertRaisesRegex(ContractError, 'parameters'):
            plan(example_plan(config=config))
        value = example_plan()
        value['config']['dtype'] = 'float16'
        with self.assertRaises(ContractError):
            plan(value)

    def test_predicted_attention_cells_before_allocation(self):
        config = ModelConfig(context_length=256, blocks=1, width=256,
                             heads=256, feed_forward_width=1).to_dict()
        self.assertLess(ModelConfig.from_dict(config).parameter_count, LIMITS['max_parameters'])
        with self.assertRaisesRegex(ContractError, 'attention'):
            plan(example_plan(config=config, prompts=['a'*239], output_tokens=16))


class ReadContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='cpu-ledger-contract-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_exact_read_is_non_mutating(self):
        path = self.root/'artifact'
        path.write_bytes(b'1234')
        before = path.stat()
        self.assertEqual(read(path, 4), b'1234')
        after = path.stat()
        self.assertEqual((before.st_ino, before.st_size, before.st_mtime_ns),
                         (after.st_ino, after.st_size, after.st_mtime_ns))
        with self.assertRaises(ContractError):
            read(path, 3)
        self.assertEqual(path.read_bytes(), b'1234')

    def test_symlink_leaf_and_ancestor_rejected(self):
        target = self.root/'target'
        target.mkdir()
        (target/'artifact').write_bytes(b'a')
        (self.root/'leaf').symlink_to(target/'artifact')
        (self.root/'linked').symlink_to(target, target_is_directory=True)
        for path in (self.root/'leaf', self.root/'linked'/'artifact'):
            with self.subTest(path=path):
                with self.assertRaises(ContractError):
                    read(path)

    def test_directory_and_fifo_rejected_without_blocking(self):
        with self.assertRaises(ContractError):
            read(self.root)
        fifo = self.root/'fifo'
        os.mkfifo(fifo)
        with self.assertRaises(ContractError):
            read(fifo)


if __name__ == '__main__':
    unittest.main()
