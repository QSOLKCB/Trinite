"""Independent fixed-buffer checks with CPU-only software tensor doubles.

No upstream module, real checkpoint, PyTorch backend, or GPU is executed here.
The oracle fills rectangular blocks rather than copying the runtime predicate.
"""
import copy
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from ncm_reference_v1 import core, runtime

LEVELS = (1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89)


def config():
    return SimpleNamespace(
        model=SimpleNamespace(prefix_len=91, attn_pattern='block_diagonal',
                              query_level0=True, use_rope=True, l0_autoregressive=False),
        tokenizer=SimpleNamespace(vqvae=SimpleNamespace(
            vector_quantizer_config=SimpleNamespace(
                levels=SimpleNamespace(manual_levels=list(LEVELS), use_manual_levels=True)))),
    )


def oracle():
    size = 91 + sum(LEVELS)
    mask = [[float('-inf')] * size for _ in range(size)]
    for row in mask:
        row[:91] = [0.0] * 91
    labels = []
    start = 91
    for label, width in enumerate(LEVELS):
        for row in range(start, start + width):
            mask[row][start:start + width] = [0.0] * width
        labels.extend([label] * width)
        start += width
    return {'attn_mask_for_masking': [[mask]], 'lvl_1L': [labels]}


def flatten(value):
    if isinstance(value, list):
        for child in value:
            yield from flatten(child)
    else:
        yield value


class Tensor:
    def __init__(self, value, dtype='float32', device='cpu'):
        self.value = value
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)
        self.layout = 'strided'
        shape, item = [], value
        while isinstance(item, list):
            shape.append(len(item))
            item = item[0] if item else None
        self.shape = tuple(shape)

    def numel(self):
        return math.prod(self.shape)

    def is_floating_point(self):
        return self.dtype in ('float32', 'float64')

    def is_complex(self):
        return self.dtype == 'complex64'

    def all(self):
        return all(flatten(self.value))

    def sum(self):
        return sum(flatten(self.value))


class TorchDouble:
    Tensor = Tensor
    strided = 'strided'
    float32 = 'float32'
    int64 = 'int64'

    @staticmethod
    def tensor(value, dtype, device):
        return Tensor(value, dtype, device)

    @staticmethod
    def equal(left, right):
        return (left.shape == right.shape and left.dtype == right.dtype
                and all(a == b for a, b in zip(flatten(left.value), flatten(right.value))))

    @staticmethod
    def isfinite(value):
        return Tensor([math.isfinite(x) for x in flatten(value.value)], 'bool')

    @staticmethod
    def isnan(value):
        return Tensor([math.isnan(x) for x in flatten(value.value)], 'bool')

    @staticmethod
    def isposinf(value):
        return Tensor([x == float('inf') for x in flatten(value.value)], 'bool')

    @staticmethod
    def isneginf(value):
        return Tensor([x == float('-inf') for x in flatten(value.value)], 'bool')


class FixedBufferReview(unittest.TestCase):
    def setUp(self):
        self.values = oracle()
        self.fixed = {
            key: Tensor(value, 'int64' if key == 'lvl_1L' else 'float32')
            for key, value in self.values.items()
        }
        self.expected = {**self.fixed, 'learned.weight': Tensor([[0.0, 0.0], [0.0, 0.0]])}
        self.state = copy.deepcopy(self.expected)
        # Learned values need not equal model initialization, only its safe schema.
        self.state['learned.weight'].value = [[0.2, -0.3], [0.5, 0.8]]

    def validate(self, state=None, expected=None, fixed=None):
        with redirect_stdout(io.StringIO()):
            return runtime.validate_state(
                TorchDouble, self.state if state is None else state,
                self.expected if expected is None else expected,
                self.fixed if fixed is None else fixed,
            )

    def test_independent_block_fill_matches_every_derived_cell(self):
        actual = runtime.generator_fixed_values(config())
        self.assertEqual(actual, self.values)
        values = list(flatten(actual['attn_mask_for_masking']))
        self.assertEqual(len(values), 413 * 413)
        self.assertEqual(values.count(0.0), 55169)
        self.assertEqual(values.count(float('-inf')), 115400)
        self.assertEqual(len(actual['lvl_1L'][0]), 322)

    def test_derived_buffer_shapes_and_dtypes_are_exact(self):
        fixed = runtime.generator_fixed_buffers(TorchDouble, config())
        self.assertEqual(set(fixed), set(self.fixed))
        for name in fixed:
            self.assertTrue(TorchDouble.equal(fixed[name], self.fixed[name]))
        self.assertEqual(fixed['attn_mask_for_masking'].shape, (1, 1, 413, 413))
        self.assertEqual(fixed['lvl_1L'].shape, (1, 322))
        self.validate()

    def test_unsupported_configuration_is_not_generalized(self):
        for name, value in [('prefix_len', 90), ('attn_pattern', 'block_causal'),
                            ('query_level0', False), ('use_rope', False),
                            ('l0_autoregressive', True)]:
            cfg = config()
            setattr(cfg.model, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime.generator_fixed_values(cfg)
        cfg = config()
        cfg.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels[-1] = 88
        with self.assertRaises(ValueError):
            runtime.generator_fixed_values(cfg)

    def test_matching_infinity_counts_do_not_authorize_wrong_positions(self):
        mask = self.state['attn_mask_for_masking'].value[0][0]
        mask[0][0], mask[0][91] = mask[0][91], mask[0][0]
        self.assertEqual(list(flatten(mask)).count(float('-inf')), 115400)
        with self.assertRaisesRegex(ValueError, 'attn_mask_for_masking'):
            self.validate()

    def test_fixed_zero_nan_and_positive_infinity_corruptions_fail(self):
        for value in (1.0, float('nan'), float('inf')):
            state = copy.deepcopy(self.state)
            state['attn_mask_for_masking'].value[0][0][0][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'attn_mask_for_masking'):
                self.validate(state=state)

    def test_negative_infinity_cannot_move_into_other_buffer_or_parameter(self):
        for name in ('learned.weight', 'other.buffer'):
            state, expected = copy.deepcopy(self.state), copy.deepcopy(self.expected)
            state[name] = Tensor([[float('-inf'), 0.0], [0.0, 0.0]])
            expected[name] = Tensor([[0.0, 0.0], [0.0, 0.0]])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.validate(state=state, expected=expected)

    def test_nan_and_positive_infinity_in_learned_values_remain_rejected(self):
        for value in (float('nan'), float('inf')):
            state = copy.deepcopy(self.state)
            state['learned.weight'].value[0][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'learned.weight'):
                self.validate(state=state)

    def test_mask_shape_dtype_layout_and_device_cannot_be_cast_away(self):
        for field, value in [('shape', (1, 1, 1, 413 * 413)), ('dtype', 'float64'),
                             ('layout', 'sparse'), ('device', SimpleNamespace(type='cuda'))]:
            state = copy.deepcopy(self.state)
            setattr(state['attn_mask_for_masking'], field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'attn_mask_for_masking'):
                self.validate(state=state)

    def test_level_buffer_requires_exact_order_shape_and_integer_dtype(self):
        for mutation in ('order', 'shape', 'dtype'):
            state = copy.deepcopy(self.state)
            tensor = state['lvl_1L']
            if mutation == 'order':
                tensor.value[0][0], tensor.value[0][1] = tensor.value[0][1], tensor.value[0][0]
            elif mutation == 'shape':
                tensor.shape = (322,)
            else:
                tensor.dtype = 'float32'
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'lvl_1L'):
                self.validate(state=state)

    def test_missing_and_extra_state_keys_are_named_and_rejected(self):
        for mutation in ('missing', 'extra'):
            state = copy.deepcopy(self.state)
            if mutation == 'missing':
                del state['lvl_1L']
                name = 'lvl_1L'
            else:
                state['surprise.weight'] = Tensor([0.0])
                name = 'surprise.weight'
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, name):
                self.validate(state=state)

    def test_expected_model_must_agree_with_independent_fixed_values(self):
        expected = copy.deepcopy(self.expected)
        expected['attn_mask_for_masking'].value[0][0][0][91] = 0.0
        with self.assertRaisesRegex(ValueError, 'constructed model'):
            self.validate(expected=expected)

    def test_arbitrary_fixed_name_cannot_exempt_a_nonfinite_weight(self):
        fixed = copy.deepcopy(self.fixed)
        state, expected = copy.deepcopy(self.state), copy.deepcopy(self.expected)
        fixed['learned.weight'] = Tensor([[float('-inf'), 0.0], [0.0, 0.0]])
        state['learned.weight'] = copy.deepcopy(fixed['learned.weight'])
        expected['learned.weight'] = copy.deepcopy(fixed['learned.weight'])
        with self.assertRaises(ValueError):
            self.validate(state=state, expected=expected, fixed=fixed)

    def test_rejected_nonfinite_tensor_has_actionable_diagnostics(self):
        self.state['learned.weight'].value = [[float('nan'), float('inf')],
                                               [float('-inf'), 0.0]]
        log = io.StringIO()
        with redirect_stdout(log), self.assertRaises(ValueError) as error:
            runtime.validate_state(TorchDouble, self.state, self.expected, self.fixed)
        self.assertIn('learned.weight', str(error.exception))
        records = [json.loads(line) for line in log.getvalue().splitlines() if line.startswith('{')]
        record = next(record for record in records if record.get('tensor') == 'learned.weight')
        self.assertEqual(record['shape'], [2, 2])
        self.assertEqual(record['dtype'], 'float32')
        self.assertEqual((record['nan'], record['positive_inf'], record['negative_inf']), (1, 1, 1))

    def test_restricted_loader_validates_schema_without_mutating_any_state(self):
        payload = b'CPU TEST DOUBLE: no serialized model or checkpoint content'
        before_state = copy.deepcopy(self.state)
        before_expected = copy.deepcopy(self.expected)
        loader = Mock(return_value={'model': self.state})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic-bytes.bin'
            path.write_bytes(payload)
            with patch.object(TorchDouble, 'load', loader, create=True), redirect_stdout(io.StringIO()):
                returned = runtime.safe_state(
                    TorchDouble, path, {'size': len(payload), 'sha256': core.digest(payload)},
                    self.expected, self.fixed,
                )
            self.assertIs(returned, self.state)
            self.assertEqual(path.read_bytes(), payload)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.kwargs, {'weights_only': True, 'map_location': 'cpu'})
        for name in self.state:
            self.assertTrue(TorchDouble.equal(self.state[name], before_state[name]), name)
            self.assertTrue(TorchDouble.equal(self.expected[name], before_expected[name]), name)

    def test_checkpoint_digest_mismatch_prevents_deserialization(self):
        payload = b'CPU TEST DOUBLE: deliberately mismatched content identity'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic-bytes.bin'
            path.write_bytes(payload)
            with patch.object(TorchDouble, 'load', create=True) as loader:
                with self.assertRaisesRegex(ValueError, 'checkpoint changed before load'):
                    runtime.safe_state(
                        TorchDouble, path, {'size': len(payload), 'sha256': '0' * 64},
                        self.expected, self.fixed,
                    )
                loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
