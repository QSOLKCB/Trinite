"""Independent pre-deserialization metadata tests using software-only doubles."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_fixed_buffer_review import Tensor, TorchDouble, oracle
from ncm_reference_v1 import core, runtime


class LoaderMetadataReview(unittest.TestCase):
    def setUp(self):
        self.fixed = {
            key: Tensor(value, 'int64' if key == 'lvl_1L' else 'float32')
            for key, value in oracle().items()
        }
        self.expected = {**self.fixed, 'learned.weight': Tensor([0.0, 0.0])}
        self.metadata = {
            'role': 'generator',
            'expected_state': self.expected,
            'fixed_buffers': self.fixed,
        }
        self.pin = {'path': 'sudoku/easy/generator.pt', 'size': 1, 'sha256': '0' * 64}

    def assert_before_io_rejection(self, metadata, errors=ValueError, pin=None):
        with patch.object(runtime, 'read_bytes') as reader:
            with patch.object(TorchDouble, 'load', create=True) as loader:
                with self.assertRaises(errors):
                    runtime.safe_state(
                        TorchDouble, Path('MUST_NOT_BE_OPENED'), self.pin if pin is None else pin,
                        **metadata,
                    )
                reader.assert_not_called()
                loader.assert_not_called()

    def test_all_role_schema_and_fixed_arguments_are_mandatory(self):
        for name in ('role', 'expected_state', 'fixed_buffers'):
            metadata = self.metadata.copy()
            del metadata[name]
            with self.subTest(omitted=name):
                self.assert_before_io_rejection(metadata, TypeError)
        self.assert_before_io_rejection({}, TypeError)

    def test_none_metadata_is_rejected_before_read_or_deserialization(self):
        for name in ('role', 'expected_state', 'fixed_buffers'):
            metadata = self.metadata.copy()
            metadata[name] = None
            with self.subTest(name=name):
                self.assert_before_io_rejection(metadata)

    def test_generator_empty_and_partial_metadata_fail_before_io(self):
        cases = [
            {'expected_state': {}},
            {'fixed_buffers': {}},
            {'fixed_buffers': {'attn_mask_for_masking': self.fixed['attn_mask_for_masking']}},
            {'fixed_buffers': {'lvl_1L': self.fixed['lvl_1L']}},
            {'expected_state': {'learned.weight': Tensor([0.0, 0.0])}},
            {'expected_state': {'lvl_1L': self.fixed['lvl_1L']}},
            {'expected_state': {'attn_mask_for_masking': self.fixed['attn_mask_for_masking']}},
        ]
        for change in cases:
            with self.subTest(keys={k: list(v) for k, v in change.items()}):
                self.assert_before_io_rejection({**self.metadata, **change})

    def test_forged_tokenizer_role_cannot_hide_generator_fixed_keys(self):
        for fixed in ({}, self.fixed):
            self.assert_before_io_rejection(
                {**self.metadata, 'role': 'tokenizer', 'fixed_buffers': fixed})
        for role in ('codec', 'GENERATOR', ''):
            with self.subTest(role=role):
                self.assert_before_io_rejection({**self.metadata, 'role': role})

    def test_checkpoint_pin_role_mismatch_fails_before_deserialization(self):
        tokenizer = {'role': 'tokenizer', 'expected_state': {'codec.weight': Tensor([0.0])},
                     'fixed_buffers': {}}
        self.assert_before_io_rejection(tokenizer)  # The pin is for generator.pt.
        for path in ('sudoku/easy/tokenizer.pt', 'other/generator.pt', None):
            pin = {**self.pin, 'path': path}
            with self.subTest(path=path):
                self.assert_before_io_rejection(self.metadata, pin=pin)

    def test_bad_schema_and_fixed_tensors_fail_before_io(self):
        for mutation in ('schema_value', 'fixed_dtype', 'fixed_value', 'fixed_extra'):
            expected, fixed = copy.deepcopy(self.expected), copy.deepcopy(self.fixed)
            if mutation == 'schema_value':
                expected['learned.weight'] = [0.0, 0.0]
            elif mutation == 'fixed_dtype':
                fixed['lvl_1L'].dtype = 'float32'
            elif mutation == 'fixed_value':
                fixed['attn_mask_for_masking'].value[0][0][0][0] = float('-inf')
            else:
                fixed['learned.weight'] = Tensor([0.0, 0.0])
            with self.subTest(mutation=mutation):
                self.assert_before_io_rejection(
                    {**self.metadata, 'expected_state': expected, 'fixed_buffers': fixed})

    def test_omitted_generator_fixed_mapping_cannot_accept_finite_wrong_mask(self):
        # All-finite masks would pass a generic finite-only validator.
        state = copy.deepcopy(self.expected)
        state['attn_mask_for_masking'].value = [[[[0.0] * 413 for _ in range(413)]]]
        with patch.object(runtime, 'read_bytes') as reader:
            with patch.object(TorchDouble, 'load', return_value={'model': state}, create=True) as loader:
                with self.assertRaises(ValueError):
                    runtime.safe_state(
                        TorchDouble, Path('MUST_NOT_BE_OPENED'), self.pin,
                        role='generator', expected_state=self.expected, fixed_buffers={},
                    )
                reader.assert_not_called()
                loader.assert_not_called()

    def test_finite_wrong_mask_is_rejected_even_with_complete_metadata(self):
        payload = b'CPU TEST DOUBLE; no serialized checkpoint'
        state = copy.deepcopy(self.expected)
        state['attn_mask_for_masking'].value = [[[[0.0] * 413 for _ in range(413)]]]
        loader = Mock(return_value={'model': state})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.bin'
            path.write_bytes(payload)
            pin = {**self.pin, 'size': len(payload), 'sha256': core.digest(payload)}
            with patch.object(TorchDouble, 'load', loader, create=True), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'fixed-buffer value mismatch'):
                    runtime.safe_state(TorchDouble, path, pin, **self.metadata)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.kwargs, {'weights_only': True, 'map_location': 'cpu'})

    def test_tokenizer_accepts_only_its_own_finite_schema_with_explicit_empty_mapping(self):
        payload = b'CPU TEST DOUBLE; no serialized tokenizer'
        expected = {'codec.weight': Tensor([0.0, 0.0])}
        state = {'codec.weight': Tensor([0.5, -0.5])}
        loader = Mock(return_value={'model': state})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.bin'
            path.write_bytes(payload)
            pin = {'path': 'sudoku/easy/tokenizer.pt', 'size': len(payload),
                   'sha256': core.digest(payload)}
            with patch.object(TorchDouble, 'load', loader, create=True), redirect_stdout(io.StringIO()):
                actual = runtime.safe_state(
                    TorchDouble, path, pin, role='tokenizer',
                    expected_state=expected, fixed_buffers={},
                )
        self.assertIs(actual, state)
        loader.assert_called_once()


if __name__ == '__main__':
    unittest.main()
