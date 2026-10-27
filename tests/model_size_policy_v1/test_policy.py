"""Allocation-free ceiling boundaries, including values larger than machine ints."""
import subprocess
import sys
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, ModelConfig
from model_size_policy_v1 import (
    MAX_MODEL_PARAMETERS, build_native_decoder, preflight_parameter_count,
    require_matching_count,
)


class PolicyTests(unittest.TestCase):
    def test_inclusive_decimal_ceiling(self):
        self.assertEqual(MAX_MODEL_PARAMETERS, 8_000_000_000)
        for count in (1, 6_752, 1_247_232, 7_999_999_999, 8_000_000_000):
            self.assertEqual(preflight_parameter_count(count), count)
            self.assertEqual(require_matching_count(count, count), count)

    def test_invalid_counts(self):
        class IntegerSubclass(int):
            pass
        for count in (0, -1, 8_000_000_001, 2**63, 2**1000, True, False,
                      8e9, float('nan'), float('inf'), '8000000000', None,
                      IntegerSubclass(1)):
            with self.subTest(count=repr(count)):
                with self.assertRaises(ContractError):
                    preflight_parameter_count(count)
                for pair in ((count, 1), (1, count)):
                    with self.assertRaises(ContractError):
                        require_matching_count(*pair)

    def test_fail_closed_mismatch(self):
        for pair in ((1, 2), (8_000_000_000, 7_999_999_999)):
            with self.assertRaisesRegex(ContractError, 'disagrees'):
                require_matching_count(*pair)

    def test_reference_count_is_not_trainable_subset(self):
        self.assertEqual(ModelConfig().parameter_count, 1_247_232)

    def test_policy_import_does_not_import_backend(self):
        subprocess.run([sys.executable, '-S', '-c',
                        'import model_size_policy_v1; import sys; '
                        'assert "torch" not in sys.modules; '
                        'assert "trinite.model" not in sys.modules'], check=True)

    def test_native_preflight_rejects_before_backend_import(self):
        config = ModelConfig()
        with patch.object(ModelConfig, 'parameter_count', 8_000_000_001):
            with patch.dict(sys.modules, {'trinite.model': None}):
                with self.assertRaises(ContractError):
                    build_native_decoder(config)

    def test_rejects_invalid_and_mutated_config_before_backend(self):
        bad = ModelConfig()
        object.__setattr__(bad, 'width', True)
        class ConfigSubclass(ModelConfig):
            pass
        with patch.dict(sys.modules, {'trinite.model': None}):
            for config in (None, {}, bad, ConfigSubclass()):
                with self.assertRaises(ContractError):
                    build_native_decoder(config)
