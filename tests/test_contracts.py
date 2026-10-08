import json
from pathlib import Path
import unittest

from trinite.contracts import (ContractError, ModelConfig, identity, json_bytes,
                              parse_json)

ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_reference_config_inventory(self):
        c = ModelConfig.load(ROOT/"configs/reference.json")
        self.assertEqual(c, ModelConfig())
        self.assertEqual(c.parameter_count, 1_247_232)
        self.assertEqual(c.content_identity, identity(json_bytes(c.to_dict())))

    def test_types_bounds_and_backend_contract(self):
        for change in ({"width": True}, {"heads": 3}, {"blocks": 0},
                       {"context_length": 257}, {"dtype": "float16"},
                       {"geometry_loss": "1"}, {"linear_bias": 0},
                       {"rms_epsilon": 0.00001}, {"vocabulary_size": True}):
            with self.subTest(change=change), self.assertRaises(ContractError):
                ModelConfig(**change)

    def test_no_unknown_or_missing_fields(self):
        d = ModelConfig().to_dict()
        for bad in ({**d, "device": "cuda"}, {k: v for k, v in d.items() if k != "dtype"}):
            with self.assertRaises(ContractError):
                ModelConfig.from_dict(bad)

    def test_strict_json_input(self):
        for raw in (b'{"a":1,"a":2}', b'\xef\xbb\xbf{}', b'{"x":NaN}',
                    b'{"x":1.0}', b'"\\ud800"', b'{"x":9007199254740992}'):
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                parse_json(raw)

    def test_canonical_artifacts_but_readable_config(self):
        self.assertEqual(parse_json(b'{ "x" : 1 }'), {"x": 1})
        with self.assertRaises(ContractError):
            parse_json(b'{ "x" : 1 }', canonical=True)
        self.assertEqual(json_bytes({"b": 2, "a": 1}), b'{"a":1,"b":2}\n')

    def test_exact_byte_identity(self):
        self.assertEqual(identity(b"abc"),
                         "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertNotEqual(identity(b"abc"), identity(b"abc\n"))


if __name__ == "__main__":
    unittest.main()
