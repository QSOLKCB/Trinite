"""Exhaustive software conformance, never an epistemic capability benchmark."""

import ast
import hashlib
import json
from dataclasses import FrozenInstanceError
from itertools import product
from pathlib import Path
import random
import sys
import unittest

from trinite.contracts import ContractError
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from declared_state import (
    CODEC_ID, MAX_OUTPUT_CHARS, OBSERVATION_ID, REGISTERS,
    decode_state, encode_state, observe_output,
)


class DeclaredStateTests(unittest.TestCase):
    def test_identity_and_literal_vectors(self):
        self.assertEqual(CODEC_ID, "trinite.declared-five-trit.v1")
        self.assertEqual(OBSERVATION_ID, "trinite.declared-output-observation.v1")
        self.assertEqual(REGISTERS, ("K", "C", "R", "A", "S"))
        for state, code in (
            ((-1, -1, -1, -1, -1), "0x00"),
            ((0, 0, 0, 0, 0), "0x79"),
            ((1, 1, 1, 1, 1), "0xF2"),
            ((-1, 1, 1, -1, -1), "0x48"),
            ((-1, -1, 1, 1, 1), "0x1A"),
        ):
            self.assertEqual(encode_state(state), code)
            self.assertEqual(decode_state(code), state)

    def test_all_243_states_against_positional_oracle(self):
        codes = set()
        for state in product((-1, 0, 1), repeat=5):
            value = sum((trit + 1) * weight for trit, weight in
                        zip(state, (81, 27, 9, 3, 1)))
            code = f"0x{value:02X}"
            self.assertEqual(encode_state(state), code)
            self.assertEqual(decode_state(code), state)
            codes.add(code)
        self.assertEqual(len(codes), 243)

    def test_all_256_byte_spellings_including_reserved_states(self):
        states = list(product((-1, 0, 1), repeat=5))
        for value in range(256):
            code = f"0x{value:02X}"
            if value < 243:
                self.assertEqual(decode_state(code), states[value])
            else:
                with self.assertRaises(ContractError):
                    decode_state(code)
                self.assertEqual(observe_output(code + " x").issues, ("invalid_state_code",))

    def test_strict_types_lengths_and_noncanonical_aliases(self):
        class IntSubclass(int):
            pass
        class StrSubclass(str):
            pass
        class TupleSubclass(tuple):
            pass
        for value in (True, False, 0.0, -1.0, None, "1", 2, -2, IntSubclass(0), [], {}):
            with self.subTest(value=value), self.assertRaises(ContractError):
                encode_state((0, 0, value, 0, 0))
        for state in ([], [0] * 5, (0,) * 4, (0,) * 6, None, iter([0] * 5),
                      TupleSubclass((0,) * 5)):
            with self.assertRaises(ContractError):
                encode_state(state)
        for code in ("00", "0X00", "0x0a", "0x000", "0x0", "-1", "0x-1", " 0x00",
                     "0x00\n", "0x00 ", "0x0x00", b"0x00", None, 0, StrSubclass("0x00")):
            with self.subTest(code=code), self.assertRaises(ContractError):
                decode_state(code)

    def test_raw_output_is_preserved_without_repair_or_normalization(self):
        raw = "0x48  2 + 3 = 5.\nCafé?\t"
        before = random.getstate()
        result = observe_output(raw)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(result.raw_output, raw)
        self.assertEqual(result.body, raw[5:])
        self.assertEqual(result.state, (-1, 1, 1, -1, -1))
        self.assertEqual(result.issues, ())
        self.assertEqual(result.schema, OBSERVATION_ID)
        with self.assertRaises(FrozenInstanceError):
            result.body = "changed"
        # A declared ASK with no question stays as-is; no claim of correctness.
        self.assertEqual(observe_output("0x48 false statement").body, "false statement")

    def test_prefix_boundary_and_leading_whitespace_do_not_shift_body(self):
        for raw in ("", " 0x48 body", "\n0x48 body", "0xFF body", "0x4a body"):
            result = observe_output(raw)
            self.assertEqual(result.raw_output, raw)
            self.assertEqual(result.issues, ("invalid_state_code",))
            self.assertIsNone(result.body)
            self.assertIsNone(result.state)
        for raw in ("0x48body", "0x480 body", "0x48\tbody", "0x48\nbody", "0x48"):
            self.assertEqual(observe_output(raw).issues, ("missing_separator",))

    def test_empty_answer_is_a_failure_not_a_successful_minimal_answer(self):
        for raw in ("0x48 ", "0x48 \t\n", "0x48 \u2003"):
            result = observe_output(raw, exact_words=0)
            self.assertEqual(result.issues, ("empty_body",))
            self.assertEqual(result.word_count, 0)

    def test_exact_count_is_not_a_maximum_and_bans_are_explicit(self):
        for body in ("one", "one two three"):
            self.assertIn("exact_word_count_mismatch", observe_output("0x48 " + body, exact_words=2).issues)
        self.assertEqual(observe_output("0x48 one\u2003two", exact_words=2).issues, ())
        self.assertEqual(observe_output("0x48 Café", banned_chars=("é",)).issues,
                         ("banned_character_present",))
        self.assertEqual(observe_output("0x48 E", banned_chars=("e",)).issues, ())
        self.assertEqual(observe_output("0x48 E", banned_chars=("e", "E")).issues,
                         ("banned_character_present",))
        # Text is not normalized to compose e + combining acute.
        self.assertEqual(observe_output("0x48 e\u0301", banned_chars=("é",)).issues, ())

    def test_constraints_validate_even_on_invalid_model_output(self):
        for count in (True, -1, 1.0, MAX_OUTPUT_CHARS + 1, "2"):
            with self.assertRaises(ContractError):
                observe_output("invalid", exact_words=count)
        for chars in (["e"], "e", ("e", "e"), ("",), ("ab",), (None,),
                      tuple(chr(x) for x in range(257))):
            with self.assertRaises(ContractError):
                observe_output("invalid", banned_chars=chars)
        self.assertEqual(observe_output("invalid", banned_chars=tuple(chr(x) for x in range(256))).issues,
                         ("invalid_state_code",))

    def test_real_length_bound_and_no_input_coercion(self):
        self.assertEqual(MAX_OUTPUT_CHARS, 65_536)
        raw = "0x48 " + "x" * (MAX_OUTPUT_CHARS - 5)
        self.assertEqual(observe_output(raw).raw_output, raw)
        for value in (raw + "x", b"0x48 x", None, [], 5):
            with self.assertRaises(ContractError):
                observe_output(value)

    def test_retained_engineering_receipt_binds_exact_sources_and_input_inventory(self):
        fixture = ROOT / "fixtures/declared-state-v1"
        receipt = json.loads((fixture / "receipt.json").read_text())
        self.assertEqual(set(receipt), {"schema", "scope", "base_commit", "sources"})
        self.assertEqual(receipt["schema"], "trinite.declared-state-engineering.v1")
        self.assertEqual(receipt["scope"], "software-conformance-only; no data admission or model evidence")
        self.assertEqual(receipt["base_commit"], "5d7e4ba2388afef0049f192dc965d0a4a5121055")
        paths = ("scripts/declared_state/__init__.py", "tests/test_declared_state.py",
                 "fixtures/declared-state-v1/input-manifest.json")
        self.assertEqual(set(receipt["sources"]), set(paths))
        for path in paths:
            self.assertEqual(receipt["sources"][path],
                             "sha256:" + hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
        inputs = json.loads((fixture / "input-manifest.json").read_text())
        self.assertEqual(set(inputs), {"schema", "admitted_for_training", "files"})
        self.assertEqual(inputs["schema"], "trinite.legacy-declared-inputs.v1")
        self.assertIs(inputs["admitted_for_training"], False)
        self.assertEqual(set(inputs["files"]), {"dataset_generator.py", "eval_harness.py",
                         "gbnf_grammar.txt", "inference_wrapper.py", "project_readme.md",
                         "state_engine.py", "training_script.py"})
        for identity in inputs["files"].values():
            self.assertRegex(identity, r"^sha256:[0-9a-f]{64}$")

    def test_isolated_stdlib_dependency_and_no_core_call_sites(self):
        source = ROOT / "scripts/declared_state/__init__.py"
        imports = []
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                imports.extend((0, name.name) for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append((node.level, node.module))
        self.assertEqual(set(imports), {(0, "dataclasses"), (0, "re"), (0, "trinite.contracts")})
        for path in (ROOT / "src/trinite").rglob("*.py"):
            if "declared_state" not in path.parts:
                self.assertNotIn("declared_state", path.read_text(), str(path))


if __name__ == "__main__":
    unittest.main()
