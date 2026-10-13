"""Codec conformance against a separate base-four arithmetic oracle."""

import ast
from itertools import product
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

from trinite.contracts import ContractError
from trinite.packed_ternary import (
    CODEC_ID, MAX_CODE_COUNT, MAX_PAYLOAD_BYTES, pack_codes, unpack_codes,
)

ROOT = Path(__file__).resolve().parents[1]


def oracle_pack(codes):
    """Independent positional arithmetic, without production bit operations."""
    alphabet = {0: 0, 1: 1, -1: 2}
    return bytes(
        sum(alphabet[code] * 4**position
            for position, code in enumerate(codes[start:start + 4]))
        for start in range(0, len(codes), 4)
    )


def oracle_unpack(payload, count):
    """Decode each complete byte using divmod, then validate data and padding."""
    if len(payload) != (count + 3) // 4:
        return None
    pairs = []
    for byte in payload:
        for _ in range(4):
            byte, pair = divmod(byte, 4)
            pairs.append(pair)
    if 3 in pairs[:count] or any(pairs[count:]):
        return None
    alphabet = {0: 0, 1: 1, 2: -1}
    return tuple(alphabet[pair] for pair in pairs[:count])


class CodecConformanceTests(unittest.TestCase):
    def test_exact_contract_identity_and_resource_bounds(self):
        self.assertEqual(CODEC_ID, "trinite.ternary-2bit.v1")
        self.assertEqual(MAX_CODE_COUNT, 1_048_576)
        self.assertEqual(MAX_PAYLOAD_BYTES, 262_144)

    def test_literal_vectors_and_empty_input(self):
        vectors = (
            ((), b""), ((0,), b"\x00"), ((1,), b"\x01"), ((-1,), b"\x02"),
            ((0, 1, -1, 0), b"\x24"), ((-1, -1, -1, -1), b"\xaa"),
            ((1, 1, 1, 1), b"\x55"), ((1, -1, 0, 1, -1), b"\x49\x02"),
        )
        for codes, payload in vectors:
            with self.subTest(codes=codes):
                self.assertEqual(pack_codes(codes), payload)
                self.assertEqual(unpack_codes(payload, len(codes)), codes)
                self.assertIs(type(pack_codes(codes)), bytes)
                self.assertIs(type(unpack_codes(payload, len(codes))), tuple)

    def test_flat_row_major_order_has_no_row_boundary_padding(self):
        rows = ((0, 1, -1), (1, 0, -1))
        codes = tuple(code for row in rows for code in row)
        # The fourth pair is the first code of row two, not row-one padding.
        self.assertEqual(pack_codes(codes), b"\x64\x08")
        self.assertEqual(unpack_codes(b"\x64\x08", 6), codes)

    def test_all_9841_ternary_sequences_through_length_eight(self):
        seen = 0
        for count in range(9):
            for codes in product((-1, 0, 1), repeat=count):
                expected = oracle_pack(codes)
                self.assertEqual(pack_codes(codes), expected)
                self.assertEqual(unpack_codes(expected, count), codes)
                seen += 1
        self.assertEqual(seen, 9841)

    def test_all_single_and_double_byte_encodings_for_every_valid_count(self):
        # Exhaustively check rejection as well as decoding, independently of
        # pack_codes. This includes every reserved pair and padding combination.
        for size, counts in ((1, range(1, 5)), (2, range(5, 9))):
            accepted = {count: 0 for count in counts}
            for values in product(range(256), repeat=size):
                payload = bytes(values)
                for count in counts:
                    expected = oracle_unpack(payload, count)
                    if expected is None:
                        with self.assertRaises(ContractError):
                            unpack_codes(payload, count)
                    else:
                        self.assertEqual(unpack_codes(payload, count), expected)
                        self.assertEqual(pack_codes(expected), payload)
                        accepted[count] += 1
            self.assertEqual(accepted, {count: 3**count for count in counts})

    def test_long_sequences_and_all_count_remainders(self):
        rng = random.Random(731)
        for count in (9, 10, 11, 12, 255, 256, 257, 258, 259, 4093, 4094, 4095, 4096):
            codes = tuple(rng.choice((-1, 0, 1)) for _ in range(count))
            with self.subTest(count=count):
                expected = oracle_pack(codes)
                self.assertEqual(pack_codes(codes), expected)
                self.assertEqual(unpack_codes(expected, count), codes)

    def test_real_maximum_and_just_above_it(self):
        codes = (-1,) * MAX_CODE_COUNT
        payload = b"\xaa" * MAX_PAYLOAD_BYTES
        self.assertEqual(pack_codes(codes), payload)
        self.assertEqual(unpack_codes(payload, MAX_CODE_COUNT), codes)
        with self.assertRaises(ContractError):
            pack_codes(codes + (0,))
        with self.assertRaises(ContractError):
            unpack_codes(payload, MAX_CODE_COUNT + 1)
        with self.assertRaises(ContractError):
            unpack_codes(payload + b"\x00", MAX_CODE_COUNT)

    def test_padding_at_counts_immediately_below_limit(self):
        for padding in (1, 2, 3):
            count = MAX_CODE_COUNT - padding
            payload = b"\x00" * MAX_PAYLOAD_BYTES
            self.assertEqual(unpack_codes(payload, count), (0,) * count)
            corrupt = payload[:-1] + bytes((4 ** (count % 4),))
            with self.assertRaisesRegex(ContractError, "padding"):
                unpack_codes(corrupt, count)

    def test_reserved_pair_at_each_position_of_long_payload(self):
        payload = bytearray(18)
        for position in range(72):
            corrupt = payload.copy()
            corrupt[position // 4] = 3 * 4**(position % 4)
            with self.assertRaisesRegex(ContractError, "reserved"):
                unpack_codes(bytes(corrupt), 72)

    def test_missing_trailing_bytes_and_count_mismatches(self):
        for payload, count in ((b"", 1), (b"\x00", 0), (b"\x00", 5),
                               (b"\x00\x00", 4), (b"\x00\x00", 9),
                               (b"\x00" * 1000, 1)):
            with self.subTest(payload_length=len(payload), count=count):
                with self.assertRaises(ContractError):
                    unpack_codes(payload, count)
        # Padding cannot distinguish actual zero codes from omitted zero codes.
        # Authenticating the external count belongs to a future artifact reader.
        self.assertEqual(unpack_codes(b"\x00", 1), (0,))
        self.assertEqual(unpack_codes(b"\x00", 4), (0, 0, 0, 0))

    def test_rejects_non_builtin_codes_without_coercion(self):
        class IntSubclass(int):
            def __eq__(self, other):
                raise AssertionError("must not compare subclass")

        for value in (True, False, 1.0, -1.0, float("nan"), float("inf"), None,
                      "1", b"1", (), [], {}, object(), IntSubclass(1), 2, -2, 10**100):
            for codes in ((value,), (0, 1, -1, 0, value)):
                with self.assertRaises(ContractError):
                    pack_codes(codes)

    def test_rejects_non_builtin_counts_and_out_of_range_counts(self):
        class IntSubclass(int):
            pass

        for count in (True, False, 0.0, 1.0, "1", None, (), object(), IntSubclass(1),
                      -1, MAX_CODE_COUNT + 1, 10**100):
            with self.assertRaises(ContractError):
                unpack_codes(b"\x00", count)

    def test_rejects_mutable_payloads_and_subclasses(self):
        class BytesSubclass(bytes):
            def __len__(self):
                raise AssertionError("must not inspect subclass")

        for payload in (None, 0, "", [], (), bytearray(b"\x00"), memoryview(b"\x00"),
                        BytesSubclass(b"\x00"), iter(b"\x00")):
            with self.assertRaises(ContractError):
                unpack_codes(payload, 1)

    def test_rejects_containers_without_consuming_user_protocols(self):
        class Hostile:
            def __iter__(self):
                raise AssertionError("must not iterate user input")

            def __len__(self):
                raise AssertionError("must not inspect user input")

        class TupleSubclass(tuple):
            def __len__(self):
                raise AssertionError("must not inspect subclass")

        for codes in (None, "", b"", [], [0], {}, {0}, iter((0,)), Hostile(),
                      TupleSubclass((0,))):
            with self.assertRaises(ContractError):
                pack_codes(codes)

    def test_import_and_execution_without_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-S", "-c",
             "import sys; from trinite.packed_ternary import pack_codes, unpack_codes; "
             "assert pack_codes((0, 1, -1, 1, 0, -1)) == bytes((100, 8)); "
             "assert unpack_codes(bytes((100, 8)), 6) == (0, 1, -1, 1, 0, -1); "
             "assert not {'torch', 'numpy', 'safetensors'} & sys.modules.keys()"],
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_codec_only_imports_shared_contract_helpers(self):
        source = (ROOT / "src/trinite/packed_ternary/__init__.py").read_text()
        imports = [node for node in ast.walk(ast.parse(source))
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(len(imports), 1)
        self.assertIsInstance(imports[0], ast.ImportFrom)
        self.assertEqual((imports[0].level, imports[0].module), (2, "contracts"))
        self.assertEqual({name.name for name in imports[0].names}, {"ContractError", "integer"})


if __name__ == "__main__":
    unittest.main()
