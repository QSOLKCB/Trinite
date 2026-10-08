import unittest

from trinite.contracts import ContractError
from trinite.tokenizer import BOS, EOS, PAD, ByteTokenizer


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.t = ByteTokenizer()

    def test_byte_vocabulary_and_unicode_spans(self):
        e = self.t.encode("Aé🙂")
        self.assertEqual(e.input_ids, (BOS, 65, 195, 169, 240, 159, 153, 130, EOS))
        self.assertEqual(e.byte_spans, (None, (0, 1), (1, 2), (2, 3), (3, 4),
                                        (4, 5), (5, 6), (6, 7), None))
        self.assertEqual(self.t.decode(e.input_ids), "Aé🙂")

    def test_exact_context_limit_and_empty(self):
        self.assertEqual(len(self.t.encode("x"*254).input_ids), 256)
        with self.assertRaises(ContractError):
            self.t.encode("x"*255)
        with self.assertRaises(ContractError):
            self.t.encode("é"*128)
        self.assertEqual(self.t.encode("").input_ids, (BOS, EOS))

    def test_answer_targets_and_padding(self):
        e = self.t.encode("é=1", answer_start=3, pad_to=8)
        self.assertEqual(e.input_ids, (BOS, 195, 169, 61, 49, EOS, PAD, PAD))
        self.assertEqual(e.loss_mask, (0, 0, 0, 0, 1, 1, 0, 0))
        self.assertEqual(e.attention_mask, (1, 1, 1, 1, 1, 1, 0, 0))
        self.assertEqual(e.byte_spans[-2:], (None, None))
        self.assertEqual(self.t.decode(e.input_ids), "é=1")

    def test_offsets_and_invalid_unicode(self):
        for args in ({"answer_start": 1}, {"answer_start": True},
                     {"answer_start": 4}, {"pad_to": 2}, {"pad_to": 257}):
            with self.subTest(args=args), self.assertRaises(ContractError):
                self.t.encode("é", **args)
        with self.assertRaises(ContractError):
            self.t.encode("\ud800")

    def test_invalid_utf8_is_retained_with_explicit_display(self):
        self.assertEqual(self.t.decode_bytes((BOS, 255, EOS)), b"\xff")
        with self.assertRaises(UnicodeDecodeError):
            self.t.decode((255,))
        self.assertEqual(self.t.decode((255,), errors="backslashreplace"), "\\xff")
        with self.assertRaises(ContractError):
            self.t.decode((255,), errors="ignore")

    def test_no_silent_special_or_type_acceptance(self):
        for tokens in ((True,), (259,), (-1,), (65, BOS), (EOS, 65),
                       (PAD, 65), (EOS, EOS), (PAD, EOS)):
            with self.subTest(tokens=tokens), self.assertRaises(ContractError):
                self.t.decode_bytes(tokens)

    def test_all_byte_codes_survive_raw_decode(self):
        self.assertEqual(self.t.decode_bytes(range(256)), bytes(range(256)))


if __name__ == "__main__":
    unittest.main()
