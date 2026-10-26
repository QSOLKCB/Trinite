"""Input-only position slots and strict representation decoding, not arithmetic QA."""
import inspect
import unittest

from trinite.contracts import ContractError
from position_coupling_v1.positions import decode_answer, operand_width, prefix_positions


class PositionSlotsTests(unittest.TestCase):
    def test_literal_coupled_addresses_no_prediction_shift(self):
        self.assertEqual(prefix_positions('12+34=', 0, mode='coupled'),
                         (0, 2, 1, 0, 2, 1, 0))
        self.assertEqual(prefix_positions('12+34=', 1, mode='coupled'),
                         (0, 2, 1, 0, 2, 1, 0, 1))
        self.assertEqual(prefix_positions('12+34=', 4, mode='coupled'),
                         (0, 2, 1, 0, 2, 1, 0, 1, 2, 3, 0))
        self.assertEqual(prefix_positions('12+34=', 4, mode='coupled', offset=7),
                         (0, 9, 8, 0, 9, 8, 0, 8, 9, 10, 0))

    def test_all_widths_offsets_and_prefixes_are_input_only(self):
        self.assertEqual(set(inspect.signature(prefix_positions).parameters),
                         {'prompt', 'emitted_count', 'mode', 'context_length', 'offset'})
        for width in range(1, 5):
            prompts = [f'{a * width}+{b * width}=' for a, b in [('0', '0'), ('1', '2'), ('9', '9')]]
            for mode in ('sequence', 'coupled'):
                for offset in (0, 3, 17):
                    full = prefix_positions(prompts[0], width + 2, mode=mode, offset=offset)
                    for prompt in prompts:
                        self.assertEqual(operand_width(prompt), width)
                        for emitted in range(width + 3):
                            actual = prefix_positions(prompt, emitted, mode=mode, offset=offset)
                            self.assertEqual(actual, full[:1 + len(prompt) + emitted])
                            self.assertEqual(len(actual), 1 + len(prompt) + emitted)
                            if mode == 'sequence':
                                self.assertEqual(actual, tuple(range(offset, offset + len(actual))))

    def test_prompt_grammar_rejects_non_ascii_unequal_or_extra_text(self):
        for prompt in (None, b'1+2=', 12, '', '1+22=', '11111+22222=', '1 +2=',
                       '1+2=\n', '1+2=3', '-1+02=', '+1+02=', '１+２=', '١+٢=', '1-2=', '1+2'):
            with self.subTest(prompt=prompt), self.assertRaises(ContractError):
                operand_width(prompt)
        self.assertEqual(operand_width('0001+0002='), 4)

    def test_slots_reject_invalid_types_bounds_and_embedding_overflow(self):
        invalid = [dict(emitted_count=x) for x in (-1, 4, True, 1.0, None)]
        invalid += [dict(mode=x) for x in ('other', None, 1, True)]
        invalid += [dict(offset=x) for x in (-1, 64, True, 1.0)]
        invalid += [dict(context_length=x) for x in (1, 257, True, 64.0, 4)]
        invalid += [dict(offset=63), dict(mode='sequence', offset=60)]
        for changes in invalid:
            arguments = dict(emitted_count=0, mode='coupled'); arguments.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                prefix_positions('1+2=', **arguments)
        self.assertEqual(prefix_positions('1+2=', 0, mode='sequence', context_length=5), tuple(range(5)))
        with self.assertRaises(ContractError):
            prefix_positions('1+2=', 1, mode='sequence', context_length=5)

    def test_decode_is_representation_only_including_wrong_answers(self):
        for prompt, tokens, expected in [
            ('1+2=', (57, 57, 257), '99'),
            ('1+2=', (48, 48, 257), '0'),
            ('12+34=', (54, 52, 48, 257), '46'),
            ('9999+9999=', (56, 57, 57, 57, 49, 257), '19998'),
        ]:
            self.assertEqual(decode_answer(prompt, tokens), expected)
        self.assertEqual(decode_answer('99+99=', (54, 52, 48, 257)), '46')

    def test_decode_strictly_rejects_malformed_without_repair(self):
        invalid = [None, '30', [51, 48, 257], (), (51, 257), (51, 48),
                   (51, 48, 257, 257), (257, 48, 257), (51, 48, 258),
                   (True, 48, 257), (51.0, 48, 257), (47, 48, 257),
                   (58, 48, 257), (51, 48, 257.0), (51, 32, 257),
                   (51, -1, 257), (51, 256, 257)]
        for tokens in invalid:
            with self.subTest(tokens=tokens):
                self.assertIsNone(decode_answer('1+2=', tokens))
        with self.assertRaises(ContractError):
            decode_answer('1+22=', (51, 48, 257))
