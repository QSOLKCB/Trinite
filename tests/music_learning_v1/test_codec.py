"""Bounded codec conformance only: no model construction or numerical runs."""
from copy import deepcopy
from fractions import Fraction
import unittest
from unittest.mock import patch

from symbolic_music_v1.core import Event, Meter, Score, to_dict
from symbolic_music_v1.corpus import generate
from symbolic_music_v1.oracle import annotate
from music_learning_v1.codec import (
    TASKS, encode_score, parse_score, parse_prompt, make_example, encode_example,
    answer_text, parse_answer, component_counts,
)


class CodecTests(unittest.TestCase):
    def test_all_rows_lossless_all_tasks_bounded_native_masks(self):
        rows = generate()
        self.assertEqual(len(rows), 160)
        for row in rows:
            raw = row['input']['score']
            self.assertEqual(parse_score(encode_score(raw)), raw)
            for task in TASKS:
                example = make_example(row['input'], task)
                self.assertEqual(parse_prompt(example['prompt']), {'task': task, 'score': raw})
                self.assertEqual(parse_answer(task, example['answer']), row['target'][task])
                encoding = encode_example(example)
                self.assertLessEqual(len(encoding.input_ids), 256)
                self.assertEqual(sum(encoding.loss_mask), len(example['answer']) + 1)
                self.assertFalse(any(encoding.loss_mask[:len(example['prompt']) + 1]))
                self.assertTrue(all(encoding.loss_mask[len(example['prompt']) + 1:]))
                self.assertEqual(encoding.input_ids[-1], 257)

    def test_independent_reader_does_not_use_writer_or_core_parser(self):
        score = to_dict(Score(Meter(3, 4, (1, 1, 1)), (
            (Event(Fraction(3, 8), 127, tie_out=True),
             Event(Fraction(3, 8), 127, tie_in=True)),), 300))
        carrier = encode_score(score)
        with patch('music_learning_v1.codec.encode_score', side_effect=AssertionError), \
             patch('music_learning_v1.codec.from_dict', side_effect=AssertionError):
            self.assertEqual(parse_score(carrier), score)
        self.assertEqual(annotate(parse_score(carrier))['attacks'], 1)

    def test_full_domain_exact_fraction_and_boundaries(self):
        for pitch in (None, 0, 63, 64, 127):
            raw = to_dict(Score(Meter(3, 4, (1, 1, 1)),
                               ((Event(Fraction(1, 12), pitch),) * 9,), 30))
            self.assertEqual(parse_score(encode_score(raw)), raw)
        raw = to_dict(Score(Meter(4, 4, (2, 2)),
                           ((Event(Fraction(1, 64), 60),) * 64,) * 8, 120))
        self.assertEqual(parse_score(encode_score(raw)), raw)
        with self.assertRaisesRegex(ValueError, '256-token'):
            make_example({'task': 'annotate-rhythm-v1', 'score': raw}, 'attacks')

    def test_forbidden_fields_and_target_mutation_rejected(self):
        source = generate()[0]['input']
        for field in ('target', 'family', 'id', 'receipt', 'manifest', 'split', 'answer'):
            bad = deepcopy(source)
            bad[field] = 'forbidden'
            with self.assertRaises(ValueError):
                make_example(bad, 'attacks')
            bad = deepcopy(source)
            bad['score'][field] = 'forbidden'
            with self.assertRaises(ValueError):
                make_example(bad, 'attacks')
        for task in ('unknown', True, None):
            with self.assertRaises(ValueError):
                make_example(source, task)
        original = make_example(source, 'attacks')
        for key, value in [('answer', '999'), ('text', original['text']+'x'),
                           ('answer_start', 0), ('task', 'rests'), ('target', 1)]:
            bad = dict(original)
            bad[key] = value
            with self.assertRaises(ValueError):
                encode_example(bad)

    def test_noncanonical_invalid_truncated_and_unicode(self):
        good = 'M1;3/4;1,1,1;120;0y230'
        self.assertEqual(parse_score(good)['bars'][0][0]['duration'], [3, 4])
        for bad in [good+'x', good[:-1], good.replace('120', '0120'),
                    good.replace('3/4', '6/8'), good.replace('0y230', '0y460'),
                    good.replace('0y230', 'zz230'), good.replace('0y230', '20232'),
                    good.replace('0y230', '0y232'), good+'/', good+';x',
                    good.replace('1,1,1', '1,2'), good.replace('120', 'é')]:
            # 1,2 is a legal alternate grouping, tested separately below.
            if bad == good.replace('1,1,1', '1,2'):
                self.assertNotEqual(parse_score(bad), parse_score(good))
                continue
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_score(bad)
        for bad in [good+'?attacks=1', good+'?unknown=', good+'?rests==', good]:
            with self.assertRaises(ValueError):
                parse_prompt(bad)

    def test_answer_parser_and_components_do_not_repair(self):
        for bad in ('01', '-1', ' 1', '1\n', '513', True):
            with self.assertRaises(ValueError):
                parse_answer('attacks', bad)
        for bad in ('2/4', '0/2', '1/0', '9/1', '0/1,', '1/4x'):
            with self.assertRaises(ValueError):
                parse_answer('bar_boundaries', bad)
        self.assertEqual(component_counts('bar_boundaries', '0/1,1/1', '0/1,3/4,3/2'),
                         {'correct': 1, 'total': 3})
        self.assertEqual(component_counts('bar_boundaries', '0/1,3/4,3/2,2/1', '0/1,3/4,3/2'),
                         {'correct': 3, 'total': 4})
        self.assertEqual(component_counts('duration', '3/2x', '3/2'), {'correct': 0, 'total': 1})
        self.assertEqual(component_counts('rests', '1', '1'), {'correct': 1, 'total': 1})


if __name__ == '__main__':
    unittest.main()
