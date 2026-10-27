import copy
import random
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
import unittest

from trinite.music_v1.core import Event, Meter, Score, from_dict, labels, timeline, to_dict


class CoreTests(unittest.TestCase):
    def score(self):
        return Score(Meter(3, 4, (1, 1, 1)), (
            (Event(F(1, 4), 60), Event(F(1, 4), None), Event(F(1, 4), 62, tie_out=True)),
            (Event(F(1, 2), 62, tie_in=True), Event(F(1, 4), 64))))

    def test_exact_labels_and_roundtrip(self):
        score = self.score()
        self.assertEqual(from_dict(to_dict(score)), score)
        self.assertEqual([p for p, _ in timeline(score)], [F(0), F(1, 4), F(1, 2), F(3, 4), F(5, 4)])
        self.assertEqual(labels(score), {
            'attacks': 3, 'rests': 1, 'ties': 1,
            'bar_boundaries': [[0, 1], [3, 4], [3, 2]],
            'group_onsets': [[0, 1], [1, 4], [1, 2], [3, 4], [1, 1], [5, 4]],
            'duration': [3, 2]})
        with self.assertRaises(FrozenInstanceError):
            score.tempo = 60

    def test_all_meters_and_groupings(self):
        for n, d, groups in [(3, 4, (1, 1, 1)), (4, 4, (1, 1, 1, 1)),
                             (5, 8, (2, 3)), (7, 8, (2, 2, 3)), (8, 16, (3, 3, 2))]:
            score = Score(Meter(n, d, groups), ((Event(F(n, d), 60),),))
            self.assertEqual(labels(score)['duration'], [F(n, d).numerator, F(n, d).denominator])
            self.assertEqual(len(labels(score)['group_onsets']), len(groups))
        a = Score(Meter(5, 8, (2, 3)), ((Event(F(5, 8), 60),),))
        b = Score(Meter(5, 8, (3, 2)), a.bars, tempo=60)
        self.assertEqual(labels(a)['duration'], labels(b)['duration'])
        self.assertNotEqual(labels(a)['group_onsets'], labels(b)['group_onsets'])

    def test_invalid_exact_types_and_bounds(self):
        for duration in [0.25, 1, True, F(0), F(-1, 4), F(1, 65), F(2)]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                Event(duration, 60)
        for pitch in [False, -1, 128, 60.0, 'C4']:
            with self.subTest(pitch=pitch), self.assertRaises(ValueError):
                Event(F(1, 4), pitch)
        for meter in [(6, 8, (3, 3)), (True, 4, (1,)), (3, 4, [1, 1, 1]),
                      (3, 4, (1, 1)), (3, 4, (True, 2))]:
            with self.subTest(meter=meter), self.assertRaises(ValueError):
                Meter(*meter)
        for tempo in [True, 0, 301, 120.0]:
            with self.subTest(tempo=tempo), self.assertRaises(ValueError):
                Score(self.score().meter, self.score().bars, tempo)

    def test_bad_bar_and_tie_rejection(self):
        meter = Meter(4, 4, (1, 1, 1, 1))
        cases = [(), ((Event(F(1, 2), 60),),),
                 ((Event(F(1), 60, tie_in=True),),),
                 ((Event(F(1), 60, tie_out=True),),),
                 ((Event(F(1, 2), 60, tie_out=True), Event(F(1, 2), 61, tie_in=True)),),
                 ((Event(F(1, 2), 60, tie_out=True), Event(F(1, 2), 60)),),
                 ((Event(F(1, 2), 60), Event(F(1, 2), 60, tie_in=True)),),
                 ((Event(F(1), 60),),) * 9]
        for bars in cases:
            with self.subTest(bars=bars), self.assertRaises(ValueError):
                Score(meter, bars)
        with self.assertRaises(ValueError):
            Event(F(1, 4), None, tie_out=True)
        with self.assertRaises(ValueError):
            Event(F(1, 4), 60, tie_out=1)

    def test_exact_tuplets_dotted_values_and_rng_unchanged(self):
        state = random.getstate()
        # Nine twelfth-note durations fill 3/4 exactly; no float-grid labels.
        score = Score(Meter(3, 4, (1, 1, 1)), ((Event(F(1, 12), 60),) * 9,))
        self.assertEqual(labels(score)['duration'], [3, 4])
        self.assertEqual(timeline(score)[-1][0], F(2, 3))
        dotted = Score(Meter(3, 4, (1, 1, 1)), ((Event(F(3, 8), 60), Event(F(3, 8), None)),))
        self.assertEqual(from_dict(to_dict(dotted)), dotted)
        self.assertEqual(random.getstate(), state)

    def test_cross_bar_pitch_and_rest_ties_rejected(self):
        meter = Meter(3, 4, (1, 1, 1))
        for second in [Event(F(3, 4), 61, tie_in=True), Event(F(3, 4), None),
                       Event(F(3, 4), 60)]:
            with self.subTest(second=second), self.assertRaises(ValueError):
                Score(meter, ((Event(F(3, 4), 60, tie_out=True),), (second,)))

    def test_strict_reader(self):
        original = to_dict(self.score())
        mutations = []
        for field, value in [('schema', 'other'), ('tempo_qpm', True), ('bars', []), ('extra', 1)]:
            changed = copy.deepcopy(original)
            changed[field] = value
            mutations.append(changed)
        for duration in [[2, 8], [1.0, 4], [True, 4], [0, 1], [1, 0], [1, 65], '1/4']:
            changed = copy.deepcopy(original)
            changed['bars'][0][0]['duration'] = duration
            mutations.append(changed)
        changed = copy.deepcopy(original)
        changed['bars'][0][0]['answer'] = 'leak'
        mutations.append(changed)
        for value in mutations:
            with self.subTest(value=value), self.assertRaises(ValueError):
                from_dict(value)


if __name__ == '__main__':
    unittest.main()
