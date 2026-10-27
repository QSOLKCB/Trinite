import copy
import unittest

from symbolic_music_v1.corpus import generate
from symbolic_music_v1.oracle import annotate


def event(a, b, pitch=60, tie_in=False, tie_out=False):
    return dict(duration=[a, b], pitch=pitch, tie_in=tie_in, tie_out=tie_out)


def score(n, d, groups, bars):
    return dict(schema='trinite.music-score.v1', meter=dict(numerator=n, denominator=d, groups=groups),
                tempo_qpm=120, bars=bars)


class OracleTests(unittest.TestCase):
    def test_all_generated_rows(self):
        rows = generate()
        self.assertEqual(len(rows), 160)
        for row in rows:
            with self.subTest(row=row['id']):
                self.assertEqual(annotate(row['input']['score']), row['target'])

    def test_hand_worked_five_eight(self):
        value = score(5, 8, [2, 3], [[event(1, 4), event(1, 8, None), event(1, 4)]])
        self.assertEqual(annotate(value), dict(attacks=2, rests=1, ties=0,
                         bar_boundaries=[[0, 1], [5, 8]], group_onsets=[[0, 1], [1, 4]], duration=[5, 8]))

    def test_hand_worked_seven_eight_cross_bar_tie(self):
        value = score(7, 8, [3, 2, 2], [
            [event(3, 8), event(1, 4, None), event(1, 4, 62, tie_out=True)],
            [event(1, 8, 62, tie_in=True), event(3, 4, 65)]])
        self.assertEqual(annotate(value), dict(attacks=3, rests=1, ties=1,
                         bar_boundaries=[[0, 1], [7, 8], [7, 4]],
                         group_onsets=[[0, 1], [3, 8], [5, 8], [7, 8], [5, 4], [3, 2]], duration=[7, 4]))

    def test_rejects_wrong_bar_ties_and_raw_types(self):
        original = score(5, 8, [2, 3], [[event(1, 4), event(3, 8)]])
        variants = []
        for field, value in [('duration', [1, 8]), ('duration', [2, 8]), ('duration', [True, 4]),
                             ('pitch', False), ('tie_in', True), ('tie_out', True), ('extra', 1)]:
            changed = copy.deepcopy(original)
            changed['bars'][0][-1][field] = value
            variants.append(changed)
        for changed in variants:
            with self.subTest(value=changed), self.assertRaises(ValueError):
                annotate(changed)
        value = score(5, 8, [2, 3], [[event(1, 4, tie_out=True), event(3, 8, 62, tie_in=True)]])
        with self.assertRaises(ValueError):
            annotate(value)
        value['bars'][0][1]['pitch'] = None
        with self.assertRaises(ValueError):
            annotate(value)


if __name__ == '__main__':
    unittest.main()
