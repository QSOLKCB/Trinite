from fractions import Fraction as F
from io import BytesIO
import unittest
import random
import wave

from trinite.music_v1.core import Event, Meter, Score, labels
from trinite.music_v1.render import wav_bytes


class RenderTests(unittest.TestCase):
    def frames(self, score, rate=8000):
        with wave.open(BytesIO(wav_bytes(score, rate)), 'rb') as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, rate))
            return wav.getnframes(), wav.readframes(wav.getnframes())

    def test_rest_silence_and_tempo_not_meter(self):
        score = Score(Meter(5, 8, (2, 3)), ((Event(F(5, 8), None),),), tempo=120)
        count, pcm = self.frames(score)
        self.assertEqual(count, 10000)
        self.assertEqual(pcm, bytes(20000))
        slower = Score(score.meter, score.bars, tempo=60)
        self.assertEqual(self.frames(slower)[0], 20000)
        self.assertEqual(labels(score), labels(slower))

    def test_tie_is_continuous_and_render_is_repeatable(self):
        meter = Meter(4, 4, (1, 1, 1, 1))
        whole = Score(meter, ((Event(F(1), 60),),))
        split = Score(meter, ((Event(F(1, 2), 60, tie_out=True), Event(F(1, 2), 60, tie_in=True)),))
        self.assertEqual(wav_bytes(whole), wav_bytes(split))
        self.assertEqual(wav_bytes(split), wav_bytes(split))
        self.assertNotEqual(self.frames(whole)[1], bytes(32000))

    def test_all_meters_groupings_exact_length_and_rng_unchanged(self):
        state = random.getstate()
        for n, d, groups in [(3, 4, (1, 1, 1)), (4, 4, (1, 1, 1, 1)),
                             (4, 4, (2, 2)), (5, 8, (2, 3)), (5, 8, (3, 2)),
                             (7, 8, (2, 2, 3)), (7, 8, (3, 2, 2)), (7, 8, (2, 3, 2)),
                             (8, 16, (2, 2, 2, 2)), (8, 16, (3, 3, 2))]:
            score = Score(Meter(n, d, groups), ((Event(F(n, d), None),),), tempo=90)
            with self.subTest(meter=(n, d), groups=groups):
                self.assertEqual(self.frames(score)[0], (n * 240 * 8000) // (d * 90))
        self.assertEqual(random.getstate(), state)

    def test_tie_across_bar_and_no_input_mutation(self):
        meter = Meter(3, 4, (1, 1, 1))
        score = Score(meter, ((Event(F(3, 4), 60, tie_out=True),),
                             (Event(F(3, 4), 60, tie_in=True),)))
        before = labels(score)
        self.assertEqual(self.frames(score)[0], 24000)
        self.assertEqual(labels(score), before)
        for rate in [True, 7999, 48001, 8000.0]:
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                wav_bytes(score, rate)


if __name__ == '__main__':
    unittest.main()
