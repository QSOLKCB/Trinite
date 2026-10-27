"""Optional illustrative PCM rendering, never a source of symbolic labels.

No samples or donor assets. Floating sine synthesis is repeatable in the same
Python/libm environment, not a cross-platform audio-byte conformance promise.
"""
from io import BytesIO
import math
import struct
import wave

from .core import timeline, validate


def wav_bytes(score, sample_rate=8000):
    """Render mono 16-bit PCM with absolute-boundary truncation to samples.

    Tied segments are one continuous oscillator/envelope, including across bars.
    Tempo is quarter notes/minute regardless of meter. This simple audition is
    not a performance model: group accents are symbolic and are not synthesized.
    Bounds: at most 64 seconds, 48 kHz, approximately 6 MiB PCM per call.
    """
    validate(score)
    if type(sample_rate) is not int or not 8000 <= sample_rate <= 48000:
        raise ValueError("sample rate must be an integer in [8000, 48000]")
    events = timeline(score)
    scale = 240 * sample_rate  # whole notes -> samples, divided by QPM

    def frame(position):
        return int(position * scale / score.tempo)

    count = frame(len(score.bars) * score.meter.duration)
    pcm = bytearray(2 * count)
    i = 0
    while i < len(events):
        onset, event = events[i]
        end = onset + event.duration
        while event.tie_out:
            i += 1
            next_onset, event = events[i]
            end = next_onset + event.duration
        if event.pitch is not None:
            first, last = frame(onset), frame(end)
            frequency = 440 * 2 ** ((event.pitch - 69) / 12)
            # A short fade at the ends of the whole sounded chain avoids clicks.
            fade = max(1, min(sample_rate // 200, (last - first) // 2))
            for index in range(first, last):
                age = index - first
                envelope = min(1, age / fade, (last - 1 - index) / fade)
                value = round(10000 * envelope * math.sin(2 * math.pi * frequency * age / sample_rate))
                struct.pack_into("<h", pcm, 2 * index, value)
        i += 1
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()
