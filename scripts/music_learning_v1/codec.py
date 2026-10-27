"""Lossless ASCII music carrier and strict, answer-masked byte adapter.

The event alphabet is base64url, not audio or compressed executable content.
Each five-character event is pitch-high, pitch-low, numerator-minus-one,
denominator-minus-one, tie flags (in + 2*out). Pitch 128 denotes a rest.
Bars are slash-separated; header fields retain signature, groups and QPM.
The independent reader does not call the writer or the core notation parser.
"""
import re

from symbolic_music_v1.core import from_dict, labels
from symbolic_music_v1.oracle import annotate

ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_'
TASKS = ('attacks', 'rests', 'ties', 'bar_boundaries', 'group_onsets', 'duration')
MAX_LENGTH = 256
_DECIMAL = r'(?:0|[1-9][0-9]*)'


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('unexpected music adapter fields')


def _decimal(text):
    if not re.fullmatch(_DECIMAL, text, re.ASCII) or len(text) > 3:
        raise ValueError('noncanonical or oversized music integer')
    return int(text)


def encode_score(value):
    """Encode every notation field; never silently simplify or truncate."""
    score = from_dict(value)
    meter = score.meter
    header = f'M1;{meter.numerator}/{meter.denominator};' + ','.join(map(str, meter.groups))
    bars = []
    for bar in score.bars:
        events = []
        for event in bar:
            pitch = 128 if event.pitch is None else event.pitch
            events.append(ALPHABET[pitch // 64] + ALPHABET[pitch % 64]
                          + ALPHABET[event.duration.numerator - 1]
                          + ALPHABET[event.duration.denominator - 1]
                          + str(int(event.tie_in) + 2 * int(event.tie_out)))
        bars.append(''.join(events))
    return header + f';{score.tempo};' + '/'.join(bars)


def parse_score(text):
    """Independent grammar parser plus integer-tick semantic validation."""
    if type(text) is not str or not text.isascii() or len(text) > 2700:
        raise ValueError('bounded ASCII score required')
    fields = text.split(';')
    if len(fields) != 5 or fields[0] != 'M1':
        raise ValueError('music carrier version or header mismatch')
    signature = fields[1].split('/')
    if len(signature) != 2:
        raise ValueError('invalid signature')
    n, d = map(_decimal, signature)
    groups = [_decimal(v) for v in fields[2].split(',')]
    tempo = _decimal(fields[3])
    bars = []
    for raw in fields[4].split('/'):
        if not raw or len(raw) % 5 or len(raw) > 320:
            raise ValueError('invalid fixed-width music events')
        events = []
        for start in range(0, len(raw), 5):
            item = raw[start:start + 5]
            if any(c not in ALPHABET for c in item[:4]) or item[4] not in '0123':
                raise ValueError('invalid music event alphabet')
            pitch = ALPHABET.index(item[0]) * 64 + ALPHABET.index(item[1])
            if pitch > 128:
                raise ValueError('invalid music pitch code')
            flags = int(item[4])
            events.append({'duration': [ALPHABET.index(item[2]) + 1, ALPHABET.index(item[3]) + 1],
                           'pitch': None if pitch == 128 else pitch,
                           'tie_in': bool(flags & 1), 'tie_out': bool(flags & 2)})
        bars.append(events)
    value = {'schema': 'trinite.music-score.v1',
             'meter': {'numerator': n, 'denominator': d, 'groups': groups},
             'tempo_qpm': tempo, 'bars': bars}
    annotate(value)
    return value


def parse_prompt(prompt):
    if type(prompt) is not str or len(prompt) > MAX_LENGTH or not prompt.endswith('='):
        raise ValueError('invalid bounded music prompt')
    parts = prompt[:-1].split('?')
    if len(parts) != 2 or parts[1] not in TASKS:
        raise ValueError('invalid music task')
    return {'task': parts[1], 'score': parse_score(parts[0])}


def answer_text(task, value):
    if task not in TASKS:
        raise ValueError('unknown music task')
    if task in ('attacks', 'rests', 'ties'):
        return str(value)
    if task == 'duration':
        return f'{value[0]}/{value[1]}'
    return ','.join(f'{a}/{b}' for a, b in value)


def make_example(input_view, task):
    """Only original input-only records are admitted, never audit/target rows."""
    _keys(input_view, ('task', 'score'))
    if input_view['task'] != 'annotate-rhythm-v1' or task not in TASKS:
        raise ValueError('unknown source or output task')
    score = input_view['score']
    carrier = encode_score(score)
    recovered = parse_score(carrier)
    if recovered != score:
        raise ValueError('lossless independent roundtrip failed')
    expected = annotate(recovered)
    if expected != labels(from_dict(score)):
        raise ValueError('independent musical label mismatch')
    prompt = carrier + '?' + task + '='
    answer = answer_text(task, expected[task])
    example = {'task': task, 'prompt': prompt, 'answer': answer,
               'text': prompt + answer, 'answer_start': len(prompt)}
    # BOS and EOS count toward the unchanged native cap. No truncation.
    if len(example['text'].encode('ascii')) + 2 > MAX_LENGTH:
        raise ValueError('music sequence exceeds native 256-token budget')
    return example


def encode_example(example):
    """Revalidate carrier/answer then delegate masking to the native tokenizer."""
    from trinite.tokenizer import ByteTokenizer
    _keys(example, ('task', 'prompt', 'answer', 'text', 'answer_start'))
    parsed = parse_prompt(example['prompt'])
    expected = make_example({'task': 'annotate-rhythm-v1', 'score': parsed['score']}, parsed['task'])
    if example != expected or type(example['answer_start']) is not int:
        raise ValueError('music example changed after adaptation')
    return ByteTokenizer(MAX_LENGTH).encode(example['text'], answer_start=example['answer_start'])


def parse_answer(task, text):
    """Parse canonical answers; malformed generations are failures, never repaired."""
    from math import gcd
    if task not in TASKS or type(text) is not str or not text.isascii() or len(text) > 254:
        raise ValueError('invalid music answer')
    if task in ('attacks', 'rests', 'ties'):
        value = _decimal(text)
        if value > 512:
            raise ValueError('music count exceeds notation bound')
        return value
    pairs = []
    for token in text.split(','):
        parts = token.split('/')
        if len(parts) != 2:
            raise ValueError('invalid answer fraction')
        a, b = map(_decimal, parts)
        if not 1 <= b <= 64 or not 0 <= a <= 8 * b or gcd(a, b) != 1:
            raise ValueError('noncanonical or out-of-range answer fraction')
        pairs.append([a, b])
    if task == 'duration':
        if len(pairs) != 1 or pairs[0][0] == 0:
            raise ValueError('invalid duration answer')
        return pairs[0]
    if not 1 <= len(pairs) <= 129:
        raise ValueError('invalid answer position count')
    return pairs


def component_counts(task, predicted, expected):
    """Exact count/fraction-position accuracy; malformed answers earn zero.

    Missing expected and extra generated positions are incorrect components.
    EOS correctness is intentionally scored separately by the native scorer.
    """
    gold = parse_answer(task, expected)
    gold_parts = gold if task in ('bar_boundaries', 'group_onsets') else [gold]
    try:
        actual = parse_answer(task, predicted)
    except ValueError:
        return {'correct': 0, 'total': len(gold_parts)}
    actual_parts = actual if task in ('bar_boundaries', 'group_onsets') else [actual]
    return {'correct': sum(a == b for a, b in zip(actual_parts, gold_parts)),
            'total': max(len(actual_parts), len(gold_parts))}
