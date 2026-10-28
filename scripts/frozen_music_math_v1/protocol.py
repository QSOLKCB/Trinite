"""Fixed exploratory arithmetic inputs and non-neural, post-generation scoring.

No numerical backend, model, checkpoint loader or training dependency. Labels
are constructed only by gold_answers()/score_records(), never input_questions().
"""
from collections import Counter
import json
import os
from pathlib import Path
import re

from .artifacts import digest as identity, read


def exact_keys(value, keys, label):
    if type(value) is not dict or set(value) != keys:
        raise ValueError(label + " fields differ")


def integer(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(label + " type/range differs")
    return value

CAP = 16
CONTEXT = 256
EOS = 257
CHECKPOINTS = tuple((subset, seed, step) for subset in ('one', 'four')
                    for seed in (17, 29) for step in (0, 64))
PROTOCOL = {
    'schema': 'trinite.frozen-music-math-protocol.v1',
    'questions': 32, 'checkpoints': 8, 'generation_cap': CAP,
    'context_length': CONTEXT, 'expected_decodes': 256,
    'max_forward_calls': 4096, 'worker_seconds': 600,
    'max_log_bytes': 1048576, 'max_output_bytes': 8388608,
    'format': 'plain ASCII a+b=', 'decoding': 'native raw greedy, EOS or fixed cap',
    'selection': 'actual retained steps 0 and 64; no checkpoint election',
    'training': False, 'arithmetic_bridge': False, 'fresh_confirmation': False,
    'learning_adequate': False, 'phase5_ready': False, 'release_ready': False,
    'interpretation': 'exploratory unseen-to-music-checkpoints; not unseen-to-project',
}

# These operands are fixed before any target result. No outcome-dependent
# selection, holdout election, generation seed search or question replacement.
_STRATA = (
    ('one-digit-no-carry', ((1, 2), (1, 3), (2, 3), (1, 5),
                           (2, 5), (3, 5), (4, 5), (2, 4))),
    ('one-digit-carry', ((1, 9), (2, 9), (3, 9), (4, 9),
                        (5, 9), (6, 9), (7, 9), (8, 9))),
    ('two-digit-no-carry', ((11, 12), (12, 22), (21, 24), (23, 33),
                           (34, 33), (43, 35), (44, 45), (54, 45))),
    ('two-digit-carry', ((17, 14), (28, 14), (26, 27), (37, 27),
                        (48, 27), (49, 37), (59, 49), (68, 57))),
)
CONSTANT_ANSWERS = ('6', '14', '5/4', '7/4')
_DECIMAL = re.compile(r'(?:0|[1-9][0-9]*)\Z', re.ASCII)


def input_questions():
    """Input-only records: no answer, answer length, tokens, or oracle output."""
    result = []
    for stratum, pairs in _STRATA:
        for a, b in pairs:
            prompt = f'{a}+{b}='
            if 1 + len(prompt.encode('ascii')) + CAP > CONTEXT:
                raise ValueError('probe prompt exceeds frozen context budget')
            result.append({'question_id': f'q{len(result):02d}',
                           'stratum': stratum, 'prompt': prompt})
    return result


def validate_questions(questions):
    if questions != input_questions():
        raise ValueError('probe questions differ from fixed input-only inventory')
    return questions


def gold_answers():
    """Independent ordinary integer addition for scoring, not model input."""
    return {f'q{i:02d}': str(a + b)
            for i, (a, b) in enumerate(pair for _, pairs in _STRATA for pair in pairs)}


def checkpoint_name(subset, seed, step):
    if type(subset) is not str or type(seed) is not int or type(step) is not int \
            or (subset, seed, step) not in CHECKPOINTS:
        raise ValueError('unknown probe checkpoint')
    return f'{subset}-{seed}-{step:03d}'


def _generated(tokens):
    if type(tokens) is not list or not 1 <= len(tokens) <= CAP:
        raise ValueError('invalid probe generated token count')
    for token in tokens:
        integer(token, 'generated token', 0, 258)
    if EOS in tokens[:-1] or (tokens[-1] != EOS and len(tokens) != CAP):
        raise ValueError('generation must terminate at first EOS or fixed cap')
    return tokens


def _metrics(rows):
    counts = Counter(tuple(r['generated_tokens']) for r in rows)
    outside = [r for r in rows if r['answer'] not in ('6', '14')]
    return {'questions': len(rows), 'correct': sum(r['correct'] for r in rows),
            'eos_terminated': sum(r['eos_terminated'] for r in rows),
            'cap_terminated': sum(not r['eos_terminated'] for r in rows),
            'malformed': sum(r['malformed'] for r in rows),
            'generated_tokens': sum(len(r['generated_tokens']) for r in rows),
            'unique_raw_outputs': len(counts),
            'output_frequencies': [{'generated_tokens': list(tokens), 'count': count}
                                   for tokens, count in sorted(counts.items())],
            'answers_other_than_6_or_14': {'questions': len(outside),
                'correct': sum(r['correct'] for r in outside)}}


def score_records(records):
    """Validate all 256 raw predictions, then score without any model calls.

    Incomplete journals are retained by the controller but cannot yield a
    completed summary. Exact byte answer plus EOS is the only primary score.
    """
    questions, gold = input_questions(), gold_answers()
    if type(records) is not list or len(records) != 256:
        raise ValueError('completed probe requires all 256 prediction records')
    scored, cells = {}, {}
    offset = 0
    for subset, seed, step in CHECKPOINTS:
        name = checkpoint_name(subset, seed, step)
        rows = []
        for question in questions:
            record = records[offset]
            offset += 1
            exact_keys(record, {'subset', 'seed', 'step', 'question_id',
                                'generated_tokens'}, 'probe raw prediction')
            if (type(record['subset']) is not str or type(record['seed']) is not int
                    or type(record['step']) is not int
                    or (record['subset'], record['seed'], record['step']) != (subset, seed, step)
                    or type(record['question_id']) is not str
                    or record['question_id'] != question['question_id']):
                raise ValueError('probe prediction inventory/order differs')
            tokens = _generated(record['generated_tokens'])
            answer = gold[question['question_id']]
            eos = tokens[-1] == EOS
            text = ''.join(chr(t) for t in tokens[:-1]) if eos and all(t < 128 for t in tokens[:-1]) else ''
            rows.append({**record, 'stratum': question['stratum'], 'answer': answer,
                         'correct': tokens == [*answer.encode('ascii'), EOS],
                         'eos_terminated': eos,
                         'malformed': not eos or _DECIMAL.fullmatch(text) is None})
        scored[name] = rows
        cells[name] = {**_metrics(rows), 'strata': {
            stratum: _metrics([r for r in rows if r['stratum'] == stratum])
            for stratum, _ in _STRATA}}
    paired = {}
    for subset in ('one', 'four'):
        for seed in (17, 29):
            initial = scored[checkpoint_name(subset, seed, 0)]
            final = scored[checkpoint_name(subset, seed, 64)]
            pairs = list(zip(initial, final))
            paired[f'{subset}-{seed}'] = {
                'initial_correct': sum(a['correct'] for a, _ in pairs),
                'final_correct': sum(b['correct'] for _, b in pairs),
                'correct_difference': sum(int(b['correct']) - int(a['correct']) for a, b in pairs),
                'newly_correct': sum(not a['correct'] and b['correct'] for a, b in pairs),
                'lost_correct': sum(a['correct'] and not b['correct'] for a, b in pairs),
                'changed_raw_outputs': sum(a['generated_tokens'] != b['generated_tokens'] for a, b in pairs),
            }
    prevalence = Counter(gold.values())
    maximum = max(prevalence.values())
    return {'schema': 'trinite.frozen-music-math-summary.v1', 'decodes': 256,
            'forward_calls': sum(len(r['generated_tokens']) for r in records),
            'cells': cells, 'paired': paired,
            'answer_frequencies': dict(sorted(prevalence.items())),
            'fixed_answer_baselines': {a: {'correct': prevalence[a], 'questions': 32}
                                       for a in CONSTANT_ANSWERS},
            'max_constant_baseline': {'correct': maximum, 'questions': 32,
                'answers': sorted(a for a, n in prevalence.items() if n == maximum)},
            'fresh_confirmation': False, 'learning_adequate': False,
            'phase5_ready': False, 'release_ready': False}


_HISTORY_SOURCE = 'scripts/balanced_addition_v1/data.py'
_HISTORY_ID = 'sha256:25b860afed659f19aee6be5362d2e4e6c9024745005a8037ba2fa3cff5f9f7da'
_PROMPTS = tuple(re.compile(pattern, re.ASCII) for pattern in (
    r'([0-9]{1,3})\+([0-9]{1,3})=',
    r'A:([0-9]{1,3}),([0-9]{1,3})=',
    r'A\(([0-9]{1,3}),([0-9]{1,3})\):',
))


def historical_overlap(root):
    """Bounded, source-bound overlap audit, explicitly not a novelty certificate.

    Covers readable fixture JSON prompt/add-formal fields plus the pinned
    balanced-addition generator's exhaustive 0..12 operand universe. Archives,
    other carriers and other generated-only corpora are not exhaustively audited.
    """
    root = Path(root)
    source = root / _HISTORY_SOURCE
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 1048576:
        raise ValueError('missing/unsafe historical addition source')
    source_id = identity(read(source, 1048576))
    if source_id != _HISTORY_ID:
        raise ValueError('historical addition source differs from frozen coverage')
    fixture_root = root / 'fixtures'
    if fixture_root.is_symlink() or not fixture_root.is_dir():
        raise ValueError('missing/unsafe fixture overlap scope')
    files = []
    for directory, dirs, names in os.walk(fixture_root, followlinks=False):
        for name in dirs + names:
            if (Path(directory) / name).is_symlink():
                raise ValueError('symlink in historical overlap scope')
        for name in names:
            if name.endswith('.json'):
                files.append(Path(directory) / name)
                if len(files) > 2048:
                    raise ValueError('historical overlap file ceiling exceeded')
    if not files:
        raise ValueError('empty historical fixture overlap scope')
    sources, prompts, families, total = {}, set(), set(), 0
    for path in sorted(files):
        if not path.is_file() or path.stat().st_size > 8388608:
            raise ValueError('unsafe/oversized historical JSON')
        raw = read(path, 8388608)
        total += len(raw)
        if len(raw) > 8388608 or total > 67108864:
            raise ValueError('historical overlap byte ceiling exceeded')
        sources[path.relative_to(root).as_posix()] = identity(raw)
        try:
            stack = [json.loads(raw)]
        except (ValueError, UnicodeError) as error:
            raise ValueError('invalid historical JSON') from error
        while stack:
            value = stack.pop()
            if type(value) is dict:
                prompt = value.get('prompt')
                if type(prompt) is str:
                    prompts.add(prompt)
                    for pattern in _PROMPTS:
                        match = pattern.fullmatch(prompt)
                        if match:
                            families.add(tuple(sorted(map(int, match.groups()))))
                if value.get('task') == 'add' and type(value.get('a')) is int and type(value.get('b')) is int:
                    families.add(tuple(sorted((value['a'], value['b']))))
                stack.extend(value.values())
            elif type(value) is list:
                stack.extend(value)
    rows = []
    for question in input_questions():
        a, b = map(int, question['prompt'][:-1].split('+'))
        rows.append({'question_id': question['question_id'],
            'exact_fixture_prompt_seen': question['prompt'] in prompts,
            'commutative_fixture_family_seen': tuple(sorted((a, b))) in families,
            'known_balanced_addition_family': a <= 12 and b <= 12})
    return {'schema': 'trinite.frozen-music-math-overlap.v1',
            'fixture_sources': sources, 'fixture_json_bytes': total,
            'known_generator_source': {_HISTORY_SOURCE: source_id},
            'known_generator_coverage': 'all ordered addition operands 0..12, two prior carriers',
            'questions': rows, 'excluded_questions': [],
            'global_freshness_established': False,
            'scope': 'fixture .json prompt/add-formal fields and pinned balanced-addition generator; excludes archives, unrecognized carriers, other generated-only corpora',
            'selection': 'fixed questions retained despite known project-history overlap; no outcome-adaptive replacement'}
