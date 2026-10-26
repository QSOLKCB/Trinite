"""One authoritative finite admission and exact matched addition schedule.

Generated labels use unit increments; separate bounded addition and grammar checks
verify each row. No model, network, observer, historical mutation or user data.
"""
from collections import Counter
from pathlib import Path
import re

from trinite.contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json
from trinite.tokenizer import ByteTokenizer
from .protocol import ARMS, STRATA, STEPS, BATCH_SIZE

POLICY = 'trinite.addition-development-data.v1'
CONTROL = ((1, 4), (2, 5), (0, 9), (3, 8), (6, 7))
EXPANSION = ((2, 3), (0, 7), (1, 8), (5, 6), (4, 9))
SUMS = (5, 7, 9, 11, 13)
CARRIERS = ('infix', 'fields')
ROOT = Path(__file__).resolve().parents[2]


def source_receipt():
    names = ('scripts/addition_development_v1/__init__.py',
             'scripts/addition_development_v1/data.py',
             'scripts/addition_development_v1/protocol.py',
             'src/trinite/contracts.py', 'src/trinite/tokenizer.py')
    return {name: identity((ROOT/name).read_bytes()) for name in names}


def oracle(formal):
    """Independent exact bounded label rule, never an answer repair."""
    exact_keys(formal, {'task', 'a', 'b'}, 'addition specification')
    if formal['task'] != 'add':
        raise ContractError('addition only')
    a = integer(formal['a'], 'operand', 0, 12)
    b = integer(formal['b'], 'operand', 0, 12)
    return str(a+b)


def parse_prompt(prompt):
    if type(prompt) is not str or len(prompt) > 16 or not prompt.isascii():
        raise ContractError('bounded addition prompt required')
    decimal = r'(0|[1-9][0-9]?)'
    match = re.fullmatch(r'A:' + decimal + ',' + decimal + '=', prompt, re.ASCII)
    if match is None:
        match = re.fullmatch(r'A\(' + decimal + ',' + decimal + r'\):', prompt, re.ASCII)
    if match is None:
        raise ContractError('addition prompt grammar mismatch')
    formal = {'task': 'add', 'a': int(match[1]), 'b': int(match[2])}
    oracle(formal)
    return formal


def _family(a, b):
    return POLICY + ':' + identity(json_bytes(sorted((a, b))))


def _row(a, b, carrier):
    if carrier not in CARRIERS:
        raise ContractError('unknown addition carrier')
    formal = {'task': 'add', 'a': a, 'b': b}
    value = a
    for _ in range(b):
        value += 1
    answer = str(value)
    if oracle(formal) != answer:
        raise ContractError('generated addition label failed independent oracle')
    prompt = f'A:{a},{b}=' if carrier == 'infix' else f'A({a},{b}):'
    if parse_prompt(prompt) != formal:
        raise ContractError('addition carrier changed semantics')
    family = tuple(sorted((a, b)))
    memberships = ['candidate'] if family in CONTROL + EXPANSION else []
    if family in CONTROL:
        memberships.insert(0, 'control')
    stratum = 'train' if memberships else ('range-length' if max(a, b) >= 10 else
              'familiar-answer' if a+b in SUMS else 'novel-answer')
    semantic = identity(json_bytes({'task': 'add', 'a': a, 'b': b}))
    core = {
        'formal': formal, 'answer': answer, 'task': 'add', 'carrier': carrier,
        'prompt': prompt, 'text': prompt+answer,
        'family_id': _family(a, b), 'semantic_identity': semantic,
        'split': 'train' if memberships else 'test', 'stratum': stratum,
        'training_arms': memberships, 'source_ids': [POLICY],
        'verifier_outcome': 'verified',
        'transformation_lineage': {'formal_identity': semantic, 'carrier': carrier},
        'coverage': {'answer_seen_in_both_training_arms': a+b in SUMS,
                     'operand_range_extrapolation': max(a, b) >= 10,
                     'operand_digit_length_extrapolation': max(a, b) >= 10},
    }
    ByteTokenizer(64).encode(core['text'], answer_start=len(prompt.encode('ascii')))
    return {**core, 'example_identity': identity(json_bytes(core))}


def _all_rows():
    return [_row(a, b, carrier) for a in range(13) for b in range(13) for carrier in CARRIERS]


def dataset():
    """Complete 0..12 development universe, with one fixed two-arm assignment."""
    rows = _all_rows()
    raw = json_bytes({'schema': POLICY, 'task': 'add', 'examples': rows})
    manifest = {
        'schema': 'trinite.addition-development-admission.v1',
        'dataset_identity': identity(raw), 'sources': source_receipt(),
        'admission': {
            'source_id': POLICY,
            'origin': 'local exhaustive nonnegative integer addition, operands 0 through 12',
            'author': 'Codex deterministic generator commissioned by Trent Slade / QSOL-IMC',
            'rights_basis': 'newly rendered mathematical facts and minimal symbolic carriers; no imported prose, teacher output or external content',
            'generation': 'unit-increment generator independently checked by bounded integer addition and separate prompt parser',
            'scope': 'this exact finite development corpus and two fixed training assignments only',
            'reviewer': 'automated source, rights, label, grammar and family audit under commissioned protocol',
            'reviewed_on': '2026-10-26', 'outcome': 'admitted',
            'limitations': 'not legal certification, blanket synthetic-data admission or fresh confirmation; public facts can overlap historical examples',
        },
        'counts': {'all_unique': len(rows),
                   **{arm: sum(arm in r['training_arms'] for r in rows) for arm in ARMS},
                   **{stratum: sum(r['stratum'] == stratum for r in rows) for stratum in STRATA}},
        'family_counts': {selector: len({r['family_id'] for r in rows
                          if selector in r['training_arms'] or r['stratum'] == selector})
                          for selector in ARMS + STRATA},
        'range_length_answer_coverage': {
            label: {'rows': sum(r['stratum'] == 'range-length' and
                       r['coverage']['answer_seen_in_both_training_arms'] == seen for r in rows),
                    'families': len({r['family_id'] for r in rows if
                       r['stratum'] == 'range-length' and
                       r['coverage']['answer_seen_in_both_training_arms'] == seen})}
            for label, seen in (('familiar-answer', True), ('novel-answer', False))},
        'exposure_class': 'development only; no fresh confirmation partition exists',
    }
    return raw, json_bytes(manifest)


def audit_bytes(raw, manifest):
    if type(raw) is not bytes or type(manifest) is not bytes or len(raw) > 1024*1024 or len(manifest) > 64*1024:
        raise ContractError('addition admission requires bounded byte artifacts')
    parse_json(raw, canonical=True)
    parse_json(manifest, canonical=True)
    expected = dataset()
    if (raw, manifest) != expected:
        raise ContractError('addition corpus/admission differs from exact reviewed source')
    rows = parse_json(raw, canonical=True)['examples']
    if len({r['example_identity'] for r in rows}) != len(rows) or len({r['prompt'] for r in rows}) != len(rows):
        raise ContractError('duplicate addition rows/prompts')
    train_families = {r['family_id'] for r in rows if r['split'] == 'train'}
    dev_families = {r['family_id'] for r in rows if r['split'] == 'test'}
    if train_families & dev_families:
        raise ContractError('addition training/development family leakage')
    for row in rows:
        if oracle(row['formal']) != row['answer'] or parse_prompt(row['prompt']) != row['formal']:
            raise ContractError('addition label/grammar mismatch')
    return parse_json(manifest, canonical=True)


def examples(selector):
    """Unique rows: training diagnostics never contain repeated schedule visits."""
    if selector not in ARMS + STRATA:
        raise ContractError('unknown addition arm or development stratum')
    raw, manifest = dataset()
    audit_bytes(raw, manifest)
    return [r for r in parse_json(raw, canonical=True)['examples']
            if selector in r['training_arms'] or r['stratum'] == selector]


def schedule(arm):
    if arm not in ARMS:
        raise ContractError('unknown addition training arm')
    unique = examples(arm)
    by_formal = {(r['formal']['a'], r['formal']['b'], r['carrier']): r for r in unique}
    pairs = CONTROL + (CONTROL if arm == 'control' else EXPANSION)
    rows = []
    for a, b in pairs:
        for left, right in ((a, b), (b, a)):
            for carrier in CARRIERS:
                rows.append(by_formal[(left, right, carrier)])
    return rows


def _hist(values):
    return dict(sorted(Counter(str(v) for v in values).items()))


def _cycle_exposure(rows):
    encoded = [ByteTokenizer(64).encode(r['text'], answer_start=len(r['prompt'].encode('ascii')))
               for r in rows]
    return {
        'visits': len(rows),
        'answer_histogram': _hist(r['answer'] for r in rows),
        'carrier_histogram': _hist(r['carrier'] for r in rows),
        'prompt_byte_lengths': _hist(len(r['prompt'].encode('ascii')) for r in rows),
        'model_input_lengths': _hist(len(e.input_ids)-1 for e in encoded),
        'target_lengths': _hist(sum(e.loss_mask) for e in encoded),
        'left_operand_histogram': _hist(r['formal']['a'] for r in rows),
        'right_operand_histogram': _hist(r['formal']['b'] for r in rows),
        'model_input_token_histogram': _hist(t for e in encoded for t in e.input_ids[:-1]),
        'model_input_tokens': sum(len(e.input_ids)-1 for e in encoded),
        'scored_target_tokens': sum(sum(e.loss_mask) for e in encoded),
    }


def exposure(arm):
    rows = schedule(arm)
    cycle = _cycle_exposure(rows)
    visits = STEPS*BATCH_SIZE
    if visits % len(rows):
        raise ContractError('addition budget must end on a complete schedule cycle')
    cycles = visits//len(rows)
    return {
        'schema': 'trinite.addition-development-exposure.v1', 'arm': arm,
        'cycle_example_identities': [r['example_identity'] for r in rows],
        'cycle_identity': identity(json_bytes([r['example_identity'] for r in rows])),
        'cycle': cycle, 'cycles': cycles, 'updates': STEPS, 'batch_size': BATCH_SIZE,
        'scheduled_visits': visits,
        'scheduled_model_input_tokens': cycles*cycle['model_input_tokens'],
        'scheduled_target_tokens': cycles*cycle['scored_target_tokens'],
        'unique_rows': len(examples(arm)),
        'unique_families': len({r['family_id'] for r in examples(arm)}),
    }


def validate_matching():
    left, right = schedule('control'), schedule('candidate')
    if len(left) != 40 or len(right) != 40:
        raise ContractError('addition cycle size differs')
    for a, b in zip(left, right):
        ea = ByteTokenizer(64).encode(a['text'], answer_start=len(a['prompt'].encode()))
        eb = ByteTokenizer(64).encode(b['text'], answer_start=len(b['prompt'].encode()))
        if (a['answer'], a['carrier'], len(a['prompt']), len(ea.input_ids), sum(ea.loss_mask)) != (
                b['answer'], b['carrier'], len(b['prompt']), len(eb.input_ids), sum(eb.loss_mask)):
            raise ContractError('addition paired-slot exposure differs')
    l, r = exposure('control'), exposure('candidate')
    if l['cycle'] != r['cycle']:
        raise ContractError('addition cycle exposure histograms differ')
    return {'schema': 'trinite.addition-development-matching.v1',
            'per_slot_answer_carrier_lengths_equal': True,
            'cycle_exposure_histograms_equal': True,
            'control': l, 'candidate': r,
            'interpretation': 'unique families and repetition differ by design; matched marginals do not prove an internal learning mechanism'}
