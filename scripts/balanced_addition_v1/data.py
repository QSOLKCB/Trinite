"""Versioned finite admission and length-matched broad-support addition schedule.

Generated labels use unit increments; separate bounded addition and grammar checks
verify each row. No model, network, observer, historical mutation or user data.
"""
from collections import Counter
from pathlib import Path
import re

from trinite.contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json
from trinite.tokenizer import ByteTokenizer
from .protocol import ARMS, STRATA, STEPS, BATCH_SIZE

POLICY = 'trinite.balanced-addition-data.v1'
CONTROL = ((1, 4), (2, 5), (0, 9), (3, 8), (6, 7))
HELDOUT = ((0, 5), (1, 6), (1, 8), (2, 4), (2, 6), (3, 3),
           (2, 9), (2, 8), (3, 9), (4, 8), (5, 8), (6, 8),
           (6, 9), (7, 7), (7, 9))
CANDIDATE = tuple((a, b) for a in range(10) for b in range(a, 10)
                  if (a, b) not in HELDOUT)
SUMS = (5, 7, 9, 11, 13)
CARRIERS = ('infix', 'fields')
ROOT = Path(__file__).resolve().parents[2]


def source_receipt():
    names = ('scripts/balanced_addition_v1/__init__.py',
             'scripts/balanced_addition_v1/data.py',
             'scripts/balanced_addition_v1/protocol.py',
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
    memberships = ['candidate'] if family in CANDIDATE else []
    if family in CONTROL:
        memberships.insert(0, 'control')
    stratum = 'train' if memberships else ('range-length' if max(a, b) >= 10 else
              'shared-answer' if a+b in SUMS else 'broadened-answer')
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
                     'answer_seen_in_candidate_training': a+b <= 18,
                     'development_not_fresh_confirmation': True,
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
        'schema': 'trinite.balanced-addition-admission.v1',
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
            for label, seen in (('shared-answer', True), ('other-answer', False))},
        'training_support': {arm: support(arm, rows) for arm in ARMS},
        'range_support': {label: {'rows': sum(r['stratum'] == 'range-length' and
            (0 if r['coverage']['answer_seen_in_both_training_arms'] else
             1 if r['coverage']['answer_seen_in_candidate_training'] else 2) == value for r in rows),
            'families': len({r['family_id'] for r in rows if r['stratum'] == 'range-length' and
            (0 if r['coverage']['answer_seen_in_both_training_arms'] else
             1 if r['coverage']['answer_seen_in_candidate_training'] else 2) == value})}
            for value, label in enumerate(('both', 'candidate-only', 'neither'))},
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
    rows = examples(arm)
    by_formal = {(r['formal']['a'], r['formal']['b'], r['carrier']): r for r in rows}
    # Four slots per family including two deliberately repeated diagonal orders.
    # The control fixes the carrier and answer-length slot pattern. Candidate
    # queues consume each family slot exactly once, never choosing by outcomes.
    control_slots = [(a, b, c) for _ in range(8) for x, y in CONTROL
                     for a, b in ((x, y), (y, x)) for c in CARRIERS]
    if arm == 'control':
        return [by_formal[key] for key in control_slots]
    queues = {(length, c): [] for length in (1, 2) for c in CARRIERS}
    for x, y in CANDIDATE:
        for a, b in ((x, y), (y, x)):
            for c in CARRIERS:
                queues[(len(str(a+b)), c)].append(by_formal[(a, b, c)])
    result = [queues[(len(str(a+b)), c)].pop(0) for a, b, c in control_slots]
    if any(queues.values()):
        raise ContractError('balanced schedule has unconsumed family slots')
    return result


def support(arm, all_rows):
    rows = [r for r in all_rows if arm in r['training_arms']]
    return {
        'answer_values': sorted({int(r['answer']) for r in rows}),
        'answer_digits': sorted(set(''.join(r['answer'] for r in rows))),
        'answer_digits_by_position': {
            str(i): sorted({r['answer'][i] for r in rows if len(r['answer']) > i})
            for i in (0, 1)},
        'left_operands': sorted({r['formal']['a'] for r in rows}),
        'right_operands': sorted({r['formal']['b'] for r in rows}),
        'families_per_answer': {str(value): len({r['family_id'] for r in rows
                                if int(r['answer']) == value}) for value in range(25)},
        'unique_rows': len(rows),
    }


def _hist(values):
    return dict(sorted(Counter(str(v) for v in values).items()))


def _cycle_exposure(rows):
    encoded = [ByteTokenizer(64).encode(r['text'], answer_start=len(r['prompt'].encode('ascii')))
               for r in rows]
    return {
        'visits': len(rows),
        'answer_histogram': _hist(r['answer'] for r in rows),
        'answer_digit_histograms_by_position': {str(i): _hist(
            r['answer'][i] for r in rows if len(r['answer']) > i) for i in (0, 1)},
        'family_visit_histogram': _hist(r['family_id'] for r in rows),
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
        'schema': 'trinite.balanced-addition-exposure.v1', 'arm': arm,
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
    if len(left) != 160 or len(right) != 160:
        raise ContractError('balanced addition cycle size differs')
    for a, b in zip(left, right):
        ea = ByteTokenizer(64).encode(a['text'], answer_start=len(a['prompt'].encode()))
        eb = ByteTokenizer(64).encode(b['text'], answer_start=len(b['prompt'].encode()))
        if (a['carrier'], len(a['prompt']), len(ea.input_ids), sum(ea.loss_mask)) != (
                b['carrier'], len(b['prompt']), len(eb.input_ids), sum(eb.loss_mask)):
            raise ContractError('balanced addition paired-slot lengths differ')
    l, r = exposure('control'), exposure('candidate')
    matched = ('visits', 'carrier_histogram', 'prompt_byte_lengths',
               'model_input_lengths', 'target_lengths', 'model_input_tokens', 'scored_target_tokens')
    if any(l['cycle'][k] != r['cycle'][k] for k in matched):
        raise ContractError('balanced addition matched exposure differs')
    raw, _ = dataset()
    rows = parse_json(raw, canonical=True)['examples']
    candidate = support('candidate', rows)
    if (candidate['answer_values'] != list(range(19)) or
        candidate['answer_digits'] != list('0123456789') or
        candidate['answer_digits_by_position'] != {'0': list('0123456789'), '1': list('012345678')} or
        candidate['left_operands'] != list(range(10)) or
        candidate['right_operands'] != list(range(10))):
        raise ContractError('broad output/operand support incomplete')
    if (set(l['cycle']['family_visit_histogram'].values()) != {32} or
        set(r['cycle']['family_visit_histogram'].values()) != {4}):
        raise ContractError('balanced per-family scheduled visits differ')
    return {'schema': 'trinite.balanced-addition-matching.v1',
            'per_slot_carrier_lengths_equal': True,
            'cycle_answer_and_operand_histograms_equal': False,
            'control': l, 'candidate': r,
            'interpretation': 'data-recipe comparison: family counts, repetition, answer and operand marginals differ; not an isolated output-digit causal test'}
