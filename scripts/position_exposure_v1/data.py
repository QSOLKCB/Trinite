"""Unchanged arithmetic admission plus exact address/role schedule accounting."""
from collections import Counter, defaultdict
from trinite.contracts import ContractError, identity, json_bytes
from trinite.tokenizer import ByteTokenizer
from position_coupling_v1.positions import prefix_positions
from position_study_v1 import data as original
from .protocol import ARMS, MODES, POLICIES, OFFSETS, STRATA, PROTOCOL, offset_for

# Reuse the original authoritative generator/admission bytes without relabelling.
dataset = original.dataset
audit_bytes = original.audit_bytes


def examples(selector):
    return original.examples('train' if selector in ARMS else selector)


def schedule(arm):
    if arm not in ARMS: raise ContractError('unknown exposure arm')
    return original.schedule('control')


def role_labels(width):
    digits = [f'operand-significance-{d}' for d in range(width, 0, -1)]
    return ['bos', *digits, 'plus', *digits, 'equals',
            *(f'answer-slot-{i}' for i in range(width + 1)), 'unscored-short-row-eos']


def coarse(role):
    if role.startswith('operand-significance-'): return 'operand'
    if role.startswith('answer-slot-'): return 'answer'
    return role


def pairs_json(pairs):
    return [{'address': p, 'role': role} for p, role in sorted(pairs)]


def exposure(arm):
    rows = schedule(arm); tok = ByteTokenizer(64)
    encoded = [tok.encode(r['text'], answer_start=len(r['prompt'])) for r in rows]
    detailed = Counter(); visits = defaultdict(Counter); hist = Counter()
    inputs = predictors = targets = 0
    cycle_offsets = Counter(offset_for(arm, cycle * len(rows)) for cycle in range(PROTOCOL['cycles']))
    for offset, repetitions in sorted(cycle_offsets.items()):
        for start in range(0, len(rows), PROTOCOL['batch_size']):
            batch = encoded[start:start + PROTOCOL['batch_size']]
            shifted_length = max(len(e.input_ids) for e in batch) - 1
            for r, e in zip(rows[start:start + len(batch)], batch):
                live = min(len(e.input_ids), shifted_length)
                ps = prefix_positions(r['prompt'], live - 1 - len(r['prompt']),
                                      mode=MODES[arm], offset=offset)
                labels = role_labels(r['width'])[:live]
                if len(ps) != len(labels): raise ContractError('exposure role length mismatch')
                detailed.update({pair: count * repetitions for pair, count in Counter(zip(ps, labels)).items()})
                hist.update({p: count * repetitions for p, count in Counter(ps).items()})
                visits[r['example_identity']][offset] += repetitions
                inputs += live * repetitions; predictors += (len(e.input_ids) - 1) * repetitions; targets += sum(e.loss_mask) * repetitions
    trained = set(detailed); coarse_trained = {(p, coarse(role)) for p, role in trained}
    probes = {}
    for stratum in ('train', *STRATA):
        needed = set()
        for r in examples(stratum):
            # A cap of D+2 calls consumes at most D+1 previous emitted tokens.
            # Actual terminal EOS stops generation and is never consumed again.
            for emitted in range(r['width'] + 2):
                ps = prefix_positions(r['prompt'], emitted, mode=MODES[arm], offset=0)
                needed.update(zip(ps, role_labels(r['width'])[:len(ps)]))
        coarse_needed = {(p, coarse(role)) for p, role in needed}
        probes[stratum] = {'required_address_coarse_roles': pairs_json(coarse_needed),
            'missing_address_coarse_roles': pairs_json(coarse_needed - coarse_trained),
            'required_address_detailed_roles': pairs_json(needed),
            'missing_address_detailed_roles': pairs_json(needed - trained),
            'coarse_roles_covered': coarse_needed <= coarse_trained,
            'detailed_roles_covered': needed <= trained}
    row_cycle = Counter(r['example_identity'] for r in rows)
    expected_offsets = OFFSETS if POLICIES[arm] == 'balanced' else (0,)
    repetitions = PROTOCOL['cycles'] // len(expected_offsets)
    balanced = all(dict(visits[key]) == {o: n * repetitions for o in expected_offsets}
                   for key, n in row_cycle.items())
    if not balanced: raise ContractError('per-row offset visits are not exactly balanced')
    return {'schema': 'trinite.position-exposure-accounting.v1', 'arm': arm,
        'cycle_identity': identity(json_bytes([r['example_identity'] for r in rows])),
        'cycle_visits': len(rows), 'cycles': PROTOCOL['cycles'],
        'scheduled_visits': len(rows) * PROTOCOL['cycles'],
        'scheduled_model_input_tokens': inputs, 'per_example_predictor_input_tokens': predictors,
        'live_unscored_eos_input_tokens': inputs - predictors, 'scheduled_target_tokens': targets,
        'training_position_row_lookups': {str(k):v for k,v in sorted(hist.items())},
        'training_address_detailed_role_counts': [{'address':p, 'role':role, 'lookups':n}
            for (p,role),n in sorted(detailed.items())],
        'per_row_offset_visits': {k:{str(o):n for o,n in sorted(v.items())} for k,v in sorted(visits.items())},
        'per_row_offset_balance_verified': balanced, 'evaluation': probes,
        'scope': PROTOCOL['coverage_limit']}


def validate_matching():
    audit_bytes(*dataset())
    result = {arm: exposure(arm) for arm in ARMS}
    for arm, e in result.items():
        for k in ('cycle_visits', 'cycles', 'scheduled_visits', 'scheduled_model_input_tokens', 'scheduled_target_tokens'):
            if e[k] != PROTOCOL[k]: raise ContractError('frozen exposure differs: ' + k)
        if POLICIES[arm] == 'balanced' and not all(v['coarse_roles_covered'] for v in e['evaluation'].values()):
            raise ContractError('balanced address/coarse-role coverage admission failed')
    return result
