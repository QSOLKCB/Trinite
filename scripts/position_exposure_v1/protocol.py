"""Frozen factorial controls; no target outcomes or adaptive selection."""
from copy import deepcopy
from trinite.contracts import ContractError
from position_study_v1.protocol import PROTOCOL as ORIGINAL

ARMS = ('sequence-fixed', 'coupled-fixed', 'sequence-balanced', 'coupled-balanced')
MODES = {arm: arm.split('-')[0] for arm in ARMS}
POLICIES = {arm: arm.split('-')[1] for arm in ARMS}
OFFSETS = tuple(range(8))
SEEDS = (0, 1, 2)
STRATA = ('unseen', 'length3', 'length4')
MILESTONES = (0, 1000, 2000, 3000, 4000)
STEPS = 4000
BATCH_SIZE = 8
CONTRASTS = {
    'coupling-fixed': ('sequence-fixed', 'coupled-fixed'),
    'coupling-balanced': ('sequence-balanced', 'coupled-balanced'),
    'offset-sequence': ('sequence-fixed', 'sequence-balanced'),
    'offset-coupled': ('coupled-fixed', 'coupled-balanced'),
}
PROTOCOL = deepcopy(ORIGINAL)
PROTOCOL.update({
    'schema': 'trinite.position-exposure-protocol.v1',
    'hypothesis': 'balanced embedding-address/coarse-role exposure changes transfer, and coupling effects depend on offset policy, conditional on unchanged arithmetic data and representation',
    'arms': {a: {'mode': MODES[a], 'offset_policy': POLICIES[a]} for a in ARMS},
    'schedule': 'position-exposure-family-cycle-offset.v1',
    'position_configuration': {
        'schema': 'trinite.position-exposure-addressing.v1',
        'slots': 'trinite.arithmetic-position-slots.v1',
        'training_offset_schedule': 'cycle-index-modulo-eight.v1',
        'balanced_offsets': list(OFFSETS), 'evaluation_offset': 0,
        'cap_policy': 'prompt-operand-width-plus-two.v1',
        'adapter': 'trinite.position-exposure-live-prefix.v1'},
    'contrasts': {k:list(v) for k,v in CONTRASTS.items()},
    'criteria_scope': 'the original family margins apply separately to each named directed contrast, treating its second condition as candidate; no pooled or omnibus success claim',
    'matched': ['all arithmetic data, labels and row order', 'effective seeded parameters',
        'optimizer', 'architecture', 'updates and token budgets', 'generation and evaluation offset zero'],
    'changed': ['addressing and deterministic training-offset policy, factorially controlled'],
    'scoring': {**ORIGINAL['scoring'], 'schema': 'trinite.position-exposure-scores.v1'},
    'coverage_gate': 'balanced arms must cover all evaluation embedding-address x coarse token-role pairs; detailed significance and answer-slot gaps are mandatory separate limitations, not passed gates',
    'coverage_limit': 'lookup exposure is not gradient support, novel significance training or arithmetic learning; the range/length/composition confound remains',
    'update_time_enforcement': 'sampled at frozen milestones; native invocations stop at cycle boundaries; per-invocation bound480; outer worker deadline1800',
})


def offset_for(arm, cursor):
    if arm not in ARMS or type(cursor) is not int or cursor < 0:
        raise ContractError('unknown exposure arm or invalid cursor')
    return (cursor // PROTOCOL['cycle_visits']) % len(OFFSETS) if POLICIES[arm] == 'balanced' else 0
