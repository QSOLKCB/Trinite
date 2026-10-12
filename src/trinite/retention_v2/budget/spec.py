"""Explicit v2 experiment/schedule specification over unchanged v1 numerical inputs."""
from collections import Counter
from dataclasses import replace
from pathlib import Path

from ...contracts import ContractError, identity, json_bytes, parse_json
from ...maths_learning import state_for as pooled_state
from ..spec import (sources as assurance_sources, corpora, diagnostic_rows,
                    expected_counts, inputs, drift)
from .plan import BudgetRetentionPlan
from ...maths_plan import MathsPlan
from ..evidence import read_bytes

PROTOCOL_IDENTITY = 'sha256:4813eef6d93d8d229cf31624d61d702c77e915a7034b7d994c12145929f0789d'
PROFILES = ('control', 'candidate')


def protocol():
    raw = read_bytes(Path(__file__).parent/'protocol.json', 16384)
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError('v2 protocol changed; freeze a new version')
    return parse_json(raw, canonical=True)


def sources():
    package = Path(__file__).resolve().parent; root = package.parents[3]
    paths = [*package.glob('*.py'), *package.glob('*.json'), root/'scripts/run_maths_retention_budget.py', root/'scripts/archive_maths_retention_budget.py']
    return {**assurance_sources(), **{str(p.relative_to(root)): identity(read_bytes(p, 1024*1024)) for p in sorted(paths)}}


def cells():
    return tuple((seed, profile) for seed in (0, 1) for profile in (PROFILES if seed == 0 else PROFILES[::-1]))


def cell_name(seed, profile):
    if type(seed) is not int or type(profile) is not str or (seed, profile) not in cells():
        raise ContractError('unknown fixed v2 retention cell')
    return f'maths-retention-budget-v2-{seed}-{profile}-dense'


def profile_settings(run_profile):
    if type(run_profile) is not str or run_profile not in ('research', 'smoke'):
        raise ContractError('unknown fixed v2 run profile')
    return protocol()['run_profiles'][run_profile]


def plan_for(seed, run_profile, profile):
    cell_name(seed, profile)
    settings = profile_settings(run_profile)['training']
    steps = protocol()[run_profile+'_updates'][profile]
    return replace(BudgetRetentionPlan.from_dict(settings), seed=seed, steps=steps)


def state_for(seed, profile, run_profile):
    plan = plan_for(seed, run_profile, profile)
    # Reuse authoritative admission, encoding, native initialization and AdamW.
    # The bounded v1 adapter builds only the unchanged unique pooled rows; it
    # never represents an 8192-update run as a v1 plan.
    common = plan.to_dict()
    common.update(schema='trinite.maths-plan.v1', steps=min(plan.steps,4096),
                  validation_every=min(plan.validation_every,4096),
                  max_seconds=min(plan.max_seconds,480),
                  max_target_tokens=min(plan.max_target_tokens,524288))
    state = pooled_state(seed, 'expanded', MathsPlan.from_dict(common))
    state.config = plan
    legacy_ids = {e['example_identity'] for e in corpora()['legacy'].examples if e['split']=='train'}
    unique = diagnostic_rows(state)
    legacy = [r for r in unique if r['identity'] in legacy_ids]
    maths = [r for r in unique if r['identity'] not in legacy_ids]
    if not legacy or not maths: raise ContractError('budget needs both admitted training streams')
    state.data['train'] = [stream[(step*4+j) % len(stream)]
                          for step in range(plan.steps) for stream in (legacy, maths) for j in range(4)]
    if sum(sum(r['loss']) for r in state.data['train']) > plan.max_target_tokens:
        raise ContractError('budget target schedule exceeds plan')
    return state


def scoring_examples(pack, run_profile):
    profile_settings(run_profile)
    if run_profile == 'research': return tuple(e for e in pack.examples if e['split']=='train')
    first = {}
    for e in pack.examples:
        if e['split'] != 'train': continue
        key = (e['task'], e['split'])
        first[key] = min(first.get(key, e['semantic_identity']), e['semantic_identity'])
    return tuple(e for e in pack.examples if e['split']=='train' and e['semantic_identity'] == first[e['task'], e['split']])

