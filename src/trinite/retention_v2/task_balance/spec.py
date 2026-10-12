"""Frozen matched-update task exposure experiment over the unchanged budget lane."""
from dataclasses import replace
from pathlib import Path

from ...contracts import ContractError, identity, parse_json
from ..budget import spec as budget_spec
from ..budget.plan import BudgetRetentionPlan
from ..spec import corpora, diagnostic_rows, expected_counts, inputs, drift
from ..evidence import read_bytes
from .schedule import MATHS_TASKS, schedule_rows, task_visits

PROTOCOL_IDENTITY = 'sha256:1c391549cad55ef72081669f0df495d7a387e379b8a34922ae72251e19a05be2'
PROFILES = ('control', 'candidate')


def protocol():
    raw = read_bytes(Path(__file__).parent / 'protocol.json', 16384)
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError('task-balance protocol changed; freeze a new version')
    value = parse_json(raw, canonical=True)
    if tuple(value['maths_task_order']) != MATHS_TASKS:
        raise ContractError('task order differs from frozen schedule')
    return value


def sources():
    package = Path(__file__).resolve().parent
    root = package.parents[3]
    paths = [*package.glob('*.py'), *package.glob('*.json'),
             root / 'scripts/run_maths_task_balance.py', root / 'scripts/archive_maths_task_balance.py']
    return {**budget_spec.sources(),
            **{str(p.relative_to(root)): identity(read_bytes(p, 1024 * 1024)) for p in sorted(paths)}}


def cells():
    return tuple((seed, profile) for seed in (0, 1)
                 for profile in (PROFILES if seed == 0 else PROFILES[::-1]))


def cell_name(seed, profile):
    if type(seed) is not int or type(profile) is not str or (seed, profile) not in cells():
        raise ContractError('unknown fixed task-balance cell')
    return f'maths-task-balance-v1-{seed}-{profile}-dense'


def profile_settings(run_profile):
    if type(run_profile) is not str or run_profile not in ('research', 'smoke'):
        raise ContractError('unknown fixed task-balance run profile')
    return protocol()['run_profiles'][run_profile]


def plan_for(seed, run_profile, profile):
    cell_name(seed, profile)
    settings = profile_settings(run_profile)['training']
    return replace(BudgetRetentionPlan.from_dict(settings), seed=seed,
                   steps=protocol()[run_profile + '_updates'][profile])


def state_for(seed, profile, run_profile):
    plan = plan_for(seed, run_profile, profile)
    # Reuse admission, tokenization, unique diagnostics, native initialization,
    # AdamW, plan bounds and the complete prior 8192-update control schedule.
    state = budget_spec.state_for(seed, 'candidate', run_profile)
    if state.config != plan:
        raise ContractError('task-balance plan differs from declared unchanged budget lane')
    packs = corpora()
    mapping = {e['example_identity']: (kind, e['task']) for kind, pack in packs.items()
               for e in pack.examples if e['split'] == 'train'}
    unique = diagnostic_rows(state)
    if {row['identity'] for row in unique} != set(mapping):
        raise ContractError('task-balance diagnostic view is not the complete training corpus')
    legacy = [row for row in unique if mapping[row['identity']][0] == 'legacy']
    maths = [row for row in unique if mapping[row['identity']][0] == 'maths']
    task_for = {row['identity']: mapping[row['identity']][1] for row in maths}
    schedule = schedule_rows(legacy, maths, task_for, plan.steps, profile)
    if profile == 'control' and schedule != state.data['train']:
        raise ContractError('task-balance control differs from prior budget schedule')
    if sum(sum(row['loss']) for row in schedule) > plan.max_target_tokens:
        raise ContractError('task-balance target schedule exceeds plan')
    state.data['train'] = schedule
    return state


def scoring_examples(pack, run_profile):
    profile_settings(run_profile)
    return budget_spec.scoring_examples(pack, run_profile)
