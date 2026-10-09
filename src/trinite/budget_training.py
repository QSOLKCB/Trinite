"""Frozen budget settings, requests, source receipts and train-only state."""
from dataclasses import replace
from datetime import datetime, timezone
import itertools
from pathlib import Path

from .contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from .foundations_training import sources as foundations_sources, state_for as foundations_state
from .budget_plan import BudgetPlan
from .foundations_data import dataset
from .foundations import read_bytes
from .training import require_environment

PROTOCOL_IDENTITY = 'sha256:9e75f2f3642970e7ca56e81b65174d5f2794e07a008379cd0d621363c58d5087'
PROTOCOL_COMMIT = '0510c28d682b4d75824d9de25a17841e17e4fd79'
RUNNER_IDENTITY = 'sha256:4ff888d0803190facdf813e140989b3ab2b8487fd7491dfe5eb293a2f6ab5cfd'
WORKLOADS = ('arithmetic', 'fold', 'lattice')
SEEDS = (0, 1, 2)
CANDIDATES = ('extended',)
MILESTONES = (1024, 2048, 3072, 4096)


def protocol():
    raw = (Path(__file__).parent / 'budget-protocol.json').read_bytes()
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError('budget protocol changed; freeze a new version')
    return parse_json(raw, canonical=True)


def sources():
    package = Path(__file__).resolve().parent
    if identity((package.parent.parent/'scripts/run_budget.py').read_bytes()) != RUNNER_IDENTITY:
        raise ContractError('budget runner source mismatch')
    return {**foundations_sources(), **{n: identity((package/n).read_bytes()) for n in ('budget.py','budget_plan.py','budget_checkpoint.py','budget_training.py','convergence.py')},
            'budget-protocol.json': PROTOCOL_IDENTITY, 'scripts/run_budget.py': RUNNER_IDENTITY}


def cells():
    return tuple(itertools.product(WORKLOADS, SEEDS, CANDIDATES))


def cell_name(workload, seed, candidate):
    if workload not in WORKLOADS or type(seed) is not int or seed not in SEEDS or candidate not in CANDIDATES:
        raise ContractError('unknown budget cell')
    return f'{workload}-{seed}-{candidate}'


def plan_for(seed, candidate):
    cell_name('arithmetic', seed, candidate)
    return replace(BudgetPlan.from_dict(protocol()['training']), seed=seed,
                   learning_rate=protocol()['candidates'][candidate])


def state_for(workload, seed, candidate, plan=None):
    cell_name(workload, seed, candidate)
    config = plan_for(seed, candidate) if plan is None else plan
    if (type(config) is not BudgetPlan or config.seed != seed
            or config.learning_rate != protocol()['candidates'][candidate]):
        raise ContractError('budget plan/seed mismatch')
    state = foundations_state(workload, 'dense', seed, config)
    # The shared loop's diagnostic evaluation uses only admitted training rows.
    state.data = {'train': state.data['train'], 'validation': state.data['train']}
    return state


def request_core():
    return {'schema': 'trinite.budget-request.v1', 'protocol': protocol(),
            'protocol_identity': PROTOCOL_IDENTITY, 'protocol_commit': PROTOCOL_COMMIT,
            'source': sources(), 'environment': require_environment(),
            'corpora': {w: {'dataset_identity': identity(dataset(w)[0]),
                           'manifest_identity': identity(dataset(w)[1])} for w in WORKLOADS}}


def freeze_request(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = json_bytes({**request_core(), 'created_at': datetime.now(timezone.utc).isoformat()})
    with path.open('xb') as stream:
        stream.write(raw)
    return {'request_identity': identity(raw), 'protocol_identity': PROTOCOL_IDENTITY, 'planned_cells': len(cells())}


def read_request(path, expected_identity):
    require_identity(expected_identity)
    raw = read_bytes(path, 128*1024)
    if identity(raw) != expected_identity:
        raise ContractError('budget request identity mismatch')
    record = parse_json(raw, canonical=True); expected = request_core()
    exact_keys(record, set(expected)|{'created_at'}, 'budget request')
    if any(json_bytes(record[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('budget request source/data/environment/protocol mismatch')
    try:
        if datetime.fromisoformat(record['created_at']).utcoffset() is None:
            raise ValueError('timezone required')
    except (ValueError, TypeError) as error:
        raise ContractError('invalid budget clock') from error
    return record, raw


