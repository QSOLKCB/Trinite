"""Frozen matched state/request contracts; one native initializer and update loop."""
from dataclasses import replace
from datetime import datetime, timezone
import itertools
from pathlib import Path

from .contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from .foundations_training import sources as parent_sources, state_for as parent_state, protocol as parent_protocol
from .foundations import read_bytes
from .learning_data import dataset, audit_bytes, source_receipt as data_sources
from .learning_plan import LearningPlan
from .training import require_environment

PROTOCOL_IDENTITY = 'sha256:b5b08b3d51824d8e685c1727434ad219fae05c3da3e0c71032c884357e574ad3'
PROTOCOL_COMMIT = 'c26d8535f97f70683c58200b58c3325eb5247d05'
RUNNER_IDENTITY = 'sha256:21ae8eda1abb2e0833019cd79394db7654ae01ea8f5b57761ce6bf93376ba094'
WORKLOADS = ('arithmetic', 'fold', 'lattice')
LANES = ('dense', 'ternary', 'four-state')
SEEDS = (0, 1, 2)


def protocol():
    raw = (Path(__file__).parent/'learning-protocol.json').read_bytes()
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError('learning protocol changed; freeze a new version')
    record = parse_json(raw, canonical=True); parent = parent_protocol()
    if record['generation_caps'] != parent['generation_caps'] or record['gates'] != parent['gates']:
        raise ContractError('shared scorer/gate contract mismatch')
    return record


def sources():
    package = Path(__file__).resolve().parent
    if identity((package.parent.parent/'scripts/run_learning.py').read_bytes()) != RUNNER_IDENTITY:
        raise ContractError('learning runner source mismatch')
    return {**parent_sources(), **data_sources(), **{n: identity((package/n).read_bytes()) for n in
        ('learning.py', 'learning_plan.py', 'learning_checkpoint.py', 'learning_training.py', 'budget_plan.py')},
        'learning-protocol.json': PROTOCOL_IDENTITY, 'scripts/run_learning.py': RUNNER_IDENTITY}


def cells():
    return tuple((w, s, lane) for w, s in itertools.product(WORKLOADS, SEEDS)
                 for lane in LANES[s:]+LANES[:s])


def cell_name(workload, seed, lane):
    if workload not in WORKLOADS or type(seed) is not int or seed not in SEEDS or lane not in LANES:
        raise ContractError('unknown fixed learning cell')
    return f'{workload}-{seed}-{lane}'


def plan_for(seed):
    cell_name('arithmetic', seed, 'dense')
    return replace(LearningPlan.from_dict(protocol()['training']), seed=seed)


def state_for(workload, seed, lane, plan=None):
    cell_name(workload, seed, lane)
    config = plan_for(seed) if plan is None else plan
    if type(config) is not LearningPlan or config.seed != seed or config.learning_rate != '0.001':
        raise ContractError('learning plan/seed mismatch')
    raw, manifest = dataset(workload); audit_bytes(raw, manifest)
    rows = parse_json(raw, canonical=True)['examples']
    mapping = {r['ordering_identity']: r['example_identity'] for r in rows}
    if len(mapping) != len(rows):
        raise ContractError('learning parent ordering map is not bijective')
    # Preserve the authoritative initializer, tokenizer, optimizer and exact
    # parent ordering. Relabel admitted item identities before any update.
    state = parent_state(workload, lane, seed, config)
    for item in state.data['train']:
        item['identity'] = mapping[item['identity']]
    state.data = {'train': state.data['train'], 'validation': state.data['train']}
    state.dataset_identity, state.manifest_identity = identity(raw), identity(manifest)
    return state


def request_core():
    settings = protocol(); root = Path(__file__).resolve().parents[2]
    prior = {n: identity(read_bytes(root/'fixtures/budget-v1'/n)) for n in ('summary.json', 'request.json')}
    if (prior['summary.json'] != settings['prior_budget_summary_identity']
            or prior['request.json'] != settings['prior_budget_request_identity']):
        raise ContractError('learning budget decision evidence mismatch')
    return {'schema': 'trinite.learning-request.v1', 'protocol': settings,
        'protocol_identity': PROTOCOL_IDENTITY, 'protocol_commit': PROTOCOL_COMMIT,
        'source': sources(), 'environment': require_environment(), 'prior_budget': prior,
        'corpora': {w: {'dataset_identity': identity(dataset(w)[0]),
                        'manifest_identity': identity(dataset(w)[1])} for w in WORKLOADS}}


def freeze_request(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = json_bytes({**request_core(), 'created_at': datetime.now(timezone.utc).isoformat()})
    with path.open('xb') as stream:
        stream.write(raw)
    return {'request_identity': identity(raw), 'protocol_identity': PROTOCOL_IDENTITY, 'planned_cells': 27}


def read_request(path, expected_identity):
    require_identity(expected_identity); raw = read_bytes(path, 128*1024)
    if identity(raw) != expected_identity:
        raise ContractError('learning request identity mismatch')
    record = parse_json(raw, canonical=True); expected = request_core()
    exact_keys(record, set(expected)|{'created_at'}, 'learning request')
    if any(json_bytes(record[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('learning request source/data/environment/protocol mismatch')
    try:
        if datetime.fromisoformat(record['created_at']).utcoffset() is None:
            raise ValueError('timezone required')
    except (ValueError, TypeError) as error:
        raise ContractError('invalid learning clock') from error
    return record, raw
