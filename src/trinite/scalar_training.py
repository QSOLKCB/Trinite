"""Frozen scalar state/request context; numerical updates remain native."""
from dataclasses import replace
from datetime import datetime, timezone
import itertools
from pathlib import Path

from .contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from .curriculum_data import dataset, audit_bytes, source_receipt as data_sources
from .learning_data import dataset as parent_dataset
from .learning_training import sources as parent_sources
from .scalar_plan import ScalarPlan
from .foundations import read_bytes
from .model import Decoder
from .tokenizer import ByteTokenizer
from .training import TrainingState, create_optimizer, evaluate, model_identity, require_environment

PROTOCOL_IDENTITY = 'sha256:d1f64a577e78d7c38f58f3cf21e7683763869dc7e50e5a7be327ab42e69be670'
PROTOCOL_COMMIT = '06edd13c3011028b18231c144b83ef8d92528c74'
WORKLOADS = ('arithmetic', 'fold', 'lattice')
LANES = ('dense', 'ternary', 'four-state')
SEEDS = (0, 1, 2)
PROJECTIONS = ('count', 'sum', 'squares', 'traversal-value', 'traversal-x', 'traversal-y', 'traversal-z')


def protocol():
    raw = read_bytes(Path(__file__).parent/'scalar-learning-protocol.json', 16384)
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError('scalar protocol changed; freeze a new version')
    return parse_json(raw, canonical=True)


def sources():
    package = Path(__file__).resolve().parent
    return {**parent_sources(), **data_sources(), **{n: identity(read_bytes(package/n)) for n in
        ('scalar_plan.py', 'scalar_training.py', 'scalar_checkpoint.py', 'scalar_learning.py')},
        'scalar-learning-protocol.json': PROTOCOL_IDENTITY,
        'scripts/run_scalar_learning.py': identity(read_bytes(package.parents[1]/'scripts/run_scalar_learning.py'))}


def cells():
    return tuple((w, s, lane) for w, s in itertools.product(WORKLOADS, SEEDS)
                 for lane in LANES[s:]+LANES[:s])


def cell_name(workload, seed, lane):
    if workload not in WORKLOADS or type(seed) is not int or seed not in SEEDS or lane not in LANES:
        raise ContractError('unknown fixed scalar cell')
    return f'{workload}-{seed}-{lane}'


def plan_for(seed):
    cell_name('arithmetic', seed, 'dense')
    return replace(ScalarPlan.from_dict(protocol()['training']), seed=seed)


def training_rows(workload, plan):
    raw, manifest = dataset(workload); audit_bytes(raw, manifest)
    parents = {r['example_identity']: r for r in parse_json(parent_dataset(workload)[0], canonical=True)['examples']}
    tokenizer = ByteTokenizer(plan.model.context_length); ranked = []
    for item in parse_json(raw, canonical=True)['examples']:
        if item['split'] != 'train': continue
        parent = parents[item['transformation_lineage']['parent_example_identity']]
        rank = identity(json_bytes({'seed': plan.seed, 'item': parent['ordering_identity']}))
        projection = PROJECTIONS.index(item['task']) if item['task'] in PROJECTIONS else 0
        e = tokenizer.encode(item['text'], answer_start=len(item['prompt'].encode()))
        ranked.append(((rank, projection), {'identity': item['example_identity'], 'ids': e.input_ids,
                       'mask': e.attention_mask, 'loss': e.loss_mask}))
    if not ranked or len({key for key, _ in ranked}) != len(ranked):
        raise ContractError('scalar exposure must have unique parent/projection ranks')
    rows = [row for _, row in sorted(ranked)]
    targets = sum(sum(rows[(i*plan.batch_size+j)%len(rows)]['loss'])
                  for i in range(plan.steps) for j in range(plan.batch_size))
    if targets > plan.max_target_tokens: raise ContractError('scalar exposure exceeds target budget')
    return raw, manifest, rows


def state_for(workload, seed, lane, plan=None):
    cell_name(workload, seed, lane); require_environment()
    plan = plan_for(seed) if plan is None else plan
    if type(plan) is not ScalarPlan or plan.seed != seed or plan.learning_rate != '0.001':
        raise ContractError('scalar plan/seed mismatch')
    raw, manifest, rows = training_rows(workload, plan)
    model = Decoder(plan.model, lane=lane, seed=seed)
    return TrainingState(plan, lane, model, create_optimizer(model, plan),
        {'train': rows, 'validation': rows}, identity(raw), identity(manifest),
        model_identity(model), evaluate(model, rows, plan.batch_size))


def request_core():
    settings = protocol(); root = Path(__file__).resolve().parents[2]
    prior_raw = {name: read_bytes(root/'fixtures/generalization-v1'/name)
                 for name in settings['prior_generalization']}
    prior = {name: identity(raw) for name, raw in prior_raw.items()}
    if prior != settings['prior_generalization']:
        raise ContractError('scalar curriculum decision evidence mismatch')
    corpora = {w: {'dataset_identity': identity(dataset(w)[0]),
                   'manifest_identity': identity(dataset(w)[1])} for w in WORKLOADS}
    if corpora != parse_json(prior_raw['report.json'], canonical=True)['curriculum_preparation']:
        raise ContractError('scalar corpus/admission differs from frozen preparation')
    return {'schema': 'trinite.scalar-learning-request.v1', 'protocol': settings,
        'protocol_identity': PROTOCOL_IDENTITY, 'protocol_commit': PROTOCOL_COMMIT,
        'source': sources(), 'environment': require_environment(), 'prior_generalization': prior,
        'corpora': corpora}


def freeze_request(path):
    raw = json_bytes({**request_core(), 'created_at': datetime.now(timezone.utc).isoformat()})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream: stream.write(raw)
    return {'request_identity': identity(raw), 'protocol_identity': PROTOCOL_IDENTITY, 'planned_cells': 27}


def read_request(path, expected_identity):
    require_identity(expected_identity); raw = read_bytes(path, 128*1024)
    if identity(raw) != expected_identity: raise ContractError('scalar request identity mismatch')
    record = parse_json(raw, canonical=True); expected = request_core()
    exact_keys(record, set(expected)|{'created_at'}, 'scalar request')
    if any(json_bytes(record[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('scalar request source/data/environment/protocol mismatch')
    try:
        if datetime.fromisoformat(record['created_at']).utcoffset() is None: raise ValueError('timezone required')
    except (TypeError, ValueError) as error: raise ContractError('invalid scalar request clock') from error
    return record, raw
