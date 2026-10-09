"""Training-only fold order development, safe replay and fresh evidence checks."""
from dataclasses import replace
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path
import resource
import time

from .contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json, require_identity
from .curriculum_data import dataset, audit_bytes
from .fold_order_plan import FoldOrderPlan
from .scalar_training import training_rows, sources as scalar_sources
from .scalar_learning import score, task_failures, closed_stage
from .comparison_checkpoint import tensor_payload, progress, restore_tensors
from .foundations import read_bytes, write, failure
from .model import Decoder
from .training import TrainingState, create_optimizer, evaluate, model_identity, require_environment, run_steps
from .observation import BundleObserver

PROTOCOL_IDENTITY = 'sha256:84012fc8d624e9d0732ecd9420baa04abe6e33d2da6608f202d6d4d2f8873366'
ERRORS = (ContractError, OSError, KeyError, TypeError, ValueError, OverflowError)


def protocol():
    raw = read_bytes(Path(__file__).parent/'fold-order-protocol.json', 16384)
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError('fold order protocol changed; freeze a new version')
    return parse_json(raw, canonical=True)


def sources():
    package = Path(__file__).resolve().parent
    return {**scalar_sources(), **{n: identity(read_bytes(package/n)) for n in
        ('fold_order_plan.py', 'fold_order.py', 'fold-order-protocol.json')},
        'scripts/run_fold_order.py': identity(read_bytes(package.parents[1]/'scripts/run_fold_order.py'))}


def cells():
    return tuple((seed, ordering) for seed in (0, 1, 2)
                 for ordering in (('mixed', 'parent') if seed == 1 else ('parent', 'mixed')))


def cell_name(seed, ordering):
    if type(seed) is not int or type(ordering) is not str or (seed, ordering) not in cells():
        raise ContractError('unknown fixed fold order cell')
    return f'fold-{seed}-{ordering}-dense'


def plan_for(seed, ordering):
    cell_name(seed, ordering)
    return replace(FoldOrderPlan.from_dict(protocol()['training']), seed=seed, ordering=ordering)


def state_for(seed, ordering, plan=None):
    cell_name(seed, ordering); require_environment()
    plan = plan_for(seed, ordering) if plan is None else plan
    if type(plan) is not FoldOrderPlan or (plan.seed, plan.ordering) != (seed, ordering):
        raise ContractError('fold order plan/cell mismatch')
    raw, manifest, original = training_rows('fold', plan)
    if {'dataset_identity': identity(raw), 'manifest_identity': identity(manifest)} != protocol()['corpus']:
        raise ContractError('fold order admission differs from frozen scalar preparation')
    rows = original if ordering == 'parent' else sorted(original, key=lambda row:
        (identity(json_bytes({'seed': seed, 'example_identity': row['identity']})), row['identity']))
    if len(rows) != protocol()['exposure']['unique_training_examples'] or len({r['identity'] for r in rows}) != len(rows):
        raise ContractError('fold order requires each admitted training example exactly once per cycle')
    targets = sum(sum(rows[(i*plan.batch_size+j)%len(rows)]['loss'])
                  for i in range(plan.steps) for j in range(plan.batch_size))
    if targets > plan.max_target_tokens:
        raise ContractError('fold order exceeds scored-target budget')
    model = Decoder(plan.model, lane='dense', seed=seed)
    return TrainingState(plan, 'dense', model, create_optimizer(model, plan),
        {'train': rows, 'validation': rows}, identity(raw), identity(manifest),
        model_identity(model), evaluate(model, rows, plan.batch_size))


def request_core():
    p = protocol(); root = Path(__file__).resolve().parents[2]
    if p['training']['steps'] * p['training']['batch_size'] != p['exposure']['unique_training_examples'] * p['exposure']['complete_passes']:
        raise ContractError('fold order protocol must cover exactly the declared complete passes')
    prior = {n: identity(read_bytes(root/'fixtures/fold-exposure-v1'/n)) for n in p['prior_exposure']}
    if prior != p['prior_exposure']:
        raise ContractError('fold order prior exposure decision mismatch')
    raw, manifest = dataset('fold'); audit_bytes(raw, manifest)
    if {'dataset_identity': identity(raw), 'manifest_identity': identity(manifest)} != p['corpus']:
        raise ContractError('fold order frozen corpus mismatch')
    return {'schema': 'trinite.fold-order-request.v1', 'protocol': p,
        'protocol_identity': PROTOCOL_IDENTITY, 'source': sources(),
        'environment': require_environment(), 'prior_exposure': prior, 'corpus': p['corpus']}


def freeze_request(path):
    raw = json_bytes({**request_core(), 'created_at': datetime.now(timezone.utc).isoformat()})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream: stream.write(raw)
    return {'request_identity': identity(raw), 'protocol_identity': PROTOCOL_IDENTITY, 'planned_cells': len(cells())}


def read_request(path, expected_identity):
    require_identity(expected_identity); raw = read_bytes(path, 128*1024)
    if identity(raw) != expected_identity: raise ContractError('fold order request identity mismatch')
    r = parse_json(raw, canonical=True); expected = request_core()
    exact_keys(r, set(expected)|{'created_at'}, 'fold order request')
    if any(json_bytes(r[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('fold order source/data/environment/protocol mismatch')
    try:
        if datetime.fromisoformat(r['created_at']).utcoffset() is None: raise ValueError('timezone required')
    except (TypeError, ValueError) as error: raise ContractError('invalid request clock') from error
    return r, raw


def context(state, request_identity):
    require_identity(request_identity)
    if (type(state.config) is not FoldOrderPlan or state.lane != 'dense'
            or state.data['validation'] is not state.data['train']):
        raise ContractError('fold order checkpoint requires its admitted training-only state')
    cell_name(state.config.seed, state.config.ordering)
    if {'dataset_identity': state.dataset_identity, 'manifest_identity': state.manifest_identity} != protocol()['corpus']:
        raise ContractError('fold order checkpoint admission mismatch')
    return {'schema': 'trinite.fold-order-checkpoint.v1', 'request_identity': request_identity,
        'protocol_identity': PROTOCOL_IDENTITY, 'source': sources(), 'environment': require_environment(),
        'plan': state.config.to_dict(), 'dataset_identity': state.dataset_identity,
        'manifest_identity': state.manifest_identity, 'initialization_identity': state.initialization_identity,
        'initial_train_loss_hex': state.initial_train_loss, 'data_view': 'train-only.v1',
        'cycle_identity': identity(json_bytes([row['identity'] for row in state.data['train']]))}


def snapshot(state, request_identity):
    payload = tensor_payload(state)
    raw = json_bytes({**context(state, request_identity), 'model_identity': model_identity(state.model),
        'step': state.step, 'cursor': state.cursor, 'target_tokens': state.target_tokens,
        'history': state.history, 'train_diagnostics': state.validation})
    if len(raw) > 8*1024*1024: raise ContractError('fold metadata exceeds budget')
    return payload, raw


def restore(payload, metadata, seed, ordering, request_identity, identities, plan=None):
    if (type(payload) is not bytes or type(metadata) is not bytes or len(payload) > 32*1024*1024
            or len(metadata) > 8*1024*1024 or (identity(payload), identity(metadata)) != identities):
        raise ContractError('fold order checkpoint identity/type/size mismatch')
    r = parse_json(metadata, canonical=True); state = state_for(seed, ordering, plan)
    expected = context(state, request_identity)
    exact_keys(r, set(expected)|{'model_identity', 'step', 'cursor', 'target_tokens', 'history',
                                 'train_diagnostics'}, 'fold order checkpoint')
    if any(json_bytes(r[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('fold order checkpoint context mismatch')
    state.step, state.cursor, state.target_tokens = r['step'], r['cursor'], r['target_tokens']
    state.history, state.validation = r['history'], r['train_diagnostics']
    progress(state)
    return restore_tensors(state, payload, r['model_identity'])


def report_for(state, request_identity, seconds_hex, rss):
    # Always score each unique admitted train item once, independently of its
    # ordering in the numerical exposure schedule. No held-out stage exists.
    scores, predictions = score(state.model, 'fold', 'train')
    visits = Counter(example for step in state.history for example in step['examples'])
    report = {'schema': 'trinite.fold-order-training.v1', 'outcome': 'completed',
        'seed': state.config.seed, 'ordering': state.config.ordering, 'lane': 'dense',
        'request_identity': request_identity, 'protocol_identity': PROTOCOL_IDENTITY,
        'source': sources(), 'model_identity': model_identity(state.model),
        'plan_identity': state.config.content_identity, 'steps': state.step,
        'target_tokens': state.target_tokens, 'initialization_identity': state.initialization_identity,
        'exposure': {'example_visits_identity': identity(json_bytes(dict(visits))),
            'unique_examples_visited': len(visits), 'total_visits': sum(visits.values()),
            'minimum_visits': min(visits.values()), 'maximum_visits': max(visits.values())},
        'schedule_identity': identity(json_bytes([h['examples'] for h in state.history])),
        'final_train_loss_hex': evaluate(state.model, state.data['train'], state.config.batch_size),
        'training_wall_seconds_hex': seconds_hex, 'peak_process_rss_kib': rss,
        'scores': scores, 'training_gate_failures': task_failures(scores, 'train'),
        'held_out_scored': False, 'scope': protocol()['scope']}
    return report, {'predictions': predictions}


def train_cell(root, seed, ordering, request, request_identity):
    _, request_raw = read_request(request, request_identity)
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    try:
        state = state_for(seed, ordering); started = time.perf_counter(); run_steps(state)
        seconds = time.perf_counter()-started
        payload, metadata = snapshot(state, request_identity)
        (root/'tensors.safetensors').write_bytes(payload); (root/'metadata.json').write_bytes(metadata)
        write(root, 'steps.json', {'steps': state.history, 'train_diagnostics': state.validation})
        report, predictions = report_for(state, request_identity, seconds.hex(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(root, 'training.json', report); write(root, 'training-predictions.json', predictions)
        read_request(request, request_identity)
        raw, manifest = dataset('fold'); collector = BundleObserver(root/'train-provenance')
        collector.record('fold-order-train', inputs={'request.json': request_raw, 'dataset.json': raw,
            'dataset-manifest.json': manifest}, outputs={n: read_bytes(root/n) for n in
            ('training.json', 'training-predictions.json', 'steps.json', 'tensors.safetensors', 'metadata.json')})
        receipt = collector.finalize(); write(root, 'train-verification.json', receipt)
        if not receipt['integrity_verified'] or receipt['errors'] or receipt['known_missing_artifacts']:
            raise ContractError('fold order evidence failed upstream verification')
        return report
    except Exception as error:
        failure(root, 'train', error); raise


def verify_cell(root, seed, ordering, request_identity):
    bound, _, _, receipt, manifest_id = closed_stage(root, 'fold', request_identity, 'train')
    report_raw = bound('training.json'); r = parse_json(report_raw, canonical=True)
    payload, metadata = bound('tensors.safetensors'), bound('metadata.json')
    state = restore(payload, metadata, seed, ordering, request_identity, (identity(payload), identity(metadata)))
    if state.step != state.config.steps: raise ContractError('fold order requires final checkpoint')
    wall = float.fromhex(r['training_wall_seconds_hex'])
    if not math.isfinite(wall) or not 0 < wall <= state.config.max_seconds or wall.hex() != r['training_wall_seconds_hex']:
        raise ContractError('invalid fold wall budget')
    integer(r['peak_process_rss_kib'], 'peak RSS', 1, 2**63-1)
    expected, predictions = report_for(state, request_identity, wall.hex(), r['peak_process_rss_kib'])
    if (json_bytes(expected) != report_raw or json_bytes(predictions) != bound('training-predictions.json')
            or bound('steps.json') != json_bytes({'steps': state.history, 'train_diagnostics': state.validation})
            or state.validation[-1]['loss_hex'] != r['final_train_loss_hex']):
        raise ContractError('fold order model/prediction/loss/history mismatch')
    return {'seed': seed, 'ordering': ordering, 'outcome': 'completed', 'report': r,
        'report_identity': identity(report_raw), 'manifest_file_identity': manifest_id, 'verification': receipt}


def validate_errors(errors):
    if (type(errors) is not dict or not set(errors) <= {cell_name(*c) for c in cells()}|{'summary'}
            or any(type(v) is not str or not v.strip() for v in errors.values())):
        raise ContractError('invalid fold order worker errors')


def summarize(root, request, request_identity, errors, *, publish=True):
    read_request(request, request_identity); validate_errors(errors); root = Path(root)
    if read_bytes(root/'request.json') != read_bytes(request):
        raise ContractError('retained fold request mismatch')
    if any((root/cell_name(*c)/n).exists() for c in cells()
           for n in ('evaluation.json', 'test-predictions.json', 'test-provenance')):
        raise ContractError('held-out artifacts are forbidden in training-only fold order')
    records = []
    for seed, ordering in cells():
        name = cell_name(seed, ordering)
        try:
            if name in errors: raise ContractError(errors[name])
            records.append(verify_cell(root/name, seed, ordering, request_identity))
        except ERRORS as error:
            records.append({'seed': seed, 'ordering': ordering, 'outcome': 'failed',
                'error': type(error).__name__+': '+str(error)})
    pair_errors = []
    complete = all(r['outcome'] == 'completed' for r in records) and not errors
    if complete:
        for seed in (0, 1, 2):
            rows = [r['report'] for r in records if r['seed'] == seed]
            if len({r['initialization_identity'] for r in rows}) != 1:
                pair_errors.append(f'initialization mismatch for seed {seed}')
            if len({identity(json_bytes(r['exposure'])) for r in rows}) != 1 or len({r['target_tokens'] for r in rows}) != 1:
                pair_errors.append(f'exposure mismatch for seed {seed}')
    groups = {str(ordering): {'passes_every_training_task_each_seed':
        [not next(r for r in records if (r['seed'], r['ordering']) == (seed, ordering))['report']['training_gate_failures']
         for seed in (0, 1, 2)]} for ordering in ('parent', 'mixed')} if complete and not pair_errors else {}
    result = {'schema': 'trinite.fold-order-summary.v1', 'request_identity': request_identity,
        'protocol_identity': PROTOCOL_IDENTITY, 'outcome': 'completed' if complete and not pair_errors else 'failed',
        'cells': records, 'groups': groups, 'worker_errors': errors, 'pairing_errors': pair_errors,
        'selected_candidate': None, 'held_out_scored': False, 'dense_learning_adequate': False,
        'format_selection_eligible': False, 'phase5_ready': False, 'release_ready': False,
        'historical_evidence_preserved': True, 'scope': protocol()['scope']}
    if publish: write(root, 'summary.json', result)
    return result
