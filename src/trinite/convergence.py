"""Training-only development orchestration; native numerical/checkpoint reuse."""
from dataclasses import replace
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import time

from .contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json, require_identity
from .foundations_training import sources as foundations_sources, state_for as foundations_state
from .foundations_plan import FoundationsPlan
from .foundations_data import dataset
from .foundations_checkpoint import snapshot, restore
from .foundations import score, check_predictions, task_failures, read_bytes, write, failure
from .training import require_environment, run_steps, model_identity, evaluate
from .observation import BundleObserver, verify_observation

PROTOCOL_IDENTITY = 'sha256:1c55d1834ad712f3a4ba14c00a486c221d540d093ed959fd1cbfc29ad21c1faf'
PROTOCOL_COMMIT = '696b251f13d2e5dc82478f7b02d7e346605f5264'
RUNNER_IDENTITY = 'sha256:bc964184ee04e95ae78ebbdf6ed2d547bc952f76b6576d80cabed0b0e599df45'
WORKLOADS = ('arithmetic', 'fold', 'lattice')
SEEDS = (0, 1, 2)
CANDIDATES = ('low', 'reference', 'high')
MILESTONES = (256, 512, 768, 1024)


def protocol():
    raw = (Path(__file__).parent / 'convergence-protocol.json').read_bytes()
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError('convergence protocol changed; freeze a new version')
    return parse_json(raw, canonical=True)


def sources():
    package = Path(__file__).resolve().parent
    if identity((package.parent.parent/'scripts/run_convergence.py').read_bytes()) != RUNNER_IDENTITY:
        raise ContractError('convergence runner source mismatch')
    return {**foundations_sources(), 'convergence.py': identity((package/'convergence.py').read_bytes()),
            'convergence-protocol.json': PROTOCOL_IDENTITY, 'scripts/run_convergence.py': RUNNER_IDENTITY}


def cells():
    return tuple(itertools.product(WORKLOADS, SEEDS, CANDIDATES))


def cell_name(workload, seed, candidate):
    if workload not in WORKLOADS or type(seed) is not int or seed not in SEEDS or candidate not in CANDIDATES:
        raise ContractError('unknown convergence cell')
    return f'{workload}-{seed}-{candidate}'


def plan_for(seed, candidate):
    cell_name('arithmetic', seed, candidate)
    return replace(FoundationsPlan.from_dict(protocol()['training']), seed=seed,
                   learning_rate=protocol()['candidates'][candidate])


def state_for(workload, seed, candidate, plan=None):
    cell_name(workload, seed, candidate)
    config = plan or plan_for(seed, candidate)
    if (not isinstance(config, FoundationsPlan) or config.seed != seed
            or config.learning_rate != protocol()['candidates'][candidate]):
        raise ContractError('convergence plan/seed mismatch')
    state = foundations_state(workload, 'dense', seed, config)
    # The shared loop's diagnostic evaluation uses only admitted training rows.
    state.data = {'train': state.data['train'], 'validation': state.data['train']}
    return state


def restore_state(payload, metadata, workload, seed, candidate, request_identity, identities, plan=None):
    config = plan or plan_for(seed, candidate)
    cell_name(workload, seed, candidate)
    if (not isinstance(config, FoundationsPlan) or config.seed != seed
            or config.learning_rate != protocol()['candidates'][candidate]):
        raise ContractError('convergence restore plan/candidate mismatch')
    if (type(payload) is not bytes or type(metadata) is not bytes or len(payload) > 32*1024*1024
            or len(metadata) > 2*1024*1024
            or (identity(payload), identity(metadata)) != identities):
        raise ContractError('convergence checkpoint identity/type/size mismatch')
    envelope = parse_json(metadata, canonical=True)
    expected = checkpoint_context(workload, seed, candidate)
    exact_keys(envelope, set(expected)|{'inner_metadata'}, 'convergence checkpoint')
    if any(json_bytes(envelope[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('convergence checkpoint context mismatch')
    inner = json_bytes(envelope['inner_metadata'])
    state = restore(payload, inner, workload, 'dense', seed, request_identity,
                    (identity(payload), identity(inner)), config)
    state.data = {'train': state.data['train'], 'validation': state.data['train']}
    return state


def checkpoint_context(workload, seed, candidate):
    cell_name(workload, seed, candidate)
    return {'schema': 'trinite.convergence-checkpoint.v1', 'protocol_identity': PROTOCOL_IDENTITY,
            'source': sources(), 'data_view': 'train-only.v1', 'workload': workload,
            'seed': seed, 'candidate': candidate}


def snapshot_state(state, workload, candidate, request_identity):
    if not isinstance(state.config, FoundationsPlan):
        raise ContractError('convergence checkpoint requires its separate bounded plan')
    context = checkpoint_context(workload, state.config.seed, candidate)
    if (state.lane != 'dense'
            or state.config.learning_rate != protocol()['candidates'][candidate]
            or state.data['validation'] is not state.data['train']):
        raise ContractError('convergence checkpoint requires its elected train-only state')
    payload, inner = snapshot(state, workload, request_identity)
    metadata = json_bytes({**context, 'inner_metadata': parse_json(inner, canonical=True)})
    if len(metadata) > 2*1024*1024:
        raise ContractError('convergence metadata exceeds budget')
    return payload, metadata


def request_core():
    return {'schema': 'trinite.convergence-request.v1', 'protocol': protocol(),
            'protocol_identity': PROTOCOL_IDENTITY, 'protocol_commit': PROTOCOL_COMMIT,
            'source': sources(), 'environment': require_environment(),
            'corpora': {w: {'dataset_identity': identity(dataset(w)[0]),
                           'manifest_identity': identity(dataset(w)[1])} for w in WORKLOADS}}


def freeze_request(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = json_bytes({**request_core(), 'created_at': datetime.now(timezone.utc).isoformat()})
    with path.open('xb') as stream:
        stream.write(raw)
    return {'request_identity': identity(raw), 'protocol_identity': PROTOCOL_IDENTITY, 'planned_cells': 27}


def read_request(path, expected_identity):
    require_identity(expected_identity)
    raw = read_bytes(path, 128*1024)
    if identity(raw) != expected_identity:
        raise ContractError('convergence request identity mismatch')
    record = parse_json(raw, canonical=True); expected = request_core()
    exact_keys(record, set(expected)|{'created_at'}, 'convergence request')
    if any(json_bytes(record[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('convergence request source/data/environment/protocol mismatch')
    try:
        if datetime.fromisoformat(record['created_at']).utcoffset() is None:
            raise ValueError('timezone required')
    except (ValueError, TypeError) as error:
        raise ContractError('invalid convergence clock') from error
    return record, raw


def stability(history, start, end, clip):
    rows = history[start:end]
    if len(rows) != end-start or not rows:
        raise ContractError('incomplete diagnostic history')
    losses = [float.fromhex(row['loss_hex']) for row in rows]
    gradients = [float.fromhex(row['gradient_norm_hex']) for row in rows]
    if any(not math.isfinite(v) or v < 0 for v in losses+gradients):
        raise ContractError('invalid diagnostic loss/gradient')
    return {'updates': len(rows), 'mean_batch_loss_hex': (math.fsum(losses)/len(rows)).hex(),
            'max_gradient_norm_hex': max(gradients).hex(),
            'clipped_updates': sum(g > float(clip) for g in gradients)}


def train_cell(root, workload, seed, candidate, request, request_identity):
    _, request_raw = read_request(request, request_identity)
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    try:
        started = time.perf_counter(); state = state_for(workload, seed, candidate)
        update_seconds = 0.; curves = []; previous = 0
        for step in MILESTONES:
            before = time.perf_counter(); run_steps(state, stop_after=step)
            update_seconds += time.perf_counter()-before
            if update_seconds > protocol()['resources']['update_seconds_total']:
                raise ContractError('convergence exceeded cumulative update budget')
            scores, predictions = score(state.model, workload, 'train')
            write(root, f'predictions-{step}.json', {'predictions': predictions})
            curves.append({'step': step, 'model_identity': model_identity(state.model),
                           'train_loss_hex': evaluate(state.model, state.data['train'], state.config.batch_size),
                           'scores': scores, 'training_gate_failures': task_failures(scores, 'train'),
                           'stability': stability(state.history, previous, step, state.config.gradient_clip)})
            previous = step
        payload, metadata = snapshot_state(state, workload, candidate, request_identity)
        (root/'tensors.safetensors').write_bytes(payload); (root/'metadata.json').write_bytes(metadata)
        write(root, 'steps.json', {'steps': state.history, 'train_diagnostics': state.validation})
        report = {'schema': 'trinite.convergence-cell.v1', 'outcome': 'completed',
                  'workload': workload, 'seed': seed, 'candidate': candidate, 'lane': 'dense',
                  'request_identity': request_identity, 'protocol_identity': PROTOCOL_IDENTITY,
                  'source': sources(), 'plan_identity': state.config.content_identity,
                  'dataset_identity': state.dataset_identity, 'manifest_identity': state.manifest_identity,
                  'initialization_identity': state.initialization_identity,
                  'model_identity': model_identity(state.model), 'target_tokens': state.target_tokens,
                  'steps': state.step, 'parameter_count': state.config.model.parameter_count,
                  'schedule_identity': identity(json_bytes([h['examples'] for h in state.history])),
                  'tensors_identity': identity(payload), 'metadata_identity': identity(metadata),
                  'update_wall_seconds_hex': update_seconds.hex(),
                  'numeric_and_scoring_wall_seconds_hex': (time.perf_counter()-started).hex(),
                  'curves': curves, 'held_out_scored': False, 'learning_adequate': False,
                  'phase5_ready': False, 'release_ready': False}
        write(root, 'report.json', report); read_request(request, request_identity)
        collector = BundleObserver(root/'provenance')
        raw, manifest = dataset(workload)
        names = ('report.json', 'steps.json', 'tensors.safetensors', 'metadata.json',
                 *(f'predictions-{s}.json' for s in MILESTONES))
        collector.record('training-convergence', inputs={'request.json': request_raw, 'dataset.json': raw,
                         'dataset-manifest.json': manifest}, outputs={n: read_bytes(root/n) for n in names})
        verified = collector.finalize(); write(root, 'verification.json', verified)
        if not verified['integrity_verified'] or verified['errors'] or verified['known_missing_artifacts']:
            raise ContractError('convergence evidence failed upstream verification')
        return report
    except Exception as error:
        failure(root, 'convergence', error)
        raise


def verified_cell(root, workload, seed, candidate, request_identity):
    root = Path(root); verify_observation(root/'provenance')
    manifest = parse_json(read_bytes(root/'provenance/manifest.json'), canonical=True)
    members = {a['content_identity'] for a in manifest['core']['artifacts']}; indexes = []
    for key in members:
        try:
            value = parse_json(read_bytes(root/'provenance/artifacts/sha256'/key[7:]))
        except ContractError:
            continue
        if type(value) is dict and value.get('schema') == 'trinite.observer-index.v1':
            indexes.append(value)
    if len(indexes) != 1:
        raise ContractError('convergence evidence requires one name index')
    names = indexes[0]['artifacts']
    raw, admitted = dataset(workload)
    inputs = {'request.json': request_identity, 'dataset.json': identity(raw),
              'dataset-manifest.json': identity(admitted)}
    if any(names.get(name) != key or key not in members for name, key in inputs.items()):
        raise ContractError('convergence evidence/request or admitted input mismatch')
    def bound(name):
        raw = read_bytes(root/name)
        if identity(raw) != names.get(name) or names[name] not in members:
            raise ContractError('convergence root copy differs from closed evidence')
        return raw
    report = parse_json(bound('report.json'), canonical=True)
    expected_context = {'schema': 'trinite.convergence-cell.v1', 'outcome': 'completed',
                        'workload': workload, 'seed': seed, 'candidate': candidate, 'lane': 'dense',
                        'request_identity': request_identity, 'protocol_identity': PROTOCOL_IDENTITY,
                        'source': sources(), 'held_out_scored': False, 'learning_adequate': False,
                        'phase5_ready': False, 'release_ready': False}
    expected_keys = set(expected_context)|{'plan_identity', 'dataset_identity', 'manifest_identity',
        'initialization_identity', 'model_identity', 'target_tokens', 'steps', 'parameter_count',
        'schedule_identity', 'tensors_identity', 'metadata_identity', 'update_wall_seconds_hex',
        'numeric_and_scoring_wall_seconds_hex', 'curves'}
    exact_keys(report, expected_keys, 'convergence report')
    if any(json_bytes(report[k]) != json_bytes(v) for k, v in expected_context.items()):
        raise ContractError('convergence cell context mismatch')
    config = plan_for(seed, candidate)
    state = restore_state(bound('tensors.safetensors'), bound('metadata.json'), workload, seed,
                          candidate, request_identity, (report['tensors_identity'], report['metadata_identity']))
    expected = {'plan_identity': config.content_identity, 'dataset_identity': state.dataset_identity,
                'manifest_identity': state.manifest_identity, 'initialization_identity': state.initialization_identity,
                'model_identity': model_identity(state.model), 'target_tokens': state.target_tokens,
                'steps': config.steps, 'parameter_count': config.model.parameter_count,
                'schedule_identity': identity(json_bytes([h['examples'] for h in state.history]))}
    if state.step != config.steps or any(json_bytes(report[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('convergence checkpoint/receipt mismatch')
    history = parse_json(bound('steps.json'), canonical=True)
    if json_bytes(history) != json_bytes({'steps': state.history, 'train_diagnostics': state.validation}):
        raise ContractError('convergence history copy mismatch')
    curves = report['curves']
    if type(curves) is not list or len(curves) != len(MILESTONES):
        raise ContractError('incomplete convergence curve')
    previous = 0
    diagnostics = {row['step']: row['loss_hex'] for row in state.validation}
    for step, curve in zip(MILESTONES, curves):
        exact_keys(curve, {'step', 'model_identity', 'train_loss_hex', 'scores', 'training_gate_failures',
                          'stability'}, 'convergence milestone')
        require_identity(curve['model_identity'])
        scores = check_predictions(workload, 'train', bound(f'predictions-{step}.json'))
        expected = {'step': step, 'train_loss_hex': diagnostics[step], 'scores': scores,
                    'training_gate_failures': task_failures(scores, 'train'),
                    'stability': stability(state.history, previous, step, config.gradient_clip)}
        if any(json_bytes(curve[k]) != json_bytes(v) for k, v in expected.items()):
            raise ContractError('convergence diagnostic/prediction mismatch')
        previous = step
    if curves[-1]['model_identity'] != model_identity(state.model):
        raise ContractError('convergence final curve/model mismatch')
    if curves[-1]['train_loss_hex'] != evaluate(state.model, state.data['train'], config.batch_size):
        raise ContractError('convergence final diagnostic loss mismatch')
    seconds = []
    for name in ('update_wall_seconds_hex', 'numeric_and_scoring_wall_seconds_hex'):
        value = float.fromhex(report[name])
        if not math.isfinite(value) or value <= 0 or value.hex() != report[name]:
            raise ContractError('invalid convergence wall time')
        seconds.append(value)
    if seconds[0] > 120 or seconds[1] < seconds[0]:
        raise ContractError('convergence wall budget mismatch')
    return report


def groups_for(records):
    indexed = {(r['workload'], r['seed'], r['candidate']): r for r in records}
    if set(indexed) != set(cells()) or len(records) != 27:
        raise ContractError('incomplete/duplicate convergence cells')
    for workload in WORKLOADS:
        for seed in SEEDS:
            runs = [indexed[workload, seed, c] for c in CANDIDATES]
            for name in ('dataset_identity', 'manifest_identity', 'initialization_identity',
                         'target_tokens', 'steps', 'parameter_count', 'schedule_identity'):
                if len({r[name] for r in runs}) != 1:
                    raise ContractError('unmatched convergence '+name)
    return {c: {'training_tasks_pass_all_cells': all(not indexed[w, s, c]['curves'][-1]['training_gate_failures']
                                                   for w in WORKLOADS for s in SEEDS),
                'failed_training_cells': [cell_name(w, s, c) for w in WORKLOADS for s in SEEDS
                                         if indexed[w, s, c]['curves'][-1]['training_gate_failures']],
                'selection': 'none; freeze a separate learning protocol before evaluation'} for c in CANDIDATES}


def summarize(root, request, request_identity, worker_errors, *, publish=True):
    read_request(request, request_identity); root = Path(root)
    valid_names = {cell_name(*cell) for cell in cells()}
    if type(worker_errors) is not dict or not set(worker_errors) <= valid_names:
        raise ContractError('unknown convergence worker errors')
    if any(type(v) is not str or not v.strip() for v in worker_errors.values()):
        raise ContractError('invalid convergence worker error')
    records = []; groups = {}; aggregate_errors = []
    for workload, seed, candidate in cells():
        name = cell_name(workload, seed, candidate)
        try:
            if name in worker_errors:
                raise ContractError(worker_errors[name])
            records.append(verified_cell(root/name, workload, seed, candidate, request_identity))
        except (ContractError, OSError, KeyError, TypeError, ValueError, OverflowError) as error:
            records.append({'workload': workload, 'seed': seed, 'candidate': candidate,
                            'outcome': 'failed', 'error': type(error).__name__+': '+str(error)})
    if all(r['outcome'] == 'completed' for r in records):
        try:
            groups = groups_for(records)
        except (ContractError, KeyError, TypeError, ValueError) as error:
            aggregate_errors.append(type(error).__name__+': '+str(error))
    result = {'schema': 'trinite.convergence-summary.v1', 'protocol_identity': PROTOCOL_IDENTITY,
              'protocol_commit': PROTOCOL_COMMIT, 'request_identity': request_identity,
              'outcome': 'completed' if groups else 'failed', 'cells': records, 'groups': groups,
              'worker_errors': worker_errors, 'aggregate_errors': aggregate_errors,
              'result_status': 'exploratory training development; no selection', 'held_out_scored': False,
              'learning_adequate': False, 'phase5_ready': False, 'release_ready': False}
    if publish:
        write(root, 'summary.json', result)
    return result
