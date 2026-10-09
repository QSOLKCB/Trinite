"""Training-only extended budget orchestration and milestone verification."""
import math
from pathlib import Path
import time

from .contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from .budget_training import (PROTOCOL_IDENTITY, PROTOCOL_COMMIT, RUNNER_IDENTITY, WORKLOADS,
    SEEDS, CANDIDATES, MILESTONES, protocol, sources, cells, cell_name, plan_for, state_for,
    request_core, freeze_request, read_request)
from .foundations_data import dataset
from .budget_checkpoint import snapshot_state, restore_state
from .convergence import stability
from .foundations import score, check_predictions, task_failures, read_bytes, write, failure
from .training import run_steps, model_identity, evaluate
from .observation import BundleObserver, verify_observation

def train_cell(root, workload, seed, candidate, request, request_identity):
    _, request_raw = read_request(request, request_identity)
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    try:
        started = time.perf_counter(); state = state_for(workload, seed, candidate, plan_for(seed,candidate))
        update_seconds = 0.; curves = []; previous = 0
        for step in MILESTONES:
            before = time.perf_counter(); run_steps(state, stop_after=step)
            update_seconds += time.perf_counter()-before
            if update_seconds > protocol()['resources']['update_seconds_total']:
                raise ContractError('budget exceeded cumulative update budget')
            scores, predictions = score(state.model, workload, 'train')
            write(root, f'predictions-{step}.json', {'predictions': predictions})
            payload, metadata = snapshot_state(state, workload, candidate, request_identity)
            (root/f'tensors-{step}.safetensors').write_bytes(payload)
            (root/f'metadata-{step}.json').write_bytes(metadata)
            curves.append({'step': step, 'model_identity': model_identity(state.model),
                           'tensors_identity': identity(payload), 'metadata_identity': identity(metadata),
                           'train_loss_hex': evaluate(state.model, state.data['train'], state.config.batch_size),
                           'scores': scores, 'training_gate_failures': task_failures(scores, 'train'),
                           'stability': stability(state.history, previous, step, state.config.gradient_clip)})
            previous = step
        payload, metadata = snapshot_state(state, workload, candidate, request_identity)
        (root/'tensors.safetensors').write_bytes(payload); (root/'metadata.json').write_bytes(metadata)
        write(root, 'steps.json', {'steps': state.history, 'train_diagnostics': state.validation})
        report = {'schema': 'trinite.budget-cell.v1', 'outcome': 'completed',
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
                 *(f'{prefix}-{s}.{suffix}' for s in MILESTONES
                   for prefix, suffix in (('predictions','json'),('tensors','safetensors'),('metadata','json'))))
        collector.record('training-budget', inputs={'request.json': request_raw, 'dataset.json': raw,
                         'dataset-manifest.json': manifest}, outputs={n: read_bytes(root/n) for n in names})
        verified = collector.finalize(); write(root, 'verification.json', verified)
        if not verified['integrity_verified'] or verified['errors'] or verified['known_missing_artifacts']:
            raise ContractError('budget evidence failed upstream verification')
        return report
    except Exception as error:
        failure(root, 'budget', error)
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
        raise ContractError('budget evidence requires one name index')
    names = indexes[0]['artifacts']
    raw, admitted = dataset(workload)
    inputs = {'request.json': request_identity, 'dataset.json': identity(raw),
              'dataset-manifest.json': identity(admitted)}
    if any(names.get(name) != key or key not in members for name, key in inputs.items()):
        raise ContractError('budget evidence/request or admitted input mismatch')
    def bound(name):
        raw = read_bytes(root/name)
        if identity(raw) != names.get(name) or names[name] not in members:
            raise ContractError('budget root copy differs from closed evidence')
        return raw
    report = parse_json(bound('report.json'), canonical=True)
    expected_context = {'schema': 'trinite.budget-cell.v1', 'outcome': 'completed',
                        'workload': workload, 'seed': seed, 'candidate': candidate, 'lane': 'dense',
                        'request_identity': request_identity, 'protocol_identity': PROTOCOL_IDENTITY,
                        'source': sources(), 'held_out_scored': False, 'learning_adequate': False,
                        'phase5_ready': False, 'release_ready': False}
    expected_keys = set(expected_context)|{'plan_identity', 'dataset_identity', 'manifest_identity',
        'initialization_identity', 'model_identity', 'target_tokens', 'steps', 'parameter_count',
        'schedule_identity', 'tensors_identity', 'metadata_identity', 'update_wall_seconds_hex',
        'numeric_and_scoring_wall_seconds_hex', 'curves'}
    exact_keys(report, expected_keys, 'budget report')
    if any(json_bytes(report[k]) != json_bytes(v) for k, v in expected_context.items()):
        raise ContractError('budget cell context mismatch')
    config = plan_for(seed, candidate)
    state = restore_state(bound('tensors.safetensors'), bound('metadata.json'), workload, seed,
                          candidate, request_identity, (report['tensors_identity'], report['metadata_identity']), config)
    expected = {'plan_identity': config.content_identity, 'dataset_identity': state.dataset_identity,
                'manifest_identity': state.manifest_identity, 'initialization_identity': state.initialization_identity,
                'model_identity': model_identity(state.model), 'target_tokens': state.target_tokens,
                'steps': config.steps, 'parameter_count': config.model.parameter_count,
                'schedule_identity': identity(json_bytes([h['examples'] for h in state.history]))}
    if state.step != config.steps or any(json_bytes(report[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('budget checkpoint/receipt mismatch')
    history = parse_json(bound('steps.json'), canonical=True)
    if json_bytes(history) != json_bytes({'steps': state.history, 'train_diagnostics': state.validation}):
        raise ContractError('budget history copy mismatch')
    curves = report['curves']
    if type(curves) is not list or len(curves) != len(MILESTONES):
        raise ContractError('incomplete budget curve')
    previous = 0
    diagnostics = {row['step']: row['loss_hex'] for row in state.validation}
    for step, curve in zip(MILESTONES, curves):
        exact_keys(curve, {'step', 'model_identity', 'train_loss_hex', 'scores', 'training_gate_failures',
                          'stability', 'tensors_identity', 'metadata_identity'}, 'budget milestone')
        require_identity(curve['model_identity'])
        milestone = restore_state(bound(f'tensors-{step}.safetensors'), bound(f'metadata-{step}.json'),
            workload, seed, candidate, request_identity,
            (curve['tensors_identity'],curve['metadata_identity']), config)
        if (milestone.step != step or milestone.history != state.history[:step]
                or milestone.validation != [v for v in state.validation if v['step'] <= step]
                or model_identity(milestone.model) != curve['model_identity']):
            raise ContractError('budget milestone checkpoint/history mismatch')
        predictions_raw = bound(f'predictions-{step}.json')
        scores = check_predictions(workload, 'train', predictions_raw)
        actual_scores, actual_predictions = score(milestone.model, workload, 'train')
        if (json_bytes({'predictions':actual_predictions}) != predictions_raw
                or json_bytes(actual_scores) != json_bytes(scores)
                or evaluate(milestone.model,milestone.data['train'],config.batch_size) != diagnostics[step]):
            raise ContractError('budget milestone model/prediction/loss mismatch')
        if step == config.steps and (curve['tensors_identity'],curve['metadata_identity']) != (
                report['tensors_identity'],report['metadata_identity']):
            raise ContractError('budget final checkpoint copy mismatch')
        expected = {'step': step, 'train_loss_hex': diagnostics[step], 'scores': scores,
                    'training_gate_failures': task_failures(scores, 'train'),
                    'stability': stability(state.history, previous, step, config.gradient_clip)}
        if any(json_bytes(curve[k]) != json_bytes(v) for k, v in expected.items()):
            raise ContractError('budget diagnostic/prediction mismatch')
        previous = step
    if curves[-1]['model_identity'] != model_identity(state.model):
        raise ContractError('budget final curve/model mismatch')
    if curves[-1]['train_loss_hex'] != evaluate(state.model, state.data['train'], config.batch_size):
        raise ContractError('budget final diagnostic loss mismatch')
    seconds = []
    for name in ('update_wall_seconds_hex', 'numeric_and_scoring_wall_seconds_hex'):
        value = float.fromhex(report[name])
        if not math.isfinite(value) or value <= 0 or value.hex() != report[name]:
            raise ContractError('invalid budget wall time')
        seconds.append(value)
    if seconds[0] > protocol()['resources']['update_seconds_total'] or seconds[1] < seconds[0]:
        raise ContractError('budget wall budget mismatch')
    return report


def groups_for(records):
    indexed = {(r['workload'], r['seed'], r['candidate']): r for r in records}
    if set(indexed) != set(cells()) or len(records) != len(cells()):
        raise ContractError('incomplete/duplicate budget cells')
    for record in records:
        config = plan_for(record['seed'],record['candidate'])
        if (record['outcome'] != 'completed' or record['steps'] != config.steps
                or record['parameter_count'] != config.model.parameter_count):
            raise ContractError('unmatched budget plan/parameter count')
    for workload in WORKLOADS:
        runs = [indexed[workload,s,'extended'] for s in SEEDS]
        for name in ('dataset_identity','manifest_identity'):
            if len({r[name] for r in runs}) != 1:
                raise ContractError('unmatched budget '+name)
    return {c: {'training_tasks_pass_all_cells': all(not indexed[w, s, c]['curves'][-1]['training_gate_failures']
                                                   for w in WORKLOADS for s in SEEDS),
                'failed_training_cells': [cell_name(w, s, c) for w in WORKLOADS for s in SEEDS
                                         if indexed[w, s, c]['curves'][-1]['training_gate_failures']],
                'selection': 'none; freeze a separate learning protocol before evaluation'} for c in CANDIDATES}


def summarize(root, request, request_identity, worker_errors, *, publish=True):
    read_request(request, request_identity); root = Path(root)
    valid_names = {cell_name(*cell) for cell in cells()}
    if type(worker_errors) is not dict or not set(worker_errors) <= valid_names:
        raise ContractError('unknown budget worker errors')
    if any(type(v) is not str or not v.strip() for v in worker_errors.values()):
        raise ContractError('invalid budget worker error')
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
    result = {'schema': 'trinite.budget-summary.v1', 'protocol_identity': PROTOCOL_IDENTITY,
              'protocol_commit': PROTOCOL_COMMIT, 'request_identity': request_identity,
              'outcome': 'completed' if groups else 'failed', 'cells': records, 'groups': groups,
              'worker_errors': worker_errors, 'aggregate_errors': aggregate_errors,
              'result_status': 'exploratory training development; no selection', 'held_out_scored': False,
              'learning_adequate': False, 'phase5_ready': False, 'release_ready': False}
    if publish:
        write(root, 'summary.json', result)
    return result
