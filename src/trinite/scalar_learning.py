"""Matched scalar-learning stages with fresh verification before held-out model calls."""
import math
from pathlib import Path
import resource
import time

from .contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json
from .scalar_training import (PROTOCOL_IDENTITY, PROTOCOL_COMMIT,
    WORKLOADS, SEEDS, LANES, protocol, sources, cells, cell_name, plan_for, state_for,
    freeze_request, read_request)
from .curriculum_data import dataset
from .scalar_checkpoint import snapshot, restore
from .foundations import (read_bytes, write, failure, score_examples, check_prediction_examples,
    task_failures as parent_task_failures, validate_pairs, comparison_groups)
from .comparison import weight_diagnostics
from .training import run_steps, evaluate, model_identity
from .observation import BundleObserver, verify_observation

ERRORS = (ContractError, OSError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError)



def task_failures(scores, stage):
    # The original gate bounds totals to 2,000. Scalar fold has 3,450 rows;
    # validate its aggregate and delegate every unchanged per-task threshold.
    exact_keys(scores, {'examples', 'correct', 'tasks', 'families', 'family_macro_accuracy_hex'}, 'scalar scores')
    integer(scores['examples'], 'scalar score count', 1, 4096)
    integer(scores['correct'], 'scalar correct count', 0, scores['examples'])
    if type(scores['tasks']) is not dict or not scores['tasks']:
        raise ContractError('scalar gates require task scores')
    failures = []; total = correct = 0
    for task, value in scores['tasks'].items():
        exact_keys(value, {'examples', 'correct', 'baseline_correct'}, 'scalar task scores')
        failures.extend(parent_task_failures({**scores, 'examples': value['examples'],
            'correct': value['correct'], 'tasks': {task: value}}, stage))
        total += value['examples']; correct += value['correct']
    if (total, correct) != (scores['examples'], scores['correct']):
        raise ContractError('scalar task totals contradict aggregate counts')
    return failures


def score(model, workload, split):
    examples = parse_json(dataset(workload)[0], canonical=True)['examples']
    return score_examples(model, examples, split, protocol()['generation_caps'][workload])


def check_predictions(workload, split, raw):
    examples = parse_json(dataset(workload)[0], canonical=True)['examples']
    return check_prediction_examples(examples, split, protocol()['generation_caps'][workload], raw)


def validate_errors(errors, *, training=False):
    allowed = {cell_name(*c) for c in cells()}
    if not training:
        allowed |= {'evaluation-matrix', 'summary', 'runner'}
    if (type(errors) is not dict or not set(errors) <= allowed
            or any(type(v) is not str or not v.strip() for v in errors.values())):
        raise ContractError('invalid scalar-learning worker errors')


def retain(root, stage, request_raw, workload, outputs, inputs=None):
    data, manifest = dataset(workload)
    collector = BundleObserver(root/(stage+'-provenance'))
    collector.record('scalar-learning-'+stage,
        inputs={'request.json': request_raw, 'dataset.json': data,
                'dataset-manifest.json': manifest, **(inputs or {})},
        outputs={n: read_bytes(root/n) for n in outputs})
    verification = collector.finalize()
    write(root, stage+'-verification.json', verification)
    if (not verification['integrity_verified'] or verification['errors']
            or verification['known_missing_artifacts']):
        raise ContractError('scalar-learning evidence failed upstream verification')


def train_cell(root, workload, seed, lane, request, request_identity):
    _, request_raw = read_request(request, request_identity)
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    try:
        state = state_for(workload, seed, lane)
        started = time.perf_counter(); run_steps(state)
        seconds = time.perf_counter()-started
        payload, metadata = snapshot(state, workload, request_identity)
        (root/'tensors.safetensors').write_bytes(payload)
        (root/'metadata.json').write_bytes(metadata)
        write(root, 'steps.json', {'steps': state.history, 'train_diagnostics': state.validation})
        scores, predictions = score(state.model, workload, 'train')
        write(root, 'training-predictions.json', {'predictions': predictions})
        report = {'schema': 'trinite.scalar-learning-training.v1', 'outcome': 'completed',
            'workload': workload, 'seed': seed, 'lane': lane,
            'request_identity': request_identity, 'protocol_identity': PROTOCOL_IDENTITY,
            'source': sources(), 'dataset_identity': state.dataset_identity,
            'manifest_identity': state.manifest_identity,
            'initialization_identity': state.initialization_identity,
            'model_identity': model_identity(state.model), 'plan_identity': state.config.content_identity,
            'parameter_count': state.config.model.parameter_count, 'steps': state.step,
            'target_tokens': state.target_tokens,
            'schedule_identity': identity(json_bytes([h['examples'] for h in state.history])),
            'tensors_identity': identity(payload), 'metadata_identity': identity(metadata),
            'training_wall_seconds_hex': seconds.hex(),
            'latent_float32_bytes': state.config.model.parameter_count*4,
            'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'initial_train_loss_hex': state.initial_train_loss,
            'final_train_loss_hex': evaluate(state.model, state.data['train'], state.config.batch_size),
            'scores': scores, 'training_gate_failures': task_failures(scores, 'train'),
            'held_out_scored': False, 'scope': protocol()['scope']}
        write(root, 'training.json', report)
        read_request(request, request_identity)
        retain(root, 'train', request_raw, workload, ('training.json', 'training-predictions.json',
                                                   'steps.json', 'tensors.safetensors', 'metadata.json'))
        return report
    except Exception as error:
        failure(root, 'train', error)
        raise


def closed_stage(root, workload, request_identity, stage):
    root = Path(root); bundle = root/(stage+'-provenance')
    manifest_raw = read_bytes(bundle/'manifest.json')
    verification = verify_observation(bundle)
    if read_bytes(bundle/'manifest.json') != manifest_raw:
        raise ContractError('scalar-learning manifest changed during verification')
    if read_bytes(root/(stage+'-verification.json')) != json_bytes(verification):
        raise ContractError('retained scalar verification receipt differs from fresh verification')
    manifest = parse_json(manifest_raw, canonical=True)
    if manifest['manifest_identity'] != verification['manifest_identity']:
        raise ContractError('scalar-learning upstream manifest identity mismatch')
    members = {a['content_identity'] for a in manifest['core']['artifacts']}
    indexes = []
    for key in members:
        try:
            value = parse_json(read_bytes(bundle/'artifacts/sha256'/key[7:]))
        except ContractError:
            continue
        if type(value) is dict and value.get('schema') == 'trinite.observer-index.v1':
            indexes.append(value)
    if len(indexes) != 1:
        raise ContractError('scalar-learning evidence requires one name index')
    names = indexes[0]['artifacts']; raw, admitted = dataset(workload)
    expected = {'request.json': request_identity, 'dataset.json': identity(raw),
                'dataset-manifest.json': identity(admitted)}
    if any(names.get(n) != key or key not in members for n, key in expected.items()):
        raise ContractError('scalar-learning evidence/request/admission mismatch')
    def bound(name):
        raw = read_bytes(root/name)
        if identity(raw) != names.get(name) or names[name] not in members:
            raise ContractError('scalar-learning root copy differs from closed evidence')
        return raw
    return bound, names, members, verification, identity(manifest_raw)


def restore_report(root, workload, seed, lane, request_identity, report):
    return restore(read_bytes(Path(root)/'tensors.safetensors'), read_bytes(Path(root)/'metadata.json'),
        workload, seed, lane, request_identity,
        (report['tensors_identity'], report['metadata_identity']), plan_for(seed))


def verified_training(root, workload, seed, lane, request_identity):
    bound, _, _, verification, manifest_file_identity = closed_stage(root, workload, request_identity, 'train')
    report = parse_json(bound('training.json'), canonical=True)
    context = {'schema': 'trinite.scalar-learning-training.v1', 'outcome': 'completed',
        'workload': workload, 'seed': seed, 'lane': lane, 'request_identity': request_identity,
        'protocol_identity': PROTOCOL_IDENTITY, 'source': sources(),
        'held_out_scored': False, 'scope': protocol()['scope']}
    exact_keys(report, set(context)|{'dataset_identity', 'manifest_identity', 'initialization_identity',
        'model_identity', 'plan_identity', 'parameter_count', 'steps', 'target_tokens', 'schedule_identity',
        'tensors_identity', 'metadata_identity', 'training_wall_seconds_hex', 'latent_float32_bytes',
        'peak_process_rss_kib', 'initial_train_loss_hex', 'final_train_loss_hex', 'scores',
        'training_gate_failures'}, 'scalar-learning training report')
    if any(json_bytes(report[k]) != json_bytes(v) for k, v in context.items()):
        raise ContractError('scalar-learning training context mismatch')
    # Bind both root checkpoint copies to the closed bundle before restoring.
    payload, metadata = bound('tensors.safetensors'), bound('metadata.json')
    state = restore(payload, metadata, workload, seed, lane, request_identity,
                    (report['tensors_identity'], report['metadata_identity']), plan_for(seed))
    scores = check_predictions(workload, 'train', bound('training-predictions.json'))
    actual_scores, actual_predictions = score(state.model, workload, 'train')
    loss = evaluate(state.model, state.data['train'], state.config.batch_size)
    expected = {'dataset_identity': state.dataset_identity, 'manifest_identity': state.manifest_identity,
        'initialization_identity': state.initialization_identity, 'model_identity': model_identity(state.model),
        'plan_identity': state.config.content_identity, 'parameter_count': state.config.model.parameter_count,
        'steps': state.config.steps, 'target_tokens': state.target_tokens,
        'schedule_identity': identity(json_bytes([h['examples'] for h in state.history])),
        'initial_train_loss_hex': state.initial_train_loss, 'final_train_loss_hex': loss,
        'latent_float32_bytes': state.config.model.parameter_count*4,
        'scores': scores, 'training_gate_failures': task_failures(scores, 'train')}
    if (state.step != state.config.steps or any(json_bytes(report[k]) != json_bytes(v) for k, v in expected.items())
            or json_bytes({'predictions': actual_predictions}) != bound('training-predictions.json')
            or json_bytes(actual_scores) != json_bytes(scores)
            or state.validation[-1]['loss_hex'] != loss):
        raise ContractError('scalar-learning checkpoint/model/prediction/loss receipt mismatch')
    if bound('steps.json') != json_bytes({'steps': state.history, 'train_diagnostics': state.validation}):
        raise ContractError('scalar-learning history copy mismatch')
    wall = float.fromhex(report['training_wall_seconds_hex'])
    if (not math.isfinite(wall) or not 0 < wall <= state.config.max_seconds
            or wall.hex() != report['training_wall_seconds_hex']):
        raise ContractError('invalid scalar-learning wall budget')
    integer(report['peak_process_rss_kib'], 'scalar-learning peak RSS', 1, 2**63-1)
    return {'report': report, 'report_identity': identity(bound('training.json')),
            'manifest_file_identity': manifest_file_identity, 'verification': verification}


def training_inventory(root, request_identity, worker_errors):
    validate_errors(worker_errors, training=True)
    records = []
    for workload, seed, lane in cells():
        name = cell_name(workload, seed, lane)
        try:
            if name in worker_errors:
                raise ContractError(worker_errors[name])
            record = verified_training(Path(root)/name, workload, seed, lane, request_identity)
            records.append({'workload': workload, 'seed': seed, 'lane': lane, 'outcome': 'completed', **record})
        except ERRORS as error:
            records.append({'workload': workload, 'seed': seed, 'lane': lane, 'outcome': 'failed',
                            'error': type(error).__name__+': '+str(error)})
    return records


def matched(records):
    keys = [(r['workload'], r['seed'], r['lane']) for r in records]
    if len(keys) != 27 or set(keys) != set(cells()):
        raise ContractError('incomplete/duplicate scalar-learning matrix')
    validate_pairs(records)


def decision_record(root, request_identity, worker_errors):
    records = training_inventory(root, request_identity, worker_errors)
    failures = []
    for row in records:
        if row['lane'] != 'dense':
            continue
        errors = (['missing/failed verified dense cell'] if row['outcome'] != 'completed'
                  else task_failures(row['report']['scores'], 'train'))
        if errors:
            failures.append({'workload': row['workload'], 'seed': row['seed'], 'errors': errors})
    pairing_errors = []
    complete = all(r['outcome'] == 'completed' for r in records)
    if complete:
        try:
            matched(records)
        except ERRORS as error:
            pairing_errors.append(type(error).__name__+': '+str(error))
    eligible = complete and not worker_errors and not failures and not pairing_errors
    return {'schema': 'trinite.scalar-learning-test-decision.v1', 'request_identity': request_identity,
        'protocol_identity': PROTOCOL_IDENTITY, 'eligible': eligible,
        'dense_training_failures': failures, 'pairing_errors': pairing_errors,
        'worker_errors': dict(worker_errors), 'training_receipts': {cell_name(r['workload'], r['seed'], r['lane']):
            {'report_identity': r['report_identity'], 'manifest_identity': r['verification']['manifest_identity'],
             'manifest_file_identity': r['manifest_file_identity']}
            for r in records if r['outcome'] == 'completed'}}, records


def evaluate_matrix(root, request, request_identity, training_errors):
    _, request_raw = read_request(request, request_identity); root = Path(root)
    decision, records = decision_record(root, request_identity, training_errors)
    decision_raw = write(root, 'test-decision.json', decision)
    errors = {}
    if decision['eligible']:
        # Freshly verify the whole matrix once, then bind each selected model to
        # those receipts. The retained decision is never a reusable authorization.
        for row in records:
            workload, seed, lane = row['workload'], row['seed'], row['lane']
            name = cell_name(workload, seed, lane); directory = root/name
            try:
                read_request(request, request_identity)
                if read_bytes(root/'test-decision.json') != decision_raw:
                    raise ContractError('scalar-learning decision changed during evaluation')
                if (identity(read_bytes(directory/'training.json')) != row['report_identity']
                        or identity(read_bytes(directory/'train-provenance/manifest.json'))
                        != row['manifest_file_identity']):
                    raise ContractError('verified training receipt changed before scoring')
                state = restore_report(directory, workload, seed, lane, request_identity, row['report'])
                scores, predictions = score(state.model, workload, 'test')
                write(directory, 'test-predictions.json', {'predictions': predictions})
                failures = task_failures(scores, 'test')
                report = {'schema': 'trinite.scalar-learning-evaluation.v1', 'outcome': 'completed',
                    'workload': workload, 'seed': seed, 'lane': lane,
                    'request_identity': request_identity, 'protocol_identity': PROTOCOL_IDENTITY,
                    'decision_identity': identity(decision_raw), 'training_identity': row['report_identity'],
                    'model_identity': model_identity(state.model), 'scores': scores,
                    'test_gate_failures': failures,
                    'learning_adequate': not row['report']['training_gate_failures'] and not failures,
                    'weights': weight_diagnostics(state.model), 'scope': protocol()['scope']}
                write(directory, 'evaluation.json', report)
                retain(directory, 'test', request_raw, workload, ('evaluation.json', 'test-predictions.json'),
                       {'test-decision.json': decision_raw, 'training.json': read_bytes(directory/'training.json')})
            except ERRORS as error:
                errors[name] = type(error).__name__+': '+str(error)
                failure(directory, 'test', error)
    write(root, 'evaluation-errors.json', errors)
    return decision, errors


def verified_evaluation(root, workload, seed, lane, request_identity, row, decision_raw):
    bound, names, members, verification, _ = closed_stage(root, workload, request_identity, 'test')
    report = parse_json(bound('evaluation.json'), canonical=True)
    state = restore_report(root, workload, seed, lane, request_identity, row['report'])
    scores = check_predictions(workload, 'test', bound('test-predictions.json'))
    actual_scores, predictions = score(state.model, workload, 'test')
    failures = task_failures(scores, 'test')
    expected = {'schema': 'trinite.scalar-learning-evaluation.v1', 'outcome': 'completed',
        'workload': workload, 'seed': seed, 'lane': lane, 'request_identity': request_identity,
        'protocol_identity': PROTOCOL_IDENTITY, 'decision_identity': identity(decision_raw),
        'training_identity': row['report_identity'], 'model_identity': model_identity(state.model),
        'scores': scores, 'test_gate_failures': failures,
        'learning_adequate': not row['report']['training_gate_failures'] and not failures,
        'weights': weight_diagnostics(state.model), 'scope': protocol()['scope']}
    if (json_bytes(report) != json_bytes(expected) or json_bytes(actual_scores) != json_bytes(scores)
            or json_bytes({'predictions': predictions}) != bound('test-predictions.json')):
        raise ContractError('scalar-learning evaluation/model/prediction receipt mismatch')
    for name, key in {'test-decision.json': identity(decision_raw), 'training.json': row['report_identity']}.items():
        if names.get(name) != key or key not in members:
            raise ContractError('scalar-learning held-out input bindings mismatch')
    return report, verification


def summarize(root, request, request_identity, worker_errors, *, publish=True):
    read_request(request, request_identity); validate_errors(worker_errors)
    root = Path(root); errors = []; groups = {}
    training_errors = parse_json(read_bytes(root/'training-worker-errors.json'), canonical=True)
    validate_errors(training_errors, training=True)
    if any(worker_errors.get(k) != v for k, v in training_errors.items()):
        raise ContractError('final worker errors omit/change training failures')
    expected, records = decision_record(root, request_identity, training_errors)
    decision = expected; decision_raw = b''
    try:
        decision_raw = read_bytes(root/'test-decision.json', 128*1024)
        if decision_raw != json_bytes(expected):
            raise ContractError('retained scalar-learning decision contradicts verified training')
        errors.extend(decision['pairing_errors'])
    except ERRORS as error:
        errors.append(type(error).__name__+': '+str(error))
    if decision['eligible'] and not errors:
        for row in records:
            w, s, lane = row['workload'], row['seed'], row['lane']; name = cell_name(w, s, lane)
            try:
                if name in worker_errors:
                    raise ContractError(worker_errors[name])
                evaluation, verification = verified_evaluation(root/name, w, s, lane, request_identity, row, decision_raw)
                row.update(evaluation=evaluation, test_verification=verification)
            except ERRORS as error:
                row.update(outcome='failed', error=type(error).__name__+': '+str(error))
        if not worker_errors and all(r['outcome'] == 'completed' for r in records):
            try:
                matched(records); groups = comparison_groups(records)
            except ERRORS as error:
                errors.append(type(error).__name__+': '+str(error))
    elif not decision['eligible']:
        if any((root/cell_name(*c)/n).exists() for c in cells()
               for n in ('evaluation.json', 'test-predictions.json', 'test-provenance')):
            errors.append('held-out artifacts exist under a blocked decision')
    failed = bool(errors or worker_errors or any(r['outcome'] == 'failed' for r in records))
    adequate = bool(groups) and all(all(groups[w]['dense']['learning_adequate_each_seed']) for w in WORKLOADS)
    result = {'schema': 'trinite.scalar-learning-summary.v1', 'request_identity': request_identity,
        'protocol_identity': PROTOCOL_IDENTITY, 'protocol_commit': PROTOCOL_COMMIT,
        'outcome': 'failed' if failed else 'completed' if groups else 'blocked',
        'test_decision': decision, 'cells': records, 'groups': groups,
        'worker_errors': dict(worker_errors), 'aggregate_errors': errors,
        'dense_scalar_learning_adequate': adequate, 'dense_learning_adequate': False,
        'composed_outputs_scored': False, 'format_selection_eligible': False,
        'phase5_ready': False, 'release_ready': False, 'result_status': 'inconclusive',
        'scope': protocol()['scope'], 'historical_evidence_preserved': True}
    if publish:
        write(root, 'summary.json', result)
    return result


def failed_summary(root, request_identity, errors):
    """Conservative inventory when the bounded summary worker cannot finish."""
    validate_errors(errors); root = Path(root)
    prior = root/'summary.json'
    if prior.exists():
        prior.rename(root/'summary-before-worker-failure.json')
    result = {'schema': 'trinite.scalar-learning-failed-inventory.v1', 'outcome': 'failed',
        'request_identity': request_identity, 'protocol_identity': PROTOCOL_IDENTITY,
        'cells': [{'workload': w, 'seed': s, 'lane': lane, 'outcome': 'unverified'} for w, s, lane in cells()],
        'groups': {}, 'worker_errors': errors, 'integrity_verified': False,
        'dense_scalar_learning_adequate': False, 'dense_learning_adequate': False,
        'composed_outputs_scored': False, 'format_selection_eligible': False,
        'phase5_ready': False, 'release_ready': False}
    write(root, 'summary.json', result)
    return result
