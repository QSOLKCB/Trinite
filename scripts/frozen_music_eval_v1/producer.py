"""Model-only admission of the complete original exposure producer packet.

Run in the isolated, pinned original source interpreter. Safe decoding and AdamW
slot validation are reused, but no optimizer is constructed and no forward,
backward or update is performed. Structural history is not numerical replay.
"""
from pathlib import Path

import torch
from trinite.comparison_checkpoint import load, _slot, _hex
from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from trinite.model import Decoder
from trinite.training import model_identity
from trinite.foundations import check_prediction_examples
from music_learning_v1.numerical import encoded
from music_exposure_v1.request import validate, cell_name
from music_exposure_v1.spec import SEEDS, CHECKPOINTS, PROTOCOL, plan_for, rows
from music_exposure_v1.reporting import summarize
from music_exposure_v1.workflow import names, finite, missing_checkpoint_files, SCOPE, CLAIMS
from music_exposure_v1.controller import bound_files
from .artifacts import inventory, read


def _schedule(record, plan):
    step = record['step']
    if type(step) is not int or step not in CHECKPOINTS:
        raise ContractError('producer milestone differs')
    if type(record['cursor']) is not int or record['cursor'] != step * 8:
        raise ContractError('producer cursor differs')
    train = encoded(rows())
    if type(record['history']) is not list or len(record['history']) != step:
        raise ContractError('producer history length differs')
    total = 0
    for i, entry in enumerate(record['history']):
        exact_keys(entry, {'step', 'loss_hex', 'gradient_norm_hex', 'target_tokens', 'examples'}, 'producer history')
        batch = [train[(i * 8 + j) % 40] for j in range(8)]
        targets = sum(sum(r['loss']) for r in batch)
        expected = {'step': i + 1, 'examples': [r['identity'] for r in batch], 'target_tokens': targets}
        if any(json_bytes(entry[k]) != json_bytes(v) for k, v in expected.items()):
            raise ContractError('producer scheduled visits differ')
        _hex(entry['loss_hex']); _hex(entry['gradient_norm_hex']); total += targets
    if type(record['target_tokens']) is not int or record['target_tokens'] != total or total > plan.max_target_tokens:
        raise ContractError('producer target count differs')
    expected_steps = [i for i in range(1, step + 1) if i % plan.validation_every == 0 or i == plan.steps]
    if type(record['validation']) is not list or len(record['validation']) != len(expected_steps):
        raise ContractError('producer diagnostic schedule differs')
    for entry, expected_step in zip(record['validation'], expected_steps):
        exact_keys(entry, {'step', 'loss_hex'}, 'producer validation')
        if type(entry['step']) is not int or entry['step'] != expected_step:
            raise ContractError('producer diagnostic step differs')
        _hex(entry['loss_hex'])
    _hex(record['initial_train_loss_hex'])


def checkpoint(payload, metadata, request, request_identity, seed, step):
    """Safely inspect all slots, copying only the model; zero model calls."""
    if type(seed) is not int or seed not in SEEDS or type(step) is not int or step not in CHECKPOINTS:
        raise ContractError('unknown producer endpoint')
    if type(payload) is not bytes or type(metadata) is not bytes or len(payload) > 16777216 or len(metadata) > 1048576:
        raise ContractError('producer checkpoint bounds differ')
    record = parse_json(metadata, canonical=True)
    plan = plan_for(seed)
    expected = {'schema': 'trinite.music-exposure-checkpoint.v1', 'seed': seed,
        'request_identity': request_identity, 'plan': plan.to_dict(),
        'dataset_identity': request['dataset_identity'], 'admission_identity': request['admission_identity'], 'step': step}
    exact_keys(record, set(expected) | {'initialization_identity', 'initial_train_loss_hex', 'model_identity',
        'cursor', 'target_tokens', 'history', 'validation'}, 'producer checkpoint')
    if any(json_bytes(record[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('producer checkpoint context differs')
    _schedule(record, plan)
    model = Decoder(plan.model, lane='dense', seed=seed)
    initial = model_identity(model)
    if record['initialization_identity'] != initial:
        raise ContractError('producer initialization differs')
    require_identity(record['model_identity'])
    try:
        tensors = load(payload)
    except Exception as error:
        raise ContractError('invalid producer safetensors') from error
    expected_names = {'model.' + n for n, _ in model.named_parameters()}
    if step:
        expected_names |= {'optimizer.' + n + '.' + slot for n, _ in model.named_parameters()
                           for slot in ('step', 'exp_avg', 'exp_avg_sq')}
    if set(tensors) != expected_names:
        raise ContractError('producer tensor inventory differs')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = tensors['model.' + name]
            if (value.device.type != 'cpu' or value.layout != torch.strided or value.dtype != parameter.dtype
                    or value.shape != parameter.shape or not bool(torch.isfinite(value).all())):
                raise ContractError('invalid producer model tensor')
            if step:
                for slot in ('step', 'exp_avg', 'exp_avg_sq'):
                    _slot(tensors['optimizer.' + name + '.' + slot], parameter, slot, step)
            parameter.copy_(value)
    model.validate_state()
    if model_identity(model) != record['model_identity'] or (step == 0 and record['model_identity'] != initial):
        raise ContractError('producer model identity differs')
    return model, record


def _score(raw):
    result = check_prediction_examples(rows(), 'train', 64, raw)
    for row in parse_json(raw, canonical=True)['predictions']:
        generated = row['generated_tokens']
        if 257 in generated[:-1] or (generated[-1] != 257 and len(generated) != 64):
            raise ContractError('producer output termination differs')
    return result


def _cell(root, raw, request, request_identity, seed, completion_identity):
    before = inventory(root)
    completion_raw = read(root / 'completion.json')
    if identity(completion_raw) != completion_identity:
        raise ContractError('producer completion identity differs')
    c = parse_json(completion_raw, canonical=True)
    exact_keys(c, {'schema', 'request_identity', 'cell', 'artifacts'}, 'producer completion')
    if (c['schema'] != 'trinite.music-exposure-completion.v1' or c['request_identity'] != request_identity
            or c['cell'] != cell_name(seed) or read(root / 'request.json') != raw):
        raise ContractError('producer completion context differs')
    exact_keys(c['artifacts'], names(), 'producer artifacts')
    required = names() | {'request.json', 'completion.json', 'attempt.json', 'worker.log'}
    if set(before) != required or {p.name for p in root.iterdir()} != required:
        raise ContractError('producer cell membership differs')
    if 'worker.log' in before:
        read(root / 'worker.log', PROTOCOL['max_worker_log_bytes'])
    bound = {n: read(root / n, PROTOCOL['max_artifact_bytes']) for n in names()}
    if any(identity(v) != c['artifacts'][n] for n, v in bound.items()):
        raise ContractError('producer artifact bytes differ')
    report = parse_json(bound['report.json'], canonical=True)
    exact_keys(report, {'schema', 'seed', 'request_identity', 'admission_identity', 'model_identity',
        'verification_scope', 'step', 'target_tokens', 'observations', 'observations_contrasts',
        'training_fit', 'update_seconds_hex', *CLAIMS}, 'producer report')
    expected = {'schema': 'trinite.music-exposure-report.v1', 'seed': seed,
        'request_identity': request_identity, 'admission_identity': request['admission_identity'],
        'verification_scope': SCOPE, 'step': 640, 'target_tokens': 10240, 'training_fit': True, **CLAIMS}
    if any(json_bytes(report[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('producer report context/fit differs')
    finite(report['update_seconds_hex'], PROTOCOL['max_update_seconds_per_seed'])
    exact_keys(report['observations'], {str(s) for s in CHECKPOINTS}, 'producer observations')
    exact_keys(report['observations_contrasts'], {str(s) for s in CHECKPOINTS}, 'producer contrasts')
    previous = None
    for step in CHECKPOINTS:
        prefix = f'step-{step:03d}'
        score_raw = bound[prefix + '-predictions.json']
        scores = _score(score_raw)
        predictions = parse_json(score_raw, canonical=True)['predictions']
        if bound[prefix + '-journal.jsonl'] != b''.join(json_bytes(p) for p in predictions):
            raise ContractError('producer journal differs')
        model, record = checkpoint(bound[prefix + '.safetensors'], bound[prefix + '-checkpoint.json'],
                                   request, request_identity, seed, step)
        if previous is not None and (record['initial_train_loss_hex'] != previous['initial_train_loss_hex']
                or record['history'][:len(previous['history'])] != previous['history']
                or record['validation'][:len(previous['validation'])] != previous['validation']):
            raise ContractError('producer milestone continuity differs')
        previous = record
        if (json_bytes(report['observations'][str(step)]) != json_bytes(scores)
                or json_bytes(report['observations_contrasts'][str(step)]) != json_bytes(summarize(score_raw))):
            raise ContractError('producer score consistency differs')
    if scores['correct'] != 40 or scores['examples'] != 40 or report['model_identity'] != record['model_identity']:
        raise ContractError('producer final fit/model differs')
    if inventory(root) != before:
        raise ContractError('producer cell mutated')
    prefix = cell_name(seed) + '/step-640'
    binding = {'seed': seed, 'step': 640, 'payload': prefix + '.safetensors',
        'metadata': prefix + '-checkpoint.json', 'payload_identity': identity(bound['step-640.safetensors']),
        'metadata_identity': identity(bound['step-640-checkpoint.json']), 'model_identity': record['model_identity'],
        'training_predictions_identity': identity(bound['step-640-predictions.json'])}
    return binding, model


def _inspect(root, request_identity, matrix_identity):
    root = Path(root)
    require_identity(request_identity); require_identity(matrix_identity)
    before = inventory(root)
    raw = read(root / 'request.json', 1048576)
    request = validate(raw, request_identity)
    matrix_raw = read(root / 'matrix.json')
    if identity(matrix_raw) != matrix_identity:
        raise ContractError('independent producer matrix identity differs')
    matrix = parse_json(matrix_raw, canonical=True)
    exact_keys(matrix, {'schema', 'request_identity', 'cells', 'outcome', 'both_seeds_fit', *CLAIMS}, 'producer matrix')
    if (matrix['schema'] != 'trinite.music-exposure-matrix.v1' or matrix['request_identity'] != request_identity
            or matrix['outcome'] != 'training_fit_passed' or matrix['both_seeds_fit'] is not True
            or any(matrix[k] is not False for k in CLAIMS)):
        raise ContractError('complete both-seed fitting producer required')
    exact_keys(matrix['cells'], {cell_name(s) for s in SEEDS}, 'producer cells')
    if {p.name for p in root.iterdir()} != {'request.json', 'matrix.json'} | {cell_name(s) for s in SEEDS}:
        raise ContractError('producer matrix membership differs')
    checkpoints, models = [], {}
    for seed in SEEDS:
        name = cell_name(seed); entry = matrix['cells'][name]
        exact_keys(entry, {'outcome', 'attempt_identity', 'completion_identity', 'retained_files'}, 'producer cell')
        if entry['outcome'] != 'completed' or bound_files(root / name) != entry['retained_files']:
            raise ContractError('producer cell inventory/completion differs')
        require_identity(entry['completion_identity'])
        attempt_raw = read(root / name / 'attempt.json')
        if identity(attempt_raw) != entry['attempt_identity']:
            raise ContractError('producer attempt identity differs')
        attempt = parse_json(attempt_raw, canonical=True)
        exact_keys(attempt, {'schema', 'cell', 'request_identity', 'outcome', 'worker_exit', 'detail',
            'completion_identity', 'wall_seconds_hex', 'missing_checkpoint_files'}, 'producer attempt')
        expected = {'schema': 'trinite.music-exposure-attempt.v1', 'cell': name,
            'request_identity': request_identity, 'outcome': 'completed', 'worker_exit': 0,
            'completion_identity': entry['completion_identity'],
            'missing_checkpoint_files': missing_checkpoint_files(root / name)}
        if any(json_bytes(attempt[k]) != json_bytes(v) for k, v in expected.items()) or type(attempt['detail']) is not str:
            raise ContractError('producer attempt context differs')
        finite(attempt['wall_seconds_hex'])
        binding, models[seed] = _cell(root / name, raw, request, request_identity, seed, entry['completion_identity'])
        checkpoints.append(binding)
    validate(raw, request_identity)
    if inventory(root) != before:
        raise ContractError('producer packet mutated')
    return {'request_identity': request_identity, 'matrix_identity': matrix_identity,
        'source': request['source'], 'environment': request['environment'],
        'admission_identity': request['admission_identity'], 'inventory': before, 'checkpoints': checkpoints,
        'safe_structure_verified': True, 'optimizer_replayed': False, 'predictions_replayed': False}, models


def inspect_matrix(root, request_identity, matrix_identity):
    return _inspect(root, request_identity, matrix_identity)[0]


def load_models(root, binding):
    actual, models = _inspect(root, binding['request_identity'], binding['matrix_identity'])
    if json_bytes(actual) != json_bytes(binding):
        raise ContractError('frozen producer binding differs')
    return models
