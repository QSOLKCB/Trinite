"""Train-only retained cell custody; audit does not replay numerical computation."""
import math
import os
from pathlib import Path
import time
from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from native_diagnostic_v1.artifacts import Store
from unseen_evaluation_v1.packet import inventory, unchanged
from .request import read, validate, cell_name
from .spec import PROTOCOL, CHECKPOINTS

SCOPE = 'source-bound retained bytes, safe native tensor structure and score consistency; no inference or optimizer replay'
CLAIMS = {'fresh_confirmation': False, 'learning_adequate': False,
          'phase5_ready': False, 'release_ready': False}
SUFFIXES = ('.safetensors', '-checkpoint.json', '-predictions.json', '-journal.jsonl')


def checkpoint_names(step):
    return {f'step-{step:03d}' + suffix for suffix in SUFFIXES}


def names():
    return {'report.json'} | set().union(*(checkpoint_names(step) for step in CHECKPOINTS))


def finite(value, maximum=None):
    try:
        number = float.fromhex(value)
    except (TypeError, ValueError) as error:
        raise ContractError('invalid exposure timing') from error
    if not math.isfinite(number) or number < 0 or number.hex() != value or (maximum is not None and number > maximum):
        raise ContractError('invalid exposure timing')
    return number


def missing_checkpoint_files(root):
    # Presence only, never a claim that incomplete observations were validated.
    present = set(inventory(root)['members'])
    return {str(step): sorted(checkpoint_names(step)-present) for step in CHECKPOINTS}


def run(root, request_raw, request_identity, seed, expected_directory):
    request = validate(request_raw, request_identity)
    cell_name(seed)
    root = Path(root)
    before = inventory(root)
    if set(before['members']) - {'.', 'worker.log'}:
        raise ContractError('exposure worker requires fresh controlled directory')
    store = Store(root, expected_directory)
    outputs, journals = {}, {}
    started = time.monotonic()
    try:
        store.write('request.json', request_raw)
        def write(name, raw):
            if time.monotonic()-started > PROTOCOL['worker_seconds']:
                raise ContractError('exposure worker deadline exceeded')
            store.write(name, raw)
            outputs[name] = identity(raw)
        def emit(prefix, record):
            if prefix not in tuple(f'step-{step:03d}' for step in CHECKPOINTS):
                raise ContractError('unknown exposure prediction journal')
            name = prefix+'-journal.jsonl'
            if name not in journals:
                store.check()
                fd = os.open(name, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600, dir_fd=store.fd)
                journals[name] = [os.fdopen(fd, 'wb'), 0]
            journal, count = journals[name]
            raw = json_bytes(record)
            if count+len(raw) > 1024*1024:
                raise ContractError('exposure journal exceeds bound')
            journal.write(raw); journal.flush(); os.fsync(journal.fileno())
            journals[name][1] += len(raw)
            store.check()
        from . import numerical
        report = numerical.train(seed, request_identity, request['admission_identity'], write, emit=emit)
        for name, (stream, _) in journals.items():
            stream.close()
            outputs[name] = identity(read(root/name, 1024*1024))
        from .reporting import summarize
        report['observations_contrasts'] = {str(step): summarize(read(root/f'step-{step:03d}-predictions.json'))
                                           for step in CHECKPOINTS}
        report.update(schema='trinite.music-exposure-report.v1', verification_scope=SCOPE, **CLAIMS)
        write('report.json', json_bytes(report))
        if set(outputs) != names():
            raise ContractError('exposure completed artifact membership differs')
        validate(request_raw, request_identity)
        completion = {'schema': 'trinite.music-exposure-completion.v1',
            'request_identity': request_identity, 'cell': cell_name(seed), 'artifacts': outputs}
        raw = json_bytes(completion); store.write('completion.json', raw)
        return identity(raw)
    finally:
        for stream, _ in journals.values():
            if not stream.closed:
                stream.close()
        store.close()


def audit(root, request_raw, request_identity, seed, expected_identity):
    root = Path(root)
    before = inventory(root)
    if before['bytes'] > 32*1024*1024:
        raise ContractError('exposure cell byte limit exceeded')
    if 'worker.log' in before['members']:
        read(root/'worker.log', PROTOCOL['max_worker_log_bytes'])
    request = validate(request_raw, request_identity)
    require_identity(expected_identity)
    raw = read(root/'completion.json')
    if identity(raw) != expected_identity:
        raise ContractError('exposure completion identity differs')
    completion = parse_json(raw, canonical=True)
    exact_keys(completion, {'schema', 'request_identity', 'cell', 'artifacts'}, 'exposure completion')
    if completion['schema'] != 'trinite.music-exposure-completion.v1' or completion['request_identity'] != request_identity \
            or completion['cell'] != cell_name(seed) or read(root/'request.json') != request_raw:
        raise ContractError('exposure completion context differs')
    exact_keys(completion['artifacts'], names(), 'exposure artifact inventory')
    required = names() | {'request.json', 'completion.json'}
    actual = set(before['members']) - {'.'}
    if not required <= actual or not actual <= required | {'worker.log', 'attempt.json'}:
        raise ContractError('exposure physical inventory differs')
    bound = {name: read(root/name) for name in names()}
    if any(identity(data) != completion['artifacts'][name] for name, data in bound.items()):
        raise ContractError('exposure bound artifact bytes differ')
    report = parse_json(bound['report.json'], canonical=True)
    exact_keys(report, {'schema', 'seed', 'request_identity', 'admission_identity', 'model_identity',
        'verification_scope', 'step', 'target_tokens', 'observations', 'observations_contrasts',
        'training_fit', 'update_seconds_hex', *CLAIMS}, 'exposure report')
    if report['schema'] != 'trinite.music-exposure-report.v1' or type(report['seed']) is not int or report['seed'] != seed \
            or report['request_identity'] != request_identity or report['admission_identity'] != request['admission_identity'] \
            or report['verification_scope'] != SCOPE or any(report[k] is not False for k in CLAIMS):
        raise ContractError('exposure report context or claims differ')
    exact_keys(report['observations'], {str(s) for s in CHECKPOINTS}, 'exposure observations')
    exact_keys(report['observations_contrasts'], {str(s) for s in CHECKPOINTS}, 'exposure contrast observations')
    from .numerical import check_score, verify_checkpoint
    from .reporting import summarize
    previous = None
    scores = {}
    for step in CHECKPOINTS:
        prefix = f'step-{step:03d}'
        data = bound[prefix+'-predictions.json']
        scores[step] = check_score(data)
        predictions = parse_json(data, canonical=True)['predictions']
        if bound[prefix+'-journal.jsonl'] != b''.join(json_bytes(p) for p in predictions):
            raise ContractError('prediction journal differs from retained complete predictions')
        metadata = parse_json(bound[prefix+'-checkpoint.json'], canonical=True)
        state = verify_checkpoint(bound[prefix+'.safetensors'], metadata, seed,
                                  request['admission_identity'], request_identity)
        if type(metadata['step']) is not int or metadata['step'] != step or state.cursor != step*8 or state.target_tokens != step*16:
            raise ContractError('exposure checkpoint progress differs')
        if previous is not None and (metadata['initial_train_loss_hex'] != previous['initial_train_loss_hex']
                or metadata['history'][:len(previous['history'])] != previous['history']
                or metadata['validation'][:len(previous['validation'])] != previous['validation']):
            raise ContractError('exposure checkpoint history or validation lineage differs')
        previous = metadata
        if json_bytes(report['observations'][str(step)]) != json_bytes(scores[step]) \
                or json_bytes(report['observations_contrasts'][str(step)]) != json_bytes(summarize(data)):
            raise ContractError('exposure report observation differs')
    if type(report['step']) is not int or report['step'] != 640 or type(report['target_tokens']) is not int or report['target_tokens'] != 10240 \
            or report['model_identity'] != previous['model_identity']:
        raise ContractError('exposure final report progress differs')
    finite(report['update_seconds_hex'], PROTOCOL['max_update_seconds_per_seed'])
    fit = scores[640]['correct'] == 40 and scores[640]['examples'] == 40
    if report['training_fit'] is not fit:
        raise ContractError('exposure training fit differs from raw whole answers')
    unchanged(root, before)
    return {'verified': True, 'report': report, 'completion_identity': expected_identity,
            'verification_scope': SCOPE, 'predictions_replayed': False, 'optimizer_replayed': False}
