"""Retained cell custody and two-seed gate; no numerical replay in audit."""
import math
import os
from pathlib import Path
import time
from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from native_diagnostic_v1.artifacts import Store
from unseen_evaluation_v1.packet import inventory, unchanged
from .request import read, validate, cell_name
from .spec import PROTOCOL, SEEDS

SCOPE = 'source-bound retained bytes, safe native tensor structure and score consistency; no inference or optimizer replay'
CLAIMS = {'fresh_confirmation': False, 'learning_adequate': False,
          'phase5_ready': False, 'release_ready': False}


def names(phase):
    if phase == 'train':
        return {'report.json'} | {f'step-{step:03d}' + suffix for step in (0, 64)
            for suffix in ('.safetensors', '-checkpoint.json', '-predictions.json', '-journal.jsonl')}
    if phase == 'heldout':
        return {'report.json', 'heldout-predictions.json', 'heldout-journal.jsonl'}
    raise ContractError('unknown contrast artifact phase')


def finite(value, maximum=None):
    try:
        number = float.fromhex(value)
    except (TypeError, ValueError) as error:
        raise ContractError('invalid contrast timing') from error
    if not math.isfinite(number) or number < 0 or number.hex() != value or (maximum is not None and number > maximum):
        raise ContractError('invalid contrast timing')
    return number


def require_gate(root, request_raw, request_identity, expected_identity):
    """Recheck both bound successful training cells before any held-out forward."""
    require_identity(expected_identity)
    raw = read(Path(root)/'gate.json', 1024*1024)
    if identity(raw) != expected_identity:
        raise ContractError('training gate identity differs')
    gate = parse_json(raw, canonical=True)
    exact_keys(gate, {'schema', 'request_identity', 'passed', 'training_completions'}, 'contrast gate')
    if gate['schema'] != 'trinite.music-contrast-gate.v1' or gate['request_identity'] != request_identity or gate['passed'] is not True:
        raise ContractError('both-seed training gate has not passed')
    exact_keys(gate['training_completions'], {cell_name(s, 'train') for s in SEEDS}, 'gate training cells')
    reports = {}
    for seed in SEEDS:
        name = cell_name(seed, 'train')
        report = audit(Path(root)/name, request_raw, request_identity, seed, 'train', gate['training_completions'][name])['report']
        if report['training_fit'] is not True:
            raise ContractError('heldout forbidden: both seeds must fit every training row')
        reports[name] = report
    if read(Path(root)/'gate.json', 1024*1024) != raw:
        raise ContractError('training gate changed while validating')
    return gate, reports


def bound_checkpoint(root, expected_completion):
    """Read only bytes committed by the gate-bound completion, even after a swap."""
    raw = read(Path(root)/'completion.json')
    if identity(raw) != expected_completion:
        raise ContractError('gate-bound training completion changed before checkpoint load')
    completion = parse_json(raw, canonical=True)
    payload = read(Path(root)/'step-064.safetensors')
    metadata_raw = read(Path(root)/'step-064-checkpoint.json')
    if (identity(payload) != completion['artifacts']['step-064.safetensors'] or
            identity(metadata_raw) != completion['artifacts']['step-064-checkpoint.json']):
        raise ContractError('gate-bound checkpoint bytes changed before heldout inference')
    return payload, parse_json(metadata_raw, canonical=True)


def run(root, request_raw, request_identity, seed, phase, expected_directory, gate_identity=None):
    request = validate(request_raw, request_identity)
    root = Path(root)
    before = inventory(root)
    if set(before['members']) - {'.', 'worker.log'}:
        raise ContractError('contrast worker requires fresh controlled directory')
    gate = None
    if phase == 'heldout':
        gate, _ = require_gate(root.parent, request_raw, request_identity, gate_identity)
    store = Store(root, expected_directory)
    outputs, journals = {}, {}
    started = time.monotonic()
    try:
        store.write('request.json', request_raw)
        def write(name, raw):
            if time.monotonic()-started > PROTOCOL['worker_seconds']:
                raise ContractError('contrast worker deadline exceeded')
            store.write(name, raw)
            outputs[name] = identity(raw)
        def emit(prefix, record):
            permitted = ('step-000', 'step-064') if phase == 'train' else ('heldout',)
            if prefix not in permitted:
                raise ContractError('unknown contrast prediction journal')
            name = prefix+'-journal.jsonl'
            if name not in journals:
                store.check()
                fd = os.open(name, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600, dir_fd=store.fd)
                journals[name] = [os.fdopen(fd, 'wb'), 0]
            journal, count = journals[name]
            raw = json_bytes(record)
            if count+len(raw) > 1024*1024:
                raise ContractError('contrast journal exceeds bound')
            journal.write(raw); journal.flush(); os.fsync(journal.fileno())
            journals[name][1] += len(raw)
            store.check()
        from . import numerical
        if phase == 'train':
            report = numerical.train(seed, request_identity, request['admission_identity'], write, emit=emit)
        elif phase == 'heldout':
            source = root.parent/cell_name(seed, 'train')
            payload, metadata = bound_checkpoint(source, gate['training_completions'][cell_name(seed, 'train')])
            # In-memory bytes are checked against the independently gate-bound completion.
            report = numerical.holdout(seed, request_identity, request['admission_identity'], payload, metadata, write, emit=emit)
            report['gate_identity'] = gate_identity
        else:
            raise ContractError('unknown contrast worker phase')
        for name, (stream, _) in journals.items():
            stream.close()
            outputs[name] = identity(read(root/name, 1024*1024))
        from .reporting import summarize
        if phase == 'train':
            report['initial_contrasts'] = summarize(read(root/'step-000-predictions.json'), 'train')
            report['final_contrasts'] = summarize(read(root/'step-064-predictions.json'), 'train')
        else:
            report['contrasts'] = summarize(read(root/'heldout-predictions.json'), 'heldout')
        report.update(schema='trinite.music-contrast-report.v1', phase=phase,
                      verification_scope=SCOPE, **CLAIMS)
        write('report.json', json_bytes(report))
        if set(outputs) != names(phase):
            raise ContractError('contrast completed artifact membership differs')
        validate(request_raw, request_identity)
        completion = {'schema': 'trinite.music-contrast-completion.v1',
            'request_identity': request_identity, 'cell': cell_name(seed, phase), 'artifacts': outputs}
        raw = json_bytes(completion); store.write('completion.json', raw)
        return identity(raw)
    finally:
        for stream, _ in journals.values():
            if not stream.closed:
                stream.close()
        store.close()


def audit(root, request_raw, request_identity, seed, phase, expected_identity, gate_identity=None):
    root = Path(root)
    before = inventory(root)
    if before['bytes'] > 32*1024*1024:
        raise ContractError('contrast cell byte limit exceeded')
    if 'worker.log' in before['members']:
        read(root/'worker.log', PROTOCOL['max_worker_log_bytes'])
    request = validate(request_raw, request_identity)
    require_identity(expected_identity)
    raw = read(root/'completion.json')
    if identity(raw) != expected_identity:
        raise ContractError('contrast completion identity differs')
    completion = parse_json(raw, canonical=True)
    exact_keys(completion, {'schema', 'request_identity', 'cell', 'artifacts'}, 'contrast completion')
    if completion['schema'] != 'trinite.music-contrast-completion.v1' or completion['request_identity'] != request_identity \
            or completion['cell'] != cell_name(seed, phase) or read(root/'request.json') != request_raw:
        raise ContractError('contrast completion context differs')
    exact_keys(completion['artifacts'], names(phase), 'contrast artifact inventory')
    required = names(phase) | {'request.json', 'completion.json'}
    actual = set(before['members']) - {'.'}
    if not required <= actual or not actual <= required | {'worker.log', 'attempt.json'}:
        raise ContractError('contrast physical inventory differs')
    bound = {name: read(root/name) for name in names(phase)}
    if any(identity(data) != completion['artifacts'][name] for name, data in bound.items()):
        raise ContractError('contrast bound artifact bytes differ')
    report = parse_json(bound['report.json'], canonical=True)
    base = {'schema', 'seed', 'phase', 'request_identity', 'admission_identity',
            'model_identity', 'verification_scope', *CLAIMS}
    expected_keys = base | ({'step', 'target_tokens', 'initial_score', 'final_score', 'training_fit',
                'update_seconds_hex', 'initial_contrasts', 'final_contrasts'} if phase == 'train'
                else {'score', 'gate_identity', 'contrasts'})
    exact_keys(report, expected_keys, 'contrast report')
    if report['schema'] != 'trinite.music-contrast-report.v1' or type(report['seed']) is not int or report['seed'] != seed \
            or report['phase'] != phase or report['request_identity'] != request_identity \
            or report['admission_identity'] != request['admission_identity'] or report['verification_scope'] != SCOPE \
            or any(report[k] is not False for k in CLAIMS):
        raise ContractError('contrast report context or claims differ')
    from .numerical import check_score, verify_checkpoint
    from .reporting import summarize
    prefixes = ('step-000', 'step-064') if phase == 'train' else ('heldout',)
    scores = {}
    for prefix in prefixes:
        data = bound[prefix+'-predictions.json']
        scores[prefix] = check_score(data, phase)
        predictions = parse_json(data, canonical=True)['predictions']
        if bound[prefix+'-journal.jsonl'] != b''.join(json_bytes(p) for p in predictions):
            raise ContractError('prediction journal differs from retained complete predictions')
    if phase == 'train':
        previous = None
        for step in (0, 64):
            prefix = f'step-{step:03d}'
            metadata = parse_json(bound[prefix+'-checkpoint.json'], canonical=True)
            state = verify_checkpoint(bound[prefix+'.safetensors'], metadata, seed,
                                      request['admission_identity'], request_identity)
            if type(metadata['step']) is not int or metadata['step'] != step or state.cursor != step*8 or state.target_tokens != step*16:
                raise ContractError('contrast checkpoint progress differs')
            if previous is not None and (metadata['initial_train_loss_hex'] != previous['initial_train_loss_hex']
                    or metadata['history'][:len(previous['history'])] != previous['history']):
                raise ContractError('contrast checkpoint history lineage differs')
            previous = metadata
        if type(report['step']) is not int or report['step'] != 64 or type(report['target_tokens']) is not int or report['target_tokens'] != 1024 \
                or report['model_identity'] != previous['model_identity']:
            raise ContractError('contrast final report progress differs')
        finite(report['update_seconds_hex'], PROTOCOL['max_update_seconds_per_seed'])
        for label, prefix in (('initial', 'step-000'), ('final', 'step-064')):
            if json_bytes(report[label+'_score']) != json_bytes(scores[prefix]) \
                    or json_bytes(report[label+'_contrasts']) != json_bytes(summarize(bound[prefix+'-predictions.json'], 'train')):
                raise ContractError('contrast report training score differs')
        fit = scores['step-064']['correct'] == 40 and scores['step-064']['examples'] == 40
        if report['training_fit'] is not fit:
            raise ContractError('contrast training fit differs from raw whole answers')
    else:
        if report['gate_identity'] != gate_identity:
            raise ContractError('heldout report gate identity differs')
        _, train_reports = require_gate(root.parent, request_raw, request_identity, gate_identity)
        if report['model_identity'] != train_reports[cell_name(seed, 'train')]['model_identity'] \
                or json_bytes(report['score']) != json_bytes(scores['heldout']) \
                or json_bytes(report['contrasts']) != json_bytes(summarize(bound['heldout-predictions.json'], 'heldout')):
            raise ContractError('heldout checkpoint or score differs')
    unchanged(root, before)
    return {'verified': True, 'report': report, 'completion_identity': expected_identity,
            'verification_scope': SCOPE, 'predictions_replayed': False, 'optimizer_replayed': False}
