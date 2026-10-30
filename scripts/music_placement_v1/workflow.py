"""Bounded retained cell/development packets; audits never replay computation."""
import math
import os
from pathlib import Path

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json
from native_diagnostic_v1.artifacts import Store
from frozen_music_eval_v1.artifacts import inventory
from .request import read, validate
from .spec import CELLS, PROTOCOL, cell_name
from .reporting import score_cell, matrix_summary

CLAIMS = {'fresh_confirmation': False, 'learning_adequate': False,
          'phase5_ready': False, 'release_ready': False}
SCOPE = 'source-bound bytes, safe tensor/progress structure and retained scores; no inference or optimizer replay'
CELL_FILES = {'final.safetensors', 'final-checkpoint.json', 'diagnostic-predictions.json', 'diagnostic-predictions.jsonl', 'report.json'}


def finite(value, maximum=None):
    try:
        number = float.fromhex(value)
    except (TypeError, ValueError) as error:
        raise ContractError('invalid timing') from error
    if not math.isfinite(number) or number < 0 or number.hex() != value or (maximum is not None and number > maximum):
        raise ContractError('timing outside fixed budget')
    return number


def bounded(root):
    files = inventory(root, max_files=128, max_bytes=128*1024*1024)
    for name in files:
        limit = PROTOCOL['max_artifact_bytes']
        if Path(name).name == 'request.json':
            limit = PROTOCOL['max_request_bytes']
        if name.endswith('.log') or name.endswith('.jsonl'):
            limit = PROTOCOL['max_journal_bytes']
        if name.endswith('.safetensors'):
            limit = PROTOCOL['max_checkpoint_bytes']
        read(Path(root)/name, limit)
    return files


def records(raw):
    return [parse_json(line, canonical=True) for line in raw.splitlines(keepends=True)]


class Journal:
    def __init__(self, store, name):
        store.check()
        self.store, self.count, self.bytes = store, 0, 0
        fd = os.open(name, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600, dir_fd=store.fd)
        self.stream = os.fdopen(fd, 'wb')
    def emit(self, record):
        raw = json_bytes(record)
        if self.bytes + len(raw) > 1024*1024:
            raise ContractError('journal exceeds fixed byte bound')
        self.store.check()
        self.stream.write(raw); self.stream.flush(); os.fsync(self.stream.fileno())
        self.bytes += len(raw); self.count += 1
        self.store.check()
    def close(self):
        self.stream.close()


def train_worker(root, raw, request_id, arm, seed, expected_directory):
    request = validate(raw, request_id)
    name = cell_name(arm, seed)
    if set(bounded(root)) - {'worker.log'}:
        raise ContractError('training worker requires fresh directory')
    store = Store(root, expected_directory)
    journal = Journal(store, 'diagnostic-predictions.jsonl')
    try:
        store.write('request.json', raw)
        from . import numerical
        report = numerical.train(arm, seed, request_id, request['admission_identity'], store.write, journal.emit)
        journal.close()
        observed = records(read(Path(root)/'diagnostic-predictions.jsonl', 1024*1024))
        summary = score_cell(observed, arm, seed, 'diagnostic')
        report = {**report, 'schema': 'trinite.music-placement-cell.v1', 'cell': name,
                  'diagnostics': summary, 'verification_scope': SCOPE, **CLAIMS}
        store.write('report.json', json_bytes(report))
        validate(raw, request_id)
        files = bounded(root)
        if set(files) != CELL_FILES | {'worker.log', 'request.json'}:
            raise ContractError('training completed inventory differs')
        completion = {'schema': 'trinite.music-placement-completion.v1', 'request_identity': request_id,
                      'cell': name, 'artifacts': {n: files[n] for n in sorted(CELL_FILES)}}
        result = json_bytes(completion)
        store.write('completion.json', result)
        return identity(result)
    finally:
        journal.close(); store.close()


def audit_cell(root, raw, request_id, arm, seed, expected_identity, *, load=False):
    before = bounded(root)
    request = validate(raw, request_id)
    name = cell_name(arm, seed)
    completion_raw = read(Path(root)/'completion.json')
    if identity(completion_raw) != expected_identity:
        raise ContractError('independent cell completion identity differs')
    completion = parse_json(completion_raw, canonical=True)
    expected = {'schema': 'trinite.music-placement-completion.v1', 'request_identity': request_id,
                'cell': name, 'artifacts': {n: before[n] for n in sorted(CELL_FILES)}}
    if json_bytes(completion) != json_bytes(expected):
        raise ContractError('cell completion/artifact binding differs')
    if not CELL_FILES | {'request.json', 'completion.json'} <= set(before) <= CELL_FILES | {'request.json', 'completion.json', 'worker.log', 'attempt.json'}:
        raise ContractError('cell membership differs')
    if read(Path(root)/'request.json', 4*1024*1024) != raw:
        raise ContractError('cell request differs')
    from . import numerical
    meta = parse_json(read(Path(root)/'final-checkpoint.json'), canonical=True)
    model, restored = numerical.verify_checkpoint(read(Path(root)/'final.safetensors', 16*1024*1024),
                                        read(Path(root)/'final-checkpoint.json'), arm, seed, request['admission_identity'], request_id)
    if restored != meta or meta['step'] != 640:
        raise ContractError('only prescribed final checkpoint admitted')
    journal = records(read(Path(root)/'diagnostic-predictions.jsonl', 1024*1024))
    summary = score_cell(journal, arm, seed, 'diagnostic')
    if read(Path(root)/'diagnostic-predictions.json') != json_bytes(journal):
        raise ContractError('diagnostic predictions/journal differ')
    report = parse_json(read(Path(root)/'report.json'), canonical=True)
    exact_keys(report, {'arm','seed','request_identity','admission_identity','step','target_tokens',
        'initialization_identity','model_identity','update_seconds_hex','diagnostic_summary','diagnostic_decodes',
        'diagnostic_forward_calls','schema','cell','diagnostics','verification_scope',*CLAIMS}, 'cell report')
    if (report.get('schema') != 'trinite.music-placement-cell.v1' or report.get('cell') != name
            or report.get('request_identity') != request_id or report['arm'] != arm or type(report['seed']) is not int or report['seed'] != seed
            or report['admission_identity'] != request['admission_identity']
            or json_bytes(report['diagnostic_summary']) != json_bytes(summary)
            or type(report['diagnostic_decodes']) is not int or report['diagnostic_decodes'] != 80
            or type(report['diagnostic_forward_calls']) is not int
            or report['diagnostic_forward_calls'] != sum(len(r['generated_tokens']) for r in journal)
            or json_bytes(report.get('diagnostics')) != json_bytes(summary)
            or report.get('verification_scope') != SCOPE
            or any(report.get(k) is not False for k in CLAIMS)
            or report.get('model_identity') != meta['model_identity']
            or report.get('initialization_identity') != meta['initialization_identity']
            or type(report.get('step')) is not int or report['step'] != 640
            or type(report.get('target_tokens')) is not int or report['target_tokens'] != 10240):
        raise ContractError('retained cell report context/score differs')
    finite(report['update_seconds_hex'], 120)
    if bounded(root) != before:
        raise ContractError('cell mutated during audit')
    result = {'report': report, 'records': journal, 'metadata': meta, 'files': before}
    if load:
        result['model'] = model
    return result


def gate_value(request_id, completions):
    return {'schema': 'trinite.music-placement-development-gate.v1', 'request_identity': request_id,
            'cells': completions, 'selection': 'all four safe completed final640 cells; no training-fit gate'}


def admit_cells(root, raw, request_id, gate, *, load=False):
    exact_keys(gate, {'schema', 'request_identity', 'cells', 'selection'}, 'development gate')
    expected_names = {cell_name(a,s) for a,s in CELLS}
    exact_keys(gate['cells'], expected_names, 'all four fixed cells')
    if json_bytes(gate) != json_bytes(gate_value(request_id, gate['cells'])):
        raise ContractError('development gate context differs')
    cells = {}
    for arm, seed in CELLS:
        name = cell_name(arm, seed)
        cells[name] = audit_cell(Path(root)/name, raw, request_id, arm, seed, gate['cells'][name], load=load)
    for seed in (17,29):
        if cells[cell_name('control',seed)]['metadata']['initialization_identity'] != cells[cell_name('swap',seed)]['metadata']['initialization_identity']:
            raise ContractError('paired initial model identities differ')
    return cells


def generation_accounting(observed):
    forwards = sum(len(r['generated_tokens']) for r in observed)
    if len(observed) != 480 or not 480 <= forwards <= 30720:
        raise ContractError('development decode/forward budget differs')
    return {'decodes': 480, 'forward_calls': forwards,
            'model_identities_unchanged': True, 'training_performed': False}


def loaded_model_identity(model):
    from trinite.training import model_identity
    return model_identity(model)


def development_worker(root, output, raw, request_id, gate_id, expected_directory):
    validate(raw, request_id)
    gate_raw = read(Path(root)/'development-gate.json')
    if identity(gate_raw) != gate_id:
        raise ContractError('independent development gate identity differs')
    gate = parse_json(gate_raw, canonical=True)
    # Every checkpoint and diagnostic is admitted before the first development call.
    cells = admit_cells(root, raw, request_id, gate, load=True)
    before = {name: value['files'] for name,value in cells.items()}
    # Admission of a later cell must not redefine an earlier model/file baseline.
    # Recheck all externally bound identities immediately before any development.
    validate(raw, request_id)
    if read(Path(root)/'development-gate.json') != gate_raw:
        raise ContractError('development gate changed during admission')
    for name, value in cells.items():
        if bounded(Path(root)/name) != before[name]:
            raise ContractError('retained cell changed during admission')
        if loaded_model_identity(value['model']) != value['metadata']['model_identity']:
            raise ContractError('loaded model changed during admission')
    if set(bounded(output)) - {'worker.log'}:
        raise ContractError('development worker requires fresh directory')
    store = Store(output, expected_directory)
    journal = Journal(store, 'predictions.jsonl')
    from . import numerical
    try:
        generation = numerical.infer_models({(a,s): cells[cell_name(a,s)]['model'] for a,s in CELLS}, 'development', journal.emit)
        for name in cells:
            if bounded(Path(root)/name) != before[name]:
                raise ContractError('development changed retained input bytes')
        journal.close()
        observed = records(read(Path(output)/'predictions.jsonl', 1024*1024))
        diagnostics = [r for a,s in CELLS for r in cells[cell_name(a,s)]['records']]
        summary = matrix_summary(diagnostics, observed)
        accounting = generation_accounting(observed)
        if json_bytes(generation) != json_bytes(dict(accounting, records=observed)):
            raise ContractError('development native generation accounting differs')
        validate(raw, request_id)
        report = {'schema': 'trinite.music-placement-development.v1', 'request_identity': request_id,
                  'gate_identity': gate_id, 'summary': summary, 'generation': accounting, 'input_files_unchanged': True,
                  'model_states_unchanged': True, 'verification_scope': SCOPE, **CLAIMS}
        store.write('report.json', json_bytes(report))
        completion = {'schema': 'trinite.music-placement-development-completion.v1',
                      'request_identity': request_id, 'gate_identity': gate_id,
                      'artifacts': {n: identity(read(Path(output)/n)) for n in ('predictions.jsonl','report.json')}}
        result = json_bytes(completion); store.write('completion.json', result)
        return identity(result)
    finally:
        journal.close(); store.close()


def audit_development(root, raw, request_id, gate_id, expected_identity):
    before = bounded(root)
    gate_raw = read(Path(root)/'development-gate.json')
    if identity(gate_raw) != gate_id:
        raise ContractError('gate identity differs')
    cells = admit_cells(root, raw, request_id, parse_json(gate_raw, canonical=True))
    output = Path(root)/'development'
    files = bounded(output)
    if set(files) != {'worker.log','predictions.jsonl','report.json','completion.json','attempt.json'}:
        raise ContractError('development inventory differs')
    completion_raw = read(output/'completion.json')
    if identity(completion_raw) != expected_identity:
        raise ContractError('development completion identity differs')
    completion = {'schema': 'trinite.music-placement-development-completion.v1',
                  'request_identity': request_id, 'gate_identity': gate_id,
                  'artifacts': {n: files[n] for n in ('predictions.jsonl','report.json')}}
    if completion_raw != json_bytes(completion):
        raise ContractError('development artifact binding differs')
    diagnostics = [r for a,s in CELLS for r in cells[cell_name(a,s)]['records']]
    observed = records(read(output/'predictions.jsonl', 1024*1024))
    summary = matrix_summary(diagnostics, observed)
    accounting = generation_accounting(observed)
    report = {'schema': 'trinite.music-placement-development.v1', 'request_identity': request_id,
              'gate_identity': gate_id, 'summary': summary, 'generation': accounting, 'input_files_unchanged': True,
              'model_states_unchanged': True, 'verification_scope': SCOPE, **CLAIMS}
    if read(output/'report.json') != json_bytes(report):
        raise ContractError('development report differs')
    if bounded(root) != before:
        raise ContractError('matrix changed during development audit')
    return report
