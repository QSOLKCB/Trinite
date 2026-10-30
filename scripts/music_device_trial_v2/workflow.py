"""Bounded retained cell/development packets; audits never replay computation."""
import math
import os
from pathlib import Path

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json
from native_diagnostic_v1.artifacts import Store
from frozen_music_eval_v1.artifacts import inventory
from .request import read, validate
from .spec import BACKENDS, PROTOCOL, TIMING_PHASES, backend_name, score_cell, matrix_summary

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


def probe_worker(root, raw, request_id, expected_directory):
    validate(raw, request_id)
    if set(bounded(root))-{'worker.log'}:raise ContractError('fresh probe directory required')
    store=Store(root,expected_directory)
    try:
        from .numerical import parity_probe, compare_probes
        payload,report=parity_probe()
        comparison=compare_probes(payload,report)
        store.write('probe.safetensors',payload)
        store.write('report.json',json_bytes(report))
        store.write('comparison.json',json_bytes(comparison))
        validate(raw,request_id)
        completion={'schema':'trinite.music-device-probe-completion.v2','request_identity':request_id,
            'artifacts':{n:identity(read(Path(root)/n,16777216)) for n in ('probe.safetensors','report.json','comparison.json')}}
        store.write('completion.json',json_bytes(completion))
        return identity(json_bytes(completion))
    finally:store.close()


def audit_probe(root,raw,request_id,expected_id):
    before=bounded(root);validate(raw,request_id)
    if set(before)-{'worker.log','attempt.json'} != {'probe.safetensors','report.json','comparison.json','completion.json'}:
        raise ContractError('probe inventory differs')
    completion_raw=read(Path(root)/'completion.json')
    expected={'schema':'trinite.music-device-probe-completion.v2','request_identity':request_id,
        'artifacts':{n:before[n] for n in ('probe.safetensors','report.json','comparison.json')}}
    if identity(completion_raw)!=expected_id or completion_raw!=json_bytes(expected):
        raise ContractError('probe completion differs')
    from .numerical import compare_probes
    comparison=compare_probes(read(Path(root)/'probe.safetensors',16777216),parse_json(read(Path(root)/'report.json'),canonical=True))
    if read(Path(root)/'comparison.json')!=json_bytes(comparison):raise ContractError('probe comparison differs')
    if type(comparison.get('passed')) is not bool:raise ContractError('probe requires explicit parity outcome')
    if bounded(root)!=before:raise ContractError('probe mutated during audit')
    return comparison


def train_worker(root,raw,request_id,backend,expected_directory,probe_id):
    request=validate(raw,request_id);backend_name(backend)
    if Path(root).name!=backend:raise ContractError('fixed backend child directory required')
    if not audit_probe(Path(root).parent/'probe',raw,request_id,probe_id)['passed']:
        raise ContractError('parity gate must pass before target training')
    if set(bounded(root))-{'worker.log'}:raise ContractError('fresh training directory required')
    store=Store(root,expected_directory);journal=Journal(store,'diagnostic-predictions.jsonl')
    try:
        store.write('request.json',raw)
        from . import numerical
        report=numerical.train(backend,request_id,request['admission_identity'],store.write,journal.emit)
        journal.close()
        observed=records(read(Path(root)/'diagnostic-predictions.jsonl',1048576))
        summary=score_cell(observed,backend,'diagnostic')
        report={**report,'schema':'trinite.music-device-cell.v2','diagnostics':summary,'verification_scope':SCOPE,**CLAIMS}
        store.write('report.json',json_bytes(report));validate(raw,request_id)
        files=bounded(root)
        if set(files)!=CELL_FILES|{'worker.log','request.json'}:raise ContractError('training inventory differs')
        completion={'schema':'trinite.music-device-completion.v2','request_identity':request_id,'backend':backend,
            'artifacts':{n:files[n] for n in sorted(CELL_FILES)}}
        store.write('completion.json',json_bytes(completion))
        return identity(json_bytes(completion))
    finally:journal.close();store.close()


def audit_cell(root,raw,request_id,backend,expected_id,*,load=False):
    backend_name(backend);before=bounded(root);request=validate(raw,request_id)
    if not CELL_FILES|{'request.json','completion.json'} <= set(before) <= CELL_FILES|{'request.json','completion.json','worker.log','attempt.json'}:
        raise ContractError('cell membership differs')
    if read(Path(root)/'request.json',4194304)!=raw:raise ContractError('cell request differs')
    completion_raw=read(Path(root)/'completion.json')
    expected={'schema':'trinite.music-device-completion.v2','request_identity':request_id,'backend':backend,
        'artifacts':{n:before[n] for n in sorted(CELL_FILES)}}
    if identity(completion_raw)!=expected_id or completion_raw!=json_bytes(expected):raise ContractError('cell completion differs')
    from . import numerical
    metadata_raw=read(Path(root)/'final-checkpoint.json')
    meta=parse_json(metadata_raw,canonical=True)
    model=numerical.load_checkpoint(read(Path(root)/'final.safetensors',16777216),metadata_raw,
        backend,request['admission_identity'],request_id)
    if meta['step']!=640:raise ContractError('only fixed final checkpoint admitted')
    observed=records(read(Path(root)/'diagnostic-predictions.jsonl',1048576))
    summary=score_cell(observed,backend,'diagnostic')
    if read(Path(root)/'diagnostic-predictions.json')!=json_bytes(observed):raise ContractError('diagnostic journal differs')
    report=parse_json(read(Path(root)/'report.json'),canonical=True)
    exact_keys(report,{'backend','seed','request_identity','admission_identity','step','target_tokens','initialization_identity',
        'model_identity','update_seconds_hex','timing_hex','diagnostic_greedy_seconds_hex','diagnostic_decodes','diagnostic_forward_calls','schema','diagnostics','verification_scope',*CLAIMS},'cell report')
    expected_context={'backend':backend,'seed':17,'request_identity':request_id,'admission_identity':request['admission_identity'],
        'step':640,'target_tokens':10240,'initialization_identity':meta['initialization_identity'],'model_identity':meta['model_identity'],
        'diagnostic_decodes':80,'diagnostic_forward_calls':sum(len(r['generated_tokens']) for r in observed),
        'schema':'trinite.music-device-cell.v2','diagnostics':summary,'verification_scope':SCOPE,**CLAIMS}
    if json_bytes({k:report[k] for k in expected_context})!=json_bytes(expected_context):raise ContractError('cell report context/score differs')
    finite(report['update_seconds_hex'],120)
    if finite(report['diagnostic_greedy_seconds_hex']) > finite(report['timing_hex']['diagnostics']):
        raise ContractError('greedy subcomponent exceeds diagnostic pipeline')
    exact_keys(report['timing_hex'],set(TIMING_PHASES),'timings')
    for value in report['timing_hex'].values():finite(value)
    if any(report['timing_hex'][k] != (0.).hex() for k in ('compile','target_warmup')):
        raise ContractError('compile and extra target warmup forbidden')
    if report['timing_hex']['updates']!=report['update_seconds_hex']:raise ContractError('update timing differs')
    if bounded(root)!=before:raise ContractError('cell changed during audit')
    result={'report':report,'metadata':meta,'records':observed,'files':before}
    if load:result['model']=model
    return result


def gate_value(request_id,completions,probe_id):
    return {'schema':'trinite.music-device-development-gate.v2','request_identity':request_id,'cells':completions,
        'probe_completion_identity':probe_id,'selection':'both safe final640 cells; no accuracy fit gate'}


def admit_cells(root,raw,request_id,gate,*,load=False):
    exact_keys(gate,{'schema','request_identity','cells','probe_completion_identity','selection'},'gate')
    exact_keys(gate['cells'],set(BACKENDS),'fixed backends')
    if gate!=gate_value(request_id,gate['cells'],gate['probe_completion_identity']):raise ContractError('gate context differs')
    if not audit_probe(Path(root)/'probe',raw,request_id,gate['probe_completion_identity'])['passed']:
        raise ContractError('failed parity cannot admit target outputs')
    cells={b:audit_cell(Path(root)/b,raw,request_id,b,gate['cells'][b],load=load) for b in BACKENDS}
    if cells['cpu']['metadata']['initialization_identity']!=cells['cuda']['metadata']['initialization_identity']:
        raise ContractError('CPU-initialized identities differ')
    return cells


def generation_accounting(observed):
    forwards=sum(len(r['generated_tokens']) for r in observed)
    if len(observed)!=240 or not 240<=forwards<=15360:raise ContractError('development budget differs')
    return {'decodes':240,'forward_calls':forwards,'model_identities_unchanged':True,'training_performed':False}


def loaded_model_identity(model):
    from .numerical import model_identity
    return model_identity(model)


def development_worker(root,output,raw,request_id,gate_id,expected_directory):
    validate(raw,request_id);gate_raw=read(Path(root)/'development-gate.json')
    if identity(gate_raw)!=gate_id:raise ContractError('independent gate identity differs')
    gate=parse_json(gate_raw,canonical=True);probe_before=bounded(Path(root)/'probe')
    cells=admit_cells(root,raw,request_id,gate,load=True)
    before={b:cells[b]['files'] for b in BACKENDS}
    validate(raw,request_id)
    if read(Path(root)/'development-gate.json')!=gate_raw or bounded(Path(root)/'probe')!=probe_before:
        raise ContractError('gate or parity packet changed during admission')
    for b in BACKENDS:
        if bounded(Path(root)/b)!=before[b] or loaded_model_identity(cells[b]['model'])!=cells[b]['metadata']['model_identity']:
            raise ContractError('earlier admitted cell changed before development')
    if set(bounded(output))-{'worker.log'}:raise ContractError('fresh development directory required')
    store=Store(output,expected_directory);journal=Journal(store,'predictions.jsonl')
    try:
        from . import numerical
        generation=numerical.infer_models({b:cells[b]['model'] for b in BACKENDS},'development',journal.emit)
        journal.close();observed=records(read(Path(output)/'predictions.jsonl',1048576))
        summary=matrix_summary([r for b in BACKENDS for r in cells[b]['records']],observed)
        accounting=generation_accounting(observed)
        exact_keys(generation,set(accounting)|{'records','timing_hex','greedy_seconds_hex'},'generation')
        if json_bytes({k:generation[k] for k in accounting}|{'records':generation['records']})!=json_bytes(accounting|{'records':observed}):
            raise ContractError('native generation accounting differs')
        exact_keys(generation['timing_hex'],set(BACKENDS),'development timings')
        for value in generation['timing_hex'].values():finite(value)
        exact_keys(generation['greedy_seconds_hex'],set(BACKENDS),'greedy subcomponent timings')
        for backend in BACKENDS:
            if finite(generation['greedy_seconds_hex'][backend])>finite(generation['timing_hex'][backend]):
                raise ContractError('greedy subcomponent exceeds development pipeline')
        for b in BACKENDS:
            if bounded(Path(root)/b)!=before[b]:raise ContractError('development changed retained cells')
        validate(raw,request_id)
        if bounded(Path(root)/'probe')!=probe_before or read(Path(root)/'development-gate.json')!=gate_raw:
            raise ContractError('development changed parity/gate inputs')
        report={'schema':'trinite.music-device-development.v2','request_identity':request_id,'gate_identity':gate_id,
            'summary':summary,'generation':accounting,'timing_hex':generation['timing_hex'],'greedy_seconds_hex':generation['greedy_seconds_hex'],
            'input_files_unchanged':True,'model_states_unchanged':True,'verification_scope':SCOPE,**CLAIMS}
        store.write('report.json',json_bytes(report))
        completion={'schema':'trinite.music-device-development-completion.v2','request_identity':request_id,'gate_identity':gate_id,
            'artifacts':{n:identity(read(Path(output)/n)) for n in ('predictions.jsonl','report.json')}}
        store.write('completion.json',json_bytes(completion));return identity(json_bytes(completion))
    finally:journal.close();store.close()


def audit_development(root,raw,request_id,gate_id,expected_id):
    before=bounded(root);gate_raw=read(Path(root)/'development-gate.json')
    if identity(gate_raw)!=gate_id:raise ContractError('gate identity differs')
    cells=admit_cells(root,raw,request_id,parse_json(gate_raw,canonical=True))
    output=Path(root)/'development';files=bounded(output)
    if set(files)!={'worker.log','predictions.jsonl','report.json','completion.json','attempt.json'}:raise ContractError('development membership differs')
    completion_raw=read(output/'completion.json')
    expected={'schema':'trinite.music-device-development-completion.v2','request_identity':request_id,'gate_identity':gate_id,
        'artifacts':{n:files[n] for n in ('predictions.jsonl','report.json')}}
    if identity(completion_raw)!=expected_id or completion_raw!=json_bytes(expected):raise ContractError('development completion differs')
    observed=records(read(output/'predictions.jsonl',1048576))
    summary=matrix_summary([r for b in BACKENDS for r in cells[b]['records']],observed)
    report=parse_json(read(output/'report.json'),canonical=True)
    exact_keys(report,{'schema','request_identity','gate_identity','summary','generation','timing_hex','greedy_seconds_hex','input_files_unchanged','model_states_unchanged','verification_scope',*CLAIMS},'development report')
    exact_keys(report['timing_hex'],set(BACKENDS),'development timings')
    for value in report['timing_hex'].values():finite(value)
    exact_keys(report['greedy_seconds_hex'],set(BACKENDS),'greedy subcomponent timings')
    for backend in BACKENDS:
        if finite(report['greedy_seconds_hex'][backend])>finite(report['timing_hex'][backend]):
            raise ContractError('greedy subcomponent exceeds development pipeline')
    expected={'schema':'trinite.music-device-development.v2','request_identity':request_id,'gate_identity':gate_id,
        'summary':summary,'generation':generation_accounting(observed),'timing_hex':report['timing_hex'],'greedy_seconds_hex':report['greedy_seconds_hex'],
        'input_files_unchanged':True,'model_states_unchanged':True,'verification_scope':SCOPE,**CLAIMS}
    if json_bytes(report)!=json_bytes(expected):raise ContractError('development retained score differs')
    if bounded(root)!=before:raise ContractError('matrix changed during audit')
    return report
