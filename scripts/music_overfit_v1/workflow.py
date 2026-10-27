"""Source-bound byte custody and record consistency, never numerical replay.

The shared native reader validates checkpoint tensor structure by allocating a
seeded model and loading safe tensors. It never forwards or replays updates.
"""
from pathlib import Path
import math
import time
from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from native_diagnostic_v1.artifacts import Store
from unseen_evaluation_v1.packet import inventory, unchanged
from music_learning_v1.workflow import require_native_admission
from .request import ROOT, read, validate, cell_name
from .spec import subset_rows

STEPS = (0, 8, 16, 32, 64)
CLAIMS = {'fresh_confirmation': False, 'learning_adequate': False,
          'phase5_ready': False, 'release_ready': False}
SCOPE = 'source-bound byte custody, native safe tensor structure and record consistency; no numerical replay'


def artifact_names():
    return {'report.json'} | {f'step-{step:03d}' + suffix for step in STEPS
                            for suffix in ('.safetensors', '-checkpoint.json', '-diagnostics.json')}


def run(root, request_raw, request_identity, seed, subset, expected_directory):
    root = Path(root).absolute()
    if root.resolve().is_relative_to(ROOT) or ROOT.is_relative_to(root.resolve()):
        raise ContractError('overfit worker output overlaps source checkout')
    before = inventory(root)
    if set(before['members']) - {'.', 'worker.log'}:
        raise ContractError('overfit worker requires a fresh controller attempt')
    request = validate(request_raw, request_identity)
    cell = cell_name(seed, subset)
    require_native_admission(request['admission_identity'])
    store = Store(root, expected_directory)
    outputs = {}
    started = time.monotonic()
    try:
        store.write('request.json', request_raw)
        def write(name, data):
            if time.monotonic() - started > 900:
                raise ContractError('overfit worker deadline exceeded')
            store.write(name, data)
            outputs[name] = identity(data)
        from .numerical import run_trial
        report = run_trial(seed, subset, request['admission_identity'], request_identity, write)
        report.update(schema='trinite.music-overfit-report.v1', verification_scope=SCOPE, **CLAIMS)
        write('report.json', json_bytes(report))
        if set(outputs) != artifact_names():
            raise ContractError('overfit completed artifact inventory differs')
        validate(request_raw, request_identity)
        completion = {'schema':'trinite.music-overfit-completion.v1',
                      'request_identity':request_identity,'cell':cell,'artifacts':outputs}
        raw = json_bytes(completion)
        store.write('completion.json', raw)
        return identity(raw)
    finally:
        store.close()


def finite_hex(value, *, maximum=None):
    try:
        number = float.fromhex(value)
    except (TypeError, ValueError) as error:
        raise ContractError('invalid diagnostic numerical encoding') from error
    if not math.isfinite(number) or number < 0 or number.hex() != value or (maximum is not None and number > maximum):
        raise ContractError('invalid diagnostic numerical observation')
    return number


def audit(root, request_raw, request_identity, seed, subset, expected_identity):
    root = Path(root)
    before = inventory(root)
    if before['bytes'] > 32 * 1024 * 1024:
        raise ContractError('overfit retained attempt exceeds bound')
    request = validate(request_raw, request_identity)
    require_identity(expected_identity)
    raw = read(root/'completion.json')
    if identity(raw) != expected_identity:
        raise ContractError('overfit completion identity differs')
    completion = parse_json(raw, canonical=True)
    exact_keys(completion, {'schema','request_identity','cell','artifacts'}, 'overfit completion')
    if (completion['schema'] != 'trinite.music-overfit-completion.v1'
            or completion['request_identity'] != request_identity
            or completion['cell'] != cell_name(seed, subset)
            or read(root/'request.json') != request_raw):
        raise ContractError('overfit completion context differs')
    exact_keys(completion['artifacts'], artifact_names(), 'overfit completed artifacts')
    names = set(before['members']) - {'.'}
    required = artifact_names() | {'request.json','completion.json'}
    if not required <= names or not names <= required | {'worker.log','attempt.json'}:
        raise ContractError('overfit completed physical membership differs')
    bound = {name:read(root/name) for name in artifact_names()}
    if any(identity(data) != completion['artifacts'][name] for name,data in bound.items()):
        raise ContractError('overfit retained artifact identity differs')
    rows = subset_rows(subset)
    from music_learning_v1.numerical import encoded, plan_for
    train = encoded(rows)
    final = None
    initial = None
    initial_loss = None
    previous_history = []
    for step in STEPS:
        prefix = f'step-{step:03d}'
        meta = parse_json(bound[prefix+'-checkpoint.json'], canonical=True)
        keys = {'schema','subset','request_identity','plan','dataset_identity','admission_identity',
                'initialization_identity','initial_train_loss_hex','model_identity','step','cursor',
                'target_tokens','history','validation'}
        exact_keys(meta, keys, 'overfit checkpoint metadata')
        expected = {'schema':'trinite.music-overfit-checkpoint.v1','subset':subset,
                    'request_identity':request_identity,'plan':plan_for(seed).to_dict(),
                    'dataset_identity':request['dataset_identity'],
                    'admission_identity':request['admission_identity'],'step':step,'cursor':step*8}
        if any(json_bytes(meta[key]) != json_bytes(value) for key,value in expected.items()):
            raise ContractError('overfit checkpoint metadata context differs')
        require_identity(meta['model_identity']); require_identity(meta['initialization_identity'])
        finite_hex(meta['initial_train_loss_hex'])
        if initial is None:
            initial = meta['initialization_identity']
            initial_loss = meta['initial_train_loss_hex']
        if meta['initialization_identity'] != initial or (step == 0 and meta['model_identity'] != initial):
            raise ContractError('overfit initialization lineage differs')
        if meta['initial_train_loss_hex'] != initial_loss:
            raise ContractError('overfit initial loss lineage differs')
        if type(meta['history']) is not list or len(meta['history']) != step:
            raise ContractError('overfit history length differs')
        if meta['history'][:len(previous_history)] != previous_history:
            raise ContractError('overfit checkpoint history prefix differs')
        previous_history = meta['history']
        targets = 0
        for i, entry in enumerate(meta['history']):
            exact_keys(entry, {'step','loss_hex','gradient_norm_hex','target_tokens','examples'}, 'overfit history')
            batch = [train[(i*8+j) % len(train)] for j in range(8)]
            count = sum(sum(row['loss']) for row in batch)
            if (type(entry['step']) is not int or type(entry['target_tokens']) is not int
                    or entry['step'] != i+1 or entry['target_tokens'] != count
                    or entry['examples'] != [row['identity'] for row in batch]):
                raise ContractError('overfit schedule history differs')
            finite_hex(entry['loss_hex']); finite_hex(entry['gradient_norm_hex']); targets += count
        if type(meta['target_tokens']) is not int or meta['target_tokens'] != targets or targets > 32768:
            raise ContractError('overfit cumulative targets differ')
        expected_validation = [] if step < 64 else [64]
        if type(meta['validation']) is not list or len(meta['validation']) != len(expected_validation):
            raise ContractError('overfit train-only diagnostic history differs')
        for entry, at in zip(meta['validation'], expected_validation):
            exact_keys(entry, {'step','loss_hex'}, 'train-only diagnostic')
            if type(entry['step']) is not int or entry['step'] != at: raise ContractError('train-only diagnostic step differs')
            finite_hex(entry['loss_hex'])
        from .numerical import verify_checkpoint
        verify_checkpoint(bound[prefix+'.safetensors'], meta, seed, subset,
                          request['admission_identity'], request_identity)
        diagnostic = parse_json(bound[prefix+'-diagnostics.json'], canonical=True)
        if (diagnostic.get('schema') != 'trinite.music-overfit-diagnostics.v1'
                or diagnostic.get('step') != step or diagnostic.get('subset') != subset
                or diagnostic.get('model_identity') != meta['model_identity']
                or diagnostic.get('unique_train_examples') != len(rows)):
            raise ContractError('overfit diagnostic context differs')
        finite_hex(diagnostic['train_loss_hex'])
        if step == 0 and diagnostic['train_loss_hex'] != initial_loss:
            raise ContractError('initial diagnostic loss differs')
        if step == 64 and diagnostic['train_loss_hex'] != meta['validation'][0]['loss_hex']:
            raise ContractError('final train diagnostic loss differs')
        from .numerical import check_diagnostic
        check_diagnostic(diagnostic, rows)
        final = meta, diagnostic
    meta, diagnostic = final
    report = parse_json(bound['report.json'], canonical=True)
    expected = {'schema':'trinite.music-overfit-report.v1','seed':seed,'subset':subset,
                'outcome':'completed','step':64,'target_tokens':meta['target_tokens'],
                'request_identity':request_identity,'model_identity':meta['model_identity'],
                'final_train_exact':diagnostic['raw_score']['correct'], 'total_train':len(rows),
                'training_fit':diagnostic['raw_score']['correct'] == len(rows),
                'verification_scope':SCOPE, **CLAIMS}
    finite_hex(report.get('update_seconds_hex'), maximum=120)
    expected['update_seconds_hex'] = report['update_seconds_hex']
    if json_bytes(report) != json_bytes(expected):
        raise ContractError('overfit report differs from final diagnostic')
    validate(request_raw, request_identity)
    unchanged(root, before)
    return {'verified':True, 'report':report, 'optimizer_replayed':False,
            'predictions_replayed':False,'tensor_structure_verified':True, **CLAIMS}
