"""Read-only, model-only import of the fixed addition producer.

No optimizer is constructed and no forward is evaluated. This is structural
custody and retained-metric verification, not numerical or optimizer replay.
The original source/environment request remains authoritative and unchanged.
"""
from pathlib import Path
from types import SimpleNamespace

import torch

from trinite.comparison_checkpoint import load, _slot, _hex
from trinite.contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json, require_identity
from trinite.model import Decoder
from trinite.observation import verify_observation
from trinite.training import model_identity
from addition_development_v1.request import read, validate, cells, cell_name
from addition_development_v1.protocol import PROTOCOL, MILESTONES, STRATA, STEPS
from addition_development_v1.data import dataset, audit_bytes, schedule
from addition_development_v1.numerical import plan_for, encoded, check_score
from addition_development_v1.workflow import artifact_names, report_for
from addition_development_v1.controller import bound_files, matrix_layout
from unseen_evaluation_v1.packet import inventory, unchanged

BINDING_KEYS = {'producer_request_identity', 'producer_matrix_identity', 'model_identities',
                'producer_source', 'producer_environment', 'cells'}


def snapshot_matrix(root):
    """Six separately bounded packets; do not apply a single-packet 128 MiB cap."""
    root = Path(root)
    layout = matrix_layout(root)
    names = {cell_name(a, s) for a, s in cells()}
    if set(layout) != names | {'request.json', 'matrix.json'}:
        raise ContractError('producer matrix membership differs')
    return {'layout': layout, 'cells': {n: inventory(root/n) for n in sorted(names)}}


def unchanged_matrix(root, snapshot):
    if matrix_layout(root) != snapshot['layout']:
        raise ContractError('producer matrix root changed')
    for name, before in snapshot['cells'].items():
        unchanged(Path(root)/name, before)


def _schedule(record, arm, plan):
    """Reconstruct exact visits/targets without evaluating losses or updates."""
    step = integer(record['step'], 'producer step', 0, plan.steps)
    if type(record['cursor']) is not int or record['cursor'] != step*plan.batch_size:
        raise ContractError('producer cursor differs')
    if type(record['history']) is not list or len(record['history']) != step:
        raise ContractError('producer history length differs')
    train = encoded(schedule(arm), plan)
    total = 0
    for i, entry in enumerate(record['history']):
        exact_keys(entry, {'step','loss_hex','gradient_norm_hex','target_tokens','examples'}, 'producer history')
        batch = [train[(i*plan.batch_size+j) % len(train)] for j in range(plan.batch_size)]
        targets = sum(sum(e['loss']) for e in batch)
        wanted = {'step': i+1, 'target_tokens': targets, 'examples': [e['identity'] for e in batch]}
        if any(json_bytes(entry[k]) != json_bytes(v) for k,v in wanted.items()):
            raise ContractError('producer scheduled visits differ')
        _hex(entry['loss_hex']); _hex(entry['gradient_norm_hex']); total += targets
    if type(record['target_tokens']) is not int or record['target_tokens'] != total or total > plan.max_target_tokens:
        raise ContractError('producer target count differs')
    expected = [i for i in range(1, step+1) if i % plan.validation_every == 0 or i == plan.steps]
    if type(record['validation']) is not list or len(record['validation']) != len(expected):
        raise ContractError('producer training diagnostic schedule differs')
    for entry, step in zip(record['validation'], expected):
        exact_keys(entry, {'step','loss_hex'}, 'producer training diagnostic')
        if type(entry['step']) is not int or entry['step'] != step:
            raise ContractError('producer training diagnostic step differs')
        _hex(entry['loss_hex'])
    _hex(record['initial_train_loss_hex'])


def checkpoint(payload, metadata, request, request_identity, arm, seed, step):
    """Delegate safe decoding and slot checks; copy only model parameters."""
    if type(payload) is not bytes or type(metadata) is not bytes or max(len(payload),len(metadata)) > PROTOCOL['max_artifact_bytes']:
        raise ContractError('producer checkpoint size/type differs')
    record = parse_json(metadata, canonical=True)
    plan = plan_for(seed)
    expected = {'schema':'trinite.addition-development-checkpoint.v1', 'arm':arm, 'seed':seed,
        'request_identity':request_identity, 'plan':plan.to_dict(),
        'dataset_identity':request['dataset_identity'], 'admission_identity':request['admission_identity']}
    exact_keys(record, set(expected)|{'initialization_identity','initial_train_loss_hex','model_identity',
        'step','cursor','target_tokens','history','validation'}, 'producer checkpoint')
    if any(json_bytes(record[k]) != json_bytes(v) for k,v in expected.items()):
        raise ContractError('producer checkpoint context differs')
    if type(record['step']) is not int or record['step'] != step:
        raise ContractError('producer checkpoint milestone differs')
    _schedule(record, arm, plan)
    model = Decoder(plan.model, lane='dense', seed=seed)
    initial = model_identity(model)
    if record['initialization_identity'] != initial:
        raise ContractError('producer seeded initialization differs')
    require_identity(record['model_identity'])
    try:
        tensors = load(payload)
    except Exception as error:
        raise ContractError('invalid producer safetensors') from error
    names = {'model.'+n for n,_ in model.named_parameters()}
    if step:
        names |= {'optimizer.'+n+'.'+slot for n,_ in model.named_parameters()
                  for slot in ('step','exp_avg','exp_avg_sq')}
    if set(tensors) != names:
        raise ContractError('producer tensor inventory differs')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = tensors['model.'+name]
            if (value.device.type != 'cpu' or value.layout != torch.strided or value.dtype != parameter.dtype
                    or value.shape != parameter.shape or not bool(torch.isfinite(value).all())):
                raise ContractError('invalid producer model tensor')
            if step:
                for slot in ('step','exp_avg','exp_avg_sq'):
                    _slot(tensors['optimizer.'+name+'.'+slot], parameter, slot, step)
            parameter.copy_(value)
    model.validate_state()
    if model_identity(model) != record['model_identity'] or (step == 0 and record['model_identity'] != initial):
        raise ContractError('producer checkpoint model identity differs')
    return model, record


def _resources(record):
    exact_keys(record, {'schema','wall_seconds_hex','update_seconds_hex','peak_rss_kib','scope','kv_payload_bytes'}, 'producer resources')
    wall = _hex(record['wall_seconds_hex']); update = _hex(record['update_seconds_hex'])
    if (record['schema'] != 'trinite.addition-development-resources.v1' or
        type(record['peak_rss_kib']) is not int or record['peak_rss_kib'] < 0 or record['kv_payload_bytes'] is not None or
        record['scope'] != 'sampled process high-water RSS includes imports; not hard memory or whole-directory cap' or
        update > PROTOCOL['update_seconds_total'] or wall > PROTOCOL['worker_seconds'] or update > wall):
        raise ContractError('producer resources differ')


def _cell(root, request_raw, request, request_identity, arm, seed, completion_identity, expected_model):
    root = Path(root); before = inventory(root)
    completion_raw = read(root/'completion.json')
    if identity(completion_raw) != completion_identity:
        raise ContractError('producer completion identity differs')
    c = parse_json(completion_raw, canonical=True)
    exact_keys(c, {'schema','request_identity','cell','artifacts','manifest_identity','verification_identity'}, 'producer completion')
    if (c['schema'] != 'trinite.addition-development-completion.v1' or c['request_identity'] != request_identity
        or c['cell'] != cell_name(arm,seed) or read(root/'request.json') != request_raw):
        raise ContractError('producer completion context differs')
    exact_keys(c['artifacts'], artifact_names(), 'producer artifacts')
    top = {Path(p).parts[0] for p in before['members'] if p != '.'}
    expected_top = artifact_names()|{'request.json','completion.json','verification.json','provenance','worker.log','attempt.json'}
    if top != expected_top:
        raise ContractError('producer cell membership differs')
    read(root/'worker.log',PROTOCOL['max_worker_log_bytes'])
    bound = {name:read(root/name) for name in artifact_names()}
    if sum(map(len,bound.values())) > PROTOCOL['max_retained_payload_bytes']:
        raise ContractError('producer retained payload bound exceeded')
    for name,raw in bound.items():
        if identity(raw) != c['artifacts'][name]:
            raise ContractError('producer artifact differs: '+name)
    vr = read(root/'verification.json'); manifest_raw = read(root/'provenance/manifest.json')
    if identity(vr) != c['verification_identity'] or identity(manifest_raw) != c['manifest_identity']:
        raise ContractError('producer custody identities differ')
    verification = verify_observation(root/'provenance')
    if (json_bytes(verification) != vr or not verification['integrity_verified'] or
        verification['errors'] or verification['known_missing_artifacts']):
        raise ContractError('producer closed provenance differs')
    indexes = []
    for item in parse_json(manifest_raw,canonical=True)['core']['artifacts']:
        key = item['content_identity']; require_identity(key)
        raw = read(root/'provenance/artifacts/sha256'/key[7:])
        try: value = parse_json(raw,canonical=True)
        except ContractError: continue
        if type(value) is dict and value.get('schema') == 'trinite.observer-index.v1':
            indexes.append(value['artifacts'])
    data, admitted = dataset(); audit_bytes(data,admitted)
    if indexes != [{**c['artifacts'],'request.json':request_identity,'dataset.json':identity(data),'admission.json':identity(admitted)}]:
        raise ContractError('producer custody names differ')
    training = {}; previous_history = []; previous_validation = []; initial_loss = None
    for step in MILESTONES:
        model, record = checkpoint(bound[f'tensors-{step}.safetensors'],bound[f'metadata-{step}.json'],
                                   request,request_identity,arm,seed,step)
        if step == 0: initial_loss = record['initial_train_loss_hex']
        if record['initial_train_loss_hex'] != initial_loss:
            raise ContractError('producer initial loss continuity differs')
        if (record['history'][:len(previous_history)] != previous_history or
            record['validation'][:len(previous_validation)] != previous_validation):
            raise ContractError('producer milestone trajectory differs')
        previous_history = record['history']; previous_validation = record['validation']
        training[str(step)] = check_score(parse_json(bound[f'training-{step}.json'],canonical=True),arm,'train',record['model_identity'])
    if record['model_identity'] != expected_model or record['step'] != STEPS:
        raise ContractError('externally committed final producer model differs')
    raw_scores = {s:bound[f'development-{s}.json'] for s in STRATA}
    development = {s:check_score(parse_json(raw,canonical=True),arm,s,expected_model) for s,raw in raw_scores.items()}
    # report_for only reads these fields; no TrainingState or optimizer exists.
    state = SimpleNamespace(model=model,initialization_identity=record['initialization_identity'],
                            step=record['step'],target_tokens=record['target_tokens'])
    report = report_for(state,arm,seed,request_identity,training,development)
    if bound['report.json'] != json_bytes(report):
        raise ContractError('producer report differs')
    _resources(parse_json(bound['resources.json'],canonical=True))
    unchanged(root,before)
    binding = {'completion_identity':completion_identity,
        'checkpoint_metadata_identity':identity(bound[f'metadata-{STEPS}.json']),
        'checkpoint_tensor_identity':identity(bound[f'tensors-{STEPS}.safetensors']),
        'model_identity':expected_model,
        'raw_score_identities':{s:identity(raw) for s,raw in raw_scores.items()}}
    return binding, model, raw_scores, report


def _inspect(root, request_identity, matrix_identity, model_identities, selected=None):
    root = Path(root)
    require_identity(request_identity); require_identity(matrix_identity)
    expected_cells = {cell_name(a,s) for a,s in cells()}
    exact_keys(model_identities,expected_cells,'external producer models')
    for key in model_identities.values(): require_identity(key)
    before = snapshot_matrix(root)
    request_raw = read(root/'request.json',1024*1024)
    request = validate(request_raw,request_identity)
    matrix_raw = read(root/'matrix.json')
    if identity(matrix_raw) != matrix_identity:
        raise ContractError('external producer matrix identity differs')
    matrix = parse_json(matrix_raw,canonical=True)
    exact_keys(matrix,{'schema','request_identity','cells','outcome','fresh_confirmation','learning_adequate','phase5_ready','release_ready'},'producer matrix')
    if (matrix['schema'] != 'trinite.addition-development-matrix.v1' or matrix['request_identity'] != request_identity
        or matrix['outcome'] != 'completed' or any(matrix[k] is not False for k in ('fresh_confirmation','learning_adequate','phase5_ready','release_ready'))):
        raise ContractError('producer matrix must be complete and non-certifying')
    exact_keys(matrix['cells'],expected_cells,'producer matrix cells')
    bindings = {}; reports = {}; chosen = None
    for arm,seed in cells():
        name = cell_name(arm,seed); entry = matrix['cells'][name]
        exact_keys(entry,{'outcome','attempt_identity','completion_identity','retained_files'},'producer matrix cell')
        if entry['outcome'] != 'completed' or bound_files(root/name) != entry['retained_files']:
            raise ContractError('producer cell completion/inventory differs')
        require_identity(entry['completion_identity'])
        attempt_raw = read(root/name/'attempt.json')
        if identity(attempt_raw) != entry['attempt_identity']:
            raise ContractError('producer attempt identity differs')
        attempt = parse_json(attempt_raw,canonical=True)
        exact_keys(attempt,{'schema','cell','request_identity','outcome','worker_exit','detail','completion_identity','wall_seconds_hex'},'producer attempt')
        expected = {'schema':'trinite.addition-development-attempt.v1','cell':name,'request_identity':request_identity,
                    'outcome':'completed','worker_exit':0,'completion_identity':entry['completion_identity']}
        if any(json_bytes(attempt[k]) != json_bytes(v) for k,v in expected.items()) or type(attempt['detail']) is not str:
            raise ContractError('producer attempt context differs')
        _hex(attempt['wall_seconds_hex'])
        bindings[name], model, scores, reports[name] = _cell(root/name,request_raw,request,request_identity,
            arm,seed,entry['completion_identity'],model_identities[name])
        if name == selected: chosen = (model,scores)
    for seed in (0,1,2):
        left,right = reports[cell_name('control',seed)],reports[cell_name('candidate',seed)]
        if left['initialization_identity'] != right['initialization_identity'] or left['target_tokens'] != right['target_tokens']:
            raise ContractError('producer paired initialization/exposure differs')
    validate(request_raw,request_identity); unchanged_matrix(root,before)
    binding = {'producer_request_identity':request_identity,'producer_matrix_identity':matrix_identity,
        'model_identities':model_identities,'producer_source':request['source'],
        'producer_environment':request['environment'],'cells':bindings}
    return binding, chosen


def inspect_matrix(root, request_identity, matrix_identity, model_identities):
    """Safe structure and source/custody only; no model forwards or update replay."""
    return _inspect(root,request_identity,matrix_identity,model_identities)[0]


def load_cell(root, binding, arm, seed):
    """Revalidate all externally committed evidence, returning only the selected final model."""
    exact_keys(binding,BINDING_KEYS,'producer binding')
    actual, selected = _inspect(root,binding['producer_request_identity'],binding['producer_matrix_identity'],
                                binding['model_identities'],cell_name(arm,seed))
    if json_bytes(actual) != json_bytes(binding):
        raise ContractError('frozen producer binding differs')
    return selected
