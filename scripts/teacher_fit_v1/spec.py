# SPDX-License-Identifier: MPL-2.0
"""FIT governance/admission. No model construction or training authority issuance."""
import os
import hashlib
from pathlib import Path
import subprocess

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite.tokenizer import ByteTokenizer
from teacher_lab_v1 import core as lab
from teacher_dataset_v1 import core as data
from . import custody as io, selection

ROOT = Path(__file__).resolve().parents[2]
DESIGN = 'configs/teacher/tt003-fit-001-v1.json'
SCHEMA = 'trinite.teacher-fit-001.v1'
STEPS = (0, 64, 128, 192, 256)
# New operational limits; the merged experimental design remains byte-identical.
LIMITS = {'native_seconds': 600, 'outer_seconds': 900, 'updates': 256,
          'target_tokens': 4096, 'batch_size': 4, 'address_space_bytes': 8*1024**3,
          'file_bytes': io.MAX_FILE, 'run_bytes': io.MAX_RUN, 'retries': 0}


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def source():
    """Complete code/config/test/decision trees, including every scoped AGENTS."""
    if git('status', '--porcelain', '--untracked-files=all').strip():
        raise ContractError('source checkout must be clean')
    sha = git('rev-parse', 'HEAD').decode().strip()
    entries = [e.split('\t',1) for e in git('ls-tree','-r','-z',sha).decode().split('\0') if e]
    blobs = {name:head.split()[2] for head,name in entries}
    names = list(blobs)
    paths = sorted(n for n in names if n and (n.split('/')[0] in
        ('src', 'scripts', 'tests', 'docs', 'configs', '.github') or '/' not in n))
    # Ignored Python/import additions must not escape the Git dirty check.
    tracked = set(paths)
    for tree in ('src', 'scripts', 'tests', 'docs', 'configs', '.github'):
        for p in (ROOT/tree).rglob('*'):
            if '__pycache__' in p.parts or p.suffix == '.pyc':
                continue
            if (p.is_file() or p.is_symlink()) and p.relative_to(ROOT).as_posix() not in tracked:
                raise ContractError('untracked source/decision input')
    members=[]
    for name in paths:
        raw=io.read(ROOT/name)
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=blobs[name]:
            raise ContractError('source bytes differ from exact Git commit')
        members.append({'path':name,'identity':identity(raw)})
    return {'commit': sha, 'members': members}


def settings():
    design = lab.source_json(DESIGN)
    expected = {'backend':'pinned-cpu-float32','batch_size':4,
        'dataset':'TT-002-source-verified-TRAIN-only','diagnostic_steps':list(STEPS),
        'gpu_allowed':False,'heldout_scored':False,'learning_rate':'0.001',
        'max_native_seconds_per_run':600,'max_outer_seconds_per_run':900,
        'max_target_tokens_per_run':4096,'model_config':'configs/teacher/gf3-teacher-90k-v1.json',
        'model_parameters':90624,'optimizer':'existing-native-AdamW','panel_rows':72,
        'panel_word_count':6,'panel_word_selection':'sha256-domain-separated-word-order-v1',
        'retries_allowed':0,'row_schedule':'sha256-domain-separated-row-id-order-independent-of-model-seed-v1',
        'schema':'trinite.teacher-fit-001-design.v1','seeds':[17,29,43],
        'steps_per_run':256,'target_execution_authorized':False,
        'training_claim':'same-panel-fit-feasibility-only','weight_distribution_authorized':False}
    if json_bytes(design) != json_bytes(expected):
        raise ContractError('merged FIT design changed')
    model = ModelConfig.from_dict(lab.source_json(design['model_config']))
    if model.parameter_count != 90624:
        raise ContractError('native teacher size changed')
    return design, model


def native_config(model, seed):
    # Frozen serialization of existing RunConfig defaults; the native adapter
    # compares this to the authoritative RunConfig before allocation.
    return {'schema':'trinite.training-plan.v1','model':model.to_dict(),'seed':seed,
        'steps':256,'batch_size':4,'learning_rate':'0.001','beta1':'0.9','beta2':'0.999',
        'optimizer_epsilon':'0.00000001','weight_decay':'0.01','gradient_clip':'1',
        'accumulation':1,'schedule':'constant','order':'seeded-hash-rank-cycle.v1',
        'loss':'answer-and-eos-target-mean.v1','validation_every':64,
        'selection':'final-step; no early stopping; no test evaluation',
        'max_seconds':600,'max_target_tokens':4096}


def admit_train(raw, model):
    """Verify all 972 original rows without opening or computing heldout labels."""
    import qutrit_conformance_v1 as q
    rows = [parse_json(line, canonical=True) for line in raw.splitlines(keepends=True)]
    selected = selection.select_panel(rows)
    tokenizer = ByteTokenizer(model.context_length)
    for row in rows:
        obs = row['observation']
        task, word, family = obs['task'], obs['word'], row['family']
        forward = data.algebraic(word, family)
        inverse = data.algebraic(word, family, True)
        if data.algebraic(forward, family, True) != word or data.algebraic(inverse, family) != word:
            raise ContractError('TRAIN inverse oracle mismatch')
        if task == 'check':
            candidate = forward if row['answer'] == '1' else str((int(forward[0])+1)%3)+forward[1:]
            answer = row['answer']
        else:
            candidate = ''
            answer = inverse if task == 'inverse' else forward
        observation = q.public(word, family, task, candidate)
        if q.expected(observation) != answer:
            raise ContractError('independent TRAIN oracle mismatch')
        prompt = q.prompt(observation)
        encoded = tokenizer.encode(prompt+answer, answer_start=len(prompt.encode('ascii')))
        expected = {'schema':data.ROW_SCHEMA,
            'id':identity(json_bytes({'schema':data.ROW_SCHEMA,'observation':observation})),
            'prompt':prompt,'observation':observation,'family':family,'split':'train',
            'answer':answer,'supervised_tokens':len(answer)+1,
            'complete_context_tokens':len(encoded.input_ids)}
        if json_bytes(row) != json_bytes(expected) or sum(encoded.loss_mask) != len(answer)+1:
            raise ContractError('original TRAIN identity/prompt/target/mask mismatch')
    by_id = {r['id']:r for r in rows}
    panel = [by_id[rid] for rid in selected['panel_ids']]
    return rows, panel, selected


def train_packet(folder, anchor):
    """Execution ingress reads only manifest, rights and TRAIN bytes."""
    folder = Path(folder)
    manifest = io.anchored(folder/'manifest.json', anchor)
    design, model = settings()
    expected_source = sorted(set(data.SOURCES) | {data.PLAN, data.plan()['source_catalog']})
    if manifest.get('source_closure') != [
            {'path':n,'identity':identity(lab.source_bytes(n))} for n in expected_source]:
        raise ContractError('TT-002 source closure drift')
    if manifest.get('schema') != data.MANIFEST_SCHEMA or manifest.get('training_authorized') is not False:
        raise ContractError('TT-002 manifest contract mismatch')
    members = manifest['files']
    if len(members) != 7 or {m['name'] for m in members} != set(data.MEMBERS)-{'manifest.json'}:
        raise ContractError('TT-002 member inventory mismatch')
    for name in ('train.jsonl', 'dataset-rights.json'):
        raw = io.read(folder/name)
        if [m for m in members if m['name']==name] != [{'name':name,'identity':identity(raw),'bytes':len(raw)}]:
            raise ContractError('TT-002 TRAIN/rights bytes mismatch')
    if io.read(folder/'dataset-rights.json') != json_bytes(data.rights()[0]):
        raise ContractError('TT-002 rights provenance mismatch')
    return admit_train(io.read(folder/'train.jsonl'), model)


def check_review(review, sha):
    if (set(review) != {'schema','commit','review_url','reviewed','checks'}
            or review['schema'] != SCHEMA+'.review' or review['commit'] != sha
            or review['reviewed'] is not True
            or not review['review_url'].startswith('https://github.com/QSOLKCB/Trinite/')
            or not review['checks']):
        raise ContractError('exact-head review evidence required')
    names = []
    for check in review['checks']:
        if (set(check) != {'name','head_sha','conclusion','url'} or check['head_sha'] != sha
                or check['conclusion'] != 'success'
                or not check['url'].startswith('https://github.com/QSOLKCB/Trinite/')):
            raise ContractError('exact-head passing CI evidence required')
        names.append(check['name'])
    if len(set(names)) != len(names) or not {'FIT contracts','TT-002 contracts','TT-003 contracts'} <= set(names):
        raise ContractError('relevant checks missing or duplicated')


def request_for(dataset, dataset_anchor, seed, experiment, review, environment):
    if type(seed) is not int or seed not in selection.SEEDS:
        raise ContractError('only predeclared FIT seeds permitted')
    source_record = source()
    check_review(review, source_record['commit'])
    design, model = settings()
    _, panel, schedule = train_packet(dataset, dataset_anchor)
    experiment = Path(experiment)
    if not experiment.is_absolute() or experiment != experiment.resolve() or experiment.is_relative_to(ROOT):
        raise ContractError('private canonical experiment directory must be outside checkout')
    from .metrics import baselines
    return {'schema':SCHEMA+'.request','status':'FROZEN_NOT_AUTHORIZED',
        'source':source_record,'review':review,'design':design,'model':model.to_dict(),
        'model_parameters':model.parameter_count,'seed':seed,'lane':'dense',
        'native_config':native_config(model,seed),
        'geometry_loss':'0','dataset_manifest_identity':dataset_anchor,
        'dataset_train_identity':identity(io.read(Path(dataset)/'train.jsonl')),
        'dataset_rights_identity':identity(io.read(Path(dataset)/'dataset-rights.json')),
        'environment':environment,'panel':panel,'selection':schedule,
        'panel_identity':identity(json_bytes(panel)),
        'schedule_identity':identity(json_bytes(schedule['visit_ids'])),
        'baselines':baselines(panel),'limits':LIMITS,
        'experiment_directory':str(experiment),'output_directory':str(experiment/f'seed-{seed}'),
        'evidence':{'checkpoints':list(STEPS),'gold_positions_per_checkpoint':sum(len(r['answer'])+1 for r in panel),
                    'greedy_positions':'all actual prefixes; 1..5 per row',
                    'vocabulary':259,'logits':'little-endian-float32','metrics':'binary64-hex',
                    'replay_equality':'exact FP32 bytes and canonical derived records',
                    'optimizer_replay':'not performed by forward replay'},
        'training_authorized':False,'heldout_scored':False,'retries_allowed':0}


def verify_request(path, anchor, dataset):
    request = io.anchored(path, anchor)
    expected = request_for(dataset, request['dataset_manifest_identity'], request['seed'],
        request['experiment_directory'], request['review'], request['environment'])
    if json_bytes(request) != json_bytes(expected):
        raise ContractError('frozen request source/data/schedule mismatch')
    return request


def approval_expected(request, request_anchor):
    # Validation shape only. There is deliberately no approval-writing command.
    return {'schema':SCHEMA+'.owner-approval','owner':'QSOLKCB',
        'scope':'one-owner-local-native-cpu-fit-attempt','request_identity':request_anchor,
        'source_commit':request['source']['commit'],'seed':request['seed'],
        'dataset_manifest_identity':request['dataset_manifest_identity'],
        'schedule_identity':request['schedule_identity'],
        'environment_identity':identity(json_bytes(request['environment'])),
        'output_directory':request['output_directory'],'limits':request['limits'],
        'training_authorized':True}


def approval(path, anchor, request, request_anchor):
    value = io.anchored(path, anchor)
    if json_bytes(value) != json_bytes(approval_expected(request, request_anchor)):
        raise ContractError('owner approval does not match this exact request/source/seed')
    return value


def approval_set(path, anchor, request, dataset):
    """All three separate grants must exist before the first target dispatch."""
    grants=io.anchored(path,anchor)
    if type(grants) is not dict or set(grants)!={'17','29','43'}:
        raise ContractError('three separate owner approvals required')
    records={}
    for seed in selection.SEEDS:
        grant=grants[str(seed)]
        if type(grant) is not dict or set(grant)!={'path','identity','request_identity'}:
            raise ContractError('invalid three-approval inventory')
        req_path=Path(request['experiment_directory'])/'requests'/f'seed-{seed}.json'
        req=verify_request(req_path,grant['request_identity'],dataset)
        if (req['seed']!=seed or req['source']!=request['source']
                or req['environment']!=request['environment'] or req['selection']!=request['selection']
                or req['experiment_directory']!=request['experiment_directory']):
            raise ContractError('approval set mixes experiments')
        records[str(seed)]=approval(grant['path'],grant['identity'],req,grant['request_identity'])
    if len({g['identity'] for g in grants.values()})!=3:
        raise ContractError('reused approval identity')
    return grants,records


def verify_approval_records(saved, request, dataset):
    grants=saved.get('inventory',{});records=saved.get('records',{})
    if (set(saved)!={'inventory','records','inventory_identity'} or set(grants)!={'17','29','43'}
            or set(records)!=set(grants) or identity(json_bytes(grants))!=saved['inventory_identity']):
        raise ContractError('retained three-approval inventory mismatch')
    for seed in selection.SEEDS:
        req=request_for(dataset,request['dataset_manifest_identity'],seed,
            request['experiment_directory'],request['review'],request['environment'])
        anchor=identity(json_bytes(req));grant=grants[str(seed)];record=records[str(seed)]
        if (set(grant)!={'path','identity','request_identity'} or grant['request_identity']!=anchor
                or identity(json_bytes(record))!=grant['identity']
                or json_bytes(record)!=json_bytes(approval_expected(req,anchor))):
            raise ContractError('retained approval differs from source-bound request')


def freeze(dataset, dataset_anchor, experiment, review_path, review_anchor, environment):
    # Full TT-002 cold verification is admission, outside the model process.
    data.verify(Path(dataset), dataset_anchor)
    review = io.anchored(review_path, review_anchor)
    requests = [request_for(dataset, dataset_anchor, seed, experiment, review, environment)
                for seed in selection.SEEDS]
    out = io.fresh(Path(experiment)/'requests')
    anchors = {}
    for request in requests:
        name = f"seed-{request['seed']}.json"
        io.record(out/name, request)
        anchors[name] = identity(json_bytes(request))
        os.chmod(out/name, 0o400)
    io.record(Path(experiment)/'request-anchors.json', anchors)
    return anchors
