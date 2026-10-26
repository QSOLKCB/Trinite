"""Bounded per-cell composition; evidence verification never replays optimizer updates."""
from pathlib import Path
import time
import resource

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from trinite.training import run_steps, model_identity
from trinite.observation import BundleObserver, verify_observation
from native_diagnostic_v1.artifacts import Store
from unseen_evaluation_v1.packet import inventory, unchanged
from .request import read, validate, cell_name
from .protocol import PROTOCOL, MILESTONES, STRATA
from .data import dataset, audit_bytes, exposure, examples
from .numerical import state_for, snapshot, restore, score, check_score, behavior_identity
from .reporting import describe, controls
from .journal import PredictionJournal


def artifact_names():
    return {'report.json','resources.json','description.json'} | {
        f'{kind}-{step}.{suffix}' for step in MILESTONES
        for kind,suffix in (('training','json'),('tensors','safetensors'),('metadata','json'))
    } | {f'development-{stratum}.json' for stratum in STRATA} | {
        f'raw-{label}.jsonl' for label in [*(f'training-{step}' for step in MILESTONES),
                                         *(f'development-{s}' for s in STRATA)]}


def journal_context(state, arm, seed, request_identity, selector):
    return {'schema':'trinite.position-study-prediction-journal.v1',
        'request_identity':request_identity,'cell':cell_name(arm,seed),
        'step':state.step,'selector':selector,'addressing':state.model.addressing,
        'model_identity':model_identity(state.model)}


def report_for(state, arm, seed, request_identity, training, development):
    return {'schema':'trinite.position-study-report.v1', 'outcome':'completed',
        'request_identity':request_identity, 'cell':cell_name(arm,seed), 'arm':arm, 'seed':seed,
        'initialization_identity':state.initialization_identity,
        'addressing':state.model.addressing,'final_behavior_identity':behavior_identity(state.model),
        'final_model_identity':model_identity(state.model), 'steps':state.step,
        'target_tokens':state.target_tokens, 'exposure':exposure(arm),
        'training':training, 'development':development,
        'controls':{s:controls(arm,s) for s in STRATA},
        'verification_scope':'source/admission/custody/safe checkpoints and retained metric consistency; not optimizer or prediction replay',
        'fresh_confirmation':False, 'learning_adequate':False, 'phase5_ready':False,
        'release_ready':False, 'cache_mode':'unsupported'}


def run(root, request_raw, request_identity, arm, seed, expected_directory):
    request = validate(request_raw, request_identity); cell_name(arm,seed)
    store = Store(root, expected_directory); outputs = {}
    started = time.monotonic(); update_seconds = 0.0
    def write(name, value, raw=False):
        data = value if raw else json_bytes(value)
        store.write(name,data); outputs[name]=data
    def observed_score(state, selector, label):
        name = f'raw-{label}.jsonl'
        journal = PredictionJournal(store,name,journal_context(state,arm,seed,request_identity,selector))
        try:
            result = score(state,arm,selector,on_prediction=journal.append)
        finally:
            journal.close()
        outputs[name] = read(Path(root)/name)
        return result
    try:
        store.write('request.json',request_raw)
        state = state_for(arm,seed)
        train_summaries = {}
        for step in MILESTONES:
            if time.monotonic()-started > PROTOCOL['worker_seconds']:
                raise ContractError('position study worker deadline exceeded')
            if step:
                before=time.monotonic();run_steps(state,stop_after=step)
                update_seconds += time.monotonic()-before
                if update_seconds > PROTOCOL['update_seconds_total']:
                    raise ContractError('cumulative position study update budget exceeded')
            result=observed_score(state,'train',f'training-{step}');train_summaries[str(step)]=result['summary']
            payload,metadata=snapshot(state,arm,seed,request_identity)
            write(f'training-{step}.json',result)
            write(f'tensors-{step}.safetensors',payload,True)
            write(f'metadata-{step}.json',metadata,True)
        # Only the frozen final step may see development rows, after all updates.
        development={}; development_records={}
        for stratum in STRATA:
            result=observed_score(state,stratum,f'development-{stratum}');development[stratum]=result['summary']
            development_records[stratum]=result
            write(f'development-{stratum}.json',result)
        write('description.json', describe(arm, parse_json(outputs[f'training-{MILESTONES[-1]}.json'], canonical=True), development_records))
        write('report.json',report_for(state,arm,seed,request_identity,train_summaries,development))
        write('resources.json',{'schema':'trinite.position-study-resources.v1',
            'wall_seconds_hex':(time.monotonic()-started).hex(),
            'update_seconds_hex':update_seconds.hex(),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope':'sampled process high-water RSS includes imports; not hard memory or whole-directory cap',
            'kv_payload_bytes':None})
        raw,admitted=dataset();audit_bytes(raw,admitted)
        store.check();provenance=Path('/proc/self/fd')/str(store.fd)/'provenance'
        observer=BundleObserver(provenance)
        observer.record('position-study-cell',inputs={'request.json':request_raw,
            'dataset.json':raw,'admission.json':admitted},outputs=outputs)
        verification=observer.finalize();store.check()
        if not verification['integrity_verified'] or verification['errors'] or verification['known_missing_artifacts']:
            raise ContractError('position study cell custody did not close')
        store.write('verification.json',json_bytes(verification))
        validate(request_raw,request_identity)
        completion={'schema':'trinite.position-study-completion.v1',
            'request_identity':request_identity,'cell':cell_name(arm,seed),
            'artifacts':{name:identity(data) for name,data in outputs.items()},
            'manifest_identity':identity(read(provenance/'manifest.json')),
            'verification_identity':identity(json_bytes(verification))}
        encoded=json_bytes(completion);store.write('completion.json',encoded)
        return identity(encoded)
    finally:
        store.close()


def audit(root, request_raw, request_identity, arm, seed, expected_identity):
    root=Path(root);before=inventory(root)
    validate(request_raw,request_identity);require_identity(expected_identity);cell_name(arm,seed)
    completion_raw=read(root/'completion.json')
    if identity(completion_raw)!=expected_identity:
        raise ContractError('position study completion identity differs')
    c=parse_json(completion_raw,canonical=True)
    exact_keys(c,{'schema','request_identity','cell','artifacts','manifest_identity','verification_identity'},'position study completion')
    if (c['schema']!='trinite.position-study-completion.v1' or
        c['request_identity']!=request_identity or c['cell']!=cell_name(arm,seed) or
        read(root/'request.json')!=request_raw):
        raise ContractError('position study completion request/cell differs')
    exact_keys(c['artifacts'],artifact_names(),'position study completed artifacts')
    top={Path(p).parts[0] for p in before['members'] if p!='.'}
    allowed=artifact_names()|{'request.json','completion.json','verification.json','provenance','worker.log','attempt.json'}
    if not top<=allowed or not (allowed-{'worker.log','attempt.json'})<=top:
        raise ContractError('position study completed directory membership differs')
    bound={name:read(root/name) for name in artifact_names()}
    if sum(map(len,bound.values()))>PROTOCOL['max_retained_payload_bytes']:
        raise ContractError('position study retained payload bound exceeded')
    for name,raw in bound.items():
        if identity(raw)!=c['artifacts'][name]:
            raise ContractError('position study retained artifact differs: '+name)
    vr=read(root/'verification.json')
    if identity(vr)!=c['verification_identity'] or identity(read(root/'provenance/manifest.json'))!=c['manifest_identity']:
        raise ContractError('position study verification/manifest identity differs')
    verification=verify_observation(root/'provenance')
    if json_bytes(verification)!=vr or not verification['integrity_verified'] or verification['errors'] or verification['known_missing_artifacts']:
        raise ContractError('position study closed provenance differs')
    manifest=parse_json(read(root/'provenance/manifest.json'),canonical=True)
    indexes=[]
    for item in manifest['core']['artifacts']:
        payload=read(root/'provenance/artifacts/sha256'/item['content_identity'][7:])
        try: value=parse_json(payload,canonical=True)
        except ContractError: continue
        if type(value)is dict and value.get('schema')=='trinite.observer-index.v1':indexes.append(value['artifacts'])
    raw,admitted=dataset();audit_bytes(raw,admitted)
    wanted={**c['artifacts'],'request.json':request_identity,'dataset.json':identity(raw),'admission.json':identity(admitted)}
    if indexes!=[wanted]:raise ContractError('position study provenance names differ')
    training={};final=None;initial=None;previous_history=[];previous_validation=[]
    for step in MILESTONES:
        state=restore(bound[f'tensors-{step}.safetensors'],bound[f'metadata-{step}.json'],arm,seed,request_identity)
        if state.step!=step:raise ContractError('position study checkpoint milestone differs')
        if step==0:
            if model_identity(state.model)!=state.initialization_identity:
                raise ContractError('position study zero-step initialization differs')
            initial=state.initialization_identity
        if state.initialization_identity!=initial:raise ContractError('position study initialization lineage differs')
        if state.history[:len(previous_history)] != previous_history or state.validation[:len(previous_validation)] != previous_validation:
            raise ContractError('position study milestone histories do not share one retained trajectory')
        previous_history=list(state.history);previous_validation=list(state.validation)
        training[str(step)]=check_score(parse_json(bound[f'training-{step}.json'],canonical=True),arm,'train',model_identity(state.model))
        final=state
        context=journal_context(state,arm,seed,request_identity,'train')
        predictions=parse_json(bound[f'training-{step}.json'],canonical=True)['predictions']
        if bound[f'raw-training-{step}.jsonl'] != b''.join(json_bytes({**context,'prediction':p}) for p in predictions):
            raise ContractError('training raw prediction journal differs')
    development={s:check_score(parse_json(bound[f'development-{s}.json'],canonical=True),arm,s,model_identity(final.model)) for s in STRATA}
    for s in STRATA:
        context=journal_context(final,arm,seed,request_identity,s)
        predictions=parse_json(bound[f'development-{s}.json'],canonical=True)['predictions']
        if bound[f'raw-development-{s}.jsonl'] != b''.join(json_bytes({**context,'prediction':p}) for p in predictions):
            raise ContractError('development raw prediction journal differs')
    description=describe(arm, parse_json(bound[f'training-{MILESTONES[-1]}.json'], canonical=True),
        {s:parse_json(bound[f'development-{s}.json'], canonical=True) for s in STRATA})
    if bound['description.json'] != json_bytes(description):
        raise ContractError('position-study descriptive raw-score accounting differs')
    expected=report_for(final,arm,seed,request_identity,training,development)
    if bound['report.json']!=json_bytes(expected):raise ContractError('position study report/context differs')
    r=parse_json(bound['resources.json'],canonical=True)
    exact_keys(r,{'schema','wall_seconds_hex','update_seconds_hex','peak_rss_kib','scope','kv_payload_bytes'},'position study resources')
    import math
    for key in ('wall_seconds_hex','update_seconds_hex'):
        value=r[key]
        try:number=float.fromhex(value)
        except (TypeError,ValueError) as error:raise ContractError('invalid resource time') from error
        if not math.isfinite(number) or number<0 or number.hex()!=value:raise ContractError('invalid resource time')
    if (r['schema']!='trinite.position-study-resources.v1' or type(r['peak_rss_kib'])is not int or
        r['peak_rss_kib']<0 or r['kv_payload_bytes']is not None or
        r['scope']!='sampled process high-water RSS includes imports; not hard memory or whole-directory cap' or
        float.fromhex(r['update_seconds_hex'])>PROTOCOL['update_seconds_total'] or
        float.fromhex(r['wall_seconds_hex'])>PROTOCOL['worker_seconds'] or
        float.fromhex(r['update_seconds_hex'])>float.fromhex(r['wall_seconds_hex'])):
        raise ContractError('resource scope/budget differs')
    validate(request_raw,request_identity);unchanged(root,before)
    return {'verified':True,'request_identity':request_identity,'completion_identity':expected_identity,
        'report':expected,'description':description,'predictions_replayed':False,'optimizer_replayed':False}
