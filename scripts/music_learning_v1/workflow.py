"""Custody and structural audit, explicitly not numerical replay."""
import os
from pathlib import Path
import time
from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json
from trinite.observation import BundleObserver, verify_observation
from native_diagnostic_v1.artifacts import Store
from unseen_evaluation_v1.packet import inventory, unchanged
from .admission import dataset, audit_bytes
from .request import read, validate, cell_name
from .protocol import PROTOCOL


def require_native_admission(expected):
    raw, admitted = dataset()
    audit_bytes(raw, admitted)
    if identity(admitted) != expected:
        raise ContractError('music source-specific native admission identity differs')
    return raw, admitted


def artifact_names():
    return {'report.json','predictions.jsonl'} | {
        stage + suffix for stage in ('untrained','trained') for suffix in
        ('-train.json','-validation.json','-test.json','-description.json',
         '.safetensors','-checkpoint.json')}


def run(root, request_raw, request_identity, seed, expected_directory):
    from .request import ROOT
    root=Path(root).absolute()
    if root.resolve().is_relative_to(ROOT) or ROOT.is_relative_to(root.resolve()):
        raise ContractError('music worker output overlaps source checkout')
    before=inventory(root)
    if set(before['members'])-{'.', 'worker.log'}:
        raise ContractError('music worker needs a fresh controller attempt')
    request=validate(request_raw,request_identity); cell_name(seed)
    raw,admitted=require_native_admission(request['admission_identity'])
    store=Store(root,expected_directory); outputs={}; started=time.monotonic()
    fd=None; journal_bytes=0
    try:
        store.write('request.json',request_raw)
        fd=os.open('predictions.jsonl',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,
                   0o600,dir_fd=store.fd)
        def write(name,data):
            if time.monotonic()-started > PROTOCOL['worker_seconds']:
                raise ContractError('music worker deadline exceeded')
            store.write(name,data); outputs[name]=data
        def journal(stage,selector,prediction):
            nonlocal journal_bytes
            store.check()
            data=json_bytes({'stage':stage,'selector':selector,'prediction':prediction})
            if journal_bytes+len(data)>PROTOCOL['max_artifact_bytes'] or store.bytes+len(data)>PROTOCOL['max_retained_payload_bytes']:
                raise ContractError('music prediction journal budget exceeded')
            # One row is flushed before beginning the next generation. Failure
            # leaves this exact prefix; no retry, repair or success-only filtering.
            view=memoryview(data)
            while view:
                n=os.write(fd,view)
                if n<=0:raise OSError('short music journal write')
                view=view[n:]
            os.fsync(fd); journal_bytes+=len(data); store.bytes+=len(data); store.check()
        from .numerical import run_trial
        report=run_trial(seed,request['admission_identity'],request_identity,write,journal)
        os.close(fd);fd=None
        outputs['predictions.jsonl']=read(Path(root)/'predictions.jsonl')
        report.update(schema='trinite.music-learning-report.v1',
            verification_scope='source/admission/custody/safe checkpoints and metrics; no optimizer or prediction replay',
            fresh_confirmation=False,learning_adequate=False,phase5_ready=False,release_ready=False)
        write('report.json',json_bytes(report))
        if set(outputs)!=artifact_names():raise ContractError('music completed artifact inventory differs')
        store.check(); provenance=Path('/proc/self/fd')/str(store.fd)/'provenance'
        observer=BundleObserver(provenance)
        observer.record('music-learning-cell',inputs={'request.json':request_raw,
            'dataset.json':raw,'admission.json':admitted},outputs=outputs)
        verification=observer.finalize();store.check()
        if not verification['integrity_verified'] or verification['errors'] or verification['known_missing_artifacts']:
            raise ContractError('music custody did not close')
        store.write('verification.json',json_bytes(verification))
        validate(request_raw,request_identity)
        completion={'schema':'trinite.music-learning-completion.v1','request_identity':request_identity,
            'cell':cell_name(seed),'artifacts':{name:identity(data) for name,data in outputs.items()},
            'manifest_identity':identity(read(Path(root)/'provenance/manifest.json')),
            'verification_identity':identity(json_bytes(verification))}
        encoded=json_bytes(completion);store.write('completion.json',encoded)
        return identity(encoded)
    finally:
        if fd is not None:os.close(fd)
        store.close()


def audit(root, request_raw, request_identity, seed, expected_identity):
    root=Path(root);before=inventory(root)
    request=validate(request_raw,request_identity);cell_name(seed)
    completion_raw=read(root/'completion.json')
    if identity(completion_raw)!=expected_identity:
        raise ContractError('music completion identity differs')
    c=parse_json(completion_raw,canonical=True)
    exact_keys(c,{'schema','request_identity','cell','artifacts','manifest_identity','verification_identity'},'music completion')
    if (c['schema']!='trinite.music-learning-completion.v1' or c['request_identity']!=request_identity
            or c['cell']!=cell_name(seed) or read(root/'request.json')!=request_raw):
        raise ContractError('music completion context differs')
    exact_keys(c['artifacts'],artifact_names(),'music completed artifacts')
    top={Path(p).parts[0] for p in before['members'] if p!='.'}
    required=artifact_names()|{'request.json','completion.json','verification.json','provenance'}
    if not required<=top or not top<=required|{'worker.log','attempt.json'}:
        raise ContractError('music completed directory membership differs')
    bound={name:read(root/name) for name in artifact_names()}
    if sum(map(len,bound.values()))>PROTOCOL['max_retained_payload_bytes']:
        raise ContractError('music retained payload budget exceeded')
    if any(identity(data)!=c['artifacts'][name] for name,data in bound.items()):
        raise ContractError('music retained artifact identity differs')
    verification=verify_observation(root/'provenance')
    vr=read(root/'verification.json')
    if (json_bytes(verification)!=vr or identity(vr)!=c['verification_identity'] or
            identity(read(root/'provenance/manifest.json'))!=c['manifest_identity'] or
            not verification['integrity_verified'] or verification['errors'] or verification['known_missing_artifacts']):
        raise ContractError('music closed custody differs')
    raw,admitted=require_native_admission(request['admission_identity'])
    manifest=parse_json(read(root/'provenance/manifest.json'),canonical=True);indexes=[]
    for item in manifest['core']['artifacts']:
        payload=read(root/'provenance/artifacts/sha256'/item['content_identity'][7:])
        try:value=parse_json(payload,canonical=True)
        except ContractError:continue
        if type(value)is dict and value.get('schema')=='trinite.observer-index.v1':indexes.append(value['artifacts'])
    wanted={**c['artifacts'],'request.json':request_identity,'dataset.json':identity(raw),'admission.json':identity(admitted)}
    if indexes!=[wanted]:raise ContractError('music custody input/output names differ')
    from .numerical import restore, check_score, schedule_receipt
    from .reporting import describe
    from trinite.training import model_identity
    journal=[];states={};descriptions={}
    for stage,step in (('untrained',0),('trained',64)):
        tensor=stage+'.safetensors';metadata=stage+'-checkpoint.json'
        state=restore(bound[tensor],bound[metadata],seed,request_identity,request['admission_identity'],
                      (c['artifacts'][tensor],c['artifacts'][metadata]))
        if state.step!=step:raise ContractError('music checkpoint stage differs')
        if stage=='untrained' and model_identity(state.model)!=state.initialization_identity:
            raise ContractError('music untrained state is not random initialization')
        states[stage]=state;records={}
        for selector in ('train','validation','test'):
            record=parse_json(bound[stage+'-'+selector+'.json'],canonical=True)
            check_score(record,selector,model_identity(state.model));records[selector]=record
            journal.extend({'stage':stage,'selector':selector,'prediction':p} for p in record['predictions'])
        descriptions[stage]=describe(records)
        if bound[stage+'-description.json']!=json_bytes(descriptions[stage]):
            raise ContractError('music description does not match raw predictions')
    if bound['predictions.jsonl']!=b''.join(json_bytes(row) for row in journal):
        raise ContractError('music prediction journal differs from complete scoring')
    final=states['trained']
    if states['untrained'].initialization_identity!=final.initialization_identity:
        raise ContractError('music checkpoint initialization lineage differs')
    report=parse_json(bound['report.json'],canonical=True)
    expected={'seed':seed,'outcome':'completed','step':64,'target_tokens':final.target_tokens,
        'request_identity':request_identity,'schedule':schedule_receipt(seed),'model_identity':model_identity(final.model),
        'schema':'trinite.music-learning-report.v1',
        'verification_scope':'source/admission/custody/safe checkpoints and metrics; no optimizer or prediction replay',
        'fresh_confirmation':False,'learning_adequate':False,'phase5_ready':False,'release_ready':False}
    # Update time is observed, not replayed. It must satisfy the frozen ceiling.
    import math
    elapsed=report.get('update_seconds_hex')
    try:number=float.fromhex(elapsed)
    except (TypeError,ValueError) as error:raise ContractError('invalid music update time') from error
    if not math.isfinite(number) or not 0<=number<=120 or number.hex()!=elapsed:
        raise ContractError('music update budget exceeded')
    expected['update_seconds_hex']=elapsed
    if json_bytes(report)!=json_bytes(expected):raise ContractError('music report differs')
    validate(request_raw,request_identity);unchanged(root,before)
    return {'verified':True,'report':expected,'descriptions':descriptions,
        'optimizer_replayed':False,'predictions_replayed':False,'fresh_confirmation':False}
