# SPDX-License-Identifier: MPL-2.0
"""Bounded no-grad observations through the unchanged native Decoder."""
import os
import time
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from teacher_diag_v1 import native as behavior
from . import core, arithmetic as ar


def forward_vector(model, prefix):
    import torch
    with torch.no_grad():
        ids=torch.tensor([prefix],dtype=torch.int64,device='cpu')
        mask=torch.ones_like(ids,dtype=torch.bool)
        vector=model(ids,mask).logits[0,-1]
        if vector.dtype!=torch.float32 or vector.device.type!='cpu' or tuple(vector.shape)!=(259,):
            raise ContractError('native output is not CPU FP32 vocabulary vector')
        raw=ar.pack(vector.tolist())
        chosen=int(vector.argmax())
    if chosen!=max(range(259),key=ar.unpack(raw).__getitem__):
        raise ContractError('native argmax/tie semantics differ')
    return raw,chosen


def check_models(models):
    from trinite.training import model_identity, require_environment
    import torch
    require_environment()
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise ContractError('strict determinism required')
    identities=[model_identity(m) for m in models]
    if identities!=[core.diag.PIN['initial_model_identity'],core.diag.PIN['teacher_model_identity']]:
        raise ContractError('original model identity mismatch')
    if any(p.grad is not None for m in models for p in m.parameters()):
        raise ContractError('unexpected model gradients')
    return identities


def observe(index, models, deadline, consume):
    before=check_models(models)
    for rec in index:
        if time.monotonic()>deadline:
            raise ContractError('cooperative forward deadline exceeded')
        raw,chosen=forward_vector(models[0 if rec['stage']=='initial' else 1],rec['prefix_tokens'])
        if rec['probe']=='greedy-prefix' and chosen!=rec['historical_greedy_token']:
            raise ContractError('historical greedy token differs at '+rec['record_id'])
        measure=ar.metrics(ar.unpack(raw),rec['positional_reference_target'])
        consume(rec,raw,measure)
    if check_models(models)!=before:
        raise ContractError('model state changed during observation')
    if time.monotonic()>deadline:
        raise ContractError('forward deadline exceeded at completion')
    return before


def run(folder,anchor,dataset,teacher,previous,out):
    request=core.verify_request(folder,anchor,dataset,teacher,previous)
    core.fresh(out,folder,dataset,teacher,previous)
    core.write(out,'request.json',core.read(folder,'request.json'))
    start=time.monotonic()
    try:
        index=core.inventory(*core.historical(dataset,teacher,previous))
        core.write(out,'index.jsonl',b''.join(json_bytes(i) for i in index))
        models=behavior.restore_models(dataset,teacher)
        measures=[]
        with (out/'logits.f32le').open('xb') as vectors,(out/'metrics.jsonl').open('xb') as metrics:
            def consume(rec,raw,measure):
                if vectors.tell()!=rec['offset_bytes']:
                    raise ContractError('vector offset mismatch')
                vectors.write(raw); vectors.flush()
                measure=dict(measure,record_id=rec['record_id'])
                metrics.write(json_bytes(measure));metrics.flush();measures.append(measure)
            ids=observe(index,models,start+request['plan']['max_execution_seconds'],consume)
            os.fsync(vectors.fileno());os.fsync(metrics.fileno())
        core.write(out,'summary.json',json_bytes(core.summarize(index,measures)))
        from trinite.training import require_environment
        core.write(out,'execution.json',json_bytes({'schema':core.SCHEMA+'.execution',
            'model_identities_before_after':ids,'environment':require_environment(),
            'gold_positions':request['gold_positions'],'greedy_positions':request['greedy_positions'],
            'optimizer_updates':0,'heldout_scored':False,'native_argmax_matches_historical':True}))
        core.verify_request(folder,anchor,dataset,teacher,previous)
        if sum(p.stat().st_size for p in out.iterdir())>request['plan']['max_result_bytes']:
            raise ContractError('result size bound exceeded')
        if time.monotonic()>start+request['plan']['max_execution_seconds']:
            raise ContractError('audit deadline exceeded')
        manifest={'schema':core.SCHEMA+'.manifest','status':'COMPLETED_READ_ONLY',
            'request_identity':anchor,'positions':len(index),'optimizer_updates':0,
            'members':[{'name':n,'identity':identity(core.read(out,n)),'bytes':(out/n).stat().st_size}
                       for n in core.RESULT_FILES[:-1]]}
        raw=json_bytes(manifest);core.write(out,'manifest.json',raw)
        return {'manifest_identity':identity(raw),'positions':len(index),'optimizer_updates':0,'status':manifest['status']}
    except Exception as exc:
        core.write(out,'failure.json',json_bytes({'successful':False,'type':type(exc).__name__,
            'message':str(exc)[:1000],'elapsed_seconds_hex':(time.monotonic()-start).hex(),
            'partial_files':{p.name:p.stat().st_size for p in out.iterdir() if p.is_file()}}))
        raise


def verify(folder,anchor,dataset,teacher,previous,out,result_anchor):
    request=core.verify_request(folder,anchor,dataset,teacher,previous)
    core.diag.check_members(out,core.RESULT_FILES)
    raw=core.read(out,'manifest.json')
    if identity(raw)!=result_anchor:
        raise ContractError('external result identity mismatch')
    manifest=parse_json(raw,canonical=True)
    expected={'schema':core.SCHEMA+'.manifest','status':'COMPLETED_READ_ONLY',
        'request_identity':anchor,'positions':request['positions'],'optimizer_updates':0,
        'members':[{'name':n,'identity':identity(core.read(out,n)),'bytes':len(core.read(out,n))}
                   for n in core.RESULT_FILES[:-1]]}
    if manifest!=expected or core.read(out,'request.json')!=core.read(folder,'request.json'):
        raise ContractError('manifest or copied request mismatch')
    index=core.jsonl(core.read(out,'index.jsonl'))
    if index!=core.inventory(*core.historical(dataset,teacher,previous)):
        raise ContractError('source-bound position inventory mismatch')
    raw_vectors=core.read(out,'logits.f32le')
    if len(raw_vectors)!=request['logit_bytes']:
        raise ContractError('incomplete or excessive vector payload')
    saved=core.jsonl(core.read(out,'metrics.jsonl'))
    if len(saved)!=len(index):
        raise ContractError('incomplete/excess metric records')
    calculated=[]
    for rec,given in zip(index,saved):
        z=ar.unpack(raw_vectors[rec['offset_bytes']:rec['offset_bytes']+ar.VECTOR.size])
        metric=dict(ar.metrics(z,rec['positional_reference_target']),record_id=rec['record_id'])
        if metric!=given:
            raise ContractError('derived metric not exactly recomputable')
        if rec['probe']=='greedy-prefix' and metric['argmax']!=rec['historical_greedy_token']:
            raise ContractError('recorded logits do not reproduce original greedy token')
        calculated.append(metric)
    if parse_json(core.read(out,'summary.json'),canonical=True)!=core.summarize(index,calculated):
        raise ContractError('summary recomputation mismatch')
    execution=parse_json(core.read(out,'execution.json'),canonical=True)
    meta=parse_json(core.diag.read_member(teacher,'metadata.json'),canonical=True)
    if execution!={'schema':core.SCHEMA+'.execution','model_identities_before_after':[
            core.diag.PIN['initial_model_identity'],core.diag.PIN['teacher_model_identity']],
            'environment':meta['environment'],'gold_positions':6804,'greedy_positions':6885,
            'optimizer_updates':0,'heldout_scored':False,'native_argmax_matches_historical':True}:
        raise ContractError('execution environment/identity receipt mismatch')
    return {'verified':True,'scope':'recorded-full-vocabulary-evidence-recomputed',
        'positions':len(index),'native_forward_replayed':False,'optimizer_updates':0}


def replay(folder,anchor,dataset,teacher,previous,out,result_anchor):
    outcome=verify(folder,anchor,dataset,teacher,previous,out,result_anchor)
    request=parse_json(core.read(folder,'request.json'),canonical=True)
    start=time.monotonic();index=core.jsonl(core.read(out,'index.jsonl'));raw=core.read(out,'logits.f32le')
    models=behavior.restore_models(dataset,teacher)
    def compare(rec,observed,measure):
        offset=rec['offset_bytes']
        if observed!=raw[offset:offset+ar.VECTOR.size]:
            raise ContractError('native FP32 bytes differ at '+rec['record_id'])
    observe(index,models,start+request['plan']['max_replay_seconds'],compare)
    core.verify_request(folder,anchor,dataset,teacher,previous)
    return dict(outcome,native_forward_replayed=True,scope='all-position-exact-fp32-native-replay')
