# SPDX-License-Identifier: MPL-2.0
"""Independent recorded recalculation and explicitly separate native forward replay."""
import math
import time
from pathlib import Path

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from . import custody as io, metrics, spec

CHECKPOINT_FILES = {'tensors.safetensors','metadata.json','predictions.json','index.jsonl',
    'logits.f32le','metrics.jsonl','summary.json','evaluation.json','weights-rights.json',
    'observation.json','prediction-progress.jsonl','manifest.json'}
PENDING_CHECKPOINT_FILES={'.pending-'+n for n in CHECKPOINT_FILES if n.endswith('.json')}


def hex_number(value):
    if type(value) is not str:
        raise ContractError('canonical hex number required')
    try:
        number=float.fromhex(value)
    except ValueError as exc:
        raise ContractError('invalid hex number') from exc
    if not math.isfinite(number) or number<0 or number.hex()!=value:
        raise ContractError('nonfinite, negative or noncanonical hex number')
    return number


def progress(request, meta, step):
    if (type(meta['step']) is not int or meta['step']!=step
            or type(meta['cursor']) is not int or meta['cursor']!=4*step
            or type(meta['target_tokens']) is not int or len(meta['history'])!=step):
        raise ContractError('checkpoint progress differs')
    rows={r['id']:r for r in request['panel']};total=0
    for i,entry in enumerate(meta['history']):
        ids=request['selection']['visit_ids'][4*i:4*(i+1)]
        targets=sum(rows[r]['supervised_tokens'] for r in ids);total+=targets
        if (set(entry)!={'step','examples','target_tokens','loss_hex','gradient_norm_hex'}
                or type(entry['step']) is not int or entry['step']!=i+1
                or type(entry['target_tokens']) is not int or entry['target_tokens']!=targets
                or entry['examples']!=ids):
            raise ContractError('history differs from frozen visits/targets')
        hex_number(entry['loss_hex']);hex_number(entry['gradient_norm_hex'])
    expected_steps=[s for s in spec.STEPS if 0<s<=step]
    if [v['step'] for v in meta['validation']]!=expected_steps:
        raise ContractError('native evaluation schedule mismatch')
    for v in meta['validation']:
        if set(v)!={'step','loss_hex'}:
            raise ContractError('native evaluation record mismatch')
        hex_number(v['loss_hex'])
    if meta['target_tokens']!=total or total>4096:
        raise ContractError('checkpoint target count mismatch')
    hex_number(meta['initial_train_loss_hex'])
    if hex_number(meta['native_elapsed_hex'])>600:
        raise ContractError('completed checkpoint over cumulative native budget')


def jsonl(path):
    return [parse_json(line,canonical=True) for line in io.read(path).splitlines(keepends=True)]


def partial_jsonl(path):
    if not Path(path).exists():return [],b''
    raw=io.read(path);cut=raw.rfind(b'\n')+1
    return [parse_json(line,canonical=True) for line in raw[:cut].splitlines(keepends=True)],raw[cut:]


def partial_checkpoint(request, folder):
    """Recalculate every completed metric line; preserve, never repair, torn tails."""
    from teacher_logit_v1 import arithmetic as ar
    folder=Path(folder)
    if {p.name for p in folder.iterdir()}-(CHECKPOINT_FILES | PENDING_CHECKPOINT_FILES):
        raise ContractError('unexpected partial checkpoint artifact')
    index,index_tail=partial_jsonl(folder/'index.jsonl')
    measures,metric_tail=partial_jsonl(folder/'metrics.jsonl')
    vectors=io.read(folder/'logits.f32le') if (folder/'logits.f32le').exists() else b''
    predictions,prediction_tail=partial_jsonl(folder/'prediction-progress.jsonl')
    if len(measures)>len(index) or len(vectors)<len(measures)*ar.VECTOR.size:
        raise ContractError('partial metric has no retained prefix/vector')
    # Producer streams index, vector, metric in that exact order. A kill may
    # leave one unmatched observation; it cannot explain arbitrary surplus.
    if len(index)>len(measures)+1 or len(vectors)>(len(measures)+1)*ar.VECTOR.size:
        raise ContractError('partial stream inventory cannot arise from declared producer')
    row_index=0;probe='gold';position=0;generated=[];finished=[]
    for i,(rec,given) in enumerate(zip(index,measures)):
        if row_index>=len(request['panel']):raise ContractError('extra partial observation')
        row=request['panel'][row_index];gold=[*row['answer'].encode('ascii'),257]
        prefix=[256,*row['prompt'].encode('ascii')]+(gold[:position] if probe=='gold' else generated)
        expected=metrics.entry(row,probe,position,prefix,i*ar.VECTOR.size)
        if rec!=expected:raise ContractError('partial diagnostic prefix/order mismatch')
        measure=ar.metrics(ar.unpack(vectors[i*ar.VECTOR.size:(i+1)*ar.VECTOR.size]),rec['target'])
        if measure!=given:raise ContractError('partial numerical record mismatch')
        position+=1
        if probe=='gold' and position==len(gold):probe='greedy';position=0
        elif probe=='greedy':
            generated.append(measure['argmax'])
            if generated[-1]==257 or position==5:
                finished.append({'id':row['id'],'tokens':generated})
                row_index+=1;probe='gold';position=0;generated=[]
    if predictions!=finished[:len(predictions)] or len(predictions)>len(finished):
        raise ContractError('partial prediction differs from complete retained argmax observations')
    # If aggregation finished before a later IO failure, verify it as well.
    if (folder/'summary.json').exists():
        if len(finished)!=len(request['panel']) or len(index)!=len(measures):
            raise ContractError('partial aggregate lacks exact complete observations')
        calculated=metrics.summarize(request['panel'],finished,index,measures)
        if io.read(folder/'summary.json')!=json_bytes(calculated):raise ContractError('partial aggregate mismatch')
    if (folder/'predictions.json').exists() and io.load(folder/'predictions.json')!=finished:
        raise ContractError('partial complete-output record differs from observations')
    return {'step_directory':folder.name,'metrics_recomputed':len(measures),
        'completed_predictions':len(predictions),'index_tail_bytes':len(index_tail),
        'metric_tail_bytes':len(metric_tail),'prediction_tail_bytes':len(prediction_tail),
        'unmatched_vector_bytes':len(vectors)-len(measures)*ar.VECTOR.size,
        'pending_json_artifacts':sorted(p.name for p in folder.iterdir() if p.name in PENDING_CHECKPOINT_FILES),
        'checkpoint_complete':False}


def checkpoint(request, anchor, folder, checkpoint_anchor, step):
    folder=Path(folder)
    manifest=io.verify_seal(folder,checkpoint_anchor)
    if ({p.name for p in folder.iterdir()}!=CHECKPOINT_FILES
            or manifest!={'schema':spec.SCHEMA+'.checkpoint-manifest',
                'request_identity':anchor,'step':step,'members':io.inventory(folder,('manifest.json',))}):
        raise ContractError('checkpoint inventory/schema mismatch')
    meta=io.load(folder/'metadata.json')
    if (set(meta)!={'schema','request_identity','seed','config','initialization_identity',
            'initial_train_loss_hex','identities','step','cursor','target_tokens','history','validation',
            'native_elapsed_hex','peak_rss_kib'} or meta['schema']!=spec.SCHEMA+'.checkpoint'
            or meta['request_identity']!=anchor or meta['seed']!=request['seed']
            or meta['config']!=request['native_config']):
        raise ContractError('checkpoint request/config/metadata mismatch')
    progress(request,meta,step)
    predictions=io.load(folder/'predictions.json');index=jsonl(folder/'index.jsonl')
    if predictions!=jsonl(folder/'prediction-progress.jsonl'):
        raise ContractError('prediction journal differs from complete outputs')
    measures=metrics.recalculate(request['panel'],predictions,index,io.read(folder/'logits.f32le'))
    if measures!=jsonl(folder/'metrics.jsonl'):
        raise ContractError('saved probabilities/ranks/NLL/margins not reproduced')
    summary=metrics.summarize(request['panel'],predictions,index,measures)
    if json_bytes(summary)!=io.read(folder/'summary.json'):
        raise ContractError('checkpoint aggregate mismatch')
    obs=io.load(folder/'observation.json')
    if obs!={'state_unchanged':True,'rng_unchanged':True,'before':meta['identities'],
            'after':meta['identities'],'optimizer_updates':0}:
        raise ContractError('observer immutability record mismatch')
    evaluation=io.load(folder/'evaluation.json')
    if set(evaluation)!={'train_loss_hex','heldout_scored','geometry_loss'} or evaluation['heldout_scored'] is not False or evaluation['geometry_loss']!='0':
        raise ContractError('native evaluation scope mismatch')
    hex_number(evaluation['train_loss_hex'])
    expected_loss=meta['initial_train_loss_hex'] if step==0 else meta['validation'][-1]['loss_hex']
    if evaluation['train_loss_hex']!=expected_loss:
        raise ContractError('checkpoint loss differs from native training evaluation')
    from teacher_training_v1.numerical import weight_rights_record
    if io.read(folder/'weights-rights.json')!=json_bytes(weight_rights_record(
            request,anchor,io.read(folder/'tensors.safetensors'))):
        raise ContractError('checkpoint rights/lineage mismatch')
    return {'step':step,'summary':summary,'metadata':meta,'evaluation':evaluation}


def verify(folder, manifest_anchor, request_anchor, dataset):
    folder=Path(folder)
    manifest=io.verify_seal(folder,manifest_anchor)
    request=spec.verify_request(folder/'request.json',request_anchor,dataset)
    if (manifest.get('schema')!=spec.SCHEMA+'.attempt-manifest'
            or manifest.get('request_identity')!=request_anchor or manifest.get('seed')!=request['seed']):
        raise ContractError('attempt manifest source/request mismatch')
    spec.approval(folder/'approval.json',manifest['approval_identity'],request,request_anchor)
    if (folder/'approval-set.json').exists():
        spec.verify_approval_records(io.load(folder/'approval-set.json'),request,dataset)
    status=io.load(folder/'supervisor.json')
    if status['status']!=manifest['status'] or status['process_resolved'] is not True:
        raise ContractError('unresolved or contradictory process outcome')
    if (folder/'environment.json').exists() and io.load(folder/'environment.json')!=request['environment']:
        raise ContractError('observed numerical environment mismatch')
    records=[]
    partials=[]
    for step in spec.STEPS:
        cp=folder/f'step-{step:03d}'
        if (cp/'manifest.json').exists():
            records.append(checkpoint(request,request_anchor,cp,identity(io.read(cp/'manifest.json')),step))
        elif cp.exists():
            partials.append(partial_checkpoint(request,cp))
    if [r['step'] for r in records]!=list(spec.STEPS[:len(records)]):
        raise ContractError('checkpoint schedule has a gap')
    initial={r['metadata']['initialization_identity'] for r in records}
    if len(initial)>1:
        raise ContractError('initialization identity changed between checkpoints')
    for prev,next_ in zip(records,records[1:]):
        if (next_['metadata']['history'][:prev['step']]!=prev['metadata']['history']
                or hex_number(next_['metadata']['native_elapsed_hex'])<hex_number(prev['metadata']['native_elapsed_hex'])):
            raise ContractError('checkpoint trajectory changed or native budget reset')
    journal_raw=io.read(folder/'progress.jsonl') if (folder/'progress.jsonl').exists() else b''
    tail=b''
    if journal_raw and not journal_raw.endswith(b'\n'):
        if status['status']=='COMPLETED':raise ContractError('truncated successful journal')
        cut=journal_raw.rfind(b'\n')+1;tail=journal_raw[cut:];journal_raw=journal_raw[:cut]
    journal=[parse_json(line,canonical=True) for line in journal_raw.splitlines(keepends=True)]
    last_elapsed=0.
    for i,event in enumerate(journal):
        if set(event)!={'history','native_elapsed_hex'}:
            raise ContractError('invalid update journal')
        elapsed=hex_number(event['native_elapsed_hex'])
        if elapsed<last_elapsed or elapsed>600:
            raise ContractError('journal cumulative native budget mismatch')
        last_elapsed=elapsed
    if journal:
        # Even updates after the last fixed checkpoint must match the full schedule.
        history=[e['history'] for e in journal]
        check={'step':len(history),'cursor':4*len(history),
            'target_tokens':sum(e['target_tokens'] for e in history),'history':history,
            'validation':[{'step':s,'loss_hex':0.0.hex()} for s in spec.STEPS if 0<s<=len(history)],
            'initial_train_loss_hex':0.0.hex(),'native_elapsed_hex':last_elapsed.hex()}
        if len(history)>256:raise ContractError('too many recorded updates')
        progress(request,check,len(history))
    for rec in records:
        if [e['history'] for e in journal[:rec['step']]]!=rec['metadata']['history']:
            raise ContractError('update journal differs from checkpoint history')
    if status['status']=='COMPLETED':
        required={'request.json','approval.json','approval-set.json','started.json','process.json','stdout.log','stderr.log',
            'environment.json','progress.jsonl','completed.json','supervisor.json','manifest.json'}
        if not required<={p.name for p in folder.iterdir()}:
            raise ContractError('successful attempt missing required artifacts')
        completed=io.load(folder/'completed.json')
        if (status['child_returncode']!=0 or status['timed_out'] or status['interrupted'] or status['parent_error']
                or hex_number(status['elapsed_seconds_hex'])>900
                or len(records)!=5 or len(journal)!=256 or (folder/'failure.json').exists()):
            raise ContractError('incomplete, failed or over-budget attempt claimed completed')
        expected={'schema':spec.SCHEMA+'.completed','checkpoints':[{'step':s,
            'manifest_identity':identity(io.read(folder/f'step-{s:03d}'/'manifest.json'))} for s in spec.STEPS],
            'optimizer_updates':256,'target_tokens':request['selection']['scheduled_target_tokens'],
            'native_elapsed_hex':records[-1]['metadata']['native_elapsed_hex'],
            'initialization_identity':next(iter(initial)),'optimizer_replayed':False,'heldout_scored':False,
            'resources':completed['resources']}
        res=completed['resources']
        if set(res)!={'peak_rss_kib','user_seconds_hex','system_seconds_hex'} or type(res['peak_rss_kib']) is not int or res['peak_rss_kib']<0:
            raise ContractError('invalid native resource observation')
        hex_number(res['user_seconds_hex']);hex_number(res['system_seconds_hex'])
        if completed!=expected:
            raise ContractError('completion receipt mismatch')
    elif status['status']!='FAILED':
        raise ContractError('unknown attempt status')
    allowed={'request.json','approval.json','approval-set.json','started.json','process.json','stdout.log','stderr.log',
        'environment.json','progress.jsonl','completed.json','failure.json','supervisor.json','manifest.json'}
    allowed|={f'step-{s:03d}' for s in spec.STEPS}
    if status['status']=='FAILED':
        allowed|={'.pending-'+n for n in allowed if n.endswith('.json')}
    if {p.name for p in folder.iterdir()}-allowed:
        raise ContractError('unexpected attempt artifacts')
    failure=io.load(folder/'failure.json') if (folder/'failure.json').exists() else None
    completed_updates=(256 if status['status']=='COMPLETED' else
        failure.get('updates_completed') if failure else 0 if status['process_started'] is False else None)
    consumed_targets=(request['selection']['scheduled_target_tokens'] if status['status']=='COMPLETED' else
        failure.get('target_tokens') if failure else 0 if status['process_started'] is False else None)
    return {'schema':spec.SCHEMA+'.recorded-verification','verified':True,'seed':request['seed'],
        'status':status['status'],'checkpoints':records,'request_identity':request_anchor,
        'manifest_identity':manifest_anchor,'supervisor':status,
        'native_forward_replayed':False,'optimizer_replayed':False,
        'journal_updates':len(journal),'journal_target_tokens':sum(e['history']['target_tokens'] for e in journal),
        'partial_journal_tail_identity':identity(tail) if tail else None,
        'failure':failure,'reported_completed_updates':completed_updates,'reported_target_tokens':consumed_targets,
        'consumption_scope':'native completed-update receipt; hard-kill unjournaled progress may be unknown',
        'partial_checkpoints':partials,
        'partial_scope':'complete partial metric records recomputed; torn/unmatched bytes retained without numerical claims',
        'tensor_scope':'raw file identities; semantic model/AdamW restoration checked by native replay',
        'heldout_scored':False}


def replay(folder, manifest_anchor, request_anchor, dataset):
    deadline=time.monotonic()+900
    outcome=verify(folder,manifest_anchor,request_anchor,dataset)
    if not outcome['checkpoints']:
        return {'schema':spec.SCHEMA+'.forward-replay','verified':True,
            'manifest_identity':manifest_anchor,'request_identity':request_anchor,
            'steps_replayed':[],'native_forward_replayed':False,'optimizer_replayed':False,
            'status':'NO_COMPLETE_CHECKPOINTS','scope':'recorded failure verified; no native replay performed'}
    folder=Path(folder);request=io.load(folder/'request.json')
    from . import native
    from teacher_logit_v1.native import forward_vector
    from trinite.training import evaluate
    if native.environment()!=request['environment']:
        raise ContractError('forward replay numerical environment mismatch')
    def forward(model,prefix):
        if time.monotonic()>deadline:raise ContractError('forward replay deadline exceeded')
        return forward_vector(model,prefix)[0]
    for cp in outcome['checkpoints']:
        path=folder/f"step-{cp['step']:03d}"
        state=native.restore(request,request_anchor,path)
        before=native.guard(state)
        index,raw,measures,predictions=metrics.observe(request['panel'],lambda prefix:forward(state.model,prefix))
        if (raw!=io.read(path/'logits.f32le') or json_bytes(predictions)!=io.read(path/'predictions.json')
                or b''.join(json_bytes(v) for v in index)!=io.read(path/'index.jsonl')
                or b''.join(json_bytes(v) for v in measures)!=io.read(path/'metrics.jsonl')
                or evaluate(state.model,state.data['train'],4)!=cp['evaluation']['train_loss_hex']):
            raise ContractError('native forward exact-byte replay mismatch')
        native.unchanged(before,native.guard(state))
    io.verify_seal(folder,manifest_anchor)
    if time.monotonic()>deadline:raise ContractError('forward replay deadline exceeded at completion')
    return {'schema':spec.SCHEMA+'.forward-replay','verified':True,
        'manifest_identity':manifest_anchor,'request_identity':request_anchor,
        'steps_replayed':[c['step'] for c in outcome['checkpoints']],
        'native_forward_replayed':True,'optimizer_replayed':False,
        'equality':'exact FP32 bytes, predictions, metrics and native loss',
        'scope':'same-host pinned environment; not independent-host reproduction'}
