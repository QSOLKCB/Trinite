# SPDX-License-Identifier: MPL-2.0
"""Execution adapter only: native Decoder/AdamW/loss/step/checkpointer unchanged."""
import hashlib
import os
from pathlib import Path
import random
import resource
import sys
import time

from trinite.contracts import ContractError, ModelConfig, identity, json_bytes
from . import custody as io, metrics, spec


def environment():
    """Actual pinned CPU runtime plus interpreter and installed implementation bytes."""
    from teacher_training_v1.numerical import configure
    import torch
    runtime = configure()
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise ContractError('warn-only determinism is not permitted')
    members = []
    # Include stdlib and installed numerical code/libraries, not just version labels.
    for label, root in (('venv',Path(sys.prefix)),('python',Path(sys.base_prefix))):
        for p in sorted(root.rglob('*')):
            if '__pycache__' in p.parts or p.suffix in ('.pyc','.pyo') or p.is_dir():
                continue
            if not p.is_file():
                raise ContractError('non-file numerical environment member')
            h=hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):
                    h.update(chunk)
            members.append({'path':label+'/'+p.relative_to(root).as_posix(),
                'sha256':h.hexdigest(),'symlink':os.readlink(p) if p.is_symlink() else None})
    libraries=[]
    mapped={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
            if line.split()[-1].startswith('/') and '.so' in line.split()[-1]}
    for name in sorted(mapped):
        h=hashlib.sha256()
        with open(name,'rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        libraries.append({'path':name,'sha256':h.hexdigest()})
    return {'runtime':runtime,'executable':str(Path(sys.executable).absolute()),
            'resolved_executable':str(Path(sys.executable).resolve()),
            'prefix':sys.prefix,'base_prefix':sys.base_prefix,'members':members,
            'mapped_libraries':libraries,'numerical_environment_variables':{key:os.environ.get(key) for key in
                ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONHASHSEED',
                 'ATEN_CPU_CAPABILITY','MKL_CBWR','LD_LIBRARY_PATH','LD_PRELOAD')}}


def config(request):
    from trinite.training import RunConfig
    cfg = RunConfig(model=ModelConfig.from_dict(request['model']),seed=request['seed'],
        steps=256,batch_size=4,learning_rate='0.001',validation_every=64,
        max_seconds=600,max_target_tokens=4096)
    if cfg.to_dict()!=request['native_config']:
        raise ContractError('native RunConfig differs from frozen optimizer/configuration')
    return cfg


def encoded_panel(request):
    from trinite.tokenizer import ByteTokenizer
    tok=ByteTokenizer(config(request).model.context_length)
    rows=[]
    for row in request['panel']:
        enc=tok.encode(row['prompt']+row['answer'],answer_start=len(row['prompt'].encode('ascii')))
        rows.append({'identity':row['id'],'ids':enc.input_ids,'mask':enc.attention_mask,'loss':enc.loss_mask})
    visits=[rows[i%len(rows)]['identity'] for i in range(1024)]
    if visits != request['selection']['visit_ids']:
        raise ContractError('native visit order differs from frozen schedule')
    return rows


def fresh_state(request, anchor):
    from trinite.model import Decoder
    from trinite.training import TrainingState, create_optimizer, evaluate, model_identity, require_environment
    require_environment()
    cfg=config(request)
    rows=encoded_panel(request)
    model=Decoder(cfg.model,lane='dense',seed=cfg.seed,parameter_budget=2_000_000)
    if sum(p.numel() for p in model.parameters()) != 90624:
        raise ContractError('native model parameter inventory differs')
    optimizer=create_optimizer(model,cfg)
    if optimizer.state:
        raise ContractError('fresh optimizer contains inherited moments')
    return TrainingState(cfg,'dense',model,optimizer,{'train':rows,'validation':rows},
        request['dataset_train_identity'],anchor,model_identity(model),evaluate(model,rows,4))


def identities(state):
    from trinite.training import model_identity
    from trinite.inspection import tensor_identity
    return {'model':model_identity(state.model),'optimizer':identity(json_bytes({name:{
        slot:tensor_identity(value) for slot,value in state.optimizer.state.get(p,{}).items()}
        for name,p in state.model.named_parameters()}))}


def guard(state):
    import torch
    from trinite.comparison_checkpoint import tensor_payload
    return (identities(state),identity(tensor_payload(state)),torch.get_rng_state().clone(),
        random.getstate(),json_bytes({'step':state.step,'cursor':state.cursor,
        'targets':state.target_tokens,'history':state.history,'validation':state.validation,
        'data':{split:[dict(e,ids=list(e['ids']),mask=list(e['mask']),loss=list(e['loss']))
                      for e in rows] for split,rows in state.data.items()},
        'config':state.config.to_dict()}),
        [(name,module.training) for name,module in state.model.named_modules()])


def unchanged(before, after):
    if (before[:2]!=after[:2] or not before[2].equal(after[2]) or before[3:]!=after[3:]):
        raise ContractError('diagnostics changed model/optimizer/progress/RNG state')


def metadata(state, request, anchor, native_elapsed):
    return {'schema':spec.SCHEMA+'.checkpoint','request_identity':anchor,
        'seed':request['seed'],'config':state.config.to_dict(),
        'initialization_identity':state.initialization_identity,
        'initial_train_loss_hex':state.initial_train_loss,'identities':identities(state),
        'step':state.step,'cursor':state.cursor,'target_tokens':state.target_tokens,
        'history':state.history,'validation':state.validation,
        'native_elapsed_hex':float(native_elapsed).hex(),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def diagnostic(state, request, anchor, folder, native_elapsed):
    from trinite.comparison_checkpoint import tensor_payload
    from trinite.training import evaluate
    from teacher_logit_v1.native import forward_vector
    from teacher_training_v1.numerical import weight_rights_record
    before=guard(state)
    folder=io.fresh(folder)
    payload=tensor_payload(state)
    io.write(folder/'tensors.safetensors',payload)
    meta=metadata(state,request,anchor,native_elapsed)
    io.record(folder/'metadata.json',meta)
    # Stream observations before completing an aggregate, so a killed diagnostic
    # retains the actual prefixes/logits already observed. Never erase partials.
    with (folder/'index.jsonl').open('xb') as idx, (folder/'logits.f32le').open('xb') as logits, \
            (folder/'metrics.jsonl').open('xb') as met, (folder/'prediction-progress.jsonl').open('xb') as pred:
        def retain(rec,raw,measure):
            for stream,payload in ((idx,json_bytes(rec)),(logits,raw),(met,json_bytes(measure))):
                stream.write(payload);stream.flush()
        def retain_prediction(value):
            pred.write(json_bytes(value));pred.flush()
        index,vectors,measures,predictions=metrics.observe(request['panel'],
            lambda prefix:forward_vector(state.model,prefix)[0],retain,retain_prediction)
        for stream in (idx,logits,met,pred):os.fsync(stream.fileno())
    io.record(folder/'predictions.json',predictions)
    io.record(folder/'summary.json',metrics.summarize(request['panel'],predictions,index,measures))
    io.record(folder/'evaluation.json',{'train_loss_hex':evaluate(state.model,state.data['train'],4),
        'heldout_scored':False,'geometry_loss':'0'})
    io.record(folder/'weights-rights.json',weight_rights_record(request,anchor,payload))
    unchanged(before,guard(state))
    io.record(folder/'observation.json',{'state_unchanged':True,'rng_unchanged':True,
        'before':before[0],'after':identities(state),'optimizer_updates':0})
    return io.seal(folder,{'schema':spec.SCHEMA+'.checkpoint-manifest','request_identity':anchor,'step':state.step})


def restore(request, anchor, folder):
    from trinite.comparison_checkpoint import restore_tensors
    state=fresh_state(request,anchor)
    meta=io.load(Path(folder)/'metadata.json')
    if (meta['request_identity']!=anchor or meta['seed']!=request['seed']
            or meta['config']!=state.config.to_dict()
            or meta['initialization_identity']!=state.initialization_identity
            or meta['initial_train_loss_hex']!=state.initial_train_loss):
        raise ContractError('checkpoint initialization/source/config mismatch')
    for name in ('step','cursor','target_tokens','history','validation'):
        setattr(state,name,meta[name])
    restore_tensors(state,io.read(Path(folder)/'tensors.safetensors'),meta['identities']['model'])
    if identities(state)!=meta['identities']:
        raise ContractError('restored model or optimizer identity mismatch')
    return state


class NativeBudget:
    """One cumulative between-update budget; a segment never gets a new allowance."""
    def __init__(self, limit=600, clock=time.monotonic):
        self.limit=limit; self.elapsed=0.; self.clock=clock

    def step(self, action):
        if self.elapsed >= self.limit:
            raise ContractError('cumulative native time budget exhausted')
        start=self.clock()
        try:
            action()
        finally:
            self.elapsed+=self.clock()-start
        if self.elapsed > self.limit:
            raise ContractError('cumulative native time budget exceeded')


def resources():
    used=resource.getrusage(resource.RUSAGE_SELF)
    return {'peak_rss_kib':used.ru_maxrss,'user_seconds_hex':used.ru_utime.hex(),
            'system_seconds_hex':used.ru_stime.hex()}


def execute(request, anchor, folder):
    """Called only after the supervisor and worker independently validate approval."""
    from trinite.training import run_steps
    folder=Path(folder); state=None; budget=NativeBudget()
    try:
        actual=environment()
        if json_bytes(actual)!=json_bytes(request['environment']):
            raise ContractError('pinned numerical implementation/environment drift')
        io.record(folder/'environment.json',actual)
        state=fresh_state(request,anchor)
        checkpoints=[]
        checkpoints.append({'step':0,'manifest_identity':diagnostic(state,request,anchor,folder/'step-000',0.)})
        with (folder/'progress.jsonl').open('xb') as progress:
            for step in range(1,257):
                budget.step(lambda:run_steps(state,stop_after=step))
                progress.write(json_bytes({'history':state.history[-1],
                    'native_elapsed_hex':budget.elapsed.hex()}))
                progress.flush();os.fsync(progress.fileno())
                if step in spec.STEPS:
                    checkpoints.append({'step':step,'manifest_identity':diagnostic(
                        state,request,anchor,folder/f'step-{step:03d}',budget.elapsed)})
        if state.target_tokens!=request['selection']['scheduled_target_tokens']:
            raise ContractError('final target schedule mismatch')
        io.record(folder/'completed.json',{'schema':spec.SCHEMA+'.completed','checkpoints':checkpoints,
            'optimizer_updates':state.step,'target_tokens':state.target_tokens,
            'native_elapsed_hex':budget.elapsed.hex(),'initialization_identity':state.initialization_identity,
            'optimizer_replayed':False,'heldout_scored':False,'resources':resources()})
    except BaseException as exc:
        io.record(folder/'failure.json',{'schema':spec.SCHEMA+'.failure','type':type(exc).__name__,
            'message':str(exc)[:2000],'updates_completed':None if state is None else state.step,
            'target_tokens':None if state is None else state.target_tokens,
            'native_elapsed_hex':budget.elapsed.hex(),'resume_authorized':False})
        raise
