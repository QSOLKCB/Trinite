"""V2 runtime-aware orchestration; the v1 mathematical primitives stay authoritative.

Checkpoint/probe formats remain v1, bound to a fresh v2 source/request identity.
No v1 globals are rebound, and no old request is migrated or resealed.
"""
from copy import deepcopy
from time import monotonic
import random
import numpy as np
import torch
from safetensors.torch import save, load

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from trinite.training import TrainingState, create_optimizer, require_optimizer
from trinite.comparison_checkpoint import _slot, _hex
from music_learning_v1.numerical import encoded
from music_placement_v1.spec import plan_for, train_rows, diagnostic_rows, development_rows, require_admission
from music_device_trial_v1.adapter import DeviceDecoder, model_identity, target_loss, evaluate, greedy, batch

BACKENDS=('cpu','cuda')
SCHEMA='trinite.music-device-checkpoint.v1'
from music_device_trial_v1.spec import PARITY_TOLERANCES as TOLERANCES
from music_device_trial_v1.numerical import (update_once, _context, _progress, snapshot,
    _rng, _restore_rng, _same_rng, _probe_examples, compare_probes)


def _runtime(backend):
    from .runtime import require_environment
    if backend not in BACKENDS:raise ContractError('fixed trial backend required')
    return require_environment(backend)


def _sync(backend):
    from .runtime import synchronize
    synchronize(backend)


def _state(backend,admission_id):
    _runtime(backend);raw=_context(admission_id);plan=plan_for(17)
    rows=encoded(train_rows('control'));validation=encoded(diagnostic_rows()[:40])
    if len(rows)!=80 or len(validation)!=40:raise ContractError('device fixed training inventory differs')
    model=DeviceDecoder(plan.model,17,backend);_sync(backend)
    initial=model_identity(model)
    _sync(backend);started=monotonic();loss=evaluate(model,rows,8);_sync(backend)
    loss_time=monotonic()-started
    return TrainingState(plan,'dense',model,create_optimizer(model,plan),{'train':rows,'validation':validation},
        identity(raw),admission_id,initial,loss), {'cpu_initialization':float.fromhex(model.cpu_initialization_seconds_hex),
        'model_transfer':float.fromhex(model.model_transfer_seconds_hex),'initial_loss':loss_time,
        'compile':0.,'target_warmup':0.}


def load_checkpoint(payload,metadata,backend,admission_id,request_id):
    """Safe model-only restore; validates optimizer bytes without construction."""
    _runtime(backend);raw=_context(admission_id);require_identity(request_id)
    if type(payload)is not bytes or type(metadata)is not bytes or len(payload)>16777216 or len(metadata)>1048576:
        raise ContractError('device checkpoint type/size mismatch')
    r=parse_json(metadata,canonical=True);plan=plan_for(17)
    context={'schema':SCHEMA,'backend':backend,'seed':17,'request_identity':request_id,'plan':plan.to_dict(),
             'dataset_identity':identity(raw),'admission_identity':admission_id}
    exact_keys(r,set(context)|{'initialization_identity','initial_train_loss_hex','model_identity','step','cursor','target_tokens','history','validation'},'device checkpoint')
    if any(json_bytes(r[k])!=json_bytes(v) for k,v in context.items()):raise ContractError('device checkpoint context mismatch')
    _progress(r,plan)
    if r['step']!=640:raise ContractError('only final640 endpoint admitted')
    model=DeviceDecoder(plan.model,17,backend)
    if model_identity(model)!=r['initialization_identity']:raise ContractError('device initial model differs')
    try:tensors=load(payload)
    except Exception as error:raise ContractError('invalid device safetensors') from error
    names={'model.'+n for n,_ in model.named_parameters()}|{'optimizer.'+n+'.'+s for n,_ in model.named_parameters() for s in ('step','exp_avg','exp_avg_sq')}
    if set(tensors)!=names:raise ContractError('device tensor inventory differs')
    with torch.no_grad():
        for name,p in model.named_parameters():
            value=tensors['model.'+name]
            if value.device.type!='cpu' or value.layout!=torch.strided or value.dtype!=torch.float32 or value.shape!=p.shape or not bool(torch.isfinite(value).all()):
                raise ContractError('invalid device model tensor')
            for slot in ('step','exp_avg','exp_avg_sq'):_slot(tensors['optimizer.'+name+'.'+slot],value,slot,640)
            p.copy_(value.to(p.device))
    model.validate_state()
    if model_identity(model)!=r['model_identity']:raise ContractError('restored model identity differs')
    return model


def infer_models(models,stage,emit):
    if stage not in ('diagnostic','development') or not callable(emit):raise ContractError('fixed inference stage/callback required')
    if (not models or set(models)-set(BACKENDS) or (stage=='diagnostic' and len(models)!=1)
            or (stage=='development' and set(models)!=set(BACKENDS))):raise ContractError('fixed backend endpoint inventory required')
    if len({id(m) for m in models.values()})!=len(models):raise ContractError('independent model objects required')
    params=[id(p) for m in models.values() for p in m.parameters()]
    if len(set(params))!=len(params):raise ContractError('shared endpoint parameters forbidden')
    for backend,model in models.items():
        _runtime(backend)
        if model.backend!=backend or model.config!=plan_for(17).model:raise ContractError('inference backend/config differs')
    rows=diagnostic_rows() if stage=='diagnostic' else development_rows()
    if len(rows)!=(80 if stage=='diagnostic' else 120):raise ContractError('measurement coverage differs')
    inputs=[{k:r[k] for k in ('example_identity','prompt')} for r in rows]
    rng=_rng(models);grad_mode=torch.is_grad_enabled();states={};hooks=[];calls=0;records=[];timing={};greedy_timing={}
    def guard():
        if not _same_rng(rng,models) or torch.is_grad_enabled()!=grad_mode:raise ContractError('device inference mutated RNG/grad mode')
        for backend in models:_runtime(backend)
        for key,state in states.items():
            m=models[key]
            if (model_identity(m)!=state['identity'] or [(n,id(p),p.requires_grad) for n,p in m.named_parameters()]!=state['params']
                    or list(m.named_buffers()) or [(n,id(v),v.training) for n,v in m.named_modules()]!=state['eval_modes']
                    or any(p.grad is not None for p in m.parameters())):raise ContractError('device inference mutated endpoint state')
    def count(module,args):
        nonlocal calls
        calls+=1
        if calls>len(models)*len(rows)*64:raise ContractError('device forward budget exceeded')
    try:
        for key,m in models.items():
            if any(p.grad is not None for p in m.parameters()):raise ContractError('dirty inference gradients')
            states[key]={'identity':model_identity(m),'params':[(n,id(p),p.requires_grad) for n,p in m.named_parameters()],
                         'modes':[(v,v.training) for v in m.modules()]}
            m.eval();states[key]['eval_modes']=[(n,id(v),v.training) for n,v in m.named_modules()]
            hooks.append(m.register_forward_pre_hook(count))
        for backend in BACKENDS:
            if backend not in models:continue
            _sync(backend);started=monotonic();greedy_seconds=0.
            for row in inputs:
                guard();start=calls;_sync(backend);greedy_started=monotonic()
                with torch.no_grad():generated=greedy(models[backend],row['prompt'],64)
                _sync(backend);greedy_seconds+=monotonic()-greedy_started
                if (type(generated)is not list or not 1<=len(generated)<=64 or any(type(t)is not int or not 0<=t<=258 for t in generated)
                        or 257 in generated[:-1] or (generated[-1]!=257 and len(generated)!=64) or calls-start!=len(generated)):
                    raise ContractError('device greedy raw/forward accounting differs')
                guard();record={'backend':backend,'seed':17,'step':640,'stage':stage,
                    'example_identity':row['example_identity'],'generated_tokens':generated}
                raw=json_bytes(record);emit(record)
                if json_bytes(record)!=raw:raise ContractError('callback mutated device prediction')
                records.append(record);guard()
            _sync(backend);timing[backend]=(monotonic()-started).hex();greedy_timing[backend]=greedy_seconds.hex()
        guard()
    finally:
        for hook in hooks:hook.remove()
        for state in states.values():
            for module,mode in state['modes']:module.training=mode
        _restore_rng(rng);torch.set_grad_enabled(grad_mode)
    return {'records':records,'decodes':len(records),'forward_calls':calls,'model_identities_unchanged':True,
            'training_performed':False,'timing_hex':timing,'greedy_seconds_hex':greedy_timing}


def train(backend,request_id,admission_id,write,emit):
    require_identity(request_id)
    if not callable(write) or not callable(emit):raise ContractError('artifact/journal callbacks required')
    state=None;elapsed=0.
    try:
        state,timings=_state(backend,admission_id)
        training_rng=_rng({backend:state.model});training_grad_mode=torch.is_grad_enabled()
        while state.step<640:
            _runtime(backend);_sync(backend);started=monotonic()
            try:update_once(state);_sync(backend)
            finally:elapsed+=monotonic()-started
            if elapsed>120:raise ContractError('cumulative device update deadline exceeded')
        timings['updates']=elapsed
        _sync(backend);started=monotonic();payload,meta=snapshot(state,backend,request_id);_sync(backend)
        write('final.safetensors',payload);write('final-checkpoint.json',meta);_sync(backend)
        timings['serialization']=monotonic()-started
        result=infer_models({backend:deepcopy(state.model)},'diagnostic',emit)
        timings['diagnostics']=float.fromhex(result['timing_hex'][backend])
        write('diagnostic-predictions.json',json_bytes(result['records']))
        if (snapshot(state,backend,request_id)!=(payload,meta) or not _same_rng(training_rng,{backend:state.model})
                or torch.is_grad_enabled()!=training_grad_mode):raise ContractError('diagnostic callback mutated training state/RNG/grad mode')
        _runtime(backend)
        return {'backend':backend,'seed':17,'request_identity':request_id,'admission_identity':admission_id,
            'step':640,'target_tokens':state.target_tokens,'initialization_identity':state.initialization_identity,
            'model_identity':model_identity(state.model),'update_seconds_hex':elapsed.hex(),
            'timing_hex':{k:v.hex() for k,v in timings.items()},'diagnostic_decodes':result['decodes'],
            'diagnostic_forward_calls':result['forward_calls'],
            'diagnostic_greedy_seconds_hex':result['greedy_seconds_hex'][backend]}
    except Exception as error:
        if state is not None:
            try:
                payload,meta=snapshot(state,backend,request_id);write('failure.safetensors',payload);write('failure-checkpoint.json',meta)
            except Exception as failure:write('failure-checkpoint-error.json',json_bytes({'type':type(failure).__name__,'message':str(failure)}))
        write('failure.json',json_bytes({'outcome':'incomplete','type':type(error).__name__,'message':str(error),
              'completed_updates':None if state is None else state.step,'update_seconds_hex':elapsed.hex()}))
        raise


def _probe_one(backend):
    _runtime(backend);plan=plan_for(17);_sync(backend);started=monotonic()
    model=DeviceDecoder(plan.model,17,backend);optimizer=create_optimizer(model,plan)
    tensors={'initial.'+n:p.detach().to('cpu').clone().contiguous() for n,p in model.named_parameters()}
    initial=model_identity(model)
    ids,mask,scored=batch(_probe_examples(),backend)
    logits=model(ids[:,:-1],mask[:,:-1]).logits
    weights=scored[:,1:];targets=int(weights.sum())
    losses=torch.nn.functional.cross_entropy(logits.reshape(-1,259),ids[:,1:].reshape(-1),reduction='none')
    loss=(losses.reshape(weights.shape)*weights).sum()/targets
    if not bool(torch.isfinite(loss)):raise ContractError('nonfinite probe loss')
    loss.backward()
    for n,p in model.named_parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise ContractError('invalid probe gradient')
        tensors['gradient.'+n]=p.grad.detach().to('cpu').clone().contiguous()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(plan.gradient_clip),error_if_nonfinite=True,foreach=False)
    optimizer.step();optimizer.zero_grad(set_to_none=True);model.validate_state()
    for n,p in model.named_parameters():
        cpu=p.detach().to('cpu').clone().contiguous();tensors['parameter.'+n]=cpu
        slots=optimizer.state[p];exact_keys(slots,{'step','exp_avg','exp_avg_sq'},'probe slots')
        for slot,v in slots.items():
            value=v.detach().to('cpu').clone().contiguous();_slot(value,cpu,slot,1)
            tensors[slot+'.'+n]=value
    tensors.update(logits=logits.detach().to('cpu').clone().contiguous(),loss=loss.detach().to('cpu').clone(),
                   gradient_norm=norm.detach().to('cpu').clone())
    _sync(backend)
    return tensors,{'backend':backend,'initialization_identity':initial,'updates':1,'visits':8,'target_tokens':targets,
                    'forward_calls':1,'forward_shape':[8,67,259],'seconds_hex':(monotonic()-started).hex()}


def parity_probe():
    """Exactly one disposable parity/warmup update on each backend, no target state."""
    _runtime('cpu');_runtime('cuda')
    # Snapshot both global RNGs after admitted CUDA initialization.
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(0).clone()
    py_rng=random.getstate();np_rng=np.random.get_state();grad_mode=torch.is_grad_enabled()
    tensors={};reports={}
    try:
        for backend in BACKENDS:
            values,reports[backend]=_probe_one(backend)
            tensors.update({backend+'.'+k:v for k,v in values.items()})
        if (not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state(0))
                or py_rng!=random.getstate() or np_rng[0]!=np.random.get_state()[0]
                or not np.array_equal(np_rng[1],np.random.get_state()[1]) or np_rng[2:]!=np.random.get_state()[2:]
                or torch.is_grad_enabled()!=grad_mode):
            raise ContractError('disposable parity probe changed RNG')
        payload=save(tensors)
        if len(payload)>16777216:raise ContractError('probe artifact exceeds bound')
        report={'schema':'trinite.music-device-probe.v1','by_backend':reports,'probe_input_identity':identity(json_bytes(_probe_examples())),
                'tolerances':TOLERANCES,'budget':{'updates':2,'visits':16,'target_tokens':32,'forward_calls':2},'payload_identity':identity(payload)}
        # Retain failed numerical comparisons as data; the controller gates on passed.
        return payload,report
    finally:
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng,0);random.setstate(py_rng);np.random.set_state(np_rng)
        torch.set_grad_enabled(grad_mode)
