"""Source-bound foundations replay using shared named tensor/progress checks."""
import torch
from safetensors.torch import load

from .contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from .comparison_checkpoint import tensor_payload, progress, _slot
from .foundations_training import state_for, sources
from .foundations_data import dataset
from .training import model_identity, require_environment

SCHEMA='trinite.foundations-checkpoint.v1'


def _context(state,workload,request_identity):
    require_identity(request_identity)
    raw,manifest=dataset(workload)
    if (state.dataset_identity,state.manifest_identity)!=(identity(raw),identity(manifest)):
        raise ContractError('foundations state/workload admission mismatch')
    return {'schema':SCHEMA,'workload':workload,'lane':state.lane,'seed':state.config.seed,
        'request_identity':request_identity,'plan':state.config.to_dict(),'source':sources(),
        'environment':require_environment(),'dataset_identity':state.dataset_identity,
        'manifest_identity':state.manifest_identity,'initialization_identity':state.initialization_identity,
        'initial_train_loss_hex':state.initial_train_loss}


def snapshot(state,workload,request_identity):
    progress(state)
    context=_context(state,workload,request_identity)
    payload=tensor_payload(state)
    metadata={**context,'model_identity':model_identity(state.model),
        'step':state.step,'cursor':state.cursor,'target_tokens':state.target_tokens,
        'history':state.history,'validation':state.validation}
    raw=json_bytes(metadata)
    if len(raw)>2*1024*1024:raise ContractError('foundations metadata exceeds budget')
    return payload,raw


def restore(payload,metadata,workload,lane,seed,request_identity,expected_identities,config=None):
    if (type(payload) is not bytes or type(metadata) is not bytes or len(payload)>32*1024*1024
            or len(metadata)>2*1024*1024 or (identity(payload),identity(metadata))!=expected_identities):
        raise ContractError('foundations snapshot identity/type/size mismatch')
    r=parse_json(metadata,canonical=True);state=state_for(workload,lane,seed,config)
    expected=_context(state,workload,request_identity)
    exact_keys(r,set(expected)|{'model_identity','step','cursor','target_tokens','history','validation'},'foundations snapshot')
    if any(json_bytes(r[k])!=json_bytes(v) for k,v in expected.items()):
        raise ContractError('foundations snapshot context mismatch')
    state.step,state.cursor,state.target_tokens=r['step'],r['cursor'],r['target_tokens']
    state.history,state.validation=r['history'],r['validation'];progress(state)
    try:tensors=load(payload)
    except Exception as error:raise ContractError('invalid foundations safetensors') from error
    names={'model.'+n for n,_ in state.model.named_parameters()}
    if state.step:names|={'optimizer.'+n+'.'+slot for n,_ in state.model.named_parameters()
                         for slot in ('step','exp_avg','exp_avg_sq')}
    if set(tensors)!=names:raise ContractError('foundations tensor inventory mismatch')
    with torch.no_grad():
        for name,p in state.model.named_parameters():
            value=tensors['model.'+name]
            if value.shape!=p.shape or value.dtype!=p.dtype or not bool(torch.isfinite(value).all()):
                raise ContractError('invalid foundations model tensor')
            p.copy_(value)
            if state.step:
                slots={}
                for slot in ('step','exp_avg','exp_avg_sq'):
                    value=tensors['optimizer.'+name+'.'+slot];_slot(value,p,slot,state.step)
                    slots[slot]=value.clone()
                state.optimizer.state[p]=slots
    progress(state)
    if model_identity(state.model)!=r['model_identity']:raise ContractError('foundations model identity mismatch')
    return state
