"""Safe comparison snapshots, with explicit expected identities and replay scope."""
import math

import torch
from safetensors.torch import load, save

from .contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json
from .training import require_environment, require_optimizer, model_identity
from .comparison_training import sources, state_for

SCHEMA = 'trinite.comparison-checkpoint.v1'


def _hex(value):
    try:
        number = float.fromhex(value)
    except (ValueError, TypeError) as error:
        raise ContractError('checkpoint requires canonical float hex') from error
    if not math.isfinite(number) or number < 0 or number.hex() != value:
        raise ContractError('checkpoint requires finite nonnegative canonical float hex')
    return number


def progress(state):
    if state.model.config != state.config.model or state.model.lane != state.lane:
        raise ContractError('comparison state model/plan/lane mismatch')
    integer(state.step, 'comparison step', 0, state.config.steps)
    if (type(state.cursor) is not int or state.cursor != state.step*state.config.batch_size
            or type(state.history) is not list or len(state.history) != state.step
            or type(state.target_tokens) is not int):
        raise ContractError('comparison progress mismatch')
    train = state.data['train']; total = 0
    for i, entry in enumerate(state.history):
        exact_keys(entry, {'step', 'loss_hex', 'gradient_norm_hex', 'target_tokens', 'examples'}, 'comparison history')
        batch = [train[(i*state.config.batch_size+j) % len(train)] for j in range(state.config.batch_size)]
        targets = sum(sum(e['loss']) for e in batch)
        if (type(entry['step']) is not int or entry['step'] != i+1
                or type(entry['target_tokens']) is not int or entry['target_tokens'] != targets
                or entry['examples'] != [e['identity'] for e in batch]):
            raise ContractError('comparison history differs from admitted schedule')
        total += targets
        _hex(entry['loss_hex']); _hex(entry['gradient_norm_hex'])
    if total != state.target_tokens or total > state.config.max_target_tokens:
        raise ContractError('comparison target count mismatch')
    expected = [i for i in range(1, state.step+1) if i % state.config.validation_every == 0 or i == state.config.steps]
    if type(state.validation) is not list or len(state.validation) != len(expected):
        raise ContractError('comparison validation schedule mismatch')
    for i, entry in zip(expected, state.validation):
        exact_keys(entry, {'step', 'loss_hex'}, 'comparison validation')
        if type(entry['step']) is not int or entry['step'] != i:
            raise ContractError('comparison validation step mismatch')
        _hex(entry['loss_hex'])
    _hex(state.initial_train_loss)
    state.model.validate_state(); require_optimizer(state)


def _slot(value, parameter, slot, step):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32
            or value.layout != torch.strided or value.shape != (() if slot == 'step' else parameter.shape)
            or not bool(torch.isfinite(value).all()) or (slot == 'step' and float(value) != step)
            or (slot == 'exp_avg_sq' and bool((value < 0).any()))):
        raise ContractError('invalid comparison optimizer tensor')


def snapshot(state, workload, request_identity):
    progress(state)
    tensors = {}
    for name, p in state.model.named_parameters():
        if p.grad is not None:
            raise ContractError('snapshot requires cleared completed-step gradients')
        tensors['model.'+name] = p.detach().clone().contiguous()
        slots = state.optimizer.state.get(p, {})
        if state.step:
            exact_keys(slots, {'step', 'exp_avg', 'exp_avg_sq'}, 'comparison optimizer slots')
            for slot, value in slots.items():
                _slot(value, p, slot, state.step)
                tensors['optimizer.'+name+'.'+slot] = value.detach().clone().contiguous()
        elif slots:
            raise ContractError('zero-step comparison has optimizer moments')
    payload = save(tensors)
    if len(payload) > 32*1024*1024:
        raise ContractError('comparison snapshot exceeds tensor budget')
    metadata = {'schema': SCHEMA, 'workload': workload, 'lane': state.lane, 'seed': state.config.seed,
        'request_identity': request_identity, 'plan': state.config.to_dict(), 'source': sources(),
        'environment': require_environment(), 'dataset_identity': state.dataset_identity,
        'manifest_identity': state.manifest_identity, 'initialization_identity': state.initialization_identity,
        'initial_train_loss_hex': state.initial_train_loss, 'model_identity': model_identity(state.model),
        'step': state.step, 'cursor': state.cursor, 'target_tokens': state.target_tokens,
        'history': state.history, 'validation': state.validation,
        'rng_scope': 'private seeded initialization; frozen training reads no global RNG and has no stochastic operations'}
    raw = json_bytes(metadata)
    if len(raw) > 1024*1024:
        raise ContractError('comparison snapshot exceeds metadata budget')
    return payload, raw


def restore(payload, metadata, workload, lane, seed, request_identity, expected_identities, config=None):
    if (type(payload) is not bytes or type(metadata) is not bytes or len(payload) > 32*1024*1024
            or len(metadata) > 1024*1024 or (identity(payload), identity(metadata)) != expected_identities):
        raise ContractError('comparison snapshot identity/type/size mismatch')
    r = parse_json(metadata, canonical=True)
    keys = {'schema', 'workload', 'lane', 'seed', 'request_identity', 'plan', 'source', 'environment',
            'dataset_identity', 'manifest_identity', 'initialization_identity', 'initial_train_loss_hex',
            'model_identity', 'step', 'cursor', 'target_tokens', 'history', 'validation', 'rng_scope'}
    exact_keys(r, keys, 'comparison snapshot')
    state = state_for(workload, lane, seed, config)
    expected = {'schema': SCHEMA, 'workload': workload, 'lane': lane, 'seed': seed,
        'request_identity': request_identity, 'plan': state.config.to_dict(), 'source': sources(),
        'environment': require_environment(), 'dataset_identity': state.dataset_identity,
        'manifest_identity': state.manifest_identity, 'initialization_identity': state.initialization_identity,
        'initial_train_loss_hex': state.initial_train_loss,
        'rng_scope': 'private seeded initialization; frozen training reads no global RNG and has no stochastic operations'}
    if any(json_bytes(r[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('comparison snapshot context mismatch')
    state.step, state.cursor, state.target_tokens = r['step'], r['cursor'], r['target_tokens']
    state.history, state.validation = r['history'], r['validation']
    progress(state)
    try:
        tensors = load(payload)
    except Exception as error:
        raise ContractError('invalid comparison safetensors') from error
    names = {'model.'+n for n, _ in state.model.named_parameters()}
    if state.step:
        names |= {'optimizer.'+n+'.'+slot for n, _ in state.model.named_parameters()
                  for slot in ('step', 'exp_avg', 'exp_avg_sq')}
    if set(tensors) != names:
        raise ContractError('comparison tensor inventory mismatch')
    with torch.no_grad():
        for name, p in state.model.named_parameters():
            value = tensors['model.'+name]
            if value.shape != p.shape or value.dtype != p.dtype or not bool(torch.isfinite(value).all()):
                raise ContractError('invalid comparison model tensor')
            p.copy_(value)
            if state.step:
                slots = {}
                for slot in ('step', 'exp_avg', 'exp_avg_sq'):
                    value = tensors['optimizer.'+name+'.'+slot]
                    _slot(value, p, slot, state.step)
                    slots[slot] = value.clone()
                state.optimizer.state[p] = slots
    progress(state)
    if model_identity(state.model) != r['model_identity']:
        raise ContractError('comparison model identity mismatch')
    return state
