"""Fixed native updates and guarded prompt-only measurements; no public resume.

Checkpoint verification is safe structure/progress validation, not numerical
optimizer replay. Development loading restores models only, never optimizers.
"""
from copy import deepcopy
from time import monotonic

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from music_learning_v1.numerical import encoded
from music_exposure_v1.numerical import _preserve_training
from .spec import (CELLS, PROTOCOL, cell_name, plan_for, train_rows,
                   diagnostic_rows, development_rows, require_admission)

CAP = 64
CHECKPOINT_SCHEMA = 'trinite.music-pitch-intervention-checkpoint.v1'


def _context(arm, seed, admission_identity):
    cell_name(arm, seed)
    require_identity(admission_identity)
    raw, receipt = require_admission(admission_identity)
    if identity(receipt) != admission_identity:
        raise ContractError('pitch admission differs')
    return raw


def state_for(arm, seed, admission_identity):
    """Fresh paired initialization, independent optimizer, common validation."""
    from trinite.model import Decoder
    from trinite.training import TrainingState, create_optimizer, evaluate, model_identity, require_environment
    raw = _context(arm, seed, admission_identity)
    require_environment()
    config = plan_for(seed)
    train = encoded(train_rows(arm))
    validation = encoded(diagnostic_rows()[:40])
    if len(train) != 80 or len(validation) != 40:
        raise ContractError('fixed pitch training/validation inventory differs')
    model = Decoder(config.model, lane='dense', seed=seed)
    return TrainingState(config, 'dense', model, create_optimizer(model, config),
        {'train': train, 'validation': validation}, identity(raw), admission_identity,
        model_identity(model), evaluate(model, train, config.batch_size))


def snapshot(state, arm, request_identity):
    from trinite.comparison_checkpoint import tensor_payload
    from trinite.training import model_identity
    cell_name(arm, state.config.seed)
    require_identity(request_identity)
    payload = tensor_payload(state)
    metadata = {'schema': CHECKPOINT_SCHEMA, 'arm': arm, 'seed': state.config.seed,
        'request_identity': request_identity, 'plan': state.config.to_dict(),
        'dataset_identity': state.dataset_identity, 'admission_identity': state.manifest_identity,
        'initialization_identity': state.initialization_identity,
        'initial_train_loss_hex': state.initial_train_loss,
        'model_identity': model_identity(state.model), 'step': state.step,
        'cursor': state.cursor, 'target_tokens': state.target_tokens,
        'history': state.history, 'validation': state.validation}
    raw = json_bytes(metadata)
    if len(payload) > 16777216 or len(raw) > 1048576:
        raise ContractError('pitch checkpoint exceeds fixed byte budget')
    return payload, raw


def _progress(record, arm, plan, require_final):
    """Verify exact admitted schedule; values are not a replay of updates."""
    from trinite.comparison_checkpoint import _hex
    step = record['step']
    if type(step) is not int or not 0 <= step <= 640 or (require_final and step != 640):
        raise ContractError('pitch checkpoint endpoint differs')
    if type(record['cursor']) is not int or record['cursor'] != step * 8:
        raise ContractError('pitch cursor differs')
    train = encoded(train_rows(arm))
    if type(record['history']) is not list or len(record['history']) != step:
        raise ContractError('pitch history differs')
    total = 0
    for i, entry in enumerate(record['history']):
        exact_keys(entry, {'step', 'loss_hex', 'gradient_norm_hex', 'target_tokens', 'examples'}, 'pitch history')
        batch = [train[(i * 8 + j) % len(train)] for j in range(8)]
        targets = sum(sum(row['loss']) for row in batch)
        expected = {'step': i + 1, 'examples': [row['identity'] for row in batch], 'target_tokens': targets}
        if any(json_bytes(entry[k]) != json_bytes(v) for k, v in expected.items()):
            raise ContractError('pitch scheduled history differs')
        _hex(entry['loss_hex']); _hex(entry['gradient_norm_hex']); total += targets
    if type(record['target_tokens']) is not int or record['target_tokens'] != total or total > plan.max_target_tokens:
        raise ContractError('pitch target count differs')
    expected_steps = [i for i in range(1, step + 1) if i % plan.validation_every == 0 or i == plan.steps]
    if type(record['validation']) is not list or len(record['validation']) != len(expected_steps):
        raise ContractError('pitch validation schedule differs')
    for entry, expected in zip(record['validation'], expected_steps):
        exact_keys(entry, {'step', 'loss_hex'}, 'pitch validation')
        if type(entry['step']) is not int or entry['step'] != expected:
            raise ContractError('pitch validation step differs')
        _hex(entry['loss_hex'])
    _hex(record['initial_train_loss_hex'])


def load_checkpoint(payload, metadata, arm, seed, admission_identity, request_identity, *, require_final=True):
    """Safe model-only restore; validate retained AdamW slots without installation."""
    import torch
    from trinite.comparison_checkpoint import load, _slot
    from trinite.model import Decoder
    from trinite.training import model_identity, require_environment
    raw = _context(arm, seed, admission_identity)
    require_identity(request_identity)
    if type(require_final) is not bool:
        raise ContractError('strict checkpoint endpoint selector required')
    if type(payload) is not bytes or type(metadata) is not bytes or len(payload) > 16777216 or len(metadata) > 1048576:
        raise ContractError('pitch checkpoint type/size differs')
    require_environment()
    record = parse_json(metadata, canonical=True)
    config = plan_for(seed)
    expected = {'schema': CHECKPOINT_SCHEMA, 'arm': arm, 'seed': seed,
        'request_identity': request_identity, 'plan': config.to_dict(),
        'dataset_identity': identity(raw), 'admission_identity': admission_identity}
    exact_keys(record, set(expected) | {'initialization_identity', 'initial_train_loss_hex',
        'model_identity', 'step', 'cursor', 'target_tokens', 'history', 'validation'}, 'pitch checkpoint')
    if any(json_bytes(record[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('pitch checkpoint context differs')
    _progress(record, arm, config, require_final)
    model = Decoder(config.model, lane='dense', seed=seed)
    initial = model_identity(model)
    if record['initialization_identity'] != initial:
        raise ContractError('pitch initialization differs')
    require_identity(record['model_identity'])
    try:
        tensors = load(payload)
    except Exception as error:
        raise ContractError('invalid pitch safetensors') from error
    names = {'model.' + n for n, _ in model.named_parameters()}
    if record['step']:
        names |= {'optimizer.' + n + '.' + slot for n, _ in model.named_parameters()
                  for slot in ('step', 'exp_avg', 'exp_avg_sq')}
    if set(tensors) != names:
        raise ContractError('pitch tensor inventory differs')
    with torch.no_grad():
        for name, p in model.named_parameters():
            value = tensors['model.' + name]
            if (value.device.type != 'cpu' or value.layout != torch.strided or value.dtype != p.dtype
                    or value.shape != p.shape or not bool(torch.isfinite(value).all())):
                raise ContractError('invalid pitch model tensor')
            if record['step']:
                for slot in ('step', 'exp_avg', 'exp_avg_sq'):
                    _slot(tensors['optimizer.' + name + '.' + slot], p, slot, record['step'])
            p.copy_(value)
    model.validate_state()
    if model_identity(model) != record['model_identity'] or (record['step'] == 0 and record['model_identity'] != initial):
        raise ContractError('pitch model identity differs')
    return model, record


def verify_checkpoint(payload, metadata, arm, seed, admission_identity, request_identity, *, require_final=True):
    """Structural verification alias, intentionally model-only and zero-forward."""
    return load_checkpoint(payload, metadata, arm, seed, admission_identity, request_identity,
                           require_final=require_final)


def infer_models(models, stage, emit):
    """All-model guards around every prompt-only call and journal callback."""
    import random
    import numpy as np
    import torch
    from trinite.comparison import greedy
    from trinite.training import model_identity
    if stage not in ('diagnostic', 'development') or not callable(emit):
        raise ContractError('fixed inference stage and journal required')
    expected = set(CELLS) if stage == 'development' else set(models)
    if set(models) != expected or not models or (stage == 'diagnostic' and len(models) != 1):
        raise ContractError('fixed inference model inventory differs')
    for arm, seed in models:
        cell_name(arm, seed)
    parameters = [id(p) for model in models.values() for p in model.parameters()]
    if len({id(model) for model in models.values()}) != len(models) or len(set(parameters)) != len(parameters):
        raise ContractError('independent inference model objects required')
    selected = diagnostic_rows() if stage == 'diagnostic' else development_rows()
    required_rows = 80 if stage == 'diagnostic' else 120
    if len(selected) != required_rows:
        raise ContractError('fixed inference row inventory differs')
    inputs = [{k: row[k] for k in ('example_identity', 'prompt')} for row in selected]
    order = [cell for cell in CELLS if cell in models]
    states, hooks, records = {}, [], []
    calls = 0
    maximum = len(models) * required_rows * CAP
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    grad_mode = torch.is_grad_enabled()

    def guard():
        now = np.random.get_state()
        if (random.getstate() != rng[0] or now[0] != rng[1][0] or not np.array_equal(now[1], rng[1][1])
                or now[2:] != rng[1][2:] or not torch.equal(torch.get_rng_state(), rng[2])):
            raise ContractError('pitch inference mutated RNG')
        for key, state in states.items():
            model = models[key]
            if (model_identity(model) != state['identity']
                    or [(n, id(p), p.requires_grad) for n,p in model.named_parameters()] != state['params']
                    or [(n,id(b)) for n,b in model.named_buffers()] != state['buffers']
                    or [(n,id(m),m.training) for n,m in model.named_modules()] != state['eval_modes']
                    or any(p.grad is not None for p in model.parameters())
                    or any(not torch.equal(b,state['values'][n]) for n,b in model.named_buffers())):
                raise ContractError('pitch inference mutated model, buffers, gradients or modes')

    def count(module, args):
        nonlocal calls
        calls += 1
        if calls > maximum:
            raise ContractError('pitch inference forward bound exceeded')

    try:
        for key in order:
            model = models[key]
            if model.config.context_length != 256 or any(p.device.type != 'cpu' or p.dtype != torch.float32
                    or p.grad is not None for p in model.parameters()):
                raise ContractError('pitch inference requires clean CPU float32 model')
            states[key] = state = {'identity': model_identity(model),
                'params': [(n,id(p),p.requires_grad) for n,p in model.named_parameters()],
                'buffers': [(n,id(b)) for n,b in model.named_buffers()],
                'values': {n:b.detach().clone() for n,b in model.named_buffers()},
                'modes': [(m,m.training) for m in model.modules()]}
            model.eval()
            state['eval_modes'] = [(n,id(m),m.training) for n,m in model.named_modules()]
            hooks.append(model.register_forward_pre_hook(count))
        for arm, seed in order:
            for row in inputs:
                prompt = row['prompt']
                if type(prompt) is not str or not prompt.isascii() or 1 + len(prompt) + CAP > 256:
                    raise ContractError('pitch prompt exceeds fixed context')
                guard(); start = calls
                with torch.no_grad():
                    generated = greedy(models[(arm,seed)], prompt, CAP)
                if (type(generated) is not list or not 1 <= len(generated) <= CAP
                        or any(type(t) is not int or not 0 <= t <= 258 for t in generated)
                        or 257 in generated[:-1] or (generated[-1] != 257 and len(generated) != CAP)
                        or calls - start != len(generated)):
                    raise ContractError('pitch native EOS/cap/forward accounting differs')
                guard()
                record = {'arm': arm, 'seed': seed, 'step': 640, 'stage': stage,
                    'example_identity': row['example_identity'], 'generated_tokens': generated}
                retained = json_bytes(record)
                emit(record)
                if json_bytes(record) != retained:
                    raise ContractError('journal callback changed raw prediction')
                records.append(record); guard()
        guard()
    finally:
        for hook in hooks:
            hook.remove()
        for state in states.values():
            for module, mode in state['modes']:
                module.training = mode
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        torch.set_grad_enabled(grad_mode)
    if len(records) != len(models) * required_rows:
        raise ContractError('pitch inference coverage incomplete')
    return {'records': records, 'decodes': len(records), 'forward_calls': calls,
            'model_identities_unchanged': True, 'training_performed': False}


def train(arm, seed, request_identity, admission_identity, write, emit=None):
    """One fresh fixed640 trajectory; diagnostic failure retains incomplete evidence."""
    from trinite.training import run_steps, model_identity
    from .reporting import score_cell
    cell_name(arm, seed); require_identity(request_identity)
    if not callable(write) or (emit is not None and not callable(emit)):
        raise ContractError('pitch training requires artifact callbacks')
    state, elapsed = None, 0.0
    try:
        state = state_for(arm, seed, admission_identity)
        while state.step < 640:
            if elapsed >= 120:
                raise ContractError('pitch cumulative update deadline exceeded')
            started = monotonic()
            try:
                run_steps(state, stop_after=state.step + 1)
            finally:
                elapsed += monotonic() - started
            if elapsed > 120:
                raise ContractError('pitch cumulative update deadline exceeded')
        with _preserve_training(state):
            payload, metadata = snapshot(state, arm, request_identity)
            write('final.safetensors', payload); write('final-checkpoint.json', metadata)
            model = deepcopy(state.model)
            result = infer_models({(arm,seed): model}, 'diagnostic', emit or (lambda record: None))
            summary = score_cell(result['records'], arm, seed, 'diagnostic')
            write('diagnostic-predictions.json', json_bytes(result['records']))
        return {'arm': arm, 'seed': seed, 'request_identity': request_identity,
            'admission_identity': admission_identity, 'step': state.step,
            'target_tokens': state.target_tokens, 'initialization_identity': state.initialization_identity,
            'model_identity': model_identity(state.model), 'update_seconds_hex': elapsed.hex(),
            'diagnostic_summary': summary, 'diagnostic_decodes': result['decodes'],
            'diagnostic_forward_calls': result['forward_calls']}
    except Exception as error:
        if state is not None:
            try:
                payload, metadata = snapshot(state, arm, request_identity)
                write('failure.safetensors', payload); write('failure-checkpoint.json', metadata)
            except Exception as checkpoint_error:
                write('failure-checkpoint-error.json', json_bytes({
                    'type': type(checkpoint_error).__name__, 'message': str(checkpoint_error)}))
        write('failure.json', json_bytes({'type': type(error).__name__, 'message': str(error),
            'completed_updates': None if state is None else state.step,
            'update_seconds_hex': elapsed.hex(), 'outcome': 'incomplete'}))
        raise
