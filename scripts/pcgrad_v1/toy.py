"""Fixed eight-step, two-parameter engineering harness, not native research.

Replay-to-resume reconstructs the entire bounded prefix and compares the full
snapshot. It never loads arbitrary tensor or optimizer state from a checkpoint.
"""
from dataclasses import dataclass, field
import torch
from trinite.contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json
from trinite.training import RunConfig, create_optimizer, require_environment, require_optimizer
from .kernel import ARMS, combine
from .spec import PROTOCOL_ID, sources, specification

MAX_CHECKPOINT = 256*1024


def values(tensor):
    return [float(x).hex() for x in tensor.detach().reshape(-1).tolist()]


@dataclass
class ToyState:
    arm: str
    seed: int
    model: torch.nn.Module
    config: RunConfig
    optimizer: torch.optim.AdamW
    rng: torch.Generator
    step: int = 0
    history: list = field(default_factory=list)


def create(arm, seed, *, profile='engineering'):
    specification(profile)  # Reject research before allocation or environment work.
    if type(arm) is not str or arm not in ARMS:
        raise ContractError('unknown toy arm')
    integer(seed, 'engineering seed', 0, 1)
    require_environment()
    rng = torch.Generator(device='cpu').manual_seed(seed)
    model = torch.nn.Module()
    model.register_parameter('weight', torch.nn.Parameter(torch.randn(2, generator=rng, dtype=torch.float32)*.02))
    config = RunConfig(seed=seed, steps=8, batch_size=2, validation_every=8)
    return ToyState(arm, seed, model, config, create_optimizer(model, config), rng)


def validate(state):
    specification()
    if type(state) is not ToyState or state.arm not in ARMS:
        raise ContractError('invalid toy state')
    integer(state.seed, 'toy seed', 0, 1)
    integer(state.step, 'toy step', 0, 8)
    if (state.config != RunConfig(seed=state.seed, steps=8, batch_size=2, validation_every=8)
            or type(state.history) is not list or len(state.history) != state.step
            or list(dict(state.model.named_parameters())) != ['weight']
            or state.model.weight.shape != (2,) or state.model.weight.dtype != torch.float32
            or state.model.weight.device.type != 'cpu' or not torch.isfinite(state.model.weight).all()
            or state.rng.device.type != 'cpu'):
        raise ContractError('toy model/config/progress mismatch')
    require_optimizer(state)


def run(state, *, stop_after=8):
    validate(state)
    require_environment()
    end = integer(stop_after, 'toy stop', state.step, 8)
    data = specification()['toy_inputs']
    while state.step < end:
        w = state.model.weight
        state.optimizer.zero_grad(set_to_none=True)
        before = values(w)
        grads, losses = [], []
        # Two separate reverse-mode passes, neither with an intervening update.
        for name in ('final', 'auxiliary'):
            x = torch.tensor(data[name]['x'], dtype=torch.float32, device='cpu')
            y = torch.tensor(data[name]['y'], dtype=torch.float32, device='cpu')
            loss = ((x @ w-y)**2).mean()
            if not bool(torch.isfinite(loss)):
                raise ContractError('nonfinite toy objective')
            grads.append(torch.autograd.grad(loss, w)[0])
            losses.append(float(loss.detach()).hex())
        if values(w) != before:
            raise ContractError('parameter snapshot changed between backwards')
        order = tuple(torch.randperm(2, generator=state.rng).tolist())
        combined = combine(*grads, arm=state.arm, order=order)
        w.grad = combined.gradient.clone()
        norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), 1., error_if_nonfinite=True, foreach=False)
        clipped = values(w.grad)
        state.optimizer.step()
        state.optimizer.zero_grad(set_to_none=True)
        state.step += 1
        state.history.append({'step': state.step, 'before_hex': before,
            'losses_hex': losses, 'original_gradients_hex': [values(g) for g in grads],
            'projected_gradients_hex': [values(g) for g in combined.projected],
            'combined_gradient_hex': values(combined.gradient), 'clipped_gradient_hex': clipped,
            'clip_input_norm_hex': float(norm).hex(), 'after_hex': values(w),
            'diagnostics': combined.diagnostics, 'backwards': 2, 'optimizer_calls': 1,
            'objective_targets': [2, 2]})
        validate(state)
    return state


def snapshot(state):
    validate(state)
    w = state.model.weight
    if w.grad is not None:
        raise ContractError('toy snapshot requires completed steps and cleared gradients')
    slots = state.optimizer.state.get(w, {})
    exact_keys(slots, {'step', 'exp_avg', 'exp_avg_sq'} if state.step else set(), 'toy AdamW')
    for name, tensor in slots.items():
        if (tensor.dtype != torch.float32 or tensor.device.type != 'cpu'
                or tensor.shape != (() if name == 'step' else (2,))
                or not bool(torch.isfinite(tensor).all())
                or (name == 'step' and float(tensor) != state.step)
                or (name == 'exp_avg_sq' and bool((tensor < 0).any()))):
            raise ContractError('invalid toy AdamW slots')
    raw = json_bytes({'schema': 'trinite.pcgrad-toy-checkpoint.v1', 'arm': state.arm,
        'seed': state.seed, 'step': state.step, 'protocol_identity': PROTOCOL_ID,
        'source': sources(), 'environment': require_environment(),
        'optimizer_config': state.config.to_dict(), 'model_hex': values(w),
        'adamw_hex': {k: values(v) for k, v in slots.items()},
        'rng_hex': bytes(state.rng.get_state().tolist()).hex(), 'history': state.history,
        'scope': 'engineering-only full-prefix replay-to-resume; not native checkpoint loading'})
    if len(raw) > MAX_CHECKPOINT:
        raise ContractError('toy checkpoint exceeds bound')
    return raw


def resume(raw, *, expected_identity, arm, seed, profile='engineering'):
    specification(profile)
    if (type(raw) is not bytes or not 1 <= len(raw) <= MAX_CHECKPOINT
            or type(expected_identity) is not str or identity(raw) != expected_identity):
        raise ContractError('toy checkpoint external identity/type/size mismatch')
    meta = parse_json(raw, canonical=True)
    if type(meta) is not dict or 'step' not in meta:
        raise ContractError('toy checkpoint requires step')
    step = integer(meta['step'], 'toy checkpoint step', 0, 8)
    state = run(create(arm, seed, profile=profile), stop_after=step)
    if snapshot(state) != raw:
        raise ContractError('toy checkpoint full source/environment/model/AdamW/RNG/history replay mismatch')
    return state
