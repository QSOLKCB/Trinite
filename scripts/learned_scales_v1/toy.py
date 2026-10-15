"""Four-update synthetic decoder conformance, never a scientific training run.

Resume reconstructs the bounded prefix and compares every byte before returning
state. It does not load untrusted tensors, optimizer slots or executable objects.
"""
from dataclasses import dataclass, field
import torch
from torch.nn import functional as F
from trinite.contracts import ContractError, ModelConfig, exact_keys, identity, integer, json_bytes, parse_json
from trinite.training import RunConfig, create_optimizer, require_environment, require_optimizer
from .model import ARMS, ScaleDecoder, ScaleLinear
from .quantizer import quantize
from .spec import PROTOCOL_ID, sources, specification

MAX_CHECKPOINT = 1048576


def values(tensor):
    return [float(x).hex() for x in tensor.detach().reshape(-1).tolist()]


def tensors(model):
    return {n: values(p) for n, p in model.named_parameters()}


def diagnostics(model):
    result = {}
    for name, m in model.named_modules():
        if isinstance(m, ScaleLinear):
            q = quantize(m.weight, m.scale)
            unit = m.weight.detach()/m.scale.detach()
            result[name] = {'scale_hex': float(m.scale.detach()).hex(),
                'codes': q.codes.reshape(-1).tolist(), 'effective_hex': values(q.weight),
                'zero_count': int((q.codes == 0).sum()), 'elements': q.codes.numel(),
                'saturated_count': int((unit.abs() >= 1).sum())}
    return result


@dataclass
class ToyState:
    arm: str
    seed: int
    model: ScaleDecoder
    config: RunConfig
    optimizer: torch.optim.AdamW
    rng: torch.Generator
    step: int = 0
    history: list = field(default_factory=list)


def create(arm, seed, *, profile='engineering'):
    spec = specification(profile)
    if type(arm) is not str or arm not in ARMS:
        raise ContractError('unknown toy arm')
    integer(seed, 'engineering seed', 0, 1)
    require_environment()
    model = ScaleDecoder(ModelConfig(**spec['model']), arm=arm, seed=seed)
    config = RunConfig(seed=seed, steps=4, batch_size=2, validation_every=4)
    return ToyState(arm, seed, model, config, create_optimizer(model, config),
                    torch.Generator(device='cpu').manual_seed(seed))


def validate(state):
    spec = specification()
    if type(state) is not ToyState or state.arm not in ARMS:
        raise ContractError('invalid toy state')
    integer(state.seed, 'toy seed', 0, 1)
    integer(state.step, 'toy step', 0, 4)
    if (state.config != RunConfig(seed=state.seed, steps=4, batch_size=2, validation_every=4)
            or type(state.model) is not ScaleDecoder or state.model.arm != state.arm
            or state.model.seed != state.seed or state.model.config != ModelConfig(**spec['model'])
            or type(state.history) is not list or len(state.history) != state.step
            or state.rng.device.type != 'cpu'):
        raise ContractError('toy model/config/progress mismatch')
    state.model.validate_state()
    require_optimizer(state)


def run(state, *, stop_after=4):
    validate(state)
    require_environment()
    end = integer(stop_after, 'toy stop', state.step, 4)
    spec = specification()
    x = torch.tensor(spec['toy_inputs'], dtype=torch.int64, device='cpu')
    targets = torch.tensor(spec['toy_targets'], dtype=torch.int64, device='cpu')
    while state.step < end:
        before = diagnostics(state.model)
        order = torch.randperm(2, generator=state.rng)
        state.optimizer.zero_grad(set_to_none=True)
        logits = state.model(x[order], torch.ones_like(x, dtype=torch.bool)).logits
        loss = F.cross_entropy(logits.reshape(-1, 259), targets[order].reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise ContractError('nonfinite toy objective')
        loss.backward()
        gradients = {n: values(p.grad) for n, p in state.model.named_parameters() if p.grad is not None}
        norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), 1., error_if_nonfinite=True, foreach=False)
        state.optimizer.step()
        # This isolated projected-scale toy is deliberately not the native loop.
        pre_projection = {n: values(p) for n, p in state.model.named_parameters() if n.endswith('.scale')}
        state.model.project_scales_()
        state.optimizer.zero_grad(set_to_none=True)
        after = diagnostics(state.model)
        state.step += 1
        state.history.append({'step': state.step, 'order': order.tolist(),
            'loss_hex': float(loss.detach()).hex(), 'clip_input_norm_hex': float(norm).hex(),
            'gradients_hex': gradients, 'scale_before_projection_hex': pre_projection,
            'before': before, 'after': after,
            'code_changes': {n: sum(a != b for a,b in zip(before[n]['codes'], after[n]['codes'])) for n in before},
            'backwards': 1, 'optimizer_calls': 1, 'targets': 6})
        validate(state)
    return state


def snapshot(state):
    validate(state)
    named = dict(state.model.named_parameters())
    if any(p.grad is not None for p in named.values()):
        raise ContractError('snapshot requires completed steps and cleared gradients')
    if any(p not in set(named.values()) for p in state.optimizer.state):
        raise ContractError('unknown optimizer parameter')
    slots = {}
    for name, p in named.items():
        current = state.optimizer.state.get(p, {})
        expected = {'step', 'exp_avg', 'exp_avg_sq'} if state.step and p.requires_grad else set()
        exact_keys(current, expected, 'learned-scale AdamW')
        for key, tensor in current.items():
            if (tensor.dtype != torch.float32 or tensor.device.type != 'cpu'
                    or tensor.shape != (() if key == 'step' else p.shape)
                    or not bool(torch.isfinite(tensor).all())
                    or (key == 'step' and float(tensor) != state.step)
                    or (key == 'exp_avg_sq' and bool((tensor < 0).any()))):
                raise ContractError('invalid learned-scale AdamW slot')
        slots[name] = {k: values(v) for k, v in current.items()}
    raw = json_bytes({'schema': 'trinite.learned-scales-toy-checkpoint.v1',
        'arm': state.arm, 'seed': state.seed, 'step': state.step,
        'protocol_identity': PROTOCOL_ID, 'source': sources(), 'environment': require_environment(),
        'inventory': state.model.inventory(), 'optimizer_config': state.config.to_dict(),
        'model_hex': tensors(state.model), 'adamw_hex': slots,
        'rng_hex': bytes(state.rng.get_state().tolist()).hex(), 'history': state.history,
        'scope': 'engineering full-prefix replay-to-resume, not native checkpoint loading'})
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
    step = integer(meta['step'], 'toy checkpoint step', 0, 4)
    state = run(create(arm, seed, profile=profile), stop_after=step)
    if snapshot(state) != raw:
        raise ContractError('toy checkpoint full source/environment/model/AdamW/RNG/history replay mismatch')
    return state
