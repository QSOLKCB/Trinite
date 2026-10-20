"""Matched pilot using the single native updater and bounded replay checkpoints."""
import torch
from trinite.contracts import ContractError, ModelConfig, exact_keys, identity, integer, json_bytes, parse_json
from trinite.training import (RunConfig, TrainingState, admitted_data, create_optimizer, evaluate,
                              model_identity, require_environment, require_optimizer, run_steps)
from trinite.comparison import greedy
from trinite.tokenizer import EOS
from .model import ARMS, RefitDecoder
from .spec import PROTOCOL_ID, inputs, sources, specification

MAX_CHECKPOINT = 2_097_152


def configuration(seed):
    integer(seed, 'pilot seed', 0, 1)
    spec = specification()
    return RunConfig(model=ModelConfig(**spec['model']), seed=seed, steps=60, batch_size=4,
                     validation_every=20, max_seconds=120, max_target_tokens=480)


def create(arm, seed):
    if type(arm) is not str or arm not in ARMS:
        raise ContractError('unknown refit pilot arm')
    require_environment()
    config = configuration(seed)
    dataset, manifest = inputs()
    data = admitted_data(dataset, manifest, config)
    model = RefitDecoder(config.model, arm=arm, seed=seed)
    return TrainingState(config, arm, model, create_optimizer(model, config), data,
                         identity(dataset), identity(manifest), model_identity(model),
                         evaluate(model, data['train'], config.batch_size))


def validate(state):
    if type(state) is not TrainingState or state.lane not in ARMS or type(state.model) is not RefitDecoder:
        raise ContractError('invalid pilot state type/arm')
    config = configuration(state.config.seed)
    dataset, manifest = inputs()
    integer(state.step, 'pilot step', 0, 60)
    integer(state.cursor, 'pilot cursor', 0, 240)
    integer(state.target_tokens, 'pilot scored targets', 0, 480)
    if type(state.history) is not list or type(state.validation) is not list:
        raise ContractError('pilot history must be explicit lists')
    if (state.config != config or state.model.arm != state.lane or state.model.seed != config.seed
            or state.model.config != config.model or state.cursor != state.step*4
            or state.target_tokens != state.step*8 or len(state.history) != state.step
            or state.dataset_identity != identity(dataset) or state.manifest_identity != identity(manifest)
            or state.data != admitted_data(dataset, manifest, config)):
        raise ContractError('pilot configuration/input/progress mismatch')
    state.model.validate_state()
    require_optimizer(state)


def run(state, *, stop_after=60):
    validate(state)
    run_steps(state, stop_after=stop_after)
    validate(state)
    return state


def values(tensor):
    return [float(x).hex() for x in tensor.detach().reshape(-1).tolist()]


def snapshot(state):
    validate(state)
    named = dict(state.model.named_parameters())
    if any(p.grad is not None for p in named.values()):
        raise ContractError('pilot checkpoint requires cleared gradients')
    if any(p not in set(named.values()) for p in state.optimizer.state):
        raise ContractError('unknown AdamW parameter')
    slots = {}
    for name, p in named.items():
        current = state.optimizer.state.get(p, {})
        exact_keys(current, {'step', 'exp_avg', 'exp_avg_sq'} if state.step else set(), 'pilot AdamW slots')
        for key, tensor in current.items():
            if (tensor.dtype != torch.float32 or tensor.device.type != 'cpu'
                    or tensor.shape != (() if key == 'step' else p.shape)
                    or not bool(torch.isfinite(tensor).all())
                    or (key == 'step' and float(tensor) != state.step)
                    or (key == 'exp_avg_sq' and bool((tensor < 0).any()))):
                raise ContractError('invalid pilot AdamW slot')
        slots[name] = {k: values(v) for k,v in current.items()}
    raw = json_bytes({'schema':'trinite.scale-refit-training-checkpoint.v1',
        'protocol_identity':PROTOCOL_ID, 'source':sources(), 'environment':require_environment(),
        'inventory':state.model.inventory(), 'config':state.config.to_dict(),
        'dataset_identity':state.dataset_identity, 'manifest_identity':state.manifest_identity,
        'initialization_identity':state.initialization_identity, 'initial_train_loss':state.initial_train_loss,
        'step':state.step, 'cursor':state.cursor, 'target_tokens':state.target_tokens,
        'history':state.history, 'validation':state.validation,
        'model_hex':{n:values(p) for n,p in named.items()}, 'adamw_hex':slots,
        'rng':'private initialization generator consumed only during construction; no update/generation RNG draws',
        'scope':'exact bounded full-prefix replay; no native checkpoint import'})
    if len(raw)>MAX_CHECKPOINT:
        raise ContractError('pilot checkpoint exceeds size bound')
    return raw


def resume(raw, *, expected_identity, arm, seed):
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_CHECKPOINT or identity(raw) != expected_identity:
        raise ContractError('pilot checkpoint identity/type/size mismatch')
    meta = parse_json(raw, canonical=True)
    if type(meta) is not dict or 'step' not in meta:
        raise ContractError('pilot checkpoint requires progress')
    step = integer(meta['step'], 'checkpoint step', 0, 60)
    state = run(create(arm, seed), stop_after=step)
    if snapshot(state) != raw:
        raise ContractError('pilot complete source/environment/input/model/AdamW/history replay mismatch')
    return state


def quality(state):
    """Only explicit train/validation views; never call historical all-split scorer."""
    validate(state)
    examples = parse_json(inputs()[0], canonical=True)['examples']
    result = {}
    for split in ('train','validation'):
        rows=[]
        for item in examples:
            if item['split'] != split:
                continue
            predicted = greedy(state.model, item['prompt'], 2)
            expected = [*item['answer'].encode(), EOS]
            rows.append({'example_identity':item['example_identity'], 'family_id':item['family_id'],
                         'generated_tokens':predicted, 'expected_tokens':expected, 'correct':predicted==expected})
        result[split]={'examples':len(rows), 'correct':sum(r['correct'] for r in rows),
                      'target_loss_hex':evaluate(state.model,state.data[split],4),'predictions':rows}
    return result


def diagnostics(model):
    from scale_refit.quantizer import refit, exact_sse_numerator
    from torch.nn import functional as F
    from .model import RefitLinear
    result = {}
    for name, module in model.named_modules():
        if type(module) is not RefitLinear:
            continue
        w = module.weight.detach()
        selected = refit(w)
        from trinite.quantizer import quantize
        baseline = quantize(w)
        n = w.shape[1]
        x = torch.tensor([[(-1 if i % 2 else 1)/256 for i in range(n)],
                          [(i % 3 - 1)/256 for i in range(n)]], dtype=torch.float32)
        dense = F.linear(x, w)
        result[name] = {'decision': selected.decision,
            'codes_preserved_at_same_weight': torch.equal(baseline.codes, selected.quantized.codes),
            'nonzero_codes': int((baseline.codes != 0).sum()),
            'weight_sse_numerators': [str(selected.baseline_sse_numerator), str(selected.selected_sse_numerator)],
            'fixed_input_output_sse_numerators': [str(exact_sse_numerator(dense,F.linear(x,baseline.weight))),
                str(exact_sse_numerator(dense,F.linear(x,selected.quantized.weight)))],
            'sse_denominator_power_of_two':298}
    return result


def report(on_cell=None):
    before, closure = require_environment(), sources()
    cells=[]
    for seed in specification()['seeds']:
        for arm in ARMS:
            state=create(arm,seed)
            checkpoints = {}
            for stop in specification()['replay_stops']:
                run(state, stop_after=stop)
                checkpoints[str(stop)] = snapshot(state)
            final = checkpoints['60']
            replayed = []
            for stop, raw in checkpoints.items():
                restored = resume(raw, expected_identity=identity(raw), arm=arm, seed=seed)
                run(restored)
                if snapshot(restored) != final:
                    raise ContractError('interrupted replay differs from uninterrupted final state')
                replayed.append(int(stop))
            cell = {'arm':arm,'seed':seed,'initial_checkpoint':parse_json(checkpoints['0'],canonical=True),
                    'final_checkpoint':parse_json(final,canonical=True),
                    'intermediate_checkpoints': {s:parse_json(raw,canonical=True) for s,raw in checkpoints.items() if s not in ('0','60')},
                    'verified_resume_stops':replayed, 'quality':quality(state),
                    'same_weight_diagnostics':diagnostics(state.model)}
            cells.append(cell)
            if on_cell is not None:
                on_cell(cell)
    if before != require_environment() or closure != sources():
        raise ContractError('pilot environment or source changed during execution')
    for seed in (0,1):
        pair=[c for c in cells if c['seed']==seed]
        if pair[0]['initial_checkpoint']['model_hex'] != pair[1]['initial_checkpoint']['model_hex']:
            raise ContractError('pilot arms did not share latent initialization')
        if [[h['examples'] for h in c['final_checkpoint']['history']] for c in pair][0] != [[h['examples'] for h in c['final_checkpoint']['history']] for c in pair][1]:
            raise ContractError('pilot arm exposure mismatch')
    return {'schema':'trinite.scale-refit-training-report.v1','protocol_identity':PROTOCOL_ID,
            'source':closure,'environment':before,'cells':cells,'test_evaluated':False,
            'claim_scope':'exploratory visible-formal-fixture pilot, not fresh generalization',
            'speed_claim':False,'format_selected':False,'release_ready':False}
