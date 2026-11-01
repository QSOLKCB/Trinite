"""Task bridge to the unchanged native updater/decoder/closed tensor reader."""
import time

import queue_evaluation_v1 as gate
from trinite.contracts import identity, json_bytes, parse_json
from . import data
from .spec import plan, sources


def configure():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from trinite.training import require_environment
    return require_environment()


def check_rows(rows):
    from procedural_dataset_v1.cases import expected_target
    expected = data.training_rows(data.inventory(), lambda r: expected_target('context',r))
    gate.require(json_bytes(rows) == json_bytes(expected), 'training rows differ from admitted inventory')


def schedule(rows, seed, fit=False):
    config = plan(seed,fit)
    selected = data.fit_rows(rows) if fit else rows
    visits = [selected[i % len(selected)] for i in range(config.steps*8)]
    targets = sum(sum(r['loss']) for r in visits)
    gate.require(targets <= config.max_target_tokens, 'scheduled supervised token ceiling')
    if fit:
        gate.require(targets == 2432, 'fit target ceiling changed')
    return dict(visits=len(visits), unique=len(selected), targets=targets,
        order_sha256=gate.digest(json_bytes([r['identity'] for r in visits])))


def state_for(rows, seed, binding, fit=False):
    from trinite.model import Decoder
    from trinite.training import TrainingState, create_optimizer, evaluate, model_identity, require_environment
    require_environment()
    check_rows(rows)
    schedule(rows,seed,fit)
    config = plan(seed,fit)
    selected = data.fit_rows(rows) if fit else rows
    model = Decoder(config.model,lane='dense',seed=seed)
    gate.require(sum(p.numel() for p in model.parameters()) == config.model.parameter_count == 24768,
                 'actual model parameter count differs from approved schedule')
    return TrainingState(config,'dense',model,create_optimizer(model,config),
        {'train':selected,'validation':selected},identity(json_bytes(selected)),binding,
        model_identity(model),evaluate(model,selected,8))


def snapshot(state, binding):
    from trinite.comparison_checkpoint import tensor_payload
    from trinite.training import model_identity, require_environment
    payload = tensor_payload(state)
    metadata = dict(schema='trinite.context-symbol-checkpoint.v1',binding=binding,
        sources=sources(),environment=require_environment(),plan=state.config.to_dict(),
        dataset_identity=state.dataset_identity,initialization_identity=state.initialization_identity,
        initial_train_loss=state.initial_train_loss,model_identity=model_identity(state.model),
        step=state.step,cursor=state.cursor,target_tokens=state.target_tokens,
        history=state.history,train_diagnostics=state.validation)
    raw = json_bytes(metadata)
    gate.require(len(raw) <= 1<<20, 'checkpoint metadata bound')
    return payload,raw


def restore(payload, raw, rows, seed, binding, expected, fit=False):
    gate.require(len(payload) <= 32<<20 and len(raw) <= 1<<20
        and [gate.digest(payload),gate.digest(raw)] == expected, 'checkpoint anchor mismatch')
    value = parse_json(raw,canonical=True)
    state = state_for(rows,seed,binding,fit)
    _, template = snapshot(state,binding)
    initial = parse_json(template,canonical=True)
    gate.keys(value,initial.keys())
    mutable = {'model_identity','step','cursor','target_tokens','history','train_diagnostics'}
    gate.require(all(json_bytes(value[k]) == json_bytes(initial[k]) for k in set(initial)-mutable),
                 'checkpoint frozen context mismatch')
    for attr,key in [('step','step'),('cursor','cursor'),('target_tokens','target_tokens'),
                     ('history','history'),('validation','train_diagnostics')]:
        setattr(state,attr,value[key])
    from trinite.comparison_checkpoint import restore_tensors
    return restore_tensors(state,payload,value['model_identity'])


def parse_output(tokens):
    # Canonical answer AND EOS only. Empty, missing EOS, BOS/PAD, extra tokens
    # and malformed byte strings are failures, never relation 0/uncertainty.
    valid = type(tokens) is list and all(type(t) is int for t in tokens)
    for answer in ('-1','0','1'):
        if valid and tokens == [*answer.encode('ascii'),257]:
            return dict(status='prediction',value=answer,tokens=tokens)
    return dict(status='malformed',tokens=tokens)


def infer(model, observation):
    from trinite.comparison import greedy
    return parse_output(greedy(model,data.prompt(observation),3))


def fit_report(model, train):
    selected = data.fit_rows(train)
    result = []
    # Only training prefixes selected for the fit set enter this function.
    by_id = {r['id']:data.observation(r) for r in data.inventory() if r['split'] == 'train'}
    for row in selected:
        raw = infer(model,by_id[row['identity']])
        result.append(dict(id=row['identity'],expected=row['answer'],raw=raw,
            exact=raw == dict(status='prediction',value=row['answer'],tokens=[*row['answer'].encode('ascii'),257])))
    return result


def require_fit(state, train):
    gate.require(type(state.step) is int and state.step == 128
        and type(state.target_tokens) is int and state.target_tokens == 2432, 'fit requires exact final checkpoint')
    report = fit_report(state.model,train)
    gate.require(len(report) == 8 and all(r['exact'] is True for r in report), 'final fit below 8/8; stop dependent stages')
    return report


def feasibility_kernel():
    """Three engineering backward/clip calls; zero optimizer steps or pilot targets."""
    import torch
    from trinite.model import Decoder
    from trinite.tokenizer import ByteTokenizer
    from trinite.training import target_loss,model_identity
    env = configure()
    model = Decoder(plan(17).model,lane='dense',seed=17)
    before = model_identity(model)
    enc = ByteTokenizer(256).encode('x'*250+'=123',answer_start=251)
    rows = [dict(ids=enc.input_ids,mask=enc.attention_mask,loss=enc.loss_mask)]*8
    timings = []
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        start = time.monotonic()
        loss,_ = target_loss(model,rows)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True,foreach=False)
        timings.append(time.monotonic()-start)
    model.zero_grad(set_to_none=True)
    start = time.monotonic()
    with torch.no_grad():
        for _ in range(2*((360+7)//8)):
            target_loss(model,rows)
        for _ in range((288+8)*3):
            target_loss(model,rows[:1])
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder)/'engineering.json'
        p.write_bytes(json_bytes(['x'*256]*648))
        p.read_bytes()
    overhead = time.monotonic()-start
    estimate = max(timings)*256*4
    whole = estimate+overhead*4
    gate.require(estimate < 120 and whole < 900, 'feasibility projection exceeds fixed schedule')
    gate.require(model_identity(model) == before, 'feasibility changed weights')
    count = sum(p.numel() for p in model.parameters())
    gate.require(count == 24768, 'actual parameter count differs')
    return dict(environment=env,parameters=count,optimizer_updates=0,gradient_calls=3,
        kernel_seconds_hex=[v.hex() for v in timings],update_projection_hex=estimate.hex(),
        diagnostic_io_seconds_hex=overhead.hex(),whole_projection_hex=whole.hex(),
        unchanged_model_identity=before)
