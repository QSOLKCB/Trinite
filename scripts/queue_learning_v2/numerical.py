"""Compose the unchanged dense native model, updater, decoder and safe reader."""
import time

import queue_evaluation_v1 as gate
from trinite.contracts import json_bytes, parse_json, identity
from queue_learning_v1.codec import GENERATION, prompt, output
from .protocol import plan, fit_rows
from .spec import sources


def configure():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from trinite.training import require_environment
    return require_environment()


def schedule(rows, seed, *, fit=False):
    config = plan(seed, fit=fit)
    from queue_learning_v1.admission import check_training_rows
    check_training_rows(rows)
    rows = fit_rows(rows) if fit else rows
    visits = [rows[i % len(rows)] for i in range(config.steps * config.batch_size)]
    count = sum(sum(r['loss']) for r in visits)
    gate.require(count <= config.max_target_tokens, 'queue scheduled supervised token budget exceeded')
    return {'unique_train': len(rows), 'scheduled_visits': len(visits), 'target_tokens': count,
            'order_sha256': gate.digest(json_bytes([r['identity'] for r in visits]))}


def state_for(rows, seed, binding, *, fit=False):
    from trinite.model import Decoder
    from trinite.training import TrainingState, create_optimizer, evaluate, model_identity, require_environment
    require_environment()
    schedule(rows, seed, fit=fit)
    rows = fit_rows(rows) if fit else rows
    config = plan(seed, fit=fit)
    model = Decoder(config.model, lane='dense', seed=seed)
    # Native updater's final diagnostic is deliberately train-only. No held-out
    # validation loss, example, label or checkpoint election enters this state.
    return TrainingState(config, 'dense', model, create_optimizer(model, config),
        {'train': rows, 'validation': rows}, identity(json_bytes(rows)), binding,
        model_identity(model), evaluate(model, rows, config.batch_size))


def snapshot(state, binding):
    from trinite.comparison_checkpoint import tensor_payload
    from trinite.training import model_identity, require_environment
    payload = tensor_payload(state)
    value = {'schema': 'trinite.queue-learning-checkpoint.v2', 'binding': binding,
             'sources': sources(), 'environment': require_environment(), 'plan': state.config.to_dict(),
             'dataset_identity': state.dataset_identity, 'initialization_identity': state.initialization_identity,
             'initial_train_loss': state.initial_train_loss, 'model_identity': model_identity(state.model),
             'step': state.step, 'cursor': state.cursor, 'target_tokens': state.target_tokens,
             'history': state.history, 'train_diagnostics': state.validation}
    raw = json_bytes(value)
    gate.require(len(raw) <= 1024 * 1024, 'queue checkpoint metadata bound')
    return payload, raw


def restore(payload, metadata, rows, seed, binding, expected, *, fit=False):
    gate.require(type(payload) is bytes and len(payload) <= 32 * 1024 * 1024
                 and type(metadata) is bytes and len(metadata) <= 1024 * 1024
                 and (gate.digest(payload), gate.digest(metadata)) == tuple(expected),
                 'queue checkpoint externally bound bytes mismatch')
    value = parse_json(metadata, canonical=True)
    state = state_for(rows, seed, binding, fit=fit)
    _, initial = snapshot(state, binding)
    context = parse_json(initial, canonical=True)
    gate.keys(value, context.keys())
    for name in set(context) - {'model_identity', 'step', 'cursor', 'target_tokens', 'history', 'train_diagnostics'}:
        gate.require(json_bytes(value[name]) == json_bytes(context[name]), 'queue checkpoint context mismatch')
    state.step, state.cursor, state.target_tokens = value['step'], value['cursor'], value['target_tokens']
    state.history, state.validation = value['history'], value['train_diagnostics']
    # This is the authoritative closed-inventory safetensors reader. No pickle,
    # executable allowlist, alternate tensor loader or recovery fallback.
    from trinite.comparison_checkpoint import restore_tensors
    return restore_tensors(state, payload, value['model_identity'])


def infer(model, observation_bytes):
    """The inference call accepts public observation and the fixed declared task."""
    from trinite.comparison import greedy
    prefix = prompt(observation_bytes)
    tokens = greedy(model, prefix, GENERATION)
    return {'status': output(tokens, observation_bytes)['status'], 'tokens': tokens}


def feasibility_kernel():
    """Exposed worst-context engineering gradient timing. ZERO optimizer steps.

    Labels here are arbitrary engineering bytes, not generated pilot outcomes.
    Discard the untrained model and its gradients. Do not claim measured updates.
    """
    from trinite.model import Decoder
    from trinite.tokenizer import ByteTokenizer
    from trinite.training import target_loss, model_identity
    import torch
    env = configure()
    model = Decoder(plan(17).model, lane='dense', seed=17)
    before = model_identity(model)
    enc = ByteTokenizer(256).encode('x' * 250 + '=123', answer_start=251)
    rows = [{'ids': enc.input_ids, 'mask': enc.attention_mask, 'loss': enc.loss_mask}] * 8
    timings = []
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        started = time.monotonic()
        loss, _ = target_loss(model, rows)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True, foreach=False)
        timings.append(time.monotonic() - started)
    model.zero_grad(set_to_none=True)
    import tempfile
    from pathlib import Path
    diagnostic_start = time.monotonic()
    with torch.no_grad():
        # Two full unique-train passes plus all heldout inference, deliberately
        # worst context; no queue label or fresh heldout target is materialized.
        for _ in range(128):
            target_loss(model, rows)
        # Always charge five worst-context forwards per generation, even if an
        # untrained greedy decoder would emit EOS early. Additional CE makes
        # this an engineering cost envelope, not a modified decoding policy.
        for _ in range(656 * GENERATION):
            target_loss(model, rows[:1])
    with tempfile.TemporaryDirectory() as folder:
        destination = Path(folder) / 'engineering.json'
        destination.write_bytes(json_bytes({'engineering': ['x' * 256] * 656}))
        destination.read_bytes()
    diagnostic_io = time.monotonic() - diagnostic_start
    whole = max(timings) * 256 * 4 + diagnostic_io * 4
    gate.require(whole < 900, 'queue diagnostic/IO projection exceeds attempt budget')
    gate.require(model_identity(model) == before, 'feasibility mutated native weights')
    estimate = max(timings) * 256 * 4  # Fourfold headroom; not an update-time measurement.
    gate.require(estimate < 120, 'queue engineering timing estimate exceeds proposed update budget')
    return {'environment': env, 'parameters': plan(17).model.parameter_count,
            'optimizer_updates': 0, 'gradient_calls': 3,
            'diagnostic_io_seconds_hex': diagnostic_io.hex(),
            'conservative_whole_attempt_seconds_hex': whole.hex(),
            'kernel_seconds_hex': [t.hex() for t in timings],
            'conservative_256_update_estimate_seconds_hex': estimate.hex(),
            'model_identity_unchanged': before,
            'scope': 'worst-context forward/backward/clip timing; no AdamW step, pilot labels, training or learned checkpoint'}
