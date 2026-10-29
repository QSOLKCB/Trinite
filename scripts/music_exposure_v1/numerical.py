"""Train-only exposure composition; unchanged native updater/scorer/safe reader.

The three diagnostics use an isolated model copy. They restore process RNG and
module modes, and reject changes to the live model/optimizer/progress/data.
There is one fresh state, one cumulative update timer, and no resume operation.
"""
from contextlib import contextmanager
from copy import deepcopy
from time import monotonic

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, require_identity
from music_learning_v1.numerical import encoded
from .spec import CHECKPOINTS, SEEDS, plan_for, rows, require_admission

GENERATION_CAP = 64
_GUARD_ID = identity(b'trinite.music-exposure-observation-guard.v1')


def _seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ContractError('exposure seed differs from frozen protocol')
    return seed


def _admitted(admission_identity):
    require_identity(admission_identity)
    raw, receipt = require_admission(admission_identity)
    if identity(receipt) != admission_identity:
        raise ContractError('exposure native admission identity differs')
    return raw


def state_for(seed, admission_identity):
    from trinite.model import Decoder
    from trinite.training import (TrainingState, create_optimizer, evaluate,
                                  model_identity, require_environment)
    _seed(seed)
    raw = _admitted(admission_identity)
    require_environment()
    config = plan_for(seed)
    train = encoded(rows())
    targets = sum(sum(train[i % len(train)]['loss'])
                  for i in range(config.steps * config.batch_size))
    if len(train) != 40 or targets != 10240:
        raise ContractError('exposure schedule requires40 rows and10240 answer/EOS targets')
    model = Decoder(config.model, lane='dense', seed=seed)
    return TrainingState(config, 'dense', model, create_optimizer(model, config),
        {'train': train, 'validation': train}, identity(raw), admission_identity,
        model_identity(model), evaluate(model, train, config.batch_size))


def snapshot(state, request_identity):
    from trinite.comparison_checkpoint import tensor_payload
    from trinite.training import model_identity
    require_identity(request_identity)
    payload = tensor_payload(state)
    metadata = {'schema': 'trinite.music-exposure-checkpoint.v1',
        'seed': state.config.seed, 'request_identity': request_identity,
        'plan': state.config.to_dict(), 'dataset_identity': state.dataset_identity,
        'admission_identity': state.manifest_identity,
        'initialization_identity': state.initialization_identity,
        'initial_train_loss_hex': state.initial_train_loss,
        'model_identity': model_identity(state.model), 'step': state.step,
        'cursor': state.cursor, 'target_tokens': state.target_tokens,
        'history': state.history, 'validation': state.validation}
    return payload, json_bytes(metadata)


@contextmanager
def _preserve_training(state):
    """Protect live training, including error paths and callback side effects.

    Native safe snapshots authenticate model/AdamW tensors and full progress;
    data and live object identities are checked separately. RNG/modes are
    deliberately restored, never used to choose examples or future updates.
    This is same-process conformance, not a hostile-code isolation boundary.
    """
    import random
    import numpy as np
    import torch
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    model, optimizer = state.model, state.optimizer
    modes = [(module, module.training) for module in model.modules()]
    parameters = [(name, id(p), p.requires_grad)
                  for name, p in model.named_parameters(remove_duplicate=False)]
    buffers = [(name, id(value)) for name, value in model.named_buffers()]
    modules = [(name, id(module)) for name, module in model.named_modules()]
    grad_enabled = torch.is_grad_enabled()
    before = snapshot(state, _GUARD_ID)
    data = deepcopy(state.data)
    try:
        yield
    finally:
        try:
            if state.model is not model or state.optimizer is not optimizer \
                    or parameters != [(n, id(p), p.requires_grad)
                                      for n, p in model.named_parameters(remove_duplicate=False)] \
                    or buffers != [(n, id(v)) for n, v in model.named_buffers()] \
                    or modules != [(n, id(m)) for n, m in model.named_modules()] \
                    or snapshot(state, _GUARD_ID) != before or state.data != data:
                raise ContractError('exposure diagnostic changed live training state')
        finally:
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            torch.set_grad_enabled(grad_enabled)
            # Direct per-module restoration preserves mixed modes, unlike train().
            for module, mode in modes:
                module.training = mode


def score(state, emit=None):
    """Only training prompts, native greedy, isolated from live optimizer state."""
    import torch
    from trinite.foundations import score_examples
    from trinite.training import model_identity
    train = rows()
    predictions = []
    with _preserve_training(state), torch.no_grad():
        model = deepcopy(state.model)
        if model_identity(model) != model_identity(state.model):
            raise ContractError('exposure diagnostic copy differs from live model')
        for row in train:
            # Baseline uses only the fixed unique training inventory. The
            # ephemeral test label selects one TRAIN row in the shared scorer.
            _, prediction = score_examples(model, train + [{**row, 'split': 'test'}],
                                           'test', GENERATION_CAP)
            if len(prediction) != 1:
                raise ContractError('exposure native scorer coverage differs')
            record = {**prediction[0], 'split': 'train'}
            if emit is not None:
                emit(record)
            predictions.append(record)
        raw = json_bytes({'predictions': predictions})
        result = check_score(raw)
    return result, raw


def check_score(raw):
    """Retained token/score consistency only; no numerical model execution."""
    from trinite.foundations import check_prediction_examples
    from trinite.contracts import parse_json
    from trinite.tokenizer import EOS
    summary = check_prediction_examples(rows(), 'train', GENERATION_CAP, raw)
    for row in parse_json(raw, canonical=True)['predictions']:
        generated = row['generated_tokens']
        if EOS in generated[:-1] or (generated[-1] != EOS and len(generated) != GENERATION_CAP):
            raise ContractError('exposure prediction is not EOS-or-cap termination')
    return summary


def train(seed, request_identity, admission_identity, write, emit=None):
    """Continuous fixed exposure, three observations, no retries or heldout API.

    The accumulator covers every single-step run_steps call (including native
    train-only validation), not snapshot/greedy overhead. The outer worker
    deadline bounds the whole process. A late final update is still incomplete.
    """
    from trinite.training import run_steps, model_identity
    _seed(seed)
    require_identity(request_identity)
    if not callable(write):
        raise ContractError('exposure training requires retained artifacts')
    state = None
    elapsed = 0.0
    summaries = {}
    try:
        state = state_for(seed, admission_identity)
        for step in CHECKPOINTS:
            while state.step < step:
                if elapsed >= state.config.max_seconds:
                    raise ContractError('exposure cumulative update deadline exceeded')
                started = monotonic()
                try:
                    run_steps(state, stop_after=state.step + 1)
                finally:
                    elapsed += monotonic() - started
                if elapsed > state.config.max_seconds:
                    raise ContractError('exposure cumulative update deadline exceeded')
            prefix = f'step-{step:03d}'
            with _preserve_training(state):
                payload, metadata = snapshot(state, request_identity)
                write(prefix + '.safetensors', payload)
                write(prefix + '-checkpoint.json', metadata)
                summary, raw = score(state,
                    None if emit is None else lambda row: emit(prefix, row))
                write(prefix + '-predictions.json', raw)
            summaries[str(step)] = summary
        final = summaries[str(CHECKPOINTS[-1])]
        return {'seed': seed, 'request_identity': request_identity,
            'admission_identity': admission_identity, 'step': state.step,
            'target_tokens': state.target_tokens, 'observations': summaries,
            'training_fit': final['correct'] == 40 and final['examples'] == 40,
            'model_identity': model_identity(state.model), 'update_seconds_hex': elapsed.hex()}
    except Exception as error:
        if state is not None:
            try:
                payload, metadata = snapshot(state, request_identity)
                write('failure.safetensors', payload)
                write('failure-checkpoint.json', metadata)
            except Exception as checkpoint_error:
                write('failure-checkpoint-error.json', json_bytes({
                    'type': type(checkpoint_error).__name__, 'message': str(checkpoint_error)}))
        write('failure.json', json_bytes({'type': type(error).__name__,
            'message': str(error), 'update_seconds_hex': elapsed.hex(),
            'completed_observations': [step for step in CHECKPOINTS if str(step) in summaries],
            'missing_observations': [step for step in CHECKPOINTS if str(step) not in summaries],
            'completed_updates': None if state is None else state.step,
            'outcome': 'incomplete'}))
        raise


def verify_checkpoint(payload, metadata, seed, admission_identity, request_identity):
    """Safe tensor/progress validation, never inference or optimizer replay.

    Caller must bind payload and metadata to the retained outer receipt. This
    reader supports partial failure checkpoints; no public resume path exists.
    """
    from trinite.model import Decoder
    from trinite.training import TrainingState, create_optimizer, model_identity, require_environment
    from trinite.comparison_checkpoint import progress, restore_tensors
    _seed(seed)
    require_identity(request_identity)
    if type(payload) is not bytes or len(payload) > 32 * 1024 * 1024:
        raise ContractError('exposure checkpoint payload type/size differs')
    exact_keys(metadata, {'schema', 'seed', 'request_identity', 'plan', 'dataset_identity',
        'admission_identity', 'initialization_identity', 'initial_train_loss_hex', 'model_identity',
        'step', 'cursor', 'target_tokens', 'history', 'validation'}, 'exposure checkpoint')
    if len(json_bytes(metadata)) > 1024 * 1024:
        raise ContractError('exposure checkpoint metadata exceeds budget')
    require_identity(metadata['model_identity'])
    raw = _admitted(admission_identity)
    require_environment()
    config = plan_for(seed)
    encoded_train = encoded(rows())
    model = Decoder(config.model, lane='dense', seed=seed)
    initial = model_identity(model)
    expected = {'schema': 'trinite.music-exposure-checkpoint.v1', 'seed': seed,
        'request_identity': request_identity, 'plan': config.to_dict(),
        'dataset_identity': identity(raw), 'admission_identity': admission_identity,
        'initialization_identity': initial}
    if any(json_bytes(metadata[key]) != json_bytes(value) for key, value in expected.items()):
        raise ContractError('exposure checkpoint context differs')
    state = TrainingState(config, 'dense', model, create_optimizer(model, config),
        {'train': encoded_train, 'validation': encoded_train}, identity(raw), admission_identity,
        initial, metadata['initial_train_loss_hex'])
    state.step, state.cursor, state.target_tokens = metadata['step'], metadata['cursor'], metadata['target_tokens']
    state.history, state.validation = metadata['history'], metadata['validation']
    progress(state)
    if state.step == 0 and metadata['model_identity'] != initial:
        raise ContractError('exposure step-zero checkpoint differs from seeded initial state')
    return restore_tensors(state, payload, metadata['model_identity'])
