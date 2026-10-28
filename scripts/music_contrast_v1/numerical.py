"""Bounded composition of native CPU learning and scoring, no execution at import.

The controller owns the two-seed, retained-receipt training gate. ``holdout`` is
an internal worker operation, never an independently selectable public CLI.
No old corpus admission is extended and no numerical updater is duplicated.
"""
from time import monotonic

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, require_identity
from music_learning_v1.numerical import encoded, plan_for
from .spec import rows, require_admission

SEEDS = (17, 29)
CHECKPOINTS = (0, 64)
GENERATION_CAP = 64


def _seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ContractError('contrast seed differs from frozen protocol')
    return seed


def _admitted(admission_identity):
    require_identity(admission_identity)
    raw, receipt = require_admission(admission_identity)
    if identity(receipt) != admission_identity:
        raise ContractError('contrast native admission identity differs')
    return raw


def state_for(seed, admission_identity):
    from trinite.model import Decoder
    from trinite.training import (TrainingState, create_optimizer, evaluate,
                                  model_identity, require_environment)
    _seed(seed)
    raw = _admitted(admission_identity)
    require_environment()
    config = plan_for(seed)
    train_rows = rows('train')
    train = encoded(train_rows)
    target_count = sum(sum(train[i % len(train)]['loss']) for i in range(512))
    if target_count != 1024:
        raise ContractError('contrast schedule must contain 1024 answer/EOS targets')
    model = Decoder(config.model, lane='dense', seed=seed)
    return TrainingState(config, 'dense', model, create_optimizer(model, config),
        {'train': train, 'validation': train}, identity(raw), admission_identity,
        model_identity(model), evaluate(model, train, config.batch_size))


def _view(selector):
    train = rows('train')
    if selector == 'train':
        return train, 'train'
    if selector == 'heldout':
        return train + rows('heldout', siblings=True), 'test'
    raise ContractError('unknown contrast scoring selector')


def score(state, selector, emit=None):
    """Native prompt-only greedy predictions. Gold enters only native scoring."""
    from trinite.foundations import score_examples, check_prediction_examples
    from trinite.training import model_identity
    baseline_and_selected, native_split = _view(selector)
    train = [r for r in baseline_and_selected if r['split'] == 'train']
    selected = [r for r in baseline_and_selected if r['split'] == native_split]
    before = model_identity(state.model)
    predictions = []
    for row in selected:
        # Use the authoritative scorer, with only the fixed training baseline.
        # A per-row test view permits immediate journaling without a new decoder.
        _, prediction = score_examples(state.model, train + [{**row, 'split': 'test'}],
                                       'test', GENERATION_CAP)
        if len(prediction) != 1:
            raise ContractError('contrast native scorer coverage differs')
        record = {**prediction[0], 'split': native_split}
        if emit is not None:
            emit(record)
        predictions.append(record)
    raw = json_bytes({'predictions': predictions})
    result = check_prediction_examples(baseline_and_selected, native_split,
                                      GENERATION_CAP, raw)
    if model_identity(state.model) != before or any(p.grad is not None for p in state.model.parameters()):
        raise ContractError('contrast score changed model or gradients')
    return result, raw


def check_score(raw, selector):
    """Retained token/score consistency, without model execution."""
    from trinite.foundations import check_prediction_examples
    from trinite.contracts import parse_json
    from trinite.tokenizer import EOS
    examples, split = _view(selector)
    summary = check_prediction_examples(examples, split, GENERATION_CAP, raw)
    for row in parse_json(raw, canonical=True)['predictions']:
        generated = row['generated_tokens']
        if EOS in generated[:-1] or (generated[-1] != EOS and len(generated) != GENERATION_CAP):
            raise ContractError('contrast prediction is not EOS-or-cap termination')
    return summary


def snapshot(state, request_identity):
    from trinite.comparison_checkpoint import tensor_payload
    from trinite.training import model_identity
    require_identity(request_identity)
    payload = tensor_payload(state)
    metadata = {'schema': 'trinite.music-contrast-checkpoint.v1',
        'seed': state.config.seed, 'request_identity': request_identity,
        'plan': state.config.to_dict(), 'dataset_identity': state.dataset_identity,
        'admission_identity': state.manifest_identity,
        'initialization_identity': state.initialization_identity,
        'initial_train_loss_hex': state.initial_train_loss,
        'model_identity': model_identity(state.model), 'step': state.step,
        'cursor': state.cursor, 'target_tokens': state.target_tokens,
        'history': state.history, 'validation': state.validation}
    return payload, json_bytes(metadata)


def train(seed, request_identity, admission_identity, write, emit=None):
    """Two fixed observations, no held-out calls, election, or adaptive retries."""
    from trinite.training import run_steps, model_identity
    _seed(seed)
    require_identity(request_identity)
    if not callable(write):
        raise ContractError('contrast training requires retained artifacts')
    state = None
    elapsed = 0.0
    summaries = {}
    try:
        state = state_for(seed, admission_identity)
        for step in CHECKPOINTS:
            while state.step < step:
                if elapsed >= state.config.max_seconds:
                    raise ContractError('contrast cumulative update deadline exceeded')
                started = monotonic()
                try:
                    run_steps(state, stop_after=state.step + 1)
                finally:
                    elapsed += monotonic() - started
                if elapsed > state.config.max_seconds:
                    raise ContractError('contrast cumulative update deadline exceeded')
            prefix = f'step-{step:03d}'
            payload, metadata = snapshot(state, request_identity)
            write(prefix + '.safetensors', payload)
            write(prefix + '-checkpoint.json', metadata)
            summary, raw = score(state, 'train',
                                 None if emit is None else lambda row: emit(prefix, row))
            write(prefix + '-predictions.json', raw)
            summaries[step] = summary
        final = summaries[64]
        return {'seed': seed, 'request_identity': request_identity,
            'admission_identity': admission_identity, 'step': state.step,
            'target_tokens': state.target_tokens, 'initial_score': summaries[0],
            'final_score': final, 'training_fit': final['correct'] == final['examples'],
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
            'message': str(error), 'update_seconds_hex': elapsed.hex()}))
        raise


def verify_checkpoint(payload, metadata, seed, admission_identity, request_identity):
    """Native safe tensor/progress validation, never a forward or update replay.

    The caller must bind both bytes and metadata to the retained outer receipt.
    Initial loss/history are structure checked; they are not recomputed here.
    """
    from trinite.model import Decoder
    from trinite.training import TrainingState, create_optimizer, model_identity, require_environment
    from trinite.comparison_checkpoint import progress, restore_tensors
    _seed(seed)
    require_identity(request_identity)
    if type(payload) is not bytes or len(payload) > 32 * 1024 * 1024:
        raise ContractError('contrast checkpoint payload type/size differs')
    exact_keys(metadata, {'schema', 'seed', 'request_identity', 'plan', 'dataset_identity',
        'admission_identity', 'initialization_identity', 'initial_train_loss_hex', 'model_identity',
        'step', 'cursor', 'target_tokens', 'history', 'validation'}, 'contrast checkpoint')
    if len(json_bytes(metadata)) > 1024 * 1024:
        raise ContractError('contrast checkpoint metadata exceeds budget')
    require_identity(metadata['model_identity'])
    raw = _admitted(admission_identity)
    require_environment()
    config = plan_for(seed)
    encoded_train = encoded(rows('train'))
    model = Decoder(config.model, lane='dense', seed=seed)
    initial = model_identity(model)
    expected = {'schema': 'trinite.music-contrast-checkpoint.v1', 'seed': seed,
        'request_identity': request_identity, 'plan': config.to_dict(),
        'dataset_identity': identity(raw), 'admission_identity': admission_identity,
        'initialization_identity': initial}
    if any(json_bytes(metadata[key]) != json_bytes(value) for key, value in expected.items()):
        raise ContractError('contrast checkpoint context differs')
    state = TrainingState(config, 'dense', model, create_optimizer(model, config),
        {'train': encoded_train, 'validation': encoded_train}, identity(raw), admission_identity,
        initial, metadata['initial_train_loss_hex'])
    state.step, state.cursor, state.target_tokens = metadata['step'], metadata['cursor'], metadata['target_tokens']
    state.history, state.validation = metadata['history'], metadata['validation']
    progress(state)
    if state.step == 0 and metadata['model_identity'] != initial:
        raise ContractError('contrast step-zero checkpoint differs from seeded initial state')
    return restore_tensors(state, payload, metadata['model_identity'])


def holdout(seed, request_identity, admission_identity, payload, metadata, write, emit=None):
    """Internal worker operation after the controller verifies BOTH train receipts."""
    state = verify_checkpoint(payload, metadata, seed, admission_identity, request_identity)
    if state.step != 64:
        raise ContractError('contrast held-out scoring requires fixed final checkpoint')
    summary, raw = score(state, 'heldout',
                         None if emit is None else lambda row: emit('heldout', row))
    write('heldout-predictions.json', raw)
    return {'seed': seed, 'request_identity': request_identity,
        'admission_identity': admission_identity, 'model_identity': metadata['model_identity'],
        'score': summary}
