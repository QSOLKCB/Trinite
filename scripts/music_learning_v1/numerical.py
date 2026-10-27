"""Prospective music-only composition of the unchanged native CPU primitives.

No model is constructed at import. Public entrypoints require the new lane's
reviewed admission gate; the PR81 conformance receipt is never native admission.
"""
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, exact_keys, parse_json
from trinite.foundations_plan import FoundationsPlan
from .codec import encode_example

SEEDS = (17, 29)
GENERATION_CAP = 64


def plan_for(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ContractError('music seed outside frozen pilot')
    return FoundationsPlan(model=ModelConfig(context_length=256, blocks=1, width=32,
        heads=4, feed_forward_width=64), seed=seed, steps=64, batch_size=8,
        learning_rate='0.001', validation_every=64, max_seconds=120,
        max_target_tokens=32768)


def examples():
    """Fresh exact public projections; custody metadata never enters the carrier."""
    from .admission import dataset, audit_bytes
    raw, receipt = dataset()
    audit_bytes(raw, receipt)
    rows = parse_json(raw, canonical=True)['examples']
    result = []
    for row in rows:
        meter = row['formal']['meter']
        if 1 + len(row['prompt'].encode('ascii')) + GENERATION_CAP > 256:
            raise ContractError('music fixed decoding cap exceeds context')
        result.append({**row, 'meter': str(meter['numerator']) + '/' + str(meter['denominator'])})
    return result


def train_rows():
    # Identity hash rank is fixed across initialization seeds. Every unique row
    # appears once before the final 32 repeated visits; no test/validation rows.
    return sorted((r for r in examples() if r['split'] == 'train'),
                  key=lambda r: r['example_identity'])


def encoded(rows):
    result = []
    for row in rows:
        value = {k: row[k] for k in ('task', 'prompt', 'answer', 'text', 'answer_start')}
        item = encode_example(value)
        result.append({'identity': row['example_identity'], 'ids': item.input_ids,
                       'mask': item.attention_mask, 'loss': item.loss_mask})
    return result


def schedule_receipt(seed):
    config = plan_for(seed)
    rows = train_rows()
    train = encoded(rows)
    visits = [train[i % len(train)] for i in range(config.steps * config.batch_size)]
    targets = sum(sum(row['loss']) for row in visits)
    if targets > config.max_target_tokens:
        raise ContractError('music frozen target budget exceeded')
    return {'unique_train_examples': len(train), 'scheduled_visits': len(visits),
            'unique_scheduled_examples': len({r['identity'] for r in visits}),
            'target_tokens': targets,
            'schedule_identity': identity(json_bytes([r['identity'] for r in visits]))}


def state_for(seed, admission_identity):
    from .workflow import require_native_admission
    require_native_admission(admission_identity)
    from .admission import dataset, audit_bytes
    raw, receipt = dataset()
    audit_bytes(raw, receipt)
    if identity(receipt) != admission_identity:
        raise ContractError('music native admission identity mismatch')
    from trinite.model import Decoder
    from trinite.training import (TrainingState, create_optimizer, evaluate,
                                  model_identity, require_environment)
    require_environment()
    config = plan_for(seed)
    schedule_receipt(seed)
    train = encoded(train_rows())
    model = Decoder(config.model, lane='dense', seed=seed)
    return TrainingState(config, 'dense', model, create_optimizer(model, config),
        {'train': train, 'validation': train}, identity(raw),
        admission_identity, model_identity(model), evaluate(model, train, config.batch_size))


def view(selector):
    if selector not in ('train', 'validation', 'test'):
        raise ContractError('unknown music scoring split')
    rows = examples()
    train = [r for r in rows if r['split'] == 'train']
    if selector == 'train':
        return train, 'train'
    # Native scorer has train/test selectors. Only this ephemeral scorer view
    # relabels validation; the outer record retains its original split identity.
    return train + [{**r, 'split': 'test'} for r in rows if r['split'] == selector], 'test'


def score(state, selector, emit):
    """Use the authoritative scorer per row and retain each prediction at once.

    Its train-only baseline pool is unchanged. A one-row ephemeral test view
    allows journaling without reimplementing greedy decoding or exact scoring.
    """
    if not callable(emit):
        raise ContractError('music scoring requires a prediction journal')
    from trinite.foundations import score_examples, check_prediction_examples
    from trinite.training import model_identity
    rows, native_split = view(selector)
    baseline_rows = [row for row in rows if row['split'] == 'train']
    selected = [row for row in rows if row['split'] == native_split]
    predictions = []
    before = model_identity(state.model)
    for item in selected:
        _, single = score_examples(state.model, baseline_rows + [{**item, 'split': 'test'}],
                                   'test', GENERATION_CAP)
        if len(single) != 1:
            raise ContractError('music per-row scorer returned wrong coverage')
        prediction = {**single[0], 'split': native_split}
        emit(prediction)
        predictions.append(prediction)
    if model_identity(state.model) != before:
        raise ContractError('music scoring changed model identity')
    summary = check_prediction_examples(rows, native_split, GENERATION_CAP,
                                        json_bytes({'predictions': predictions}))
    return {'selector': selector, 'model_identity': before,
            'summary': summary, 'predictions': predictions}


def check_score(record, selector, expected_model):
    from trinite.foundations import check_prediction_examples
    exact_keys(record, {'selector', 'model_identity', 'summary', 'predictions'}, 'music scores')
    if record['selector'] != selector or record['model_identity'] != expected_model:
        raise ContractError('music scoring context differs')
    rows, native_split = view(selector)
    summary = check_prediction_examples(rows, native_split, GENERATION_CAP,
                                       json_bytes({'predictions': record['predictions']}))
    if json_bytes(summary) != json_bytes(record['summary']):
        raise ContractError('music summary differs from complete retained predictions')
    return summary


def snapshot(state, request_identity):
    from trinite.comparison_checkpoint import tensor_payload
    from trinite.training import model_identity
    payload = tensor_payload(state)
    metadata = {'schema': 'trinite.music-learning-checkpoint.v1',
        'request_identity': request_identity, 'plan': state.config.to_dict(),
        'dataset_identity': state.dataset_identity, 'admission_identity': state.manifest_identity,
        'initialization_identity': state.initialization_identity,
        'initial_train_loss_hex': state.initial_train_loss, 'model_identity': model_identity(state.model),
        'step': state.step, 'cursor': state.cursor, 'target_tokens': state.target_tokens,
        'history': state.history, 'validation': state.validation}
    return payload, json_bytes(metadata)


def restore(payload, metadata, seed, request_identity, admission_identity, expected_identities):
    # Expected byte identities must come from independently bound outer custody.
    if (type(payload) is not bytes or type(metadata) is not bytes
            or len(payload) > 32 * 1024 * 1024 or len(metadata) > 1024 * 1024
            or (identity(payload), identity(metadata)) != expected_identities):
        raise ContractError('music checkpoint identity/type/size mismatch')
    from trinite.comparison_checkpoint import progress, restore_tensors
    value = parse_json(metadata, canonical=True)
    state = state_for(seed, admission_identity)
    expected = {'schema': 'trinite.music-learning-checkpoint.v1',
        'request_identity': request_identity, 'plan': state.config.to_dict(),
        'dataset_identity': state.dataset_identity, 'admission_identity': admission_identity,
        'initialization_identity': state.initialization_identity,
        'initial_train_loss_hex': state.initial_train_loss}
    exact_keys(value, set(expected) | {'model_identity', 'step', 'cursor', 'target_tokens',
                                     'history', 'validation'}, 'music checkpoint')
    if any(json_bytes(value[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('music checkpoint context mismatch')
    state.step, state.cursor, state.target_tokens = value['step'], value['cursor'], value['target_tokens']
    state.history, state.validation = value['history'], value['validation']
    progress(state)
    return restore_tensors(state, payload, value['model_identity'])


def run_trial(seed, admission_identity, request_identity, write, journal):
    """Controller-only future execution; retain files immediately, never overwrite.

    An outer deadline/custody controller is required. This function is not a CLI
    and does not grant admission, authorization, replay or hardware attestation.
    """
    from .workflow import require_native_admission
    require_native_admission(admission_identity)
    from trinite.training import run_steps
    from time import monotonic
    from .reporting import describe
    try:
        state = state_for(seed, admission_identity)
        update_seconds = 0.0
        for stage in ('untrained', 'trained'):
            if stage == 'trained':
                started = monotonic()
                try:
                    run_steps(state)
                    update_seconds = monotonic() - started
                    if update_seconds > state.config.max_seconds:
                        raise ContractError('music updates and final train diagnostic exceeded elapsed budget')
                except Exception:
                    # Best-effort safe checkpoint after a completed native boundary;
                    # preserve the original failure if partial gradients forbid it.
                    try:
                        payload, metadata = snapshot(state, request_identity)
                        write('update-failure.safetensors', payload)
                        write('update-failure-checkpoint.json', metadata)
                    except Exception as checkpoint_error:
                        write('update-checkpoint-failure.json', json_bytes({
                            'type': type(checkpoint_error).__name__, 'message': str(checkpoint_error)}))
                    raise
            records = {}
            for selector in ('train', 'validation', 'test'):
                records[selector] = score(state, selector,
                    lambda prediction, s=selector: journal(stage, s, prediction))
                write(stage + '-' + selector + '.json', json_bytes(records[selector]))
            write(stage + '-description.json', json_bytes(describe(records)))
            payload, metadata = snapshot(state, request_identity)
            write(stage + '.safetensors', payload)
            write(stage + '-checkpoint.json', metadata)
        return {'seed': seed, 'outcome': 'completed', 'step': state.step,
                'target_tokens': state.target_tokens, 'request_identity': request_identity,
                'schedule': schedule_receipt(seed), 'model_identity': records['train']['model_identity'],
                'update_seconds_hex': update_seconds.hex()}
    except Exception as error:
        write('failure.json', json_bytes({'outcome': 'failed', 'type': type(error).__name__,
                                         'message': str(error)}))
        raise
