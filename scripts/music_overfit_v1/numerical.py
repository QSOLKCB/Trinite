"""Train-only composition of unchanged native primitives; no execution at import."""
from time import monotonic

from trinite.contracts import ContractError, identity, json_bytes
from music_learning_v1.numerical import encoded, plan_for
from .spec import subset_rows, validate_rows

CHECKPOINTS = (0, 8, 16, 32, 64)
GENERATION_CAP = 64
PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-5


def state_for(seed, subset, admission_identity):
    from music_learning_v1.workflow import require_native_admission
    from trinite.model import Decoder
    from trinite.training import (TrainingState, create_optimizer, evaluate,
                                  model_identity, require_environment)
    raw, _ = require_native_admission(admission_identity)
    require_environment()
    config = plan_for(seed)
    rows = subset_rows(subset)
    train = encoded(rows)
    if sum(sum(train[i % len(train)]['loss']) for i in range(512)) > config.max_target_tokens:
        raise ContractError('overfit schedule exceeds target ceiling')
    model = Decoder(config.model, lane='dense', seed=seed)
    state = TrainingState(config, 'dense', model, create_optimizer(model, config),
        {'train': train, 'validation': train}, identity(raw), admission_identity,
        model_identity(model), evaluate(model, train, config.batch_size))
    # Native validation is explicitly an alias for these unique train rows.
    state.music_subset = subset
    return state


def _prediction(vector, target, *, retain_logits=False):
    import torch
    if vector.shape != (259,) or not bool(torch.isfinite(vector).all()):
        raise ContractError('invalid diagnostic logits')
    winner = int(vector.argmax())
    result = {'target': target, 'argmax': winner,
        'rank': 1 + int((vector > vector[target]).sum()),
        'target_probability_hex': float(torch.softmax(vector, dim=0)[target]).hex(),
        'target_logit_hex': float(vector[target]).hex(),
        'argmax_logit_hex': float(vector[winner]).hex(),
        'target_minus_argmax_hex': float(vector[target] - vector[winner]).hex()}
    if retain_logits:
        result['logits_hex'] = [float(value).hex() for value in vector]
    return result


def inspect(state, rows):
    """Gold-prefix observations and raw prompt-only decoding stay separate."""
    import torch
    from trinite.training import batch, evaluate, model_identity
    from trinite.foundations import score_examples
    from trinite.tokenizer import EOS
    rows = validate_rows(rows, state.music_subset)
    train = encoded(rows)
    if train != state.data['train'] or train != state.data['validation']:
        raise ContractError('diagnostic rows differ from admitted update rows')
    before = model_identity(state.model)
    if any(p.grad is not None for p in state.model.parameters()):
        raise ContractError('diagnostic requires cleared gradients')
    observations = []
    with torch.no_grad():
        for row, encoding in zip(rows, train):
            ids, attention, scored = batch([encoding])
            p, a = len(row['prompt'].encode('ascii')), len(row['answer'].encode('ascii'))
            targets = [*row['answer'].encode('ascii'), EOS]
            if (list(ids[0, p+1:].tolist()) != targets
                    or list(scored[0].tolist()) != [False] * (p+1) + [True] * (a+1)
                    or not bool(attention.all())):
                raise ContractError('answer/EOS shifted mask alignment differs')
            logits = state.model(ids[:, :-1], attention[:, :-1]).logits[0]
            positions = []
            for offset, target in enumerate(targets):
                positions.append({'target_position': p+1+offset,
                    'logit_position': p+offset, **_prediction(logits[p+offset], target)})
            boundaries = {}
            for label, index, target in (('first_answer', p, targets[0]), ('gold_prefix_eos', p+a, EOS)):
                prefix = ids[:, :index+1]
                observed = state.model(prefix, attention[:, :index+1]).logits[0, -1]
                delta = float((observed - logits[index]).abs().max())
                if not torch.allclose(observed, logits[index], atol=PARITY_ATOL, rtol=PARITY_RTOL):
                    raise ContractError('teacher-forced/prefix boundary logits disagree')
                boundaries[label] = {'logit_position': index,
                    'prefix_max_abs_difference_hex': delta.hex(),
                    **_prediction(logits[index], target, retain_logits=True),
                    'prefix_logits_hex': [float(value).hex() for value in observed]}
            observations.append({'example_identity': row['example_identity'],
                'task': row['task'], 'positions': positions, **boundaries})
    # Authoritative scorer supplies raw outputs and exact answer-plus-EOS score.
    summary, predictions = score_examples(state.model, rows, 'train', GENERATION_CAP)
    for observation, prediction in zip(observations, predictions):
        expected, generated = prediction['expected_tokens'], prediction['generated_tokens']
        first = next((i for i in range(max(len(expected), len(generated)))
                      if i >= len(expected) or i >= len(generated) or expected[i] != generated[i]), None)
        observation['free_generation'] = prediction
        observation['first_divergent_position'] = first
        observation['termination'] = 'eos' if generated and generated[-1] == EOS else 'cap'
        observation['teacher_forced_all_targets_correct'] = all(
            position['target'] == position['argmax'] for position in observation['positions'])
        if generated[0] != observation['first_answer']['argmax']:
            raise ContractError('prompt-only first answer differs from teacher-forced first answer')
    result = {'schema': 'trinite.music-overfit-diagnostics.v1', 'step': state.step,
        'subset': state.music_subset, 'model_identity': before,
        'train_loss_hex': evaluate(state.model, train, state.config.batch_size),
        'unique_train_examples': len(rows), 'raw_score': summary, 'rows': observations}
    if model_identity(state.model) != before or any(p.grad is not None for p in state.model.parameters()):
        raise ContractError('diagnostic mutated model or gradients')
    return result


def snapshot(state, request_identity):
    from trinite.comparison_checkpoint import tensor_payload
    from trinite.training import model_identity
    payload = tensor_payload(state)
    metadata = {'schema': 'trinite.music-overfit-checkpoint.v1',
        'subset': state.music_subset, 'request_identity': request_identity,
        'plan': state.config.to_dict(), 'dataset_identity': state.dataset_identity,
        'admission_identity': state.manifest_identity,
        'initialization_identity': state.initialization_identity,
        'initial_train_loss_hex': state.initial_train_loss,
        'model_identity': model_identity(state.model), 'step': state.step,
        'cursor': state.cursor, 'target_tokens': state.target_tokens,
        'history': state.history, 'validation': state.validation}
    return payload, json_bytes(metadata)


def run_trial(seed, subset, admission_identity, request_identity, write):
    """Controller-only future execution, fixed final selection and cumulative cap."""
    from trinite.training import run_steps, model_identity
    state = None
    update_seconds = 0.0
    try:
        state = state_for(seed, subset, admission_identity)
        rows = subset_rows(subset)
        for step in CHECKPOINTS:
            while state.step < step:
                if update_seconds >= state.config.max_seconds:
                    raise ContractError('overfit cumulative update deadline exceeded')
                started = monotonic()
                try:
                    run_steps(state, stop_after=state.step + 1)
                finally:
                    update_seconds += monotonic() - started
                if update_seconds > state.config.max_seconds:
                    raise ContractError('overfit cumulative update deadline exceeded')
            name = f'step-{step:03d}'
            payload, metadata = snapshot(state, request_identity)
            write(name + '.safetensors', payload)
            write(name + '-checkpoint.json', metadata)
            diagnostic = inspect(state, rows)
            write(name + '-diagnostics.json', json_bytes(diagnostic))
        correct = sum(row['free_generation']['correct'] for row in diagnostic['rows'])
        return {'seed': seed, 'subset': subset, 'outcome': 'completed',
            'step': state.step, 'target_tokens': state.target_tokens,
            'request_identity': request_identity, 'model_identity': model_identity(state.model),
            'final_train_exact': correct, 'total_train': len(rows),
            'training_fit': correct == len(rows), 'update_seconds_hex': update_seconds.hex()}
    except Exception as error:
        # Retain all completed boundaries and the failure, without resealing/retry.
        if state is not None:
            try:
                payload, metadata = snapshot(state, request_identity)
                write('failure.safetensors', payload)
                write('failure-checkpoint.json', metadata)
            except Exception as checkpoint_error:
                write('failure-checkpoint-error.json', json_bytes({
                    'type': type(checkpoint_error).__name__, 'message': str(checkpoint_error)}))
        write('failure.json', json_bytes({'outcome': 'failed', 'type': type(error).__name__,
            'message': str(error), 'update_seconds_hex': update_seconds.hex()}))
        raise


def check_diagnostic(record, rows):
    """Read-only record consistency, never a forward or optimizer replay."""
    import math
    from trinite.contracts import exact_keys, integer, require_identity
    from trinite.foundations import check_prediction_examples
    from trinite.tokenizer import EOS

    def finite(value):
        if type(value) is not str:
            raise ContractError('diagnostic requires canonical finite float hex')
        try:
            number = float.fromhex(value)
        except (ValueError, OverflowError) as error:
            raise ContractError('diagnostic requires canonical finite float hex') from error
        if not math.isfinite(number) or number.hex() != value:
            raise ContractError('diagnostic requires canonical finite float hex')
        return number

    prediction_keys = {'target', 'argmax', 'rank', 'target_probability_hex',
        'target_logit_hex', 'argmax_logit_hex', 'target_minus_argmax_hex'}

    def prediction(value, target, extras):
        exact_keys(value, prediction_keys | extras, 'overfit prediction')
        integer(value['target'], 'diagnostic target', 0, 258)
        integer(value['argmax'], 'diagnostic argmax', 0, 258)
        integer(value['rank'], 'diagnostic rank', 1, 259)
        if value['target'] != target:
            raise ContractError('diagnostic target differs')
        probability = finite(value['target_probability_hex'])
        target_logit = finite(value['target_logit_hex'])
        maximum = finite(value['argmax_logit_hex'])
        margin = finite(value['target_minus_argmax_hex'])
        if not 0 <= probability <= 1 or margin > 0 or maximum < target_logit:
            raise ContractError('diagnostic probabilities/logits invalid')
        if not math.isclose(margin, target_logit-maximum, abs_tol=PARITY_ATOL, rel_tol=PARITY_RTOL):
            raise ContractError('diagnostic margin differs')
        if 'logits_hex' in extras:
            vectors = []
            for key in ('logits_hex', 'prefix_logits_hex'):
                values = value[key]
                if type(values) is not list or len(values) != 259:
                    raise ContractError('diagnostic logit inventory differs')
                vectors.append([finite(v) for v in values])
            full, prefix = vectors
            winner = max(range(259), key=full.__getitem__)
            if (winner != value['argmax'] or full[target] != target_logit
                    or full[winner] != maximum
                    or 1 + sum(v > full[target] for v in full) != value['rank']):
                raise ContractError('diagnostic logit-derived prediction differs')
            if any(abs(a-b) > PARITY_ATOL + PARITY_RTOL*abs(a) for a,b in zip(full,prefix)):
                raise ContractError('diagnostic prefix parity differs')
            delta = finite(value['prefix_max_abs_difference_hex'])
            if delta < 0 or not math.isclose(delta, max(abs(a-b) for a,b in zip(full,prefix)),
                                           abs_tol=PARITY_ATOL, rel_tol=PARITY_RTOL):
                raise ContractError('diagnostic prefix difference differs')
            expected_probability = math.exp(full[target]-maximum)/sum(math.exp(v-maximum) for v in full)
            if not math.isclose(probability, expected_probability, abs_tol=PARITY_ATOL, rel_tol=PARITY_RTOL):
                raise ContractError('diagnostic target probability differs')

    exact_keys(record, {'schema','step','subset','model_identity','train_loss_hex',
                       'unique_train_examples','raw_score','rows'}, 'overfit diagnostic')
    rows = validate_rows(rows, record['subset'])
    if record['schema'] != 'trinite.music-overfit-diagnostics.v1' or type(record['step']) is not int or record['step'] not in CHECKPOINTS:
        raise ContractError('diagnostic schema/checkpoint differs')
    require_identity(record['model_identity'])
    if finite(record['train_loss_hex']) < 0:
        raise ContractError('negative diagnostic loss')
    if type(record['unique_train_examples']) is not int or record['unique_train_examples'] != len(rows):
        raise ContractError('diagnostic train count differs')
    observed = record['rows']
    if type(observed) is not list or len(observed) != len(rows):
        raise ContractError('diagnostic coverage differs')
    if any(type(item) is not dict or 'free_generation' not in item for item in observed):
        raise ContractError('diagnostic prediction absent')
    summary = check_prediction_examples(rows, 'train', GENERATION_CAP,
        json_bytes({'predictions':[item['free_generation'] for item in observed]}))
    predictions = []
    for item, row in zip(observed, rows):
        exact_keys(item, {'example_identity','task','positions','first_answer','gold_prefix_eos',
            'free_generation','first_divergent_position','termination',
            'teacher_forced_all_targets_correct'}, 'overfit diagnostic row')
        if (item['example_identity'] != row['example_identity'] or item['task'] != row['task']
                or item['free_generation']['example_identity'] != row['example_identity']):
            raise ContractError('diagnostic ordered row identity differs')
        p = len(row['prompt'].encode('ascii'))
        targets = [*row['answer'].encode('ascii'), EOS]
        positions = item['positions']
        if type(positions) is not list or len(positions) != len(targets):
            raise ContractError('diagnostic target coverage differs')
        for offset, (position, target) in enumerate(zip(positions, targets)):
            prediction(position, target, {'target_position','logit_position'})
            if (type(position['target_position']) is not int or position['target_position'] != p+1+offset
                    or type(position['logit_position']) is not int or position['logit_position'] != p+offset):
                raise ContractError('diagnostic target/logit alignment differs')
        for label, offset in (('first_answer',0), ('gold_prefix_eos',len(targets)-1)):
            boundary = item[label]
            prediction(boundary, targets[offset], {'logit_position','logits_hex',
                'prefix_logits_hex','prefix_max_abs_difference_hex'})
            if type(boundary['logit_position']) is not int or boundary['logit_position'] != p+offset:
                raise ContractError('diagnostic boundary position differs')
            if any(boundary[key] != positions[offset][key] for key in prediction_keys):
                raise ContractError('diagnostic boundary/position summary differs')
        generated = item['free_generation']['generated_tokens']
        expected = item['free_generation']['expected_tokens']
        if EOS in generated[:-1] or (generated[-1] != EOS and len(generated) != GENERATION_CAP):
            raise ContractError('diagnostic generation termination is not native greedy')
        first = next((i for i in range(max(len(expected),len(generated)))
                      if i >= len(expected) or i >= len(generated) or expected[i] != generated[i]), None)
        if (item['first_divergent_position'] != first
                or (first is not None and type(item['first_divergent_position']) is not int)
                or item['termination'] != ('eos' if generated and generated[-1] == EOS else 'cap')
                or type(item['teacher_forced_all_targets_correct']) is not bool
                or item['teacher_forced_all_targets_correct'] != all(v['target'] == v['argmax'] for v in positions)
                or not generated or generated[0] != item['first_answer']['argmax']):
            raise ContractError('diagnostic generated/teacher-forced summary differs')
        predictions.append(item['free_generation'])
    if json_bytes(summary) != json_bytes(record['raw_score']):
        raise ContractError('diagnostic raw score differs')
    return summary


def verify_checkpoint(payload, metadata, seed, subset, admission_identity, request_identity):
    """Native safe tensor/progress validation, without forwards or update replay.

    The caller must first bind these bytes to the independently retained outer
    commitment. Initial loss and history are structure-checked, not recomputed.
    """
    from trinite.contracts import exact_keys, require_identity
    from music_learning_v1.workflow import require_native_admission
    from trinite.model import Decoder
    from trinite.training import TrainingState, create_optimizer, model_identity, require_environment
    from trinite.comparison_checkpoint import progress, restore_tensors
    if type(payload) is not bytes or len(payload) > 32 * 1024 * 1024:
        raise ContractError('overfit checkpoint payload type/size differs')
    exact_keys(metadata, {'schema','subset','request_identity','plan','dataset_identity',
        'admission_identity','initialization_identity','initial_train_loss_hex','model_identity',
        'step','cursor','target_tokens','history','validation'}, 'overfit checkpoint')
    if len(json_bytes(metadata)) > 1024 * 1024:
        raise ContractError('overfit checkpoint metadata exceeds budget')
    require_identity(request_identity)
    require_identity(metadata['model_identity'])
    raw, _ = require_native_admission(admission_identity)
    require_environment()
    config = plan_for(seed)
    train = encoded(subset_rows(subset))
    model = Decoder(config.model, lane='dense', seed=seed)
    initial_identity = model_identity(model)
    expected = {'schema':'trinite.music-overfit-checkpoint.v1','subset':subset,
        'request_identity':request_identity,'plan':config.to_dict(),
        'dataset_identity':identity(raw),'admission_identity':admission_identity,
        'initialization_identity':initial_identity}
    if any(json_bytes(metadata[key]) != json_bytes(value) for key,value in expected.items()):
        raise ContractError('overfit checkpoint context differs')
    state = TrainingState(config, 'dense', model, create_optimizer(model, config),
        {'train':train,'validation':train}, identity(raw), admission_identity,
        initial_identity, metadata['initial_train_loss_hex'])
    state.music_subset = subset
    state.step, state.cursor, state.target_tokens = metadata['step'], metadata['cursor'], metadata['target_tokens']
    state.history, state.validation = metadata['history'], metadata['validation']
    progress(state)
    if state.step == 0 and metadata['model_identity'] != initial_identity:
        raise ContractError('overfit zero-step checkpoint differs from seeded initialization')
    return restore_tensors(state, payload, metadata['model_identity'])
