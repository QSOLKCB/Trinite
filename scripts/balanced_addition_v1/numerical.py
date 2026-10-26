"""Compose admitted rows with the unchanged native updater, scorer and tensor reader."""
from dataclasses import dataclass

from trinite.budget_plan import BudgetPlan
from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json
from trinite.model import Decoder
from trinite.tokenizer import ByteTokenizer
from trinite.training import TrainingState, create_optimizer, evaluate, model_identity, require_environment
from trinite.comparison_checkpoint import tensor_payload, restore_tensors, progress
from trinite.foundations import score_examples, check_prediction_examples
from .data import dataset, audit_bytes, examples, schedule
from .protocol import STEPS, BATCH_SIZE, MILESTONES, PROTOCOL
from .request import cell_name


@dataclass(frozen=True)
class AdditionPlan(BudgetPlan):
    schema: str = 'trinite.balanced-addition-plan.v1'
    steps: int = 4000
    validation_every: int = 1000
    max_target_tokens: int = 76800
    order: str = 'length-matched-balanced-addition-slot-cycle.v1'

    def __post_init__(self):
        if self.schema != 'trinite.balanced-addition-plan.v1' or self.order != 'length-matched-balanced-addition-slot-cycle.v1':
            raise ContractError('unknown addition plan')
        common = self.to_dict()
        common.update(schema='trinite.budget-training-plan.v1', order='seeded-hash-rank-cycle.v1')
        BudgetPlan.from_dict(common)


def plan_for(seed):
    cell_name('control', seed)
    return AdditionPlan(seed=seed)


def encoded(rows, plan):
    tok = ByteTokenizer(plan.model.context_length)
    result = []
    for row in rows:
        e = tok.encode(row['text'], answer_start=len(row['prompt'].encode('ascii')))
        result.append({'identity': row['example_identity'], 'ids': e.input_ids,
                       'mask': e.attention_mask, 'loss': e.loss_mask})
    return result


def state_for(arm, seed, plan=None):
    cell_name(arm, seed); require_environment()
    config = plan_for(seed) if plan is None else plan
    if type(config) is not AdditionPlan or config.seed != seed:
        raise ContractError('addition plan type/seed differs')
    raw, admitted = dataset(); audit_bytes(raw, admitted)
    unique = examples(arm)
    visits = schedule(arm)
    train = encoded(visits, config)
    validation = encoded(unique, config)  # unique TRAIN rows only, never held-out data
    targets = sum(sum(train[(i*config.batch_size+j) % len(train)]['loss'])
                  for i in range(config.steps) for j in range(config.batch_size))
    if targets > config.max_target_tokens:
        raise ContractError('addition schedule exceeds target budget')
    model = Decoder(config.model, lane='dense', seed=seed)
    return TrainingState(config, 'dense', model, create_optimizer(model, config),
        {'train': train, 'validation': validation}, identity(raw), identity(admitted),
        model_identity(model), evaluate(model, validation, config.batch_size))


def snapshot(state, arm, seed, request_identity):
    cell_name(arm, seed)
    payload = tensor_payload(state)
    metadata = {'schema': 'trinite.balanced-addition-checkpoint.v1', 'arm': arm,
        'seed': seed, 'request_identity': request_identity, 'plan': state.config.to_dict(),
        'dataset_identity': state.dataset_identity, 'admission_identity': state.manifest_identity,
        'initialization_identity': state.initialization_identity,
        'initial_train_loss_hex': state.initial_train_loss, 'model_identity': model_identity(state.model),
        'step': state.step, 'cursor': state.cursor, 'target_tokens': state.target_tokens,
        'history': state.history, 'validation': state.validation}
    return payload, json_bytes(metadata)


def restore(payload, metadata, arm, seed, request_identity, plan=None):
    if type(metadata) is not bytes or len(metadata) > 16*1024*1024:
        raise ContractError('addition checkpoint metadata bound')
    r = parse_json(metadata, canonical=True)
    state = state_for(arm, seed, plan)
    expected = {'schema': 'trinite.balanced-addition-checkpoint.v1', 'arm': arm,
        'seed': seed, 'request_identity': request_identity, 'plan': state.config.to_dict(),
        'dataset_identity': state.dataset_identity, 'admission_identity': state.manifest_identity,
        'initialization_identity': state.initialization_identity,
        'initial_train_loss_hex': state.initial_train_loss}
    exact_keys(r, set(expected)|{'model_identity','step','cursor','target_tokens','history','validation'}, 'addition checkpoint')
    if any(json_bytes(r[k]) != json_bytes(v) for k,v in expected.items()):
        raise ContractError('addition checkpoint context mismatch')
    state.step, state.cursor, state.target_tokens = r['step'], r['cursor'], r['target_tokens']
    state.history, state.validation = r['history'], r['validation']
    progress(state)
    return restore_tensors(state, payload, r['model_identity'])


def view(arm, selector):
    unique = examples(arm)
    if selector == 'train':
        return unique, 'train'
    return unique + examples(selector), 'test'


def score(state, arm, selector):
    rows, split = view(arm, selector)
    summary, predictions = score_examples(state.model, rows, split, PROTOCOL['generation_cap'])
    return {'model_identity': model_identity(state.model), 'summary': summary, 'predictions': predictions}


def check_score(record, arm, selector, expected_model):
    exact_keys(record, {'model_identity','summary','predictions'}, 'addition scores')
    if record['model_identity'] != expected_model:
        raise ContractError('addition scoring checkpoint identity differs')
    rows, split = view(arm, selector)
    actual = check_prediction_examples(rows, split, PROTOCOL['generation_cap'],
                                      json_bytes({'predictions': record['predictions']}))
    if json_bytes(actual) != json_bytes(record['summary']):
        raise ContractError('addition retained summary differs from predictions')
    return actual
