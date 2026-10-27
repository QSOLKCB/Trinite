"""Compose existing numerical operations; no updater, loss or tensor codec fork."""
from dataclasses import dataclass
import time

from trinite.budget_plan import BudgetPlan
from trinite.contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json
from trinite.tokenizer import ByteTokenizer, EOS
from trinite.training import TrainingState, create_optimizer, evaluate, model_identity, require_environment
from trinite.training import run_steps as native_run_steps
from trinite.comparison_checkpoint import tensor_payload, restore_tensors, progress
from trinite.foundations import score_examples, check_prediction_examples, metrics
from .adapter import ExposureDecoder, addressing, generation_cap
from .data import dataset, audit_bytes, examples, schedule
from .protocol import ARMS, MODES, PROTOCOL, offset_for


@dataclass(frozen=True)
class ExposurePlan(BudgetPlan):
    schema: str = 'trinite.position-exposure-plan.v1'
    steps: int = 4000
    validation_every: int = 1000
    max_target_tokens: int = 121600
    order: str = 'position-exposure-family-cycle-offset.v1'

    def __post_init__(self):
        if self.schema != 'trinite.position-exposure-plan.v1' or self.order != PROTOCOL['schedule']:
            raise ContractError('unknown position exposure plan')
        common = self.to_dict()
        common.update(schema='trinite.budget-training-plan.v1',order='seeded-hash-rank-cycle.v1')
        BudgetPlan.from_dict(common)


def plan_for(seed):
    from .request import cell_name
    cell_name(ARMS[0],seed)
    return ExposurePlan(seed=seed)


def encoded(rows, plan):
    tok = ByteTokenizer(plan.model.context_length)
    result = []
    for row in rows:
        e = tok.encode(row['text'],answer_start=len(row['prompt']))
        result.append({'identity':row['example_identity'],'ids':e.input_ids,
                       'mask':e.attention_mask,'loss':e.loss_mask})
    return result


def state_for(arm, seed, plan=None):
    from .request import cell_name
    cell_name(arm,seed); require_environment()
    config = plan_for(seed) if plan is None else plan
    if type(config) is not ExposurePlan or config.seed != seed:
        raise ContractError('study plan type/seed differs')
    raw, admitted = dataset(); audit_bytes(raw,admitted)
    train = encoded(schedule(arm),config)
    validation = encoded(examples('train'),config)  # unique TRAIN only
    targets = sum(sum(train[(i*config.batch_size+j)%len(train)]['loss'])
                  for i in range(config.steps) for j in range(config.batch_size))
    if targets > config.max_target_tokens:
        raise ContractError('study schedule exceeds target budget')
    model = ExposureDecoder(config.model,arm=arm,seed=seed)
    return TrainingState(config,'dense',model,create_optimizer(model,config),
        {'train':train,'validation':validation},identity(raw),identity(admitted),
        model_identity(model),evaluate(model,validation,config.batch_size))



def offset_progress(state, arm):
    cycle = PROTOCOL['cycle_visits']
    return {'schema': 'trinite.position-exposure-cursor.v1',
        'policy': addressing(arm), 'cursor': state.cursor,
        'cycle_index': state.cursor // cycle, 'cycle_cursor': state.cursor % cycle,
        'next_offset': offset_for(arm, state.cursor)}


def run_steps(state, *, stop_after=None):
    """Only orchestration: native updater owns loss, clipping, AdamW and history."""
    if type(state.config) is not ExposurePlan or type(state.model) is not ExposureDecoder:
        raise ContractError('exposure updater requires its explicit plan/model')
    state.model.require_idle()
    progress(state)
    end = state.config.steps if stop_after is None else integer(
        stop_after, 'stop step', state.step, state.config.steps)
    cycle = PROTOCOL['cycle_visits']
    if cycle % state.config.batch_size:
        raise ContractError('exposure batch crosses cycle boundary')
    deadline = time.monotonic() + state.config.max_seconds
    while state.step < end:
        if time.monotonic() > deadline:
            raise ContractError('exposure invocation exceeded its time budget')
        boundary = (state.cursor // cycle + 1) * cycle
        steps = (boundary - state.cursor) // state.config.batch_size
        if steps <= 0:
            raise ContractError('invalid exposure cycle cursor')
        with state.model.training_offset(offset_for(state.model.arm, state.cursor)):
            native_run_steps(state, stop_after=min(end, state.step + steps))
        if time.monotonic() > deadline:
            raise ContractError('exposure invocation exceeded its time budget')


def behavior_identity(model):
    return identity(json_bytes({'parameters':model_identity(model),'addressing':model.addressing}))


def snapshot(state, arm, seed, request_identity):
    from .request import cell_name
    cell_name(arm,seed)
    if state.model.addressing != addressing(arm):
        raise ContractError('checkpoint position mode differs')
    state.model.require_idle()
    payload = tensor_payload(state)
    metadata = {'schema':'trinite.position-exposure-checkpoint.v1','arm':arm,'seed':seed,
        'addressing':state.model.addressing,'behavior_identity':behavior_identity(state.model),
        'request_identity':request_identity,'plan':state.config.to_dict(),
        'dataset_identity':state.dataset_identity,'admission_identity':state.manifest_identity,
        'initialization_identity':state.initialization_identity,
        'initial_train_loss_hex':state.initial_train_loss,'model_identity':model_identity(state.model),
        'step':state.step,'cursor':state.cursor,'target_tokens':state.target_tokens,
        'history':state.history,'validation':state.validation,
        'offset_progress': offset_progress(state, arm)}
    return payload,json_bytes(metadata)


def restore(payload, metadata, arm, seed, request_identity, plan=None):
    if type(metadata) is not bytes or len(metadata)>16*1024*1024:
        raise ContractError('study checkpoint metadata bound')
    r = parse_json(metadata,canonical=True)
    state = state_for(arm,seed,plan)
    expected = {'schema':'trinite.position-exposure-checkpoint.v1','arm':arm,'seed':seed,
        'addressing':addressing(arm),'request_identity':request_identity,
        'plan':state.config.to_dict(),'dataset_identity':state.dataset_identity,
        'admission_identity':state.manifest_identity,'initialization_identity':state.initialization_identity,
        'initial_train_loss_hex':state.initial_train_loss}
    exact_keys(r,set(expected)|{'model_identity','behavior_identity','step','cursor','target_tokens','history','validation','offset_progress'},'study checkpoint')
    if any(json_bytes(r[k]) != json_bytes(v) for k,v in expected.items()):
        raise ContractError('study checkpoint context mismatch')
    state.step,state.cursor,state.target_tokens = r['step'],r['cursor'],r['target_tokens']
    state.history,state.validation = r['history'],r['validation']
    progress(state)
    if json_bytes(r['offset_progress']) != json_bytes(offset_progress(state, arm)):
        raise ContractError('checkpoint offset schedule/cursor differs')
    state = restore_tensors(state,payload,r['model_identity'])
    if behavior_identity(state.model) != r['behavior_identity']:
        raise ContractError('study checkpoint behavior mismatch')
    return state


def view(arm, selector):
    unique = examples(arm)
    return (unique,'train') if selector == 'train' else (unique+examples(selector),'test')


def scoring_groups(arm, selector):
    rows,split = view(arm,selector)
    # Native scorer's baseline is training-only within this explicit width view.
    # Include all training widths so width3/4 groups also have a training majority.
    for width in sorted({r['width'] for r in rows if r['split']==split}):
        selected = [r for r in rows if r['split']=='train' or r['width']==width]
        if split == 'train': selected = [r for r in selected if r['width']==width]
        yield width,selected,split


def score(state, arm, selector, on_prediction=None):
    if state.model.addressing != addressing(arm):
        raise ContractError('score position mode differs')
    state.model.require_idle()
    predictions = []
    for width,rows,split in scoring_groups(arm,selector):
        # Explicit new semantics: prompt-derived width cap; unchanged native greedy.
        training = [r for r in rows if r['split']=='train']
        for item in (r for r in rows if r['split']==split):
            # Keep the native baseline view and score exactly one selected row.
            # Use one selected item + original training-only majority pool.
            if split == 'test':
                _,out = score_examples(state.model,training+[item],split,generation_cap(item['prompt']))
            else:
                # Native scorer accepts only train/test. A single-row train view
                # would silently change the majority baseline; use a test view
                # with the complete training pool, then restore train identity.
                _,out = score_examples(state.model,training+[{**item,'split':'test'}],
                                       'test',generation_cap(item['prompt']))
                out[0]['split'] = 'train'
            predictions.extend(out)
            if on_prediction is not None: on_prediction(out[0])
    return {'schema':PROTOCOL['scoring']['schema'],
        'model_identity':model_identity(state.model),'addressing':state.model.addressing,
        'summary':metrics(predictions),'predictions':predictions}


def check_score(record, arm, selector, expected_model):
    exact_keys(record,{'schema','model_identity','addressing','summary','predictions'},'study scores')
    if (record['schema'] != PROTOCOL['scoring']['schema'] or
            record['model_identity'] != expected_model or record['addressing'] != addressing(arm)):
        raise ContractError('study scoring checkpoint/addressing differs')
    if type(record['predictions']) is not list:
        raise ContractError('study predictions require list')
    retained = record['predictions']; seen = set()
    for width,rows,split in scoring_groups(arm,selector):
        selected = {r['example_identity']:r for r in rows if r['split']==split}
        group = [p for p in retained if p.get('example_identity') in selected]
        check_prediction_examples(rows,split,width+2,json_bytes({'predictions':group}))
        for p in group:
            out = p['generated_tokens']; cap = generation_cap(selected[p['example_identity']]['prompt'])
            if (EOS in out and (out[-1] != EOS or EOS in out[:-1])) or (EOS not in out and len(out) != cap):
                raise ContractError('prediction violates actual EOS stop or prompt-derived cap')
            seen.add(p['example_identity'])
    if len(seen) != len(retained): raise ContractError('unknown/duplicate study predictions')
    actual = metrics(retained)
    if json_bytes(actual) != json_bytes(record['summary']):
        raise ContractError('study summary differs from raw predictions')
    return actual
