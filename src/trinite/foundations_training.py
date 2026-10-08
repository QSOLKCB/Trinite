"""Frozen foundations inputs and state; authoritative native numerical loop."""
from dataclasses import replace
from pathlib import Path

from .contracts import ContractError, identity, json_bytes, parse_json
from .foundations_data import dataset, audit_bytes, source_receipt as data_sources
from .foundations_plan import FoundationsPlan
from .foundations_oracle import WORKLOADS
from .comparison_training import sources as comparison_sources, LANES, SEEDS
from .model import Decoder
from .tokenizer import ByteTokenizer
from .training import TrainingState, create_optimizer, evaluate, model_identity, require_environment

PROTOCOL_IDENTITY='sha256:16063ebcb2f918b6c6bee9074f5949967c8fca4afd20a4245bf65e12789ca34d'
PROTOCOL_COMMIT='d85743aae0e732006c9a44d25c3455a4eee6cee0'
RUNNER_IDENTITY='sha256:55e60166b60dca660059fe32d727e556f897a42a63d4de749d4912d3f17665be'


def protocol():
    raw=(Path(__file__).resolve().parent/'foundations-protocol.json').read_bytes()
    if identity(raw)!=PROTOCOL_IDENTITY:raise ContractError('foundations protocol changed; freeze a new version')
    return parse_json(raw,canonical=True)


def sources():
    root=Path(__file__).resolve().parent
    result={**comparison_sources(),**data_sources(),**{n:identity((root/n).read_bytes()) for n in
        ('foundations_plan.py','foundations_training.py','foundations_checkpoint.py','foundations.py')}}
    result['scripts/run_foundations.py']=RUNNER_IDENTITY
    return result


def state_for(workload,lane,seed,config=None):
    if workload not in WORKLOADS or lane not in LANES or type(seed) is not int or seed not in SEEDS:
        raise ContractError('unknown fixed foundations cell')
    require_environment()
    plan=config or replace(FoundationsPlan.from_dict(protocol()['training']),seed=seed)
    if not isinstance(plan,FoundationsPlan) or plan.seed!=seed:raise ContractError('foundations plan/seed mismatch')
    raw,manifest=dataset(workload);audit_bytes(raw,manifest)
    examples=parse_json(raw,canonical=True)['examples'];data={s:[] for s in ('train','validation')}
    tokenizer=ByteTokenizer(plan.model.context_length)
    for item in examples:
        if item['split'] not in data:continue
        e=tokenizer.encode(item['text'],answer_start=len(item['prompt'].encode()))
        data[item['split']].append({'identity':item['example_identity'],'ids':e.input_ids,
            'mask':e.attention_mask,'loss':e.loss_mask})
    data['train'].sort(key=lambda e:identity(json_bytes({'seed':seed,'item':e['identity']})))
    targets=sum(sum(data['train'][(i*plan.batch_size+j)%len(data['train'])]['loss'])
        for i in range(plan.steps) for j in range(plan.batch_size))
    if targets>plan.max_target_tokens:raise ContractError('foundations schedule exceeds target budget')
    model=Decoder(plan.model,lane=lane,seed=seed)
    return TrainingState(plan,lane,model,create_optimizer(model,plan),data,identity(raw),identity(manifest),
        model_identity(model),evaluate(model,data['train'],plan.batch_size))
