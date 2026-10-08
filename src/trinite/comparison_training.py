"""Admitted comparison state construction; shared native numerical update loop."""
from dataclasses import replace
from pathlib import Path

import torch

from .contracts import ContractError, identity, json_bytes, parse_json
from .data import build_dataset, audit_bytes
from .comparison_data import dataset as qec_dataset, audit_bytes as audit_qec
from .qec_source import verify_pin
from .model import Decoder
from .tokenizer import ByteTokenizer
from .training import (RunConfig, TrainingState, require_environment, source_receipt,
                       model_identity, evaluate, create_optimizer)

PROTOCOL_IDENTITY = 'sha256:09acdbff49e6c458974b82de18345e544d0f2c727e350db8f4e6d4f4e8344ca0'
PROTOCOL_COMMIT = '3bc49f478ce0997b0ab46d36c79b9eb7ba7319cd'
LANES = ('dense', 'ternary', 'four-state')
WORKLOADS = ('formal', 'qutrit', 'ququart')
SEEDS = (0, 1, 2)
RUNNER_IDENTITY = 'sha256:7405704e7b0a2ff7a2c4080315aea5c872192f3371a14fdf6db4f4b2397f950c'


def protocol():
    raw = (Path(__file__).resolve().parent/'comparison-protocol.json').read_bytes()
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError('comparison protocol changed; freeze a new version')
    return parse_json(raw, canonical=True)


def sources():
    root = Path(__file__).resolve().parent
    result = {**source_receipt(), **{n: identity((root/n).read_bytes()) for n in (
        'comparison.py', 'comparison_training.py', 'comparison_checkpoint.py', 'comparison_data.py', 'qec_source.py',
        'comparison-protocol.json', 'geometry.py', '_geo_reference.py', 'geometry-pin.json')}}
    result.update({'qec/'+n: v for n, v in verify_pin()['files'].items()})
    # Local packaging wrappers are also executable source.
    for p in sorted((root/'_qec').rglob('__init__.py')):
        result[str(p.relative_to(root))] = identity(p.read_bytes())
    result['scripts/run_comparison.py'] = RUNNER_IDENTITY
    return result


def corpus(workload):
    if workload == 'formal':
        data, manifest = build_dataset()
        raw, evidence = json_bytes(data), json_bytes(manifest)
        audit_bytes(raw, evidence)
        return raw, evidence
    if workload not in WORKLOADS:
        raise ContractError('unknown comparison workload')
    raw, evidence = qec_dataset(workload)
    audit_qec(raw, evidence)
    return raw, evidence


def state_for(workload, lane, seed, config=None):
    if workload not in WORKLOADS or lane not in LANES or type(seed) is not int or seed not in SEEDS:
        raise ContractError('unknown fixed comparison cell')
    require_environment()
    config = config or replace(RunConfig.from_dict(protocol()['training']), seed=seed)
    if config.seed != seed:
        raise ContractError('comparison seed differs from plan')
    raw, manifest = corpus(workload)
    data = parse_json(raw, canonical=True)
    selected = {s: [] for s in ('train', 'validation')}
    tokenizer = ByteTokenizer(config.model.context_length)
    for example in data['examples']:
        if example['split'] not in selected:
            continue
        encoded = tokenizer.encode(example['text'], answer_start=len(example['prompt'].encode()))
        selected[example['split']].append({'identity': example['example_identity'],
            'ids': encoded.input_ids, 'mask': encoded.attention_mask, 'loss': encoded.loss_mask})
    selected['train'].sort(key=lambda e: identity(json_bytes({'seed': seed, 'item': e['identity']})))
    if not selected['train'] or not selected['validation']:
        raise ContractError('empty comparison split')
    scheduled_targets = sum(sum(selected['train'][(i*config.batch_size+j) % len(selected['train'])]['loss'])
                            for i in range(config.steps) for j in range(config.batch_size))
    if scheduled_targets > config.max_target_tokens:
        raise ContractError('comparison target schedule exceeds plan')
    model = Decoder(config.model, lane=lane, seed=seed)
    optimizer = create_optimizer(model, config)
    state = TrainingState(config, lane, model, optimizer, selected, identity(raw), identity(manifest),
                          model_identity(model), evaluate(model, selected['train'], config.batch_size))
    return state


