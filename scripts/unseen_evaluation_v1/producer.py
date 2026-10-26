"""Read-only producer custody and inference-only import; never constructs an optimizer.

This is a new structural importer, not the old audit or a numerical replay.
The original source/environment guard remains authoritative and unchanged.
"""
from pathlib import Path

import torch

from trinite.budget_plan import BudgetPlan
from trinite.budget_training import sources as budget_sources, PROTOCOL_IDENTITY
from trinite.comparison_checkpoint import load, _slot, _hex
from trinite.contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json, require_identity
from trinite.model import Decoder
from trinite.observation import verify_observation
from trinite.tokenizer import ByteTokenizer
from trinite.training import model_identity, require_environment
from native_diagnostic_v1.request import read, validate, corpus, training_examples, MILESTONES, PROTOCOL
from native_diagnostic_v1.mechanics import validate_preflight
from native_diagnostic_v1.measurements import validate_report
from native_diagnostic_v1.workflow import validate_resources
from .packet import inventory, unchanged


FINAL_STEP = 4096
BINDING_KEYS = {'producer_request_identity', 'producer_completion_identity', 'model_identity',
                'checkpoint_metadata_identity', 'checkpoint_tensor_identity',
                'producer_source', 'producer_environment', 'seed'}

def _schedule(record, plan):
    """Reconstruct tokenization and declared visit order, not updates or losses."""
    tokenizer = ByteTokenizer(plan.model.context_length)
    rows = training_examples()
    rows.sort(key=lambda r: identity(json_bytes({'seed': plan.seed, 'item': r['example_identity']})))
    encoded = [(r['example_identity'], sum(tokenizer.encode(r['text'],
                answer_start=len(r['prompt'].encode('ascii'))).loss_mask)) for r in rows]
    step = record['step']
    integer(step, 'producer step', 0, plan.steps)
    if type(record['cursor']) is not int or record['cursor'] != step * plan.batch_size:
        raise ContractError('producer cursor mismatch')
    if type(record['history']) is not list or len(record['history']) != step:
        raise ContractError('producer history length mismatch')
    targets = 0
    for index, entry in enumerate(record['history']):
        exact_keys(entry, {'step', 'loss_hex', 'gradient_norm_hex', 'target_tokens', 'examples'}, 'producer history')
        batch = [encoded[(index * plan.batch_size + j) % len(encoded)] for j in range(plan.batch_size)]
        count = sum(n for _, n in batch)
        expected = {'step': index + 1, 'target_tokens': count, 'examples': [key for key, _ in batch]}
        if any(json_bytes(entry[k]) != json_bytes(v) for k, v in expected.items()):
            raise ContractError('producer scheduled visits mismatch')
        _hex(entry['loss_hex']); _hex(entry['gradient_norm_hex'])
        targets += count
    if type(record['target_tokens']) is not int or record['target_tokens'] != targets or targets > plan.max_target_tokens:
        raise ContractError('producer target count mismatch')
    expected_steps = [i for i in range(1, step + 1) if i % plan.validation_every == 0 or i == plan.steps]
    diagnostics = record['train_diagnostics']
    if type(diagnostics) is not list or len(diagnostics) != len(expected_steps):
        raise ContractError('producer train diagnostic schedule mismatch')
    for entry, expected in zip(diagnostics, expected_steps):
        exact_keys(entry, {'step', 'loss_hex'}, 'producer train diagnostic')
        if type(entry['step']) is not int or entry['step'] != expected:
            raise ContractError('producer train diagnostic step mismatch')
        _hex(entry['loss_hex'])
    _hex(record['initial_train_loss_hex'])


def _checkpoint(payload, metadata, request, request_identity, step):
    """Delegate safe decoding/slot rules; import only named model parameters."""
    record = parse_json(metadata, canonical=True)
    plan = BudgetPlan.from_dict(request['plan'])
    expected = {'schema': 'trinite.budget-checkpoint.v1', 'data_view': 'train-only.v1',
        'protocol_identity': PROTOCOL_IDENTITY, 'workload': 'arithmetic', 'seed': request['seed'],
        'candidate': 'extended', 'lane': 'dense', 'request_identity': request_identity,
        'plan': request['plan'], 'source': budget_sources(), 'environment': require_environment(),
        'dataset_identity': request['dataset_identity'], 'manifest_identity': request['manifest_identity']}
    exact_keys(record, set(expected) | {'initialization_identity', 'initial_train_loss_hex',
        'model_identity', 'step', 'cursor', 'target_tokens', 'history', 'train_diagnostics'}, 'producer checkpoint')
    if any(json_bytes(record[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('producer checkpoint source/environment/context mismatch')
    if type(record['step']) is not int or record['step'] != step:
        raise ContractError('producer checkpoint milestone mismatch')
    _schedule(record, plan)
    model = Decoder(plan.model, lane='dense', seed=plan.seed)
    initial = model_identity(model)
    if record['initialization_identity'] != initial:
        raise ContractError('producer seeded initialization mismatch')
    require_identity(record['model_identity'])
    try:
        tensors = load(payload)
    except Exception as error:
        raise ContractError('invalid producer safetensors') from error
    names = {'model.' + name for name, _ in model.named_parameters()}
    if step:
        names |= {'optimizer.' + name + '.' + slot for name, _ in model.named_parameters()
                  for slot in ('step', 'exp_avg', 'exp_avg_sq')}
    if set(tensors) != names:
        raise ContractError('producer tensor inventory mismatch')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = tensors['model.' + name]
            if (value.device.type != 'cpu' or value.layout != torch.strided or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())):
                raise ContractError('invalid producer model tensor')
            if step:
                for slot in ('step', 'exp_avg', 'exp_avg_sq'):
                    _slot(tensors['optimizer.' + name + '.' + slot], parameter, slot, step)
            parameter.copy_(value)
    model.validate_state()
    if model_identity(model) != record['model_identity'] or (step == 0 and record['model_identity'] != initial):
        raise ContractError('producer checkpoint model identity mismatch')
    return model, record


def _inspect(root, request_identity, completion_identity, expected_model_identity):
    root = Path(root)
    for value in (request_identity, completion_identity, expected_model_identity):
        require_identity(value)
    snapshot = inventory(root)
    total = 0
    def retained(name):
        nonlocal total
        raw = read(root / name)
        total += len(raw)
        if total > PROTOCOL['max_retained_payload_bytes']:
            raise ContractError('producer retained payload exceeds bound')
        return raw
    request_raw = retained('request.json')
    request = validate(request_raw, request_identity)
    completion_raw = retained('completion.json')
    if identity(completion_raw) != completion_identity:
        raise ContractError('producer completion identity mismatch')
    completion = parse_json(completion_raw, canonical=True)
    exact_keys(completion, {'schema', 'request_identity', 'artifacts', 'manifest_identity', 'verification_identity'}, 'producer completion')
    if completion['schema'] != 'trinite.native-diagnostic-completion.v1' or completion['request_identity'] != request_identity:
        raise ContractError('producer completion context mismatch')
    names = {'preflight.json', 'report.json', 'resources.json'} | {
        f'{kind}-{step}.{suffix}' for step in MILESTONES
        for kind, suffix in (('diagnostics', 'json'), ('tensors', 'safetensors'), ('metadata', 'json'))}
    exact_keys(completion['artifacts'], names, 'producer artifacts')
    bound = {}
    for name, wanted in completion['artifacts'].items():
        require_identity(wanted)
        raw = retained(name)
        if identity(raw) != wanted:
            raise ContractError('producer artifact identity mismatch: ' + name)
        bound[name] = raw
    verification_raw = retained('verification.json')
    manifest_raw = read(root / 'provenance/manifest.json')
    if (identity(verification_raw) != completion['verification_identity']
            or identity(manifest_raw) != completion['manifest_identity']):
        raise ContractError('producer custody identity mismatch')
    verified = verify_observation(root / 'provenance')
    if (not verified['integrity_verified'] or verified['errors'] or verified['known_missing_artifacts']
            or json_bytes(verified) != verification_raw):
        raise ContractError('producer custody verification failed')
    indexes = []
    manifest = parse_json(manifest_raw, canonical=True)
    for item in manifest['core']['artifacts']:
        key = item['content_identity']; require_identity(key)
        raw = read(root / 'provenance/artifacts/sha256' / key[7:])
        try:
            value = parse_json(raw, canonical=True)
        except ContractError:
            continue
        if type(value) is dict and value.get('schema') == 'trinite.observer-index.v1':
            indexes.append(value['artifacts'])
    data_raw, data_manifest = corpus()
    expected_names = {**completion['artifacts'], 'request.json': request_identity,
                      'dataset.json': identity(data_raw), 'dataset-manifest.json': identity(data_manifest)}
    if indexes != [expected_names]:
        raise ContractError('producer custody name index mismatch')
    examples = training_examples()
    first = final = final_record = initial_loss = None
    for step in MILESTONES:
        model, record = _checkpoint(bound[f'tensors-{step}.safetensors'],
                                    bound[f'metadata-{step}.json'], request, request_identity, step)
        diagnostic = parse_json(bound[f'diagnostics-{step}.json'], canonical=True)
        validate_report(diagnostic, examples, PROTOCOL['generation_cap'])
        if (diagnostic['model_identity'] != record['model_identity']
                or any(r['context_length'] != model.config.context_length for r in diagnostic['rows'])):
            raise ContractError('producer diagnostic model/context mismatch')
        if first is None:
            first = record['model_identity']
            initial_loss = record['initial_train_loss_hex']
        elif record['initial_train_loss_hex'] != initial_loss:
            raise ContractError('producer initial-loss continuity mismatch')
        final, final_record = model, record
    if model_identity(final) != expected_model_identity:
        raise ContractError('externally bound producer final model differs')
    expected_report = {'schema': 'trinite.native-diagnostic-report.v1', 'outcome': 'completed',
        'request_identity': request_identity, 'seed': request['seed'], 'milestones': list(MILESTONES),
        'initialization_identity': first, 'final_model_identity': expected_model_identity,
        'steps': final_record['step'], 'target_tokens': final_record['target_tokens'],
        'parameter_count': final.config.parameter_count, 'cache_mode': 'unsupported',
        'held_out_scored': False, 'learning_adequate': False, 'phase5_ready': False, 'release_ready': False,
        'verification_scope': 'custody, request binding and safe checkpoint structure; not full optimizer replay'}
    if bound['report.json'] != json_bytes(expected_report):
        raise ContractError('producer report context mismatch')
    validate_preflight(parse_json(bound['preflight.json'], canonical=True))
    validate_resources(parse_json(bound['resources.json'], canonical=True))
    validate(request_raw, request_identity)
    binding = {'producer_request_identity': request_identity, 'producer_completion_identity': completion_identity,
        'model_identity': expected_model_identity, 'checkpoint_metadata_identity': identity(bound[f'metadata-{FINAL_STEP}.json']),
        'checkpoint_tensor_identity': identity(bound[f'tensors-{FINAL_STEP}.safetensors']),
        'producer_source': request['source'], 'producer_environment': request['environment'], 'seed': request['seed']}
    unchanged(root, snapshot)
    return binding, final


def inspect_packet(root, request_identity, completion_identity, expected_model_identity):
    """Verify custody/context/tensor structure only; no inference/update replay."""
    return _inspect(root, request_identity, completion_identity, expected_model_identity)[0]


def load_model(root, binding):
    """Recheck external binding, then return the final CPU model without an optimizer."""
    exact_keys(binding, BINDING_KEYS, 'producer binding')
    actual, model = _inspect(root, binding['producer_request_identity'],
                             binding['producer_completion_identity'], binding['model_identity'])
    if json_bytes(actual) != json_bytes(binding):
        raise ContractError('producer frozen binding mismatch')
    return model
