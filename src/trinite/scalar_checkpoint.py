"""Matched larger checkpoints sharing the authoritative named-tensor reader."""
from .contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from .comparison_checkpoint import tensor_payload, progress, restore_tensors
from .training import model_identity, require_environment
from .scalar_plan import ScalarPlan
from .scalar_training import sources, PROTOCOL_IDENTITY, cell_name, state_for
from .curriculum_data import dataset

METADATA_LIMIT = 8*1024*1024


def context(state, workload, request_identity):
    if type(state.config) is not ScalarPlan:
        raise ContractError('scalar-learning checkpoint requires its separate bounded plan')
    cell_name(workload, state.config.seed, state.lane); require_identity(request_identity)
    raw, manifest = dataset(workload)
    if (state.config.learning_rate != '0.001' or state.data['validation'] is not state.data['train']
            or (state.dataset_identity, state.manifest_identity) != (identity(raw), identity(manifest))):
        raise ContractError('scalar-learning checkpoint requires its admitted train-only state')
    return {'schema': 'trinite.scalar-learning-checkpoint.v1', 'data_view': 'train-only.v1',
        'protocol_identity': PROTOCOL_IDENTITY, 'workload': workload, 'seed': state.config.seed,
        'lane': state.lane, 'request_identity': request_identity, 'plan': state.config.to_dict(),
        'source': sources(), 'environment': require_environment(),
        'dataset_identity': state.dataset_identity, 'manifest_identity': state.manifest_identity,
        'initialization_identity': state.initialization_identity, 'initial_train_loss_hex': state.initial_train_loss}


def snapshot(state, workload, request_identity):
    progress(state); elected = context(state, workload, request_identity)
    payload = tensor_payload(state)
    raw = json_bytes({**elected, 'model_identity': model_identity(state.model),
        'step': state.step, 'cursor': state.cursor, 'target_tokens': state.target_tokens,
        'history': state.history, 'train_diagnostics': state.validation})
    if len(raw) > METADATA_LIMIT:
        raise ContractError('scalar-learning metadata exceeds budget')
    return payload, raw


def restore(payload, metadata, workload, seed, lane, request_identity, identities, plan=None):
    if (type(payload) is not bytes or type(metadata) is not bytes
            or len(payload) > 32*1024*1024 or len(metadata) > METADATA_LIMIT
            or (identity(payload), identity(metadata)) != identities):
        raise ContractError('scalar-learning checkpoint identity/type/size mismatch')
    record = parse_json(metadata, canonical=True); state = state_for(workload, seed, lane, plan)
    expected = context(state, workload, request_identity)
    exact_keys(record, set(expected)|{'model_identity', 'step', 'cursor', 'target_tokens',
                                     'history', 'train_diagnostics'}, 'scalar-learning checkpoint')
    if any(json_bytes(record[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError('scalar-learning checkpoint context mismatch')
    state.step, state.cursor, state.target_tokens = record['step'], record['cursor'], record['target_tokens']
    state.history, state.validation = record['history'], record['train_diagnostics']
    progress(state)
    return restore_tensors(state, payload, record['model_identity'])
