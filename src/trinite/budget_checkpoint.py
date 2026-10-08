"""Source-bound train-only budget checkpoints using shared tensor/progress laws."""
from .contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from .comparison_checkpoint import tensor_payload, progress, restore_tensors
from .training import model_identity, require_environment
from .budget_plan import BudgetPlan

METADATA_LIMIT = 8*1024*1024


def context(state, workload, candidate, request_identity):
    # Lazy orchestration imports prevent a checkpoint/orchestration import cycle.
    from .budget import sources, PROTOCOL_IDENTITY, cell_name, plan_for
    from .foundations_data import dataset
    if type(state.config) is not BudgetPlan:
        raise ContractError('budget checkpoint requires its separate bounded plan')
    cell_name(workload, state.config.seed, candidate); require_identity(request_identity)
    raw, manifest = dataset(workload)
    if (state.lane != 'dense' or state.config.learning_rate != plan_for(state.config.seed,candidate).learning_rate
            or state.data['validation'] is not state.data['train']
            or (state.dataset_identity,state.manifest_identity) != (identity(raw),identity(manifest))):
        raise ContractError('budget checkpoint requires its elected train-only state')
    return {'schema':'trinite.budget-checkpoint.v1', 'data_view':'train-only.v1',
            'protocol_identity':PROTOCOL_IDENTITY, 'workload':workload, 'seed':state.config.seed,
            'candidate':candidate, 'lane':'dense', 'request_identity':request_identity,
            'plan':state.config.to_dict(), 'source':sources(), 'environment':require_environment(),
            'dataset_identity':state.dataset_identity, 'manifest_identity':state.manifest_identity,
            'initialization_identity':state.initialization_identity,
            'initial_train_loss_hex':state.initial_train_loss}


def snapshot_state(state, workload, candidate, request_identity):
    progress(state); elected = context(state,workload,candidate,request_identity)
    payload = tensor_payload(state)
    metadata = json_bytes({**elected,'model_identity':model_identity(state.model),
        'step':state.step,'cursor':state.cursor,'target_tokens':state.target_tokens,
        'history':state.history,'train_diagnostics':state.validation})
    if len(metadata)>METADATA_LIMIT:
        raise ContractError('budget metadata exceeds budget')
    return payload,metadata


def restore_state(payload,metadata,workload,seed,candidate,request_identity,identities,plan=None):
    from .budget import state_for
    if (type(payload) is not bytes or type(metadata) is not bytes
            or len(payload)>32*1024*1024 or len(metadata)>METADATA_LIMIT
            or (identity(payload),identity(metadata))!=identities):
        raise ContractError('budget checkpoint identity/type/size mismatch')
    record = parse_json(metadata,canonical=True)
    state = state_for(workload,seed,candidate,plan)
    expected = context(state,workload,candidate,request_identity)
    exact_keys(record,set(expected)|{'model_identity','step','cursor','target_tokens','history',
                                   'train_diagnostics'},'budget checkpoint')
    if any(json_bytes(record[k])!=json_bytes(v) for k,v in expected.items()):
        raise ContractError('budget checkpoint context mismatch')
    state.step,state.cursor,state.target_tokens=record['step'],record['cursor'],record['target_tokens']
    state.history,state.validation=record['history'],record['train_diagnostics']
    progress(state)
    return restore_tensors(state,payload,record['model_identity'])
