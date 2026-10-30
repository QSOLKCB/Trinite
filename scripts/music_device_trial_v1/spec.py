"""Fixed same-build CPU/CUDA control pilot, independent of curriculum outcome."""
from trinite.contracts import ContractError, identity, json_bytes
from music_placement_v1.spec import (train_rows, diagnostic_rows, development_rows,
                                    plan_for, dataset, schedule_receipt)
BACKENDS = ('cpu', 'cuda')
SEED = 17
TIMING_PHASES = ('cpu_initialization','model_transfer','initial_loss','updates','diagnostics','serialization','compile','target_warmup')
PARITY_TOLERANCES = {k:{'atol_hex':a.hex(),'rtol_hex':r.hex()} for k,(a,r) in {
    'logits':(1e-5,1e-4), 'loss':(1e-6,1e-5), 'gradient':(1e-5,1e-3),
    'parameter':(2e-6,1e-4), 'exp_avg':(1e-6,1e-3), 'exp_avg_sq':(1e-8,1e-3),
    'gradient_norm':(1e-5,1e-3)}.items()}
PROTOCOL = {
    'schema': 'trinite.music-device-trial-protocol.v1',
    'backends': list(BACKENDS), 'seed': SEED, 'arm': 'control',
    'steps': 640, 'batch_size': 8, 'generation_cap': 64,
    'training_visits_per_cell': 5120, 'target_tokens_per_cell': 10240,
    'aggregate_updates': 1280, 'aggregate_visits': 10240, 'aggregate_targets': 20480,
    'diagnostic_rows_per_cell': 80, 'diagnostic_unique_prompts': 70,
    'development_rows_per_cell': 120, 'max_decodes': 400, 'max_decode_forwards': 25600,
    'probe_updates': 2, 'probe_visits': 16, 'probe_targets': 32,
    'probe_input_length': 67, 'probe_scope': 'disposable nonmusical one-update CPU/CUDA parity; not target training; fresh target workers receive no persistent warmup',
    'max_update_seconds_per_cell': 120, 'worker_seconds': 900,
    'aggregate_worker_seconds': 3600,
    'timing': 'sum of probe, two train and one development worker ceilings, not whole-controller deadline; CPU one thread; synchronized CUDA phases; cold worker wall includes overhead',
    'max_worker_log_bytes': 1048576, 'max_journal_bytes': 1048576,
    'max_request_bytes': 4194304, 'max_checkpoint_bytes': 16777216,
    'max_artifact_bytes': 16777216, 'max_total_payload_bytes': 134217728,
    'parity_tolerances': PARITY_TOLERANCES,
    'initialization': 'exact shared CPU-initialized bytes, independent models and optimizers',
    'numerics': 'float32, no TF32/autocast/compile/fused/foreach; same CUDA-wheel environment for both cells',
    'selection': 'both final640 endpoints; no election/resume/retry/repair/fallback',
    'development_gate': 'both safe completed final640 cells and diagnostic journals, no accuracy gate',
    'interpretation': 'one fixed-order same-build compatibility/timing pair, not robust speed benchmark or backend-by-curriculum interaction',
    'final_tensor_equality_required': False, 'fresh_confirmation': False,
    'gpu_curriculum_benefit_established': False, 'learning_adequate': False,
    'phase5_ready': False, 'release_ready': False,
}

def backend_name(backend):
    if type(backend) is not str or backend not in BACKENDS:
        raise ContractError('only fixed CPU/CUDA cells admitted')
    return backend


def score_cell(records, backend, stage):
    """Backend labels are transport metadata, never input or target scoring data."""
    from music_placement_v1.reporting import score_cell as original
    backend_name(backend)
    count = 80 if stage == 'diagnostic' else 120 if stage == 'development' else None
    if type(records) is not list or len(records) != count:
        raise ContractError('complete fixed device measurement required')
    normalized = []
    for r in records:
        if (type(r) is not dict or set(r) != {'backend','seed','step','stage','example_identity','generated_tokens'}
                or r['backend'] != backend or type(r['seed']) is not int or r['seed'] != SEED):
            raise ContractError('device record context differs')
        normalized.append({'arm':'control', **{k:v for k,v in r.items() if k != 'backend'}})
    result = original(normalized, 'control', SEED, stage)
    return {'backend': backend, 'scores': result}


def matrix_summary(diagnostic_records, development_records):
    if len(diagnostic_records) != 160 or len(development_records) != 240:
        raise ContractError('both fixed backend inventories required')
    result = {}
    for stage, records, size in (('diagnostic',diagnostic_records,80),('development',development_records,120)):
        result[stage] = {b:score_cell(records[i*size:(i+1)*size],b,stage)['scores']
                         for i,b in enumerate(BACKENDS)}
    metrics = ('exact_plus_eos','complete_groups','both_correct_edges','changed_output_edges')
    result['cuda_minus_cpu'] = {stage: {view: {m: result[stage]['cuda']['by_sibling'][view][m]-result[stage]['cpu']['by_sibling'][view][m]
                          for m in metrics} for view in result[stage]['cpu']['by_sibling']}
                          for stage in ('diagnostic','development')}
    result['raw_backend_pairs'] = {}
    for stage, records, rows in (('diagnostic', diagnostic_records, diagnostic_rows()),
                                  ('development', development_records, development_rows())):
        size=len(rows); cpu=records[:size]; cuda=records[size:]
        views={}
        for view in sorted({r['sibling'] for r in rows}):
            selected=[(row,left,right) for row,left,right in zip(rows,cpu,cuda) if row['sibling']==view]
            equal=sum(left['generated_tokens']==right['generated_tokens'] for row,left,right in selected)
            both=sum(left['generated_tokens']==right['generated_tokens']==[*row['answer'].encode('ascii'),257] for row,left,right in selected)
            views[view]={'pairs':len(selected),'identical_raw_outputs':equal,'different_raw_outputs':len(selected)-equal,
                         'both_correct':both,'identical_but_wrong':equal-both}
        result['raw_backend_pairs'][stage]=views
    result.update(decodes=400, generated_tokens=sum(len(r['generated_tokens']) for r in diagnostic_records+development_records),
        scope='retained score comparison; no tensor equality or accuracy gate; no inference replay',
        fresh_confirmation=False, gpu_curriculum_benefit_established=False)
    return result
