"""New finite train-only admission; historical contrast corpus stays unchanged."""
from collections import Counter
from dataclasses import replace
from trinite.contracts import ContractError, identity, json_bytes
from music_contrast_v1.spec import rows as contrast_rows
from music_learning_v1.numerical import plan_for as original_plan

SEEDS = (17, 29)
CHECKPOINTS = (0, 64, 640)
PROTOCOL = {
    'schema': 'trinite.music-exposure-protocol.v1', 'seeds': list(SEEDS),
    'steps': 640, 'batch_size': 8, 'checkpoints': list(CHECKPOINTS),
    'generation_cap': 64, 'train_rows': 40, 'heldout_decodes': 0,
    'scheduled_visits_per_seed': 5120, 'visits_per_row': 128,
    'answer_eos_targets_per_seed': 10240,
    'max_decodes': 240, 'max_decode_forwards': 15360,
    'max_update_seconds_per_seed': 120, 'worker_seconds': 900,
    'aggregate_update_seconds': 240, 'aggregate_worker_seconds': 1800,
    'max_worker_log_bytes': 1048576, 'max_artifact_bytes': 16777216,
    'max_total_payload_bytes': 134217728,
    'initialization': 'fresh dense native seeds, no resume or pretraining',
    'selection': 'fixed step640 only; no election, retry, adaptive stop or expansion',
    'evaluation': 'train-only raw greedy at0/64/640; no heldout route even after fit',
    'validation': 'native validation aliases training rows; every64 updates',
    'timing': 'one cumulative monotonic update timer including native loss validation; excludes snapshots and greedy diagnostics; separate whole-worker deadline',
    'diagnostics': 'isolated model copy; original optimizer/progress/model and RNG preserved',
    'interpretation': 'exposure feasibility only, not generalization or causal comparison to previous four-row task',
    'arithmetic_bridge': False, 'geometry_loss': '0', 'fresh_confirmation': False,
    'learning_adequate': False, 'phase5_ready': False, 'release_ready': False,
}


def plan_for(seed):
    # Preserve optimizer, validation interval, model and ALL existing ceilings.
    return replace(original_plan(seed), steps=PROTOCOL['steps'])


def rows():
    return contrast_rows('train')


def dataset():
    train = rows()
    if len(train) != 40 or any(r['split'] != 'train' for r in train):
        raise ContractError('exposure admits exactly the40 original training rows')
    raw = json_bytes({'schema': 'trinite.music-exposure-data.v1',
                      'protocol': PROTOCOL, 'examples': train})
    receipt = json_bytes({'schema': 'trinite.music-exposure-admission.v1',
        'dataset_identity': identity(raw), 'protocol_identity': identity(json_bytes(PROTOCOL)),
        'origin': 'exact original musical contrast TRAIN rows only; original deterministic symbolic patterns',
        'verification': 'unchanged contrast generator, rational and integer-tick labels, compact parser and masks',
        'boundary': 'new train-only exposure admission; historical admissions remain unchanged'})
    return raw, receipt


def require_admission(expected):
    raw, receipt = dataset()
    if identity(receipt) != expected:
        raise ContractError('exposure admission identity differs')
    return raw, receipt


def schedule_receipt(step=640):
    if type(step) is not int or step not in CHECKPOINTS:
        raise ContractError('unknown exposure observation step')
    train = rows()
    visits = [train[i % len(train)] for i in range(step*8)]
    counts = Counter(r['example_identity'] for r in visits)
    return {'step': step, 'scheduled_visits': len(visits), 'unique_train_rows': 40,
        'visits_per_example': {r['example_identity']: counts[r['example_identity']] for r in train},
        'target_tokens': len(visits)*2,
        'live_input_tokens': sum(len(r['prompt'])+2 for r in visits),
        'answer_histogram': dict(Counter(r['answer'] for r in visits)),
        'schedule_identity': identity(json_bytes([r['example_identity'] for r in visits]))}
