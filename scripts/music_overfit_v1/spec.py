"""Frozen train membership and concentrated-exposure specification."""
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from music_learning_v1.admission import dataset, audit_bytes

SEEDS = (17, 29)
SUBSETS = ('one', 'four')
CELLS = tuple((seed, subset) for subset in SUBSETS for seed in SEEDS)
CHECKPOINTS = (0, 8, 16, 32, 64)
PROTOCOL = {
    'schema': 'trinite.music-overfit-protocol.v1',
    'seeds': list(SEEDS), 'subsets': list(SUBSETS), 'checkpoints': list(CHECKPOINTS),
    'steps': 64, 'batch_size': 8, 'generation_cap': 64,
    'max_target_tokens': 32768, 'max_update_seconds': 120, 'worker_seconds': 900,
    'max_worker_log_bytes': 1048576, 'max_artifact_bytes': 16777216,
    'max_retained_payload_bytes': 33554432,
    'aggregate_cells': 4, 'aggregate_update_seconds': 480, 'aggregate_worker_seconds': 3600,
    'selection': 'fixed final checkpoint; no adaptive stopping, election, retry or expansion',
    'subset_selection': 'one: shortest attacks answer then identity; four: first two distinct attacks answers and first two distinct duration answers in identity order',
    'diagnostics': 'selected unique original-train rows only; native validation field aliases these rows, not heldout',
    'fit': 'all selected raw answers plus EOS exact at step64 in both seeds, separately by subset',
    'negative_scope': 'failure at fixed cap is inconclusive about eventual fit or music learning',
    'prefix_parity_rtol': '0.00001', 'prefix_parity_atol': '0.00001',
    'geometry_loss': '0', 'e8': False, 'arithmetic_bridge': False,
    'fresh_confirmation': False, 'learning_adequate': False, 'phase5_ready': False, 'release_ready': False,
}


def admitted():
    raw, receipt = dataset()
    audit_bytes(raw, receipt)
    return raw, receipt, parse_json(raw, canonical=True)['examples']


def subset_rows(subset):
    if type(subset) is not str or subset not in SUBSETS:
        raise ContractError('unknown frozen overfit subset')
    _, _, all_rows = admitted()
    train = sorted((r for r in all_rows if r['split'] == 'train'), key=lambda r: r['example_identity'])
    if subset == 'one':
        return [min((r for r in train if r['task'] == 'attacks'),
                    key=lambda r: (len(r['answer'].encode('ascii')), r['example_identity']))]
    selected = []
    for task in ('attacks', 'duration'):
        answers = set()
        for row in train:
            if row['task'] == task and row['answer'] not in answers:
                selected.append(row); answers.add(row['answer'])
                if len(answers) == 2:
                    break
        if len(answers) != 2:
            raise ContractError('frozen contrasting train answers unavailable')
    return selected


def validate_rows(rows, subset=None):
    if type(rows) is not list or not rows:
        raise ContractError('nonempty canonical train rows required')
    _, _, all_rows = admitted()
    canonical = {r['example_identity']: r for r in all_rows if r['split'] == 'train'}
    ids = []
    for row in rows:
        if type(row) is not dict or row.get('example_identity') not in canonical:
            raise ContractError('row is not an originally admitted train member')
        key = row['example_identity']
        if json_bytes(row) != json_bytes(canonical[key]) or key in ids:
            raise ContractError('changed or repeated unique train row')
        ids.append(key)
    if subset is not None and json_bytes(rows) != json_bytes(subset_rows(subset)):
        raise ContractError('rows differ from frozen subset')
    return [canonical[key] for key in ids]


def schedule_receipt(subset):
    rows = subset_rows(subset)
    visits = [rows[i % len(rows)] for i in range(512)]
    targets = sum(r['answer_token_count'] for r in visits)
    if targets > PROTOCOL['max_target_tokens']:
        raise ContractError('overfit target ceiling exceeded')
    return {'example_identities': [r['example_identity'] for r in rows],
            'unique_train_examples': len(rows), 'scheduled_visits': 512,
            'visits_per_example': 512 // len(rows), 'target_tokens': targets,
            'schedule_identity': identity(json_bytes([r['example_identity'] for r in visits]))}
