"""One-factor family/placement-template replacement allocation on original training structures only."""
from collections import Counter
from copy import deepcopy

from trinite.contracts import ContractError, identity, json_bytes
from music_contrast_v1.spec import rows as original_rows
from music_exposure_v1.spec import plan_for as original_plan
from music_learning_v1.codec import make_example, encode_example, parse_prompt
from symbolic_music_v1.core import from_dict, labels
from symbolic_music_v1.oracle import annotate

SEEDS = (17, 29)
ARMS = ('control', 'swap')
CELLS = (('control', 17), ('swap', 17), ('swap', 29), ('control', 29))
PROTOCOL = {
    'schema': 'trinite.music-placement-protocol.v1',
    'seeds': list(SEEDS), 'arms': list(ARMS),
    'cell_order': [[arm, seed] for arm, seed in CELLS],
    'steps': 640, 'batch_size': 8, 'final_step': 640,
    'generation_cap': 64, 'context_length': 256,
    'train_pool_rows': 80, 'unique_structural_train_rows': 40,
    'unique_surface_train_rows': {'control': 40, 'swap': 70},
    'visits_per_structural_row': 128,
    'visits_per_layout_per_structure': {'control': {'original': 128}, 'swap': {'original': 64, 'swapped': 64}},
    'unchanged_surface_rows': 10, 'diagnostic_unique_prompt_surfaces': 70,
    'placement_templates': {'a': {'original': [0, 6], 'swapped': [4, 2]}, 'b': {'original': [4, 2], 'swapped': [0, 6]}},
    'scheduled_visits_per_cell': 5120, 'target_tokens_per_cell': 10240,
    'aggregate_updates': 2560, 'aggregate_scheduled_visits': 20480,
    'aggregate_target_tokens': 40960,
    'diagnostic_decodes_per_cell': 80, 'development_decodes_per_cell': 120,
    'diagnostic_decodes': 320, 'development_decodes': 480,
    'max_decodes': 800, 'max_decode_forwards': 51200,
    'max_update_seconds_per_cell': 120, 'worker_seconds': 900,
    'development_worker_seconds': 900, 'aggregate_update_seconds': 480,
    'aggregate_worker_seconds': 4500,
    'timing': '4500 is sum of fixed worker ceilings, not whole-controller wall time; update timer includes train-only loss validation',
    'max_worker_log_bytes': 1048576, 'max_journal_bytes': 1048576,
    'max_request_bytes': 4194304, 'max_checkpoint_bytes': 16777216,
    'max_artifact_bytes': 16777216, 'max_total_payload_bytes': 134217728,
    'initialization': 'fresh paired native dense seeds; independent states and optimizers; no checkpoint import',
    'intervention': 'replace alternate canonical40 cycles with the other training parent rest/tie placement template; durations, pitch60, tempo and labels unchanged',
    'estimand': 'joint rest/tie-template replacement allocation at fixed structural visits/updates/targets; canonical layout exposure halves for altered variants; swapped cycle last; not all-position coverage',
    'validation': 'same canonical40 training rows every64 updates; no selection',
    'development_barrier': 'all four safe final checkpoints and complete80-row diagnostics; no accuracy fit gate',
    'selection': 'all four prescribed final640 cells; no election, retry, extension, resume or tuning',
    'generation': 'CPU float32 native prompt-only greedy cap64; no oracle, repair or forced EOS',
    'primary': 'paired per-seed canonical development complete quartets, both-correct edges and exact+EOS',
    'secondary': 'swapped-layout diagnostic pairs, pitch/tempo development pairs; rest/tie strata and training-fit regressions retained',
    'unit': 'two structural development parents; meter/sibling/seed renderings are not independent subjects',
    'pitch_scope': 'training pitch60 only in both arms; existing pitch72 development is correlated exposed evidence, not fresh confirmation',
    'rest_scope': 'joint swap of existing a/b rest/tie placement templates; cannot isolate rest from tie effects; no new positions or c/d layouts',
    'development_scope': 'previously exposed c/d development probes; no fresh confirmation',
    'decode_accounting': 'greedy forwards only; training and loss-validation forwards are separate',
    'accuracy_pass_threshold': None,
    'arithmetic_bridge': False, 'geometry_loss': '0', 'fresh_confirmation': False,
    'learning_adequate': False, 'phase5_ready': False, 'release_ready': False,
}


def cell_name(arm, seed):
    if type(arm) is not str or arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ContractError('unknown fixed placement intervention cell')
    return arm + '-' + str(seed)


def plan_for(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ContractError('unknown fixed placement intervention seed')
    return original_plan(seed)


def _swap_rows():
    """Swap existing a/b placement templates, not durations or family membership."""
    result = []
    for row in original_rows():
        score = deepcopy(row['formal'])
        tie, rest_position = {'a': (4, 2), 'b': (0, 6)}[row['parent_id']]
        rest, tied = (c == '1' for c in row['variant_id'])
        for i, event in enumerate(event for bar in score['bars'] for event in bar):
            event['pitch'] = None if rest and i == rest_position else 60
            event['tie_in'] = bool(tied and i == tie + 1)
            event['tie_out'] = bool(tied and i == tie)
        if annotate(score) != labels(from_dict(score)) or annotate(score)['attacks'] != int(row['answer']):
            raise ContractError('placement transform changed independent labels')
        adapted = make_example({'task': 'annotate-rhythm-v1', 'score': score}, 'attacks')
        encoded = encode_example(adapted)
        if (parse_prompt(adapted['prompt'])['score'] != score
                or len(adapted['prompt']) != len(row['prompt'])
                or sum(encoded.loss_mask) != 2 or adapted['answer'] != row['answer']):
            raise ContractError('placement transform carrier/length/loss parity differs')
        value = {key: deepcopy(value) for key, value in row.items() if key != 'example_identity'}
        value.update(adapted, formal=score, sibling='swap')
        result.append({**value, 'example_identity': identity(json_bytes(value))})
    return result


def diagnostic_rows():
    """Common80-row a/b measurement, independent of arm and never c/d."""
    return original_rows() + _swap_rows()


def train_rows(arm):
    if type(arm) is not str or arm not in ARMS:
        raise ContractError('unknown placement intervention arm')
    return original_rows() + (original_rows() if arm == 'control' else _swap_rows())


def development_rows():
    return original_rows('heldout', True)


def schedule_receipt(arm):
    train = train_rows(arm)
    visits = [train[index % 80] for index in range(5120)]
    structural = Counter((r['parent_id'], r['meter'], r['variant_id']) for r in visits)
    surfaces = Counter(r['prompt'] for r in visits)
    renderings = Counter(r['example_identity'] for r in visits)
    if len(structural) != 40 or set(structural.values()) != {128}:
        raise ContractError('structural exposure allocation differs')
    return {'scheduled_visits': 5120, 'target_tokens': 10240,
        'unique_structural_rows': 40, 'train_pool_rows': 80,
        'unique_surface_rows': len(surfaces),
        'visits_per_surface': dict(sorted((identity(k.encode()),v) for k,v in surfaces.items())),
        'visits_per_rendering': dict(sorted(renderings.items())),
        'visits_per_structure': {'/'.join(key): value for key, value in sorted(structural.items())},
        'live_input_tokens': sum(len(r['prompt']) + 2 for r in visits),
        'answer_histogram': dict(sorted(Counter(r['answer'] for r in visits).items())),
        'schedule_identity': identity(json_bytes([r['example_identity'] for r in visits]))}


def dataset():
    diagnostics, development = diagnostic_rows(), development_rows()
    pools = {arm: train_rows(arm) for arm in ARMS}
    if ({r['parent_id'] for r in diagnostics} != {'a', 'b'}
            or {r['parent_id'] for r in development} != {'c', 'd'}
            or len(diagnostics) != 80 or len(development) != 120
            or len({r['prompt'] for r in diagnostics}) != 70
            or len({r['prompt'] for r in development}) != 120
            or {r['prompt'] for r in diagnostics} & {r['prompt'] for r in development}):
        raise ContractError('fixed training/development structure or surface inventory differs')
    if any(r['split'] != 'train' for pool in pools.values() for r in pool):
        raise ContractError('development data entered training')
    if pools != {'control': original_rows() + original_rows(), 'swap': original_rows() + _swap_rows()}:
        raise ContractError('exact prescribed placement templates differ')
    control, treatment = pools['control'], pools['swap']
    for left, right in zip(control, treatment):
        if ((left['parent_id'], left['meter'], left['variant_id'], left['answer'], len(left['prompt']))
                != (right['parent_id'], right['meter'], right['variant_id'], right['answer'], len(right['prompt']))):
            raise ContractError('paired structural order/labels/token-length parity differs')
        lf, rf = deepcopy(left['formal']), deepcopy(right['formal'])
        for score in (lf, rf):
            for bar in score['bars']:
                for event in bar:
                    event['pitch'] = 60
                    event['tie_in'] = event['tie_out'] = False
        if lf != rf:
            raise ContractError('intervention changes duration/meter/pitch/tempo outside placement')
    if any(event['pitch'] not in (None,60) or row['formal']['tempo_qpm'] != 120
           for pool in pools.values() for row in pool for bar in row['formal']['bars'] for event in bar):
        raise ContractError('training pitch/tempo factor changed')
    # Renderings with no rest/tie are intentionally duplicate prompt surfaces.
    if sum(a['prompt'] == b['prompt'] for a,b in zip(diagnostics[:40],diagnostics[40:])) != 10:
        raise ContractError('fixed no-rest/no-tie duplication differs')
    raw = json_bytes({'schema': 'trinite.music-placement-data.v1',
        'protocol': PROTOCOL, 'training_pools': pools, 'validation_rows': original_rows(),
        'diagnostic_rows': diagnostics, 'development_rows': development,
        'schedules': {arm: schedule_receipt(arm) for arm in ARMS}})
    receipt = json_bytes({'schema': 'trinite.music-placement-admission.v1',
        'dataset_identity': identity(raw), 'protocol_identity': identity(json_bytes(PROTOCOL)),
        'origin': 'original commissioned symbolic music only; valid label-preserving placement-template swap, no imported music/teacher',
        'verification': 'exact finite regeneration; rational/integer oracle, lossless carrier and paired allocation checks',
        'scope': 'new placement replacement study only; historical admissions unchanged; c/d exposed development'})
    return raw, receipt


def require_admission(expected):
    raw, receipt = dataset()
    if type(expected) is not str or identity(receipt) != expected:
        raise ContractError('placement intervention admission identity differs')
    return raw, receipt
