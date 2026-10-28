"""Finite original contrast corpus, independent labels and prospective admission."""
from collections import Counter
from fractions import Fraction
from trinite.contracts import ContractError, identity, json_bytes
from music_learning_v1.codec import make_example, encode_example, parse_prompt
from symbolic_music_v1.core import from_dict, labels
from symbolic_music_v1.oracle import annotate

SEEDS = (17, 29)
# Whole structural parent families stay on one side across EVERY meter/sibling.
# Distinct duration partitions distinguish parents even after stripping pitch/tempo/header.
PARENTS = (
    ('a', 'train', (1, 1, 2, 4), 0, 6),
    ('b', 'train', (1, 2, 2, 3), 4, 2),
    ('c', 'heldout', (1, 1, 3, 3), 1, 7),
    ('d', 'heldout', (2, 2, 2, 2), 5, 0),
)
METERS = ((3, 4, (1, 1, 1)), (4, 4, (2, 2)), (5, 8, (2, 3)),
          (7, 8, (2, 2, 3)), (8, 16, (4, 4)))
PROTOCOL = {
    'schema': 'trinite.music-contrast-protocol.v1', 'seeds': list(SEEDS),
    'steps': 64, 'batch_size': 8, 'generation_cap': 64,
    'train_rows': 40, 'heldout_canonical_rows': 40, 'heldout_sibling_rows': 80,
    'scheduled_visits_per_seed': 512, 'answer_eos_targets_per_seed': 1024,
    'training_decodes': 160, 'conditional_heldout_decodes': 240,
    'max_decodes': 400, 'max_decode_forwards': 25600,
    'max_update_seconds_per_seed': 120, 'worker_seconds': 900,
    'aggregate_update_seconds': 240, 'aggregate_worker_seconds': 3600,
    'max_worker_log_bytes': 1048576, 'max_artifact_bytes': 16777216,
    'max_total_payload_bytes': 134217728,
    'initialization': 'fresh dense native seeds; no checkpoint pretraining',
    'task': 'attacks only; note/rest and valid tie/reattack contrasts',
    'selection': 'fixed step64, no election, retry, extension or heldout tuning',
    'heldout_gate': 'both seeds must fit all40 canonical training rows with exact answer plus EOS',
    'evaluation': 'initial/final train; only final gated heldout canonical/pitch/tempo siblings',
    'validation': 'none; native validation field aliases training rows',
    'primary': 'complete four-variant groups and both-correct contrast edges; raw exact+EOS alongside',
    'music_pretraining_transfer': False, 'arithmetic_bridge': False, 'geometry_loss': '0',
    'fresh_confirmation': False, 'learning_adequate': False,
    'phase5_ready': False, 'release_ready': False,
}


def _score(parent, meter, rest, tied, sibling):
    _, _, weights, tie, rest_position = parent
    n, d, groups = meter
    pitch = 72 if sibling == 'pitch' else 60
    tempo = 180 if sibling == 'tempo' else 120
    events = []
    for i, weight in enumerate(weights * 2):
        duration = Fraction(n * weight, d * 8)
        events.append({'duration': [duration.numerator, duration.denominator],
                       'pitch': None if rest and i == rest_position else pitch,
                       'tie_in': bool(tied and i == tie + 1),
                       'tie_out': bool(tied and i == tie)})
    return {'schema': 'trinite.music-score.v1',
            'meter': {'numerator': n, 'denominator': d, 'groups': list(groups)},
            'tempo_qpm': tempo, 'bars': [events[:4], events[4:]]}


def rows(selector='train', siblings=False):
    if selector not in ('train', 'heldout') or type(siblings) is not bool:
        raise ContractError('unknown contrast selector')
    if siblings and selector != 'heldout':
        raise ContractError('invariant siblings are heldout probes only')
    result = []
    for parent in PARENTS:
        name, split, _, tie, rest_position = parent
        if split != selector:
            continue
        if rest_position in (tie, tie + 1):
            raise ContractError('rest must not invalidate tested tie')
        for n, d, groups in METERS:
            group = f'{name}-{n}-{d}'
            for sibling in (('canonical', 'pitch', 'tempo') if siblings else ('canonical',)):
                for rest, tied in ((False, False), (False, True), (True, False), (True, True)):
                    score = _score(parent, (n, d, groups), rest, tied, sibling)
                    expected = annotate(score)
                    if expected != labels(from_dict(score)) or expected['attacks'] != 8-int(rest)-int(tied):
                        raise ContractError('independent attacks oracle disagrees')
                    adapted = make_example({'task': 'annotate-rhythm-v1', 'score': score}, 'attacks')
                    encoded = encode_example(adapted)
                    if parse_prompt(adapted['prompt'])['score'] != score or sum(encoded.loss_mask) != 2:
                        raise ContractError('contrast carrier/mask mismatch')
                    value = {**adapted, 'formal': score, 'family_id': name,
                             'parent_id': name, 'group_id': group, 'sibling': sibling,
                             'variant_id': f'{int(rest)}{int(tied)}',
                             'meter': f'{n}/{d}', 'split': 'train' if selector == 'train' else 'test'}
                    result.append({**value, 'example_identity': identity(json_bytes(value))})
    return result


def dataset():
    train, heldout = rows(), rows('heldout', True)
    all_rows = train + heldout
    if len(train) != 40 or len(heldout) != 120 or len({r['prompt'] for r in all_rows}) != 160:
        raise ContractError('contrast inventory or cross-split collision')
    for group in {r['group_id'] for r in all_rows}:
        for sibling in {r['sibling'] for r in all_rows if r['group_id'] == group}:
            quartet = [r for r in all_rows if r['group_id'] == group and r['sibling'] == sibling]
            if len(quartet) != 4 or len({len(r['prompt']) for r in quartet}) != 1 \
                    or Counter(r['answer'] for r in quartet) != Counter({'7': 2, '6': 1, '8': 1}):
                raise ContractError('contrast quartet parity failed')
    raw = json_bytes({'schema': 'trinite.music-contrast-data.v1', 'protocol': PROTOCOL,
                      'parents': [[a,b,list(c),d,e] for a,b,c,d,e in PARENTS],
                      'meters': [[a,b,list(c)] for a,b,c in METERS], 'examples': all_rows})
    receipt = json_bytes({'schema': 'trinite.music-contrast-admission.v1',
        'dataset_identity': identity(raw), 'protocol_identity': identity(json_bytes(PROTOCOL)),
        'origin': 'original deterministic symbolic patterns commissioned by QSOL-IMC; no imported music or teacher answers',
        'verification': 'full finite regeneration, independent integer-tick and rational oracles, lossless compact carrier and native masks',
        'boundary': 'this exact finite new dataset only; original native and music admissions unchanged'})
    return raw, receipt


def require_admission(expected):
    raw, receipt = dataset()
    if expected != identity(receipt):
        raise ContractError('contrast admission identity mismatch')
    return raw, receipt


def validate_rows(value, selector='train', siblings=False):
    if json_bytes(value) != json_bytes(rows(selector, siblings)):
        raise ContractError('contrast rows differ from exact admitted inventory')
    return value


def schedule_receipt():
    # Complete four-variant groups; no variant ID, label or split enters prompts.
    train = rows()
    visits = [train[i % len(train)] for i in range(512)]
    counts = Counter(r['example_identity'] for r in visits)
    return {'scheduled_visits': 512, 'unique_train_rows': 40,
            'visits_per_example': dict(sorted(counts.items())),
            'target_tokens': 1024,
            'live_input_tokens': sum(len(r['prompt'])+2 for r in visits),
            'answer_histogram': dict(Counter(r['answer'] for r in visits)),
            'schedule_identity': identity(json_bytes([r['example_identity'] for r in visits]))}
