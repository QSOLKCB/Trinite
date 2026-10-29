"""Frozen finite musical-family evaluation; labels are offline scoring only."""
from collections import Counter, defaultdict

from trinite.contracts import ContractError, identity, json_bytes
from music_contrast_v1.spec import dataset, rows
from music_contrast_v1.reporting import _metrics, baselines, summarize

CAP = 64
CONTEXT = 256
SEEDS = (17, 29)
STEP = 640
EOS = 257
PROTOCOL = {
    'schema': 'trinite.frozen-music-eval-protocol.v1',
    'seeds': list(SEEDS), 'step': STEP, 'generation_cap': CAP,
    'context_length': CONTEXT, 'training_updates': 0,
    'train_replay_decodes': 80, 'heldout_decodes': 240,
    'max_decodes': 320, 'max_decode_forwards': 20480,
    'admission_worker_seconds': 600, 'worker_seconds': 900,
    'max_worker_log_bytes': 1048576, 'max_journal_bytes': 1048576,
    'max_request_bytes': 4194304, 'max_artifact_bytes': 33554432,
    'max_checkpoint_bytes': 16777216, 'max_original_payload_bytes': 134217728,
    'max_total_payload_bytes': 8388608,
    'selection': 'both prescribed final640 checkpoints only; no election, resume, retry or replacement',
    'admission': 'complete independently bound original producer artifacts; model-only safe restoration; no optimizer construction',
    'replay_gate': 'both seeds reproduce every retained final training raw token sequence and all40 exact answer+EOS before any heldout call',
    'generation': 'CPU float32 native prompt-only greedy; no gold prefix, oracle, repair, forced EOS or training',
    'primary': 'canonical raw exact+EOS, complete quartets and both-correct contrast edges separately for each seed',
    'unit': 'two heldout structural parents; meter rows, pitch/tempo siblings and seeds are not independent subjects',
    'controls': 'fit unique canonical training rows only; exact formal oracle is offline scoring ceiling only',
    'scope': 'existing visible musical development families; finite eight-event 8-rest-tie rule remains possible',
    'fresh_confirmation': False, 'general_music_established': False,
    'arbitrary_counting_established': False, 'arithmetic_transfer_established': False,
    'learning_adequate': False, 'phase5_ready': False, 'release_ready': False,
}


def data_contract():
    """Regenerate the historical finite corpus and a separate evaluation admission."""
    raw, original_admission = dataset()
    train, heldout = rows(), rows('heldout', True)
    if {r['parent_id'] for r in train} != {'a', 'b'} or {r['parent_id'] for r in heldout} != {'c', 'd'}:
        raise ContractError('frozen structural family split differs')
    if Counter(r['sibling'] for r in heldout) != {'canonical': 40, 'pitch': 40, 'tempo': 40}:
        raise ContractError('frozen sibling inventory differs')
    if {r['prompt'] for r in train} & {r['prompt'] for r in heldout}:
        raise ContractError('exact train/heldout prompt collision')
    triples = defaultdict(list)
    for row in heldout:
        if row['family_id'] != row['parent_id'] or row['split'] != 'test':
            raise ContractError('heldout parent assignment differs')
        triples[(row['group_id'], row['variant_id'])].append(row)
    if len(triples) != 40 or any({r['sibling'] for r in group} != {'canonical', 'pitch', 'tempo'}
                                or len({r['answer'] for r in group}) != 1 for group in triples.values()):
        raise ContractError('heldout transformation lineage differs')
    contract = {
        'schema': 'trinite.frozen-music-eval-data.v1',
        'historical_dataset_identity': identity(raw),
        'historical_admission_identity': identity(original_admission),
        'train_rows': 40, 'heldout_rows': 120,
        'train_parents': ['a', 'b'], 'heldout_parents': ['c', 'd'],
        'heldout_structural_parent_count': 2,
        'heldout_canonical_rows': 40, 'heldout_canonical_quartets': 10,
        'heldout_canonical_contrast_edges': 40,
        'heldout_correlated_sibling_rows': 80, 'transformation_triples': 40,
        'exact_prompt_overlap': 0,
        'ordered_train_identities': [r['example_identity'] for r in train],
        'ordered_heldout_identities': [r['example_identity'] for r in heldout],
        'independence': 'two structural parents; do not pool siblings or seeds into independent subjects',
    }
    admission = {'schema': 'trinite.frozen-music-eval-admission.v1',
                 'data_contract_identity': identity(json_bytes(contract)),
                 'protocol_identity': identity(json_bytes(PROTOCOL)),
                 'scope': 'exact regenerated finite historical corpus; original admissions unchanged'}
    return {**contract, 'admission_identity': identity(json_bytes(admission))}


def input_rows():
    """Only identity, prompt and split cross the inference input boundary."""
    data_contract()
    return [{key: row[key] for key in ('example_identity', 'prompt', 'split')}
            for row in rows() + rows('heldout', True)]


def _row_metrics(selected, outputs):
    return {'rows': len(selected),
            'exact_plus_eos': sum(outputs[r['example_identity']] == [*r['answer'].encode('ascii'), EOS]
                                  for r in selected),
            'eos_terminated': sum(outputs[r['example_identity']][-1] == EOS for r in selected),
            'cap_terminated': sum(outputs[r['example_identity']][-1] != EOS for r in selected),
            'distinct_raw_outputs': len({tuple(outputs[r['example_identity']]) for r in selected}),
            'generated_tokens': sum(len(outputs[r['example_identity']]) for r in selected)}


def _summary(selected, generated, selector):
    predictions = []
    outputs = {}
    for row, tokens in zip(selected, generated):
        gold = [*row['answer'].encode('ascii'), EOS]
        outputs[row['example_identity']] = tokens
        predictions.append({'example_identity': row['example_identity'], 'family_id': row['family_id'],
            'task': row['task'], 'split': row['split'], 'expected_tokens': gold,
            'generated_tokens': tokens, 'correct': tokens == gold,
            'baseline_tokens': [55, EOS], 'baseline_correct': gold == [55, EOS]})
    # Historical scorer remains the authoritative quartet/edge/control implementation.
    result = summarize(json_bytes({'predictions': predictions}), selector)
    result['raw'] = _row_metrics(selected, outputs)
    for field in ('parent_id', 'task'):
        result['by_' + ('parent' if field == 'parent_id' else field)] = {
            value: _metrics([r for r in selected if r[field] == value], outputs)
            for value in sorted({r[field] for r in selected})}
    result['by_variant'] = {variant: _row_metrics([r for r in selected if r['variant_id'] == variant], outputs)
                            for variant in sorted({r['variant_id'] for r in selected})}
    canonical = [r for r in selected if r['sibling'] == 'canonical']
    result['canonical_primary'] = _metrics(canonical, outputs)
    result['canonical_controls'] = baselines(canonical)
    result['canonical_strata'] = {
        field: {value: (_row_metrics if field == 'variant_id' else _metrics)(
            [r for r in canonical if r[field] == value], outputs)
            for value in sorted({r[field] for r in canonical})}
        for field in ('parent_id', 'meter', 'task', 'variant_id')}
    if selector == 'heldout':
        result['correlated_siblings'] = _metrics([r for r in selected if r['sibling'] != 'canonical'], outputs)
        pairs = {}
        for sibling in ('pitch', 'tempo'):
            canonical_by_id = {(r['group_id'], r['variant_id']): r for r in canonical}
            siblings = [r for r in selected if r['sibling'] == sibling]
            pairs[sibling] = {'pairs': len(siblings), 'identical_raw_outputs': 0, 'both_correct': 0}
            for row in siblings:
                base = canonical_by_id[(row['group_id'], row['variant_id'])]
                left, right = outputs[base['example_identity']], outputs[row['example_identity']]
                pairs[sibling]['identical_raw_outputs'] += left == right
                pairs[sibling]['both_correct'] += left == right == [*row['answer'].encode('ascii'), EOS]
        result['transformation_pairs'] = pairs
    return result


def score_records(records):
    """Strict complete ordered journal scoring; no inference or replay attestation."""
    if type(records) is not list or len(records) != PROTOCOL['max_decodes']:
        raise ContractError('complete fixed320 decode journal required')
    result = {'schema': 'trinite.frozen-music-eval-scores.v1', 'by_seed': {},
              'heldout_structural_parents': 2, 'independent_subjects_not_decodes': True,
              'scope': 'raw-token score consistency only; retained-output equality checked separately by worker'}
    cursor = 0
    for stage, selector in (('train_replay', 'train'), ('heldout', 'heldout')):
        selected = rows(selector, selector == 'heldout')
        for seed in SEEDS:
            generated = []
            for row in selected:
                record = records[cursor]; cursor += 1
                if type(record) is not dict or set(record) != {'seed', 'step', 'stage', 'example_identity', 'generated_tokens'}:
                    raise ContractError('unknown decode fields')
                expected = {'seed': seed, 'step': STEP, 'stage': stage, 'example_identity': row['example_identity']}
                if any(json_bytes(record[k]) != json_bytes(v) for k, v in expected.items()):
                    raise ContractError('fixed decode order or checkpoint differs')
                tokens = record['generated_tokens']
                if type(tokens) is not list or not 1 <= len(tokens) <= CAP or any(type(t) is not int or not 0 <= t <= 258 for t in tokens) \
                        or EOS in tokens[:-1] or (tokens[-1] != EOS and len(tokens) != CAP):
                    raise ContractError('raw generation must terminate at EOS or fixed cap')
                generated.append(tokens)
            summary = _summary(selected, generated, selector)
            if stage == 'train_replay' and summary['raw']['exact_plus_eos'] != 40:
                raise ContractError('both training replay gates must pass before heldout')
            result['by_seed'].setdefault(str(seed), {})[stage] = summary
    result['decodes'] = len(records)
    result['forward_calls'] = sum(len(r['generated_tokens']) for r in records)
    return result
