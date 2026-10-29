"""Strict retained-output scoring and paired descriptive contrasts, never selection."""
from collections import Counter

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from music_contrast_v1.reporting import _metrics, baselines
from .spec import CELLS, SEEDS, cell_name, diagnostic_rows, development_rows

EOS = 257


def _rows(stage):
    if type(stage) is not str or stage not in ('diagnostic', 'development'):
        raise ContractError('unknown fixed measurement stage')
    return diagnostic_rows() if stage == 'diagnostic' else development_rows()


def _row_metrics(selected, outputs):
    return {'rows': len(selected),
        'exact_plus_eos': sum(outputs[r['example_identity']] == [*r['answer'].encode('ascii'), EOS] for r in selected),
        'eos_terminated': sum(outputs[r['example_identity']][-1] == EOS for r in selected),
        'cap_terminated': sum(outputs[r['example_identity']][-1] != EOS for r in selected),
        'distinct_raw_outputs': len({tuple(outputs[r['example_identity']]) for r in selected}),
        'generated_tokens': sum(len(outputs[r['example_identity']]) for r in selected)}


def score_cell(records, arm, seed, stage):
    """Every required row remains, even if canonical training fit declines."""
    name = cell_name(arm, seed)
    selected = _rows(stage)
    if type(records) is not list or len(records) != len(selected):
        raise ContractError('complete fixed cell measurement required')
    outputs = {}
    for record, row in zip(records, selected):
        if type(record) is not dict or set(record) != {'arm', 'seed', 'step', 'stage', 'example_identity', 'generated_tokens'}:
            raise ContractError('unknown measurement record fields')
        context = {'arm': arm, 'seed': seed, 'step': 640, 'stage': stage,
                   'example_identity': row['example_identity']}
        if any(json_bytes(record[k]) != json_bytes(v) for k, v in context.items()):
            raise ContractError('fixed measurement order/context differs')
        tokens = record['generated_tokens']
        if (type(tokens) is not list or not 1 <= len(tokens) <= 64
                or any(type(t) is not int or not 0 <= t <= 258 for t in tokens)
                or EOS in tokens[:-1] or (tokens[-1] != EOS and len(tokens) != 64)):
            raise ContractError('raw generation must end at EOS or frozen cap')
        outputs[row['example_identity']] = tokens
    canonical = [r for r in selected if r['sibling'] == 'canonical']
    by_sibling = {s: _metrics([r for r in selected if r['sibling'] == s], outputs)
                  for s in sorted({r['sibling'] for r in selected})}
    canon_map = {(r['group_id'], r['variant_id']): r for r in canonical}
    pairs = {}
    for sibling in sorted(set(by_sibling) - {'canonical'}):
        siblings = [r for r in selected if r['sibling'] == sibling]
        identical = both = 0
        for row in siblings:
            base = canon_map[(row['group_id'], row['variant_id'])]
            left, right = outputs[base['example_identity']], outputs[row['example_identity']]
            identical += left == right
            both += left == right == [*row['answer'].encode('ascii'), EOS]
        pairs[sibling] = {'pairs': len(siblings), 'identical_raw_outputs': identical,
                          'changed_raw_outputs': len(siblings) - identical, 'both_correct': both}
    result = {'cell': name, 'stage': stage, 'raw': _row_metrics(selected, outputs),
        'all': _metrics(selected, outputs), 'by_sibling': by_sibling,
        'canonical_primary': by_sibling['canonical'],
        'controls': baselines(selected), 'canonical_controls': baselines(canonical),
        'transformation_pairs': pairs,
        'canonical_fit': by_sibling['canonical']['exact_plus_eos'] == 40,
        'by_parent': {p: _metrics([r for r in selected if r['parent_id'] == p], outputs)
                      for p in sorted({r['parent_id'] for r in selected})},
        'by_meter': {m: _metrics([r for r in selected if r['meter'] == m], outputs)
                     for m in sorted({r['meter'] for r in selected})},
        'by_task': {t: _metrics([r for r in selected if r['task'] == t], outputs)
                    for t in sorted({r['task'] for r in selected})},
        'by_variant': {v: _row_metrics([r for r in selected if r['variant_id'] == v], outputs)
                       for v in sorted({r['variant_id'] for r in selected})},
        'canonical_strata': {field: {value: (_row_metrics if field == 'variant_id' else _metrics)(
            [r for r in canonical if r[field] == value], outputs)
            for value in sorted({r[field] for r in canonical})}
            for field in ('parent_id', 'meter', 'task', 'variant_id')},
        'output_distributions': {s: {','.join(str(t) for t in tokens): count
            for tokens, count in sorted(Counter(tuple(outputs[r['example_identity']])
                for r in selected if r['sibling'] == s).items())} for s in by_sibling}}
    return result


def score_records(records, stage):
    """Exact four-cell stage inventory; rows/seeds are not independent subjects."""
    selected = _rows(stage)
    if type(records) is not list or len(records) != len(selected) * len(CELLS):
        raise ContractError('complete fixed four-cell measurement required')
    cells = {}
    for i, (arm, seed) in enumerate(CELLS):
        cells[cell_name(arm, seed)] = score_cell(records[i*len(selected):(i+1)*len(selected)], arm, seed, stage)
    return {'stage': stage, 'by_cell': cells, 'decodes': len(records),
        'decode_forwards': sum(len(r['generated_tokens']) for r in records),
        'scope': 'retained raw-token scoring only; no inference or optimizer replay',
        'structural_parent_count': 2, 'independent_subjects_not_decodes': True}


HISTORICAL_JOURNAL = 'fixtures/frozen-music-eval-result-v1/original-attempt/predictions.jsonl'
HISTORICAL_IDENTITY = 'sha256:b0939e4508697d4909e08acefea50ac568ae008092f1952068145c814355a56f'


def _historical_controls(development_records):
    """Ancillary exact comparable control outputs only; disagreement never gates."""
    from .bootstrap import ROOT, read
    raw = read(ROOT / HISTORICAL_JOURNAL, 1048576)
    if identity(raw) != HISTORICAL_IDENTITY:
        raise ContractError('immutable historical development journal differs')
    historical = [parse_json(line, canonical=True) for line in raw.splitlines(keepends=True)]
    selected = development_rows()
    by_seed = {}
    for seed in SEEDS:
        prior = {r['example_identity']: r['generated_tokens'] for r in historical
                 if r['seed'] == seed and r['step'] == 640 and r['stage'] == 'heldout'}
        current = {r['example_identity']: r['generated_tokens'] for r in development_records
                   if r['arm'] == 'control' and r['seed'] == seed
                   and r['step'] == 640 and r['stage'] == 'development'}
        expected = {r['example_identity'] for r in selected}
        if set(prior) != expected or set(current) != expected:
            raise ContractError('historical comparable control inventory differs')
        def counts(rows):
            return {'rows': len(rows), 'identical_raw_outputs': sum(
                current[r['example_identity']] == prior[r['example_identity']] for r in rows)}
        by_seed[str(seed)] = {'all': counts(selected), 'by_sibling': {
            sibling: counts([r for r in selected if r['sibling'] == sibling])
            for sibling in ('canonical', 'pitch', 'tempo')}}
    if read(ROOT / HISTORICAL_JOURNAL, 1048576) != raw:
        raise ContractError('historical journal changed during comparison')
    return {'historical_journal_identity': HISTORICAL_IDENTITY, 'by_seed': by_seed,
        'scope': 'ancillary historical reproducibility diagnostic; same seed/final640/example identity, control development only; exact retained raw-token equality, no model replay',
        'interpretation': 'mismatch is descriptive, not automatically a defect; environment differences can explain divergence; contemporaneous paired comparison remains primary',
        'admission_gate': False, 'checkpoint_or_seed_election': False,
        'causal_effect_established': False, 'independent_replication': False}


def matrix_summary(diagnostic_records, development_records):
    diagnostic = score_records(diagnostic_records, 'diagnostic')
    development = score_records(development_records, 'development')
    paired = {}
    metrics = ('exact_plus_eos', 'complete_groups', 'both_correct_edges', 'changed_output_edges')
    for seed in SEEDS:
        entry = {}
        for stage, summary in (('diagnostic', diagnostic), ('development', development)):
            control = summary['by_cell'][cell_name('control', seed)]
            pitch = summary['by_cell'][cell_name('pitch', seed)]
            entry[stage] = {
                'pitch_minus_control': {view: {m: pitch['by_sibling'][view][m] - control['by_sibling'][view][m]
                                              for m in metrics} for view in control['by_sibling']},
                'pitch_pair_both_correct_delta': pitch['transformation_pairs']['pitch']['both_correct'] - control['transformation_pairs']['pitch']['both_correct'],
                'pitch_pair_identical_delta': pitch['transformation_pairs']['pitch']['identical_raw_outputs'] - control['transformation_pairs']['pitch']['identical_raw_outputs'],
                'canonical_rest_only_exact_delta': pitch['canonical_strata']['variant_id']['10']['exact_plus_eos'] - control['canonical_strata']['variant_id']['10']['exact_plus_eos'],
                'canonical_exact_decreased': pitch['canonical_primary']['exact_plus_eos'] < control['canonical_primary']['exact_plus_eos'],
            }
        dc = diagnostic['by_cell'][cell_name('control', seed)]
        dp = diagnostic['by_cell'][cell_name('pitch', seed)]
        entry['control_canonical_training_fit'] = dc['canonical_fit']
        entry['pitch_canonical_training_fit'] = dp['canonical_fit']
        entry['lost_control_perfect_training_fit'] = dc['canonical_fit'] and not dp['canonical_fit']
        paired[str(seed)] = entry
    return {'schema': 'trinite.music-pitch-intervention-scores.v1',
        'diagnostic': diagnostic, 'development': development, 'paired_by_seed': paired,
        'historical_control_reproducibility': _historical_controls(development_records),
        'decodes': 800, 'decode_forwards': diagnostic['decode_forwards'] + development['decode_forwards'],
        'interpretation': 'descriptive paired replacement-allocation effects; no accuracy pass threshold or seed election',
        'fresh_confirmation': False, 'arbitrary_pitch_established': False,
        'arithmetic_transfer_established': False, 'learning_adequate': False,
        'phase5_ready': False, 'release_ready': False}
