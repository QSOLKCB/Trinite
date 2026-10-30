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
        input_strata = {name: {'pairs': 0, 'identical_raw_outputs': 0, 'both_correct': 0}
                        for name in ('changed_input', 'identical_input')}
        for row in siblings:
            base = canon_map[(row['group_id'], row['variant_id'])]
            left, right = outputs[base['example_identity']], outputs[row['example_identity']]
            identical += left == right
            correct = left == right == [*row['answer'].encode('ascii'), EOS]
            both += correct
            bucket = input_strata['identical_input' if base['prompt'] == row['prompt'] else 'changed_input']
            bucket['pairs'] += 1
            bucket['identical_raw_outputs'] += left == right
            bucket['both_correct'] += correct
        pairs[sibling] = {'pairs': len(siblings), 'identical_raw_outputs': identical,
                          'changed_raw_outputs': len(siblings) - identical, 'both_correct': both,
                          'by_input_change': input_strata}
    result = {'cell': name, 'stage': stage,
        'unique_prompt_surfaces': len({r['prompt'] for r in selected}),
        'rendering_rows': len(selected), 'raw': _row_metrics(selected, outputs),
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


def matrix_summary(diagnostic_records, development_records):
    diagnostic = score_records(diagnostic_records, 'diagnostic')
    development = score_records(development_records, 'development')
    paired = {}
    metrics = ('exact_plus_eos', 'complete_groups', 'both_correct_edges', 'changed_output_edges')
    for seed in SEEDS:
        entry = {}
        for stage, summary in (('diagnostic', diagnostic), ('development', development)):
            control = summary['by_cell'][cell_name('control', seed)]
            treatment = summary['by_cell'][cell_name('swap', seed)]
            entry[stage] = {
                'paired_transform': 'swap' if stage == 'diagnostic' else 'pitch',
                'swap_minus_control': {view: {m: treatment['by_sibling'][view][m] - control['by_sibling'][view][m]
                                              for m in metrics} for view in control['by_sibling']},
                'transformation_pair_both_correct_delta': treatment['transformation_pairs']['swap' if stage == 'diagnostic' else 'pitch']['both_correct'] - control['transformation_pairs']['swap' if stage == 'diagnostic' else 'pitch']['both_correct'],
                'transformation_pair_identical_delta': treatment['transformation_pairs']['swap' if stage == 'diagnostic' else 'pitch']['identical_raw_outputs'] - control['transformation_pairs']['swap' if stage == 'diagnostic' else 'pitch']['identical_raw_outputs'],
                'canonical_rest_only_exact_delta': treatment['canonical_strata']['variant_id']['10']['exact_plus_eos'] - control['canonical_strata']['variant_id']['10']['exact_plus_eos'],
                'canonical_exact_decreased': treatment['canonical_primary']['exact_plus_eos'] < control['canonical_primary']['exact_plus_eos'],
            }
        for bucket in ('changed_input', 'identical_input'):
            cc = diagnostic['by_cell'][cell_name('control', seed)]['transformation_pairs']['swap']['by_input_change'][bucket]
            pp = diagnostic['by_cell'][cell_name('swap', seed)]['transformation_pairs']['swap']['by_input_change'][bucket]
            entry['diagnostic'][bucket + '_swap_pairs'] = {
                'pairs_per_arm': cc['pairs'],
                'identical_output_delta': pp['identical_raw_outputs'] - cc['identical_raw_outputs'],
                'both_correct_delta': pp['both_correct'] - cc['both_correct']}
        dc = diagnostic['by_cell'][cell_name('control', seed)]
        dp = diagnostic['by_cell'][cell_name('swap', seed)]
        entry['control_canonical_training_fit'] = dc['canonical_fit']
        entry['swap_canonical_training_fit'] = dp['canonical_fit']
        entry['lost_control_perfect_training_fit'] = dc['canonical_fit'] and not dp['canonical_fit']
        paired[str(seed)] = entry
    return {'schema': 'trinite.music-placement-scores.v1',
        'diagnostic': diagnostic, 'development': development, 'paired_by_seed': paired,

        'decodes': 800, 'decode_forwards': diagnostic['decode_forwards'] + development['decode_forwards'],
        'interpretation': 'descriptive paired placement-template replacement effects; no accuracy pass threshold or seed election',
        'fresh_confirmation': False, 'arbitrary_pitch_established': False,
        'arithmetic_transfer_established': False, 'learning_adequate': False,
        'phase5_ready': False, 'release_ready': False}
