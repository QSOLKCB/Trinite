"""New-producer descriptive adapter over retained scores; no extra inference.

The historical failure-map classifier is pure and reused unchanged. Its fixed
producer importer and full-domain report are deliberately NOT used for this lane.
"""
from collections import Counter
from trinite.contracts import ContractError
from failure_map_v1.measurements import classify, CATEGORIES
from .data import examples
from .protocol import STRATA


def counts(rows):
    families = {}
    for row in rows:
        families.setdefault(row['family_id'], []).append(row)
    return {'rows': len(rows), 'families': len(families),
            'correct': sum(r['prediction']['correct'] for r in rows),
            'all_renderings_correct_families': sum(all(r['prediction']['correct'] for r in group)
                                                   for group in families.values()),
            'status': 'observed' if rows else 'untested_no_admitted_cases',
            'categories': {c: sum(r['category'] == c for r in rows) for c in CATEGORIES}}


def pairs(rows):
    """Subset-local carrier and nontrivial order pairs; consistency is not accuracy."""
    by = {(r['a'], r['b'], r['carrier']): r for r in rows}
    groups = {'carrier': [], 'operand_order': []}
    def pair(left, right):
        lp, rp = left['prediction'], right['prediction']
        same = lp['generated_tokens'] == rp['generated_tokens']
        both = lp['correct'] and rp['correct']
        category = ('both_correct' if both else 'both_wrong_same' if same else
                    'both_wrong_different' if not lp['correct'] and not rp['correct'] else 'one_correct')
        return {'left': left['example_identity'], 'right': right['example_identity'],
                'consistent': same, 'both_correct': both, 'category': category}
    for (a, b, carrier), row in sorted(by.items()):
        if carrier == 'infix':
            groups['carrier'].append(pair(row, by[a,b,'fields']))
        if a < b:
            groups['operand_order'].append(pair(row, by[b,a,carrier]))
    return {name: {'pairs': values, 'pair_count': len(values),
                   'consistent': sum(p['consistent'] for p in values),
                   'both_correct': sum(p['both_correct'] for p in values),
                   'categories': dict(sorted(Counter(p['category'] for p in values).items()))}
            for name, values in groups.items()}


def annotate(arm, items, record):
    by_id = {r['example_identity']: r for r in items}
    rows = []
    for prediction in record['predictions']:
        item = by_id[prediction['example_identity']]
        key = ('answer_seen_in_both_training_arms' if arm == 'control'
               else 'answer_seen_in_candidate_training')
        a, b = item['formal']['a'], item['formal']['b']
        seen = arm in item['training_arms']
        rows.append({'example_identity': item['example_identity'], 'family_id': item['family_id'],
            'a': a, 'b': b, 'carrier': item['carrier'], 'prompt': item['prompt'],
            'answer': item['answer'], 'original_split': item['split'], 'stratum': item['stratum'],
            'seen_in_this_arm': seen, 'answer_seen_in_training': item['coverage'][key],
            'scientific_exposure': 'seen_training_anchor' if seen else 'shared_development_not_fresh_confirmation',
            'joint_range_digit_length_shift': max(a,b) >= 10,
            'prediction': prediction, 'category': classify(prediction['generated_tokens'],
                                                           prediction['expected_tokens'])})
    if len(rows) != len(by_id) or {p['example_identity'] for p in rows} != set(by_id):
        raise ContractError('balanced descriptive coverage differs')
    return rows


def group(rows):
    return {'total': counts(rows), 'metamorphic': pairs(rows),
            'by_answer': {str(v): counts([p for p in rows if p['answer'] == str(v)])
                          for v in range(25)},
            'by_answer_support': {label: counts([p for p in rows
                                   if p['answer_seen_in_training'] == seen_answer])
                                  for label, seen_answer in (('seen', True), ('unseen', False))}}


def describe(arm, training_record, development_records):
    """Keep exposed training and identical shared-development facts separate."""
    if set(development_records) != set(STRATA):
        raise ContractError('balanced descriptive strata differ')
    train = annotate(arm, examples(arm), training_record)
    anchors = {r['example_identity'] for r in examples('control')}
    seen = [p for p in train if p['example_identity'] in anchors]
    if {p['example_identity'] for p in seen} != anchors:
        raise ContractError('balanced original-control anchor coverage differs')
    development = {s: annotate(arm, examples(s), development_records[s]) for s in STRATA}
    in_range = development['shared-answer'] + development['broadened-answer']
    return {'schema': 'trinite.balanced-addition-description.v1',
            'original_control_seen_anchors': group(seen),
            'unique_training': group(train),
            'anchor_scope': '20 already trained renderings, not generalisation or old-checkpoint replay',
            'primary_in_range_development': group(in_range),
            'shared_development': {s: group(rows) for s, rows in development.items()},
            'rows': {'train': train, **development},
            'unobserved': 'control does not score candidate-only training families; no pooled338 comparison',
            'scope': 'retained raw predictions only; empty groups untested; families and renderings correlated; no selection; consistency is not correctness',
            'additional_inference_executed': False}
