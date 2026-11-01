"""Full denominators, frozen training-only baselines, descriptive comparisons."""
from collections import Counter
from fractions import Fraction

import queue_evaluation_v1 as gate
from trinite.contracts import json_bytes
from . import data
from .numerical import parse_output

METHODS = ('main-17','main-29','majority','context-ignoring')
STATUSES = ('prediction','malformed','failure','timeout','not_attempted')


def ratio(n,d):
    return dict(numerator=n,denominator=d) if d else None


def majority(train):
    counts = Counter(r['answer'] for r in train)
    return min(counts,key=lambda x:(-counts[x],int(x)))


def shortcut(obs):
    # Deliberately ignores context possibilities: always uses rule row zero.
    # Still respects public symbol mask/query; no fitted/test constants.
    truths = {obs['table'][0][s] == obs['query'] for s,active in enumerate(obs['symbol_mask']) if active}
    return '0' if len(truths) == 2 else '1' if True in truths else '-1'


def prediction(answer):
    return parse_output([*answer.encode('ascii'),257])


def empty():
    return [dict(method=m,id=r['id'],raw=dict(status='not_attempted'))
        for m in METHODS for r in data.inventory() if r['split'] != 'train']


def check_predictions(predictions, complete=False):
    expected = [(r['method'],r['id']) for r in empty()]
    gate.require([(r['method'],r['id']) for r in predictions] == expected, 'prediction inventory/order mismatch')
    for row in predictions:
        raw = row['raw']
        gate.require(raw['status'] in STATUSES, 'unknown prediction status')
        if raw['status'] in ('prediction','malformed'):
            gate.require(json_bytes(parse_output(raw['tokens'])) == json_bytes(raw), 'raw tokens/status mismatch')
        else:
            gate.require(raw == dict(status=raw['status']), 'failure record fields')
        if complete:
            gate.require(raw['status'] in ('prediction','malformed'), 'all predictions required before scoring')


def context_pairs(rows):
    groups = {}
    for r in rows:
        req = {k:v for k,v in r['request'].items() if k != 'context_mask'}
        groups.setdefault(json_bytes(req),[]).append(r['id'])
    # Includes all pairwise context interventions, not only changed labels.
    from itertools import combinations
    return [p for ids in groups.values() for p in combinations(ids,2)]


def summary(predictions, targets=None):
    check_predictions(predictions,complete=targets is not None)
    by_key = {(r['method'],r['id']):r['raw'] for r in predictions}
    result = {}
    if targets is not None:
        by_key.update({('oracle',k):prediction(v) for k,v in targets.items()})
    for method in (*METHODS,'oracle'):
        result[method] = {}
        for split in ('validation','test'):
            rows = [r for r in data.inventory() if r['split'] == split]
            raw = [by_key.get((method,r['id']),dict(status='not_attempted')) for r in rows]
            n = len(rows)
            value = dict(planned=n,statuses={s:sum(x['status']==s for x in raw) for s in STATUSES},
                valid_output_coverage=ratio(sum(x['status']=='prediction' for x in raw),n),
                exact_accuracy=None,per_class_recall=None,macro_f1=None,context_switch_consistency=None,
                family_accuracy=None)
            if targets is not None:
                pred = {r['id']:by_key[method,r['id']].get('value') for r in rows}
                correct = sum(pred[r['id']] == targets[r['id']] for r in rows)
                recalls, f1 = {}, []
                for cls in ('-1','0','1'):
                    tp = sum(pred[r['id']] == cls == targets[r['id']] for r in rows)
                    support = sum(targets[r['id']] == cls for r in rows)
                    guessed = sum(pred[r['id']] == cls for r in rows)
                    recalls[cls] = ratio(tp,support)
                    f1.append(Fraction(2*tp,support+guessed) if support+guessed else Fraction(0))
                macro = sum(f1,Fraction(0))/3
                pairs = context_pairs(rows)
                changed = [p for p in pairs if targets[p[0]] != targets[p[1]]]
                both = lambda pairs: sum(pred[a] == targets[a] and pred[b] == targets[b] for a,b in pairs)
                families = sorted({r['family'] for r in rows})
                value.update(exact_accuracy=ratio(correct,n),per_class_recall=recalls,
                    macro_f1=ratio(macro.numerator,macro.denominator),
                    context_switch_consistency=dict(all_pairs=ratio(both(pairs),len(pairs)),
                        changed_label_pairs=ratio(both(changed),len(changed)),
                        definition='both answers correct; malformed/missing count as pair failure'),
                    family_accuracy={f:ratio(sum(pred[r['id']]==targets[r['id']] for r in rows if r['family']==f),
                                             sum(r['family']==f for r in rows)) for f in families})
            result[method][split] = value
    comparisons = {}
    if targets is not None:
        for split in ('validation','test'):
            def accuracy(m):
                q = result[m][split]['exact_accuracy']
                return Fraction(q['numerator'],q['denominator'])
            comparisons[split] = all(result[m][split]['statuses']['prediction']==result[m][split]['planned']
                and accuracy(m)>accuracy('majority') and accuracy(m)>accuracy('context-ignoring')
                for m in ('main-17','main-29'))
    return dict(schema='trinite.context-symbol-scores.v1',heldout_scored=targets is not None,
        methods=result,descriptive_gain_over_both_baselines=comparisons,
        oracle_role='correctness reference; no learned training budget; excluded from gain test',
        limits='two seeds and correlated public formal families; no significance or blindness claim')
