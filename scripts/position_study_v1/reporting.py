"""Descriptive metrics from retained outputs only, never generation feedback."""
from collections import Counter

from position_coupling_v1.positions import decode_answer, prefix_positions
from trinite.tokenizer import EOS
from trinite.contracts import ContractError, identity, json_bytes
from .data import examples
from .protocol import STRATA, PROTOCOL, SEEDS

CATEGORIES = ('exact_correct','wrong_value','immediate_eos','premature_eos',
              'malformed_output','missing_eos')


def output_flags(tokens, expected):
    body = tokens[:-1] if tokens and tokens[-1] == EOS else tokens
    return {'missing_eos': EOS not in tokens,
        'early_eos': EOS in tokens and len(body) < len(expected)-1,
        'malformed': any(not 48<=t<=57 for t in body),
        'wrong_digit_count':len(body) != len(expected)-1}


def classify(tokens, expected):
    # New representation semantics: leading zeroes are valid reversed fixed-width
    # digits. Historical failure-map grammar/cap5 cannot classify this lane.
    if tokens == expected: return 'exact_correct'
    flags = output_flags(tokens,expected)
    if tokens == [EOS]: return 'immediate_eos'
    if flags['missing_eos']: return 'missing_eos'
    if flags['malformed']: return 'malformed_output'
    if flags['early_eos']: return 'premature_eos'
    return 'wrong_value'


def annotate(items, record):
    by_id = {r['example_identity']:r for r in items}
    result = []
    for p in record['predictions']:
        item = by_id[p['example_identity']]
        fixed = decode_answer(item['prompt'],tuple(p['generated_tokens']))
        tokens=p['generated_tokens']
        body=tokens[:-1] if tokens and tokens[-1]==EOS else tokens
        decoded=(str(int(''.join(chr(t) for t in reversed(body))))
                 if body and all(48<=t<=57 for t in body) else None)
        result.append({**item,'prediction':p,'fixed_width_decoded':fixed,'decoded_numeric':decoded,
            'numeric_correct': decoded is not None and int(decoded)==item['formal']['a']+item['formal']['b'],
            'output_flags':output_flags(p['generated_tokens'],p['expected_tokens']),
            'category':classify(p['generated_tokens'],p['expected_tokens'])})
    return result


def counts(rows):
    families = {}
    for r in rows: families.setdefault(r['family_id'],[]).append(r)
    return {'rows':len(rows),'families':len(families),
        'correct':sum(r['prediction']['correct'] for r in rows),
        'numeric_correct':sum(r['numeric_correct'] for r in rows),
        'all_renderings_correct_families':sum(all(r['prediction']['correct'] for r in rs) for rs in families.values()),
        'status':'observed' if rows else 'untested_no_admitted_cases',
        'output_flags':{f:sum(r['output_flags'][f] for r in rows)
            for f in ('missing_eos','early_eos','malformed','wrong_digit_count')},
        'categories':{c:sum(r['category']==c for r in rows) for c in CATEGORIES}}


def perturbation_inventory(rows):
    pairs = {'operand_reversal':[],'equal_sum':[],'increment':[],'padded_rendering':[]}
    for i,left in enumerate(rows):
        a,b = left['formal']['a'],left['formal']['b']
        for right in rows[i+1:]:
            c,d = right['formal']['a'],right['formal']['b']
            kind = None
            if (a,b)==(c,d) and left['width'] != right['width']:
                kind = 'padded_rendering'
            elif left['width'] == right['width']:
                if (a,b)==(d,c) and a!=b: kind = 'operand_reversal'
                elif left['family_id']!=right['family_id'] and a+b==c+d: kind = 'equal_sum'
                elif (a==c and abs(b-d)==1) or (b==d and abs(a-c)==1): kind = 'increment'
            if kind is None: continue
            delta = c+d-a-b
            pairs[kind].append({'left':left['example_identity'],'right':right['example_identity'],
                                'expected_numeric_delta':delta})
    return pairs


def perturbations(rows):
    by_id={r['example_identity']:r for r in rows}
    pairs=perturbation_inventory(rows)
    for kind,items in pairs.items():
        for pair in items:
            left,right=by_id[pair['left']],by_id[pair['right']]
            lv,rv = left['decoded_numeric'],right['decoded_numeric']
            delta=pair['expected_numeric_delta']
            numeric_relation = lv is not None and rv is not None and int(rv)-int(lv)==delta
            if kind == 'padded_rendering': consistent = numeric_relation
            elif kind == 'increment':
                # Sensitivity is separately visible even for malformed/wrong output.
                consistent = left['prediction']['generated_tokens'] != right['prediction']['generated_tokens']
            else: consistent = left['prediction']['generated_tokens'] == right['prediction']['generated_tokens']
            both = left['prediction']['correct'] and right['prediction']['correct']
            pair.update({'consistent':consistent,
                'numeric_relation':numeric_relation,'both_correct':both,
                'consistent_and_correct':consistent and both})
    return {kind:{'pairs':ps,'pair_count':len(ps),
        'consistent':sum(p['consistent'] for p in ps),
        'numeric_relation':sum(p['numeric_relation'] for p in ps),
        'both_correct':sum(p['both_correct'] for p in ps),
        'consistent_and_correct':sum(p['consistent_and_correct'] for p in ps)} for kind,ps in pairs.items()}


def design_summary():
    """No model execution: exact paired-probe and input-address inventories."""
    result={}
    for s in ('train',*STRATA):
        rows=examples(s);pairs=perturbation_inventory(rows)
        result[s]={'rows':len(rows),'families':len({r['family_id'] for r in rows}),
            'width_histogram':dict(sorted(Counter(str(r['width']) for r in rows).items())),
            'pair_counts':{k:len(v) for k,v in pairs.items()},
            'pair_inventory_identity':identity(json_bytes(pairs)),
            'full_prefix_max_length':max(1+len(r['prompt'])+r['width']+2 for r in rows),
            'position_rows_at_full_cap':{mode:sorted({p for r in rows for p in
                prefix_positions(r['prompt'],r['width']+2,mode=mode)}) for mode in ('sequence','coupled')}}
        if result[s]['pair_counts'] != PROTOCOL['perturbation_pair_counts'][s]:
            raise ContractError('frozen perturbation pair inventory differs')
    return result


def group(rows):
    features = ('zero_operand','repeated_digits','carry_count','longest_carry_chain','historical_0_to_12')
    families = {r['family_id'] for r in rows}
    return {'total':counts(rows), 'perturbations':perturbations(rows),
        'family_results':{f:counts([r for r in rows if r['family_id']==f]) for f in sorted(families)},
        'by_width':{str(w):counts([r for r in rows if r['width']==w]) for w in range(1,5)},
        'by_feature':{f:{str(v):counts([r for r in rows if r['features'][f]==v])
            for v in ((False,True) if f in ('zero_operand','repeated_digits','historical_0_to_12') else range(5))}
            for f in features}}


def describe(arm, training_record, development_records):
    train = annotate(examples(arm),training_record)
    dev = {s:annotate(examples(s),development_records[s]) for s in STRATA}
    return {'schema':'trinite.position-study-description.v1',
        'seen_training_anchors':group(train),
        'development':{s:group(rs) for s,rs in dev.items()},
        'scope':'all metrics derived from retained raw tokens; correlated renderings/families, no fresh confirmation or additional inference',
        'additional_inference_executed':False}


def controls(arm, stratum):
    train,rows = examples(arm),examples(stratum)
    lookup = {r['prompt']:r['answer'] for r in train}
    most = min(Counter(r['answer'] for r in train),key=lambda s:(-sum(r['answer']==s for r in train),s))
    return {'exact_oracle_correct':len(rows),
        'training_lookup_correct':sum(lookup.get(r['prompt'])==r['answer'] for r in rows),
        'training_lookup_misses':sum(r['prompt'] not in lookup for r in rows),
        'constant_zero_correct':sum(int(r['answer'][::-1])==0 for r in rows),
        'training_majority_answer':most,
        'training_majority_correct':sum(r['answer']==most for r in rows),
        'constant_eos_correct':0,
        'scope':'non-neural controls, not fallback/repair; exact zero control uses input-derived width and EOS'}


def criteria(descriptions):
    """Integer comparisons over family denominators; no automatic research gate."""
    frozen = PROTOCOL['criteria']; pairs = {}
    for seed in SEEDS:
        left = descriptions.get(f'control-{seed}'); right = descriptions.get(f'candidate-{seed}')
        if left is None or right is None: continue
        lc = left['seen_training_anchors']['total']; rc = right['seen_training_anchors']['total']
        n = rc['families']; c = rc['all_renderings_correct_families']; l = lc['all_renderings_correct_families']
        result = {'seen_candidate_floor':100*c >= frozen['minimum_candidate_seen_percent']*n,
            'seen_regression_within_allowance':100*(l-c) <= frozen['maximum_seen_regression_percentage_points']*n}
        for s in STRATA:
            a,b = left['development'][s]['total'],right['development'][s]['total']
            n = b['families']; gain = b['all_renderings_correct_families']-a['all_renderings_correct_families']
            result[s+'_gain'] = 100*gain >= frozen['minimum_each_seed_unseen_gain_percentage_points' if s=='unseen' else 'minimum_each_seed_length_gain_percentage_points']*n
            if s!='unseen': result[s+'_floor'] = 100*b['all_renderings_correct_families'] >= frozen['minimum_candidate_each_seed_length_percent']*n
        pairs[str(seed)] = result
    return {'per_seed':pairs,'all_descriptive_criteria_met':len(pairs)==len(SEEDS) and all(all(v.values()) for v in pairs.values()),
        'interpretation':'descriptive conditions only; mixed/negative conditions retained; no learning, confirmation or release gate'}
