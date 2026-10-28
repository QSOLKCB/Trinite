"""Complete contrast sensitivity and invariant siblings, never answer repair."""
from collections import Counter, defaultdict
from trinite.contracts import ContractError, exact_keys, json_bytes, parse_json
from .spec import rows

EOS = 257


def _prediction_inventory(raw, selector):
    data = parse_json(raw, canonical=True)
    exact_keys(data, {'predictions'}, 'contrast predictions')
    selected = rows(selector, selector == 'heldout')
    expected = {r['example_identity']: r for r in selected}
    if type(data['predictions']) is not list or len(data['predictions']) != len(expected):
        raise ContractError('incomplete contrast predictions')
    outputs = {}
    for record in data['predictions']:
        exact_keys(record, {'example_identity','family_id','task','split','expected_tokens',
            'generated_tokens','correct','baseline_tokens','baseline_correct'}, 'contrast prediction')
        key = record['example_identity']
        if type(key) is not str or key not in expected or key in outputs:
            raise ContractError('unknown or duplicate contrast example')
        row = expected[key]
        tokens = record['generated_tokens']
        if type(tokens) is not list or not 1 <= len(tokens) <= 64 \
                or any(type(t) is not int or not 0 <= t <= 258 for t in tokens) \
                or EOS in tokens[:-1] or (tokens[-1] != EOS and len(tokens) != 64):
            raise ContractError('invalid raw EOS/cap generation')
        gold = [*row['answer'].encode('ascii'), EOS]
        correct = tokens == gold
        fields = {'family_id': row['family_id'], 'task': 'attacks', 'split': row['split'],
                  'expected_tokens': gold, 'correct': correct,
                  'baseline_tokens': [55, EOS], 'baseline_correct': gold == [55, EOS]}
        if any(json_bytes(record[k]) != json_bytes(v) for k,v in fields.items()):
            raise ContractError('contrast prediction context/score differs')
        outputs[key] = tokens
    return selected, outputs


def _metrics(selected, outputs):
    groups = defaultdict(dict)
    for r in selected:
        groups[(r['group_id'],r['sibling'])][r['variant_id']] = r
    correct = lambda r: outputs[r['example_identity']] == [*r['answer'].encode('ascii'), EOS]
    complete = 0
    both_correct = 0
    sensitivity = 0
    for quartet in groups.values():
        if set(quartet) != {'00','01','10','11'}:
            raise ContractError('incomplete contrast group')
        complete += all(correct(r) for r in quartet.values())
        # Two rest edits and two tie edits, all with changed oracle answers.
        for a,b in (('00','10'),('01','11'),('00','01'),('10','11')):
            left,right = quartet[a],quartet[b]
            both_correct += correct(left) and correct(right)
            sensitivity += outputs[left['example_identity']] != outputs[right['example_identity']]
    return {'rows': len(selected), 'exact_plus_eos': sum(correct(r) for r in selected),
        'groups': len(groups), 'complete_groups': complete,
        'contrast_edges': 4*len(groups), 'both_correct_edges': both_correct,
        'changed_output_edges': sensitivity,
        'eos_terminated': sum(outputs[r['example_identity']][-1] == EOS for r in selected),
        'generated_tokens': sum(len(outputs[r['example_identity']]) for r in selected)}


def _key(row, mode):
    if mode == 'global': return 'all'
    if mode == 'task_meter': return row['task']+':'+row['meter']
    if mode == 'header': return ';'.join(row['prompt'].split(';')[:4])
    if mode == 'length': return str(len(row['prompt']))
    if mode == 'header_length': return _key(row,'header')+':'+_key(row,'length')
    raise ContractError('unknown baseline')


def baselines(selected):
    train = rows()
    result = {}
    for mode in ('global','task_meter','header','length','header_length'):
        counts = defaultdict(Counter)
        for row in train: counts[_key(row,mode)][row['answer']] += 1
        fitted = {key:min(c, key=lambda a:(-c[a],a)) for key,c in counts.items()}
        outputs = {r['example_identity']:[*fitted.get(_key(r,mode),'7').encode('ascii'),EOS] for r in selected}
        result[mode] = {**_metrics(selected,outputs),
                       'unknown_key_fallback_rows':sum(_key(r,mode) not in fitted for r in selected),
                       'fit_scope':'unique canonical TRAIN rows only; unknown keys use train global majority7'}
    oracle = {r['example_identity']:[*r['answer'].encode('ascii'),EOS] for r in selected}
    result['exact_formal_oracle'] = {**_metrics(selected,oracle),
        'scope':'post-generation scoring ceiling, never model input or generated answer repair'}
    return result


def summarize(raw, selector):
    selected, outputs = _prediction_inventory(raw,selector)
    result = {'all':_metrics(selected,outputs),
        'by_sibling':{s:_metrics([r for r in selected if r['sibling']==s],outputs)
                      for s in sorted({r['sibling'] for r in selected})},
        'by_meter':{m:_metrics([r for r in selected if r['meter']==m],outputs)
                    for m in sorted({r['meter'] for r in selected})},
        'baselines':baselines(selected)}
    if selector == 'heldout':
        triples=defaultdict(list)
        for r in selected:triples[(r['group_id'],r['variant_id'])].append(r)
        result['invariance']={'triples':len(triples),
            'identical_raw_outputs':sum(len({tuple(outputs[r['example_identity']]) for r in triple})==1 for triple in triples.values()),
            'all_three_correct':sum(all(outputs[r['example_identity']]==[*r['answer'].encode('ascii'),EOS] for r in triple) for triple in triples.values())}
    else:
        result['invariance']={'triples':0,'identical_raw_outputs':0,'all_three_correct':0}
    return result
