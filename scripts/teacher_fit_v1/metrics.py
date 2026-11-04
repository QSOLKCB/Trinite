# SPDX-License-Identifier: MPL-2.0
"""Recomputable TRAIN diagnostics from complete native vocabulary observations."""
from collections import Counter
import math

from trinite.contracts import ContractError, json_bytes
from teacher_diag_v1.core import classify
from teacher_logit_v1 import arithmetic as ar

BOS, EOS = 256, 257


def task_name(row):
    task = row['observation']['task']
    return 'check_'+row['answer'] if task == 'check' else task


def score(panel, predictions):
    if len(predictions) != len(panel) or len({p['id'] for p in predictions}) != len(panel):
        raise ContractError('prediction inventory missing or duplicated')
    groups = {}
    classified = []
    for row, pred in zip(panel, predictions):
        tokens = pred['tokens']
        if (set(pred) != {'id','tokens'} or pred['id'] != row['id'] or not tokens
                or (EOS in tokens and (tokens[-1] != EOS or tokens.count(EOS) != 1))
                or (EOS not in tokens and len(tokens) != 5)):
            raise ContractError('noncanonical complete greedy output')
        result = classify(tokens, row['answer'], row['observation']['task'])
        classified.append(result)
        for key in ('all', task_name(row), 'circuit:'+row['family']):
            group = groups.setdefault(key, {'rows':0,'exact':0,'valid':0,'premature_eos':0,
                'missing_eos':0,'invalid':0,'wrong_length':0,'eos_positions':{},'outputs':{}})
            group['rows'] += 1
            group['exact'] += int(result['exact'])
            group['valid'] += int(result['format_valid'])
            eos_pos = tokens.index(EOS) if EOS in tokens else None
            group['premature_eos'] += int(eos_pos is not None and eos_pos < len(row['answer']))
            group['missing_eos'] += int(eos_pos is None)
            group['invalid'] += int(result['classification'] == 'invalid_symbol')
            group['wrong_length'] += int(result['answer_length'] != len(row['answer']))
            eos_key = str(eos_pos)
            group['eos_positions'][eos_key] = group['eos_positions'].get(eos_key,0)+1
            sequence = ','.join(map(str,tokens))
            group['outputs'][sequence] = group['outputs'].get(sequence,0)+1
    for group in groups.values():
        group['distinct_outputs'] = len(group['outputs'])
    positions = []
    vectors = [(r,p) for r,p in zip(panel,predictions) if r['observation']['task'] != 'check']
    for pos in range(4):
        positions.append({'position':pos,'rows':len(vectors),
            'targets':dict(sorted(Counter(r['answer'][pos] for r,p in vectors).items())),
            'correct':sum(len(p['tokens'])>pos and p['tokens'][pos]==ord(r['answer'][pos]) for r,p in vectors),
            'nonzero_targets':sum(r['answer'][pos]!='0' for r,p in vectors),
            'correct_nonzero':sum(r['answer'][pos]!='0' and len(p['tokens'])>pos
                                  and p['tokens'][pos]==ord(r['answer'][pos]) for r,p in vectors),
            'zero_outputs':sum(len(p['tokens'])>pos and p['tokens'][pos]==48 for r,p in vectors)})
    return {'groups':groups,'vector_positions':positions,'classified':classified,
            'exact_fit':groups['all']['exact']==72 and groups['all']['rows']==72}


def baselines(panel):
    return {str(value):score(panel,[{'id':r['id'],'tokens':
        [48+value]*len(r['answer'])+[EOS]} for r in panel]) for value in (0,1)}


def entry(row, probe, position, prefix, offset):
    gold = [*row['answer'].encode('ascii'), EOS]
    return {'row_id':row['id'],'probe':probe,'position':position,'prefix_tokens':list(prefix),
            'target':gold[position] if position < len(gold) else None,'offset_bytes':offset}


def observe(panel, forward, retain=None, retain_prediction=None):
    """forward(prefix) returns exact FP32 bytes; no RNG, sampling or interventions."""
    index, vectors, measures, predictions = [], bytearray(), [], []
    def consume(row, probe, pos, prefix):
        raw = forward(prefix)
        rec = entry(row,probe,pos,prefix,len(vectors))
        measure = ar.metrics(ar.unpack(raw),rec['target'])
        if retain is not None:
            retain(rec,raw,measure)
        index.append(rec); vectors.extend(raw); measures.append(measure)
        return measure['argmax']
    for row in panel:
        prompt = [BOS,*row['prompt'].encode('ascii')]
        gold = [*row['answer'].encode('ascii'),EOS]
        for pos in range(len(gold)):
            consume(row,'gold',pos,prompt+gold[:pos])
        generated = []
        for pos in range(5):
            chosen = consume(row,'greedy',pos,prompt+generated)
            generated.append(chosen)
            if chosen == EOS:
                break
        predictions.append({'id':row['id'],'tokens':generated})
        if retain_prediction is not None:
            retain_prediction(predictions[-1])
    return index, bytes(vectors), measures, predictions


def recalculate(panel, predictions, index, vectors):
    """Rebuild every position and greedy choice; never trust recorded aggregates."""
    score(panel,predictions)
    expected = []
    for row,pred in zip(panel,predictions):
        prompt = [BOS,*row['prompt'].encode('ascii')]
        gold = [*row['answer'].encode('ascii'),EOS]
        for probe,tokens in (('gold',gold),('greedy',pred['tokens'])):
            for pos in range(len(tokens)):
                expected.append(entry(row,probe,pos,prompt+tokens[:pos],len(expected)*ar.VECTOR.size))
    if json_bytes(index) != json_bytes(expected) or len(vectors) != len(expected)*ar.VECTOR.size:
        raise ContractError('checkpoint position inventory or vector length mismatch')
    measures = []
    by_id = {p['id']:p['tokens'] for p in predictions}
    for rec in index:
        start = rec['offset_bytes']
        measure = ar.metrics(ar.unpack(vectors[start:start+ar.VECTOR.size]), rec['target'])
        if rec['probe']=='greedy' and measure['argmax'] != by_id[rec['row_id']][rec['position']]:
            raise ContractError('greedy output differs from full vocabulary argmax')
        measures.append(measure)
    return measures


def summarize(panel, predictions, index, measures):
    result = score(panel,predictions)
    by_id = {r['id']:r for r in panel}
    groups = {}
    residue = [dict(position=p,total=0,correct_best_residue=0,correct_non_eos=0,
        nonzero_targets=0,correct_nonzero_best_residue=0,zero_best_residue=0) for p in range(4)]
    for rec,m in zip(index,measures):
        if rec['probe'] != 'gold':
            continue
        row = by_id[rec['row_id']]
        for key in ('all',task_name(row),'target:'+str(rec['target']),
                    'terminal_eos' if rec['target']==EOS else 'answer_position'):
            groups.setdefault(key,[]).append(m)
        if row['observation']['task']!='check' and rec['position']<4:
            g=residue[rec['position']]; target=rec['target']
            g['total']+=1; g['correct_best_residue']+=int(m['best_residue_token']==target)
            g['correct_non_eos']+=int(m['target_is_best_non_eos'])
            g['nonzero_targets']+=int(target!=48)
            g['correct_nonzero_best_residue']+=int(target!=48 and m['best_residue_token']==target)
            g['zero_best_residue']+=int(m['best_residue_token']==48)
    result['gold_prefix'] = {key:{'positions':len(ms),
        'mean_target_nll_hex':(math.fsum(float.fromhex(m['target_nll_hex']) for m in ms)/len(ms)).hex(),
        'mean_eos_target_margin_hex':(math.fsum(float.fromhex(m['eos_target_margin_hex']) for m in ms)/len(ms)).hex(),
        'mean_eos_rest_log_odds_hex':(math.fsum(float.fromhex(m['eos_rest_log_odds_hex']) for m in ms)/len(ms)).hex(),
        'target_argmax':sum(m['target_rank']==1 for m in ms),
        'zero_best_residue':sum(m['best_residue_token']==48 for m in ms)} for key,ms in sorted(groups.items())}
    result['gold_vector_positions']=residue
    result['baseline_exact']={v:s['groups']['all']['exact'] for v,s in baselines(panel).items()}
    result['improvement_over_baseline']={v:result['groups']['all']['exact']-n
        for v,n in result['baseline_exact'].items()}
    return result
