"""Deterministic support/error descriptions; never training, inference or causal proof.

The entrypoint wrapper authenticates this module's complete source/input closure
before invocation and rechecks it afterwards. Only previously retained outputs
are joined here; the finite placement universe is a support inventory, not data
for a new experiment.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import json

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from music_contrast_v1 import spec as contrast
from music_contrast_v1.reporting import _metrics
from music_exposure_v1 import spec as exposure
from music_pitch_intervention_v1 import spec as pitch
from music_placement_v1 import spec as placement
from music_learning_v1.codec import encode_example, parse_prompt
from symbolic_music_v1.core import from_dict, labels
from symbolic_music_v1.oracle import annotate

ROLES = ('rest', 'tie_start', 'tie_end', 'attack')
VARIANTS = ('00', '01', '10', '11')
EOS = 257


def _inputs():
    entries=[]
    for pr, folder, steps in (('PR91','music-contrast-result-20261028-v1',(0,64)),
                               ('PR93','music-exposure-result-20261029-v1',(0,64,640))):
        for seed in (17,29):
            for step in steps:
                entries.append({'path':f'fixtures/{folder}/train-{seed}--step-{step:03d}-journal.jsonl',
                                'evidence':pr,'seed':seed,'step':step,'stage':'train','arm':'control','backend':'native_cpu'})
    entries.append({'path':'fixtures/frozen-music-eval-result-v1/original-attempt/predictions.jsonl','evidence':'PR95','arm':'control','backend':'native_cpu'})
    for pr,folder,arms in (('PR97','music-pitch-result-v1/original-run',('control','pitch')),
                           ('PR99_cpu','music-placement-result-v1/cpu',('control','swap')),
                           ('PR101_cpu','music-robustness-result-v1/cpu',('control','swap'))):
        for arm in arms:
            for seed in (17,29):
                entries.append({'path':f'fixtures/{folder}/{arm}-{seed}/diagnostic-predictions.jsonl',
                                'evidence':pr,'backend':'native_cpu'})
        entries.append({'path':f'fixtures/{folder}/development/predictions.jsonl','evidence':pr,'backend':'native_cpu'})
    for backend in ('cpu','cuda'):
        entries.append({'path':f'fixtures/music-placement-result-v1/device/{backend}/diagnostic-predictions.jsonl',
                        'evidence':'PR99_device','arm':'control'})
    entries.append({'path':'fixtures/music-placement-result-v1/device/development/predictions.jsonl','evidence':'PR99_device','arm':'control'})
    return entries


INPUTS = _inputs()
INPUT_PATHS = ('fixtures/music-contrast-result-20261028-v1/train-17--step-000-journal.jsonl', 'fixtures/music-contrast-result-20261028-v1/train-17--step-064-journal.jsonl', 'fixtures/music-contrast-result-20261028-v1/train-29--step-000-journal.jsonl', 'fixtures/music-contrast-result-20261028-v1/train-29--step-064-journal.jsonl', 'fixtures/music-exposure-result-20261029-v1/train-17--step-000-journal.jsonl', 'fixtures/music-exposure-result-20261029-v1/train-17--step-064-journal.jsonl', 'fixtures/music-exposure-result-20261029-v1/train-17--step-640-journal.jsonl', 'fixtures/music-exposure-result-20261029-v1/train-29--step-000-journal.jsonl', 'fixtures/music-exposure-result-20261029-v1/train-29--step-064-journal.jsonl', 'fixtures/music-exposure-result-20261029-v1/train-29--step-640-journal.jsonl', 'fixtures/frozen-music-eval-result-v1/original-attempt/predictions.jsonl', 'fixtures/music-pitch-result-v1/original-run/control-17/diagnostic-predictions.jsonl', 'fixtures/music-pitch-result-v1/original-run/control-29/diagnostic-predictions.jsonl', 'fixtures/music-pitch-result-v1/original-run/pitch-17/diagnostic-predictions.jsonl', 'fixtures/music-pitch-result-v1/original-run/pitch-29/diagnostic-predictions.jsonl', 'fixtures/music-pitch-result-v1/original-run/development/predictions.jsonl', 'fixtures/music-placement-result-v1/cpu/control-17/diagnostic-predictions.jsonl', 'fixtures/music-placement-result-v1/cpu/control-29/diagnostic-predictions.jsonl', 'fixtures/music-placement-result-v1/cpu/swap-17/diagnostic-predictions.jsonl', 'fixtures/music-placement-result-v1/cpu/swap-29/diagnostic-predictions.jsonl', 'fixtures/music-placement-result-v1/cpu/development/predictions.jsonl', 'fixtures/music-robustness-result-v1/cpu/control-17/diagnostic-predictions.jsonl', 'fixtures/music-robustness-result-v1/cpu/control-29/diagnostic-predictions.jsonl', 'fixtures/music-robustness-result-v1/cpu/swap-17/diagnostic-predictions.jsonl', 'fixtures/music-robustness-result-v1/cpu/swap-29/diagnostic-predictions.jsonl', 'fixtures/music-robustness-result-v1/cpu/development/predictions.jsonl', 'fixtures/music-placement-result-v1/device/cpu/diagnostic-predictions.jsonl', 'fixtures/music-placement-result-v1/device/cuda/diagnostic-predictions.jsonl', 'fixtures/music-placement-result-v1/device/development/predictions.jsonl')


def _ratio(value):
    return f'{value.numerator}/{value.denominator}'


def row_features(row):
    """Validate exact carrier/target masks and describe role-specific contexts."""
    formal=row['formal']; parsed=parse_prompt(row['prompt'])
    oracle=annotate(formal)
    if (parsed['score']!=formal or parsed['task']!='attacks' or oracle!=labels(from_dict(formal))
            or str(oracle['attacks'])!=row['answer']):
        raise ContractError('audit oracle/carrier disagreement')
    adapted={k:row[k] for k in ('task','prompt','answer','text','answer_start')}
    encoded=encode_example(adapted)
    expected=[*row['answer'].encode('ascii'),EOS]
    masked=[token for token,mask in zip(encoded.input_ids,encoded.loss_mask) if mask]
    if (masked!=expected or sum(encoded.loss_mask)!=2 or row['answer_start']!=len(row['prompt'].encode('ascii'))
            or tuple(encoded.input_ids)!=(256,*row['text'].encode('ascii'),EOS)
            or tuple(encoded.loss_mask)!=(0,*([0]*len(row['prompt'])),1,1)
            or not all(encoded.attention_mask)):
        raise ContractError('audit actual prompt/target/mask alignment differs')
    events=[event for bar in formal['bars'] for event in bar]
    durations=[Fraction(*event['duration']) for event in events]
    bar_length=Fraction(formal['meter']['numerator'],formal['meter']['denominator'])
    onset=Fraction(0);contexts=[]
    for index,event in enumerate(events):
        roles=[]
        if event['pitch'] is None:roles.append('rest')
        if event['tie_out']:roles.append('tie_start')
        if event['tie_in']:roles.append('tie_end')
        if event['pitch'] is not None and not event['tie_in']:roles.append('attack')
        contexts.append({'absolute_index':index,'bar_index':index//4,'bar_relative_index':index%4,
            'onset':_ratio(onset),'bar_relative_onset':_ratio(onset-(index//4)*bar_length),
            'normalized_bar_onset':_ratio((onset-(index//4)*bar_length)/bar_length),
            'duration':_ratio(durations[index]),'normalized_duration':_ratio(durations[index]/bar_length),
            'first_event':index==0,'last_event':index==7,'bar_start':index%4==0,'bar_end':index%4==3,
            'tie_crosses_bar':bool((event['tie_out'] and index%4==3) or (event['tie_in'] and index%4==0)),
            'previous_duration':None if index==0 else _ratio(durations[index-1]),
            'next_duration':None if index==7 else _ratio(durations[index+1]),'roles':roles})
        onset+=durations[index]
    return {'example_identity':row['example_identity'],'parent':row['parent_id'],'meter':row['meter'],
        'variant':row['variant_id'],'sibling':row['sibling'],'answer':row['answer'],
        'prompt_bytes':len(row['prompt'].encode('ascii')),'prompt_sha256':identity(row['prompt'].encode('ascii')),
        'answer_start_byte':row['answer_start'],'masked_target_token_positions':[i for i,mask in enumerate(encoded.loss_mask) if mask],
        'attended_tokens':sum(encoded.attention_mask),'masked_target_tokens':expected,
        'rest_present':any('rest' in event['roles'] for event in contexts),
        'tie_present':any('tie_start' in event['roles'] for event in contexts),'events':contexts}


def support(rows, features):
    result={'rows':len(rows),'unique_prompts':len({r['prompt'] for r in rows}),
            'parents':sorted({r['parent_id'] for r in rows}),
            'answer_histogram':dict(sorted(Counter(r['answer'] for r in rows).items())),
            'variant_histogram':dict(sorted(Counter(r['variant_id'] for r in rows).items())), 'roles':{}}
    for role in ROLES:
        contexts=[event for row in rows for event in features[row['example_identity']]['events'] if role in event['roles']]
        present=sum(any(role in event['roles'] for event in features[row['example_identity']]['events']) for row in rows)
        result['roles'][role]={'present_rows':present,'absent_rows':len(rows)-present,
            **{key:sorted({event[key] for event in contexts},key=lambda x:(x is not None,str(x))) for key in
               ('absolute_index','bar_relative_index','onset','bar_relative_onset','normalized_bar_onset','duration','normalized_duration','previous_duration','next_duration','first_event','last_event','bar_start','bar_end','tie_crosses_bar')}}
    result['joint_rest_tie_placements']=sorted({(next(event['absolute_index'] for event in features[r['example_identity']]['events'] if 'rest' in event['roles']),
         next(event['absolute_index'] for event in features[r['example_identity']]['events'] if 'tie_start' in event['roles']))
         for r in rows if r['variant_id']=='11'})
    result['joint_rest_tie_placements']=[{'rest_index':rest,'tie_start_index':tie} for rest,tie in result['joint_rest_tie_placements']]
    return result


def finite_placement_universe():
    """All 7 adjacent tie starts × 6 disjoint rest positions, no model calls."""
    base=deepcopy(next(r for r in contrast.rows() if r['variant_id']=='11')['formal'])
    result=[]
    for tie in range(7):
        for rest in range(8):
            if rest in (tie,tie+1):continue
            formal=deepcopy(base)
            for index,event in enumerate(event for bar in formal['bars'] for event in bar):
                event['pitch']=None if index==rest else 60
                event['tie_out']=index==tie;event['tie_in']=index==tie+1
            expected=annotate(formal)
            if expected!=labels(from_dict(formal)) or expected['attacks']!=6:raise ContractError('support universe invalid')
            result.append({'rest_index':rest,'tie_start_index':tie,'tie_end_index':tie+1,'cross_bar_tie':tie==3})
    if len(result)!=42:raise ContractError('finite support cardinality differs')
    return {'scope':'support-only enumeration; not generated curriculum, model evaluation or a proposed split',
            'valid_adjacent_tie_plus_disjoint_rest_placements':42,'without_cross_bar_ties':36,'placements':result}


def _context_key(event):
    return (event['absolute_index'],event['bar_relative_index'],event['onset'],event['duration'],
            event['previous_duration'],event['next_duration'])


def support_difference(train, development, features):
    result={}
    for role in ('rest','tie_start','tie_end'):
        left=[e for r in train for e in features[r['example_identity']]['events'] if role in e['roles']]
        right=[e for r in development for e in features[r['example_identity']]['events'] if role in e['roles']]
        result[role]={'train_absolute_indices':sorted({e['absolute_index'] for e in left}),
            'development_absolute_indices':sorted({e['absolute_index'] for e in right}),
            'unseen_development_absolute_indices':sorted({e['absolute_index'] for e in right}-{e['absolute_index'] for e in left}),
            'train_bar_relative_indices':sorted({e['bar_relative_index'] for e in left}),
            'development_bar_relative_indices':sorted({e['bar_relative_index'] for e in right}),
            'unseen_development_bar_relative_indices':sorted({e['bar_relative_index'] for e in right}-{e['bar_relative_index'] for e in left}),
            'development_event_occurrences':len(right),
            'coarse_absolute_rational_onset_overlap':sorted({e['onset'] for e in left}&{e['onset'] for e in right}),
            'coarse_bar_relative_rational_onset_overlap':sorted({e['bar_relative_onset'] for e in left}&{e['bar_relative_onset'] for e in right}),
            'unsupported_position_onset_local_duration_context_occurrences':sum(_context_key(e) not in {_context_key(t) for t in left} for e in right)}
        def full_contexts(rows):
            return [(r['meter'], tuple(tuple(e['duration']) for bar in r['formal']['bars'] for e in bar), _context_key(event))
                    for r in rows for event in features[r['example_identity']]['events'] if role in event['roles']]
        train_contexts=set(full_contexts(train));development_contexts=full_contexts(development)
        result[role]['unsupported_meter_conditioned_full_duration_context_occurrences']=sum(value not in train_contexts for value in development_contexts)
        result[role]['context_scope']='coarse rational-time overlap is not equality of meter, position and full duration-partition context'
    return result


def _raw_histogram(tokens):
    return dict(sorted(Counter(json.dumps(value,separators=(',',':')) for value in tokens).items()))


def _simple(rows,outputs):
    correct=sum(outputs[r['example_identity']]==[*r['answer'].encode('ascii'),EOS] for r in rows)
    return {'rows':len(rows),'correct':correct,'wrong':len(rows)-correct,
            'gold_answer_histogram':dict(sorted(Counter(r['answer'] for r in rows).items())),
            'raw_output_histogram':_raw_histogram([outputs[r['example_identity']] for r in rows])}


def retained_sections(row_map,features,inputs):
    groups=defaultdict(list);files=[]
    for entry in INPUTS:
        raw=inputs[entry['path']]
        if type(raw) is not bytes:raise ContractError('authenticated input bytes required')
        if len(raw)>1024*1024:raise ContractError('audit input exceeds fixed bound')
        records=[parse_json(line,canonical=True) for line in raw.splitlines(keepends=True)]
        files.append({'path':entry['path'],'identity':identity(raw),'records':len(records)})
        for record in records:
            row=row_map.get(record['example_identity'])
            if row is None:raise ContractError('retained row outside exact regenerated corpus')
            tokens=record['generated_tokens']
            if (type(tokens) is not list or not 1<=len(tokens)<=64 or any(type(t) is not int or not 0<=t<=258 for t in tokens)
                    or EOS in tokens[:-1] or (tokens[-1]!=EOS and len(tokens)!=64)):
                raise ContractError('retained raw output violates frozen decoding')
            if 'expected_tokens' in record and record['expected_tokens']!=[*row['answer'].encode('ascii'),EOS]:raise ContractError('retained gold differs')
            if 'correct' in record and record['correct'] is not (tokens==[*row['answer'].encode('ascii'),EOS]):raise ContractError('retained score differs')
            context={key:record.get(key,entry.get(key)) for key in ('seed','step','stage','arm','backend')}
            if any(context[key] is None for key in context):raise ContractError('retained context missing')
            if (type(context['seed']) is not int or context['seed'] not in (17,29)
                    or type(context['step']) is not int or context['step'] not in (0,64,640)
                    or context['arm'] not in ('control','pitch','swap') or context['backend'] not in ('native_cpu','cpu','cuda')
                    or context['stage'] not in ('train','train_replay','heldout','diagnostic','development')):
                raise ContractError('retained context outside fixed studies')
            expected_split='test' if context['stage'] in ('heldout','development') else 'train'
            if row['split']!=expected_split:raise ContractError('retained stage/split mismatch')
            key=(entry['evidence'],context['arm'],context['backend'],context['seed'],context['step'],context['stage'],row['sibling'])
            groups[key].append((row,tokens))
    sections=[]
    for key,pairs in sorted(groups.items()):
        selected=[r for r,t in pairs];outputs={r['example_identity']:t for r,t in pairs}
        if len(outputs)!=len(pairs) or len(pairs)!=40:raise ContractError('fixed retained section must have40unique rows')
        item=dict(zip(('evidence','arm','backend','seed','step','stage','view'),key))
        item.update(metrics=_metrics(selected,outputs),by_variant={variant:_simple([r for r in selected if r['variant_id']==variant],outputs) for variant in VARIANTS},
                    by_parent={parent:_simple([r for r in selected if r['parent_id']==parent],outputs) for parent in sorted({r['parent_id'] for r in selected})},
                    answer_histogram=_raw_histogram([tokens for row,tokens in pairs]))
        rest=[r for r in selected if r['variant_id']=='10']
        item['rest_only_cases']=[{'example_identity':r['example_identity'],'parent':r['parent_id'],'meter':r['meter'],
            'gold':[55,EOS],'raw':outputs[r['example_identity']], 'correct':outputs[r['example_identity']]==[55,EOS],
            'rest_context':next(e for e in features[r['example_identity']]['events'] if 'rest' in e['roles'])} for r in rest]
        # Frozen paired variants have equal labels for rest-only and tie-only;
        # disagreement is descriptive, not a learned-mechanism diagnosis.
        by_group=defaultdict(dict)
        for r in selected:by_group[r['group_id']][r['variant_id']]=r
        pairs01_10=[]
        for group,variants in sorted(by_group.items()):
            if set(variants)!=set(VARIANTS):raise ContractError('incomplete retained quartet')
            tie,rest=variants['01'],variants['10'];left=outputs[tie['example_identity']];right=outputs[rest['example_identity']]
            pairs01_10.append({'group':group,'tie_only_raw':left,'rest_only_raw':right,'same_gold':[55,EOS],
                               'raw_equal':left==right,'both_correct':left==right==[55,EOS]})
        item['same_gold_tie_only_vs_rest_only']=pairs01_10;sections.append(item)
    return {'files':files,'sections':sections,'total_retained_records':sum(f['records'] for f in files),
            'scope':'descriptive joined artifacts; reused prompts, seeds, views and repeat runs are not independent subjects'}


def build_report(inputs):
    if type(inputs) is not dict or set(inputs)!=set(INPUT_PATHS):
        raise ContractError('exact authenticated audit input inventory required')
    train=contrast.rows();dev=contrast.rows('heldout',True);pitch_diag=pitch.diagnostic_rows();placement_diag=placement.diagnostic_rows()
    all_rows=train+dev+pitch_diag+placement_diag;row_map={}
    for row in all_rows:
        previous=row_map.setdefault(row['example_identity'],row)
        if previous!=row:raise ContractError('identity maps to conflicting regenerated row')
    features={key:row_features(row) for key,row in row_map.items()}
    pools={'original':train,'pitch_control':pitch.train_rows('control'),'pitch_treatment':pitch.train_rows('pitch'),
           'placement_control':placement.train_rows('control'),'placement_treatment':placement.train_rows('swap')}
    for rows in pools.values():
        if {r['parent_id'] for r in rows}!={'a','b'} or {r['prompt'] for r in rows}&{r['prompt'] for r in dev}:
            raise ContractError('train/development support boundary differs')
    unchanged=sum(left['prompt']==right['prompt'] for left,right in zip(placement_diag[:40],placement_diag[40:]))
    if len(placement_diag)!=80 or len({r['prompt'] for r in placement_diag})!=70 or unchanged!=10:
        raise ContractError('placement slots/surfaces differ')
    schedules={'original_64':contrast.schedule_receipt(),'exposure_640':exposure.schedule_receipt(640),
               'pitch_control':pitch.schedule_receipt('control'),'pitch_treatment':pitch.schedule_receipt('pitch'),
               'placement_control':placement.schedule_receipt('control'),'placement_treatment':placement.schedule_receipt('swap')}
    return {'schema':'trinite.music-task-audit.v1','mode':'model-free retrospective task/support and retained-error audit',
        'corpus':{'training_structural_parents':['a','b'],'exposed_development_structural_parents':['c','d'],
                  'canonical_training_rows':40,'development_rows':120,'canonical_development_rows':40,
                  'oracle_and_actual_prompt_target_mask_alignment_verified':True,
                  'validated_unique_row_identities':len(row_map),'exact_train_development_prompt_overlap':0},
        'placement':{'diagnostic_slots':80,'unique_prompts':70,'changed_input_pairs':30,'identical_input_pairs':10,
                     'scope':'template swap retains aggregate role positions; variant00 duplicates add no rest/tie feature support'},
        'support':{**{name:support(rows,features) for name,rows in pools.items()},'development':support(dev,features)},
        'support_by_parent':{name:{parent:support([r for r in rows if r['parent_id']==parent],features) for parent in sorted({r['parent_id'] for r in rows})} for name,rows in {**pools,'development':dev}.items()},
        'support_mismatch':{name:support_difference(rows,dev,features) for name,rows in pools.items()},
        'support_universe':finite_placement_universe(),'schedules':schedules,
        'row_contexts':[features[key] for key in sorted(features)],'retained':retained_sections(row_map,features,inputs),
        'claims':{'training_performed':False,'model_executed':False,'new_curriculum_generated':False,
                  'causal_mechanism_identified':False,'fresh_confirmation':False,'independent_replication':False,
                  'gpu_diagnosed':False,'hardware_attested':False},
        'interpretation':['Role-specific position/context support differs between a/b and exposed c/d; a rest is not covered merely because a tie used that index.',
            'The oracle rule is eight minus rest-presence minus valid-tie-presence; the audit does not establish which shortcut or representation the model learned.',
            'Same-gold rest-only/tie-only contrasts and retained histograms describe failure patterns, not why they occurred causally.',
            'All views/meters and repeated seeds/runs remain correlated; no significance, arbitrary counting, musical reasoning or transfer claim follows.',
            'Finite42-position enumeration is support inventory only; no new data split, target evaluation, training or proposed intervention is performed.']}
