# SPDX-License-Identifier: MPL-2.0
"""Frozen identities, exact probe inventory and independently recomputable evidence."""
from pathlib import Path
from collections import Counter, defaultdict
import math
import os
import stat
import subprocess
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from teacher_diag_v1 import core as diag
from teacher_diag_v1 import native as behavior
from teacher_lab_v1 import core as lab
from . import arithmetic as ar

SCHEMA = 'trinite.teacher-logit.v2'
PREVIOUS_REQUEST = 'sha256:c68c774a6cafafe76bf49ff43ce4e4be6d6c5ee1a3d93d233e0b39eadc3a0616'
PREVIOUS_RESULT = 'sha256:18522822e789682ac07e2390792c931e5559fa66f7bba7d9413b28bd0ffecd1f'
PLAN = 'configs/teacher/tt003-logit-001-v2.json'
CLOSURE = ('scripts/run_teacher_logit.py', 'scripts/teacher_logit_v1/__init__.py',
           'scripts/teacher_logit_v1/arithmetic.py', 'scripts/teacher_logit_v1/core.py',
           'scripts/teacher_logit_v1/native.py', PLAN, 'docs/TEACHER-LOGIT-001-V2.md',
           'docs/TEACHER-LOGIT-FIX-001.md',
           'tests/teacher_logit_v1/test_contract.py', '.github/workflows/teacher-logit.yml')
RESULT_FILES = ('request.json', 'index.jsonl', 'logits.f32le', 'metrics.jsonl',
                'summary.json', 'execution.json', 'manifest.json')
LIMITS = {'request.json': 1024*1024, 'index.jsonl': 16*1024*1024,
          'logits.f32le': 14181804, 'metrics.jsonl': 48*1024*1024,
          'summary.json': 32*1024*1024, 'execution.json': 65536,
          'manifest.json': 65536}


def read(folder, name):
    root = Path(folder)
    if root.is_symlink() or not root.is_dir() or name not in LIMITS:
        raise ContractError('unsafe artifact directory/member')
    p = root/name
    info = p.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > LIMITS[name]:
        raise ContractError('unsafe or oversized artifact')
    with p.open('rb') as f:
        raw = f.read(LIMITS[name]+1)
    if len(raw) > LIMITS[name]:
        raise ContractError('artifact exceeded bound')
    return raw


def write(folder, name, raw):
    with (Path(folder)/name).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def fresh(out, *protected):
    out = diag.require_safe_output(out, *protected)
    if out.exists() or out.is_symlink() or not out.parent.is_dir():
        raise ContractError('fresh output directory required')
    out.mkdir(mode=0o700)
    return out


def jsonl(raw):
    if not raw.endswith(b'\n'):
        raise ContractError('incomplete JSONL')
    return [parse_json(line+b'\n', canonical=True) for line in raw.splitlines()]


def historical(dataset, teacher, previous):
    # Prior result contains the bound request plus its three result artifacts.
    raw = diag.read_member(previous, 'request.json')
    if identity(raw) != PREVIOUS_REQUEST or parse_json(raw, canonical=True) != diag.build_request(dataset, teacher):
        raise ContractError('previous request/source mismatch')
    # verify() requires a single-file request directory; use its checks directly
    # through an explicit read-only adapter, never rewrite previous artifacts.
    diag.check_members(previous, diag.RESULT_FILES)
    manifest_raw = diag.read_member(previous, 'manifest.json')
    if identity(manifest_raw) != PREVIOUS_RESULT:
        raise ContractError('previous manifest identity mismatch')
    manifest = parse_json(manifest_raw, canonical=True)
    members = [{'name': n, 'identity': identity(diag.read_member(previous, n)),
                'bytes': len(diag.read_member(previous, n))}
               for n in behavior.OUTPUT_FILES]
    if members != manifest['members']:
        raise ContractError('previous member mismatch')
    predictions = jsonl(diag.read_member(previous, 'predictions.jsonl'))
    _, request, metadata, _ = diag.historical_inputs(dataset, teacher)
    rows, visited = diag.scheduled_rows(metadata, request, dataset)
    if len(predictions) != len(rows):
        raise ContractError('previous prediction inventory mismatch')
    for r, p in zip(rows, predictions):
        if (r['id'], r['family'], r['answer'], r['observation']['task'], r['id'] in visited) != (
                p['id'], p['family'], p['expected'], p['task'], p['visited']):
            raise ContractError('previous row/schedule mismatch')
    calc = behavior.summary_for(predictions, dict(parse_json(raw, canonical=True), request_identity=PREVIOUS_REQUEST))
    if calc != parse_json(diag.read_member(previous, 'summary.json'), canonical=True):
        raise ContractError('previous classifications/summary mismatch')
    hist = {s: Counter(tuple(p[s]['tokens']) for p in predictions) for s in ('initial', 'final')}
    if hist != {'initial': Counter({(61,)*5: 972}), 'final': Counter({(48,257):891, (48,48,257):81})}:
        raise ContractError('historical greedy inventory changed')
    return rows, visited, predictions


def inventory(rows, visited, predictions):
    if len(rows) != 972 or len(visited) != 128 or len(predictions) != 972:
        raise ContractError('expected TRAIN inventory required')
    result = []; seen = set()
    for stage in ('initial', 'final'):
        for probe in ('gold-prefix', 'greedy-prefix'):
            for row, old in zip(rows, predictions):
                if row['split'] != 'train' or row['family'] not in ('S1','S2','W') or row['id'] != old['id']:
                    raise ContractError('nontrain/mismatched row')
                answer = list(row['answer'].encode('ascii')); targets = answer+[257]
                generated = old[stage]['tokens']
                if len(generated) > 5 or 257 in generated[:-1]:
                    raise ContractError('historical sequence continues after EOS')
                sequence = targets if probe == 'gold-prefix' else generated
                for pos in range(len(sequence)):
                    prefix = [256, *row['prompt'].encode('ascii'), *sequence[:pos]]
                    if len(prefix) > 128:
                        raise ContractError('prefix exceeds original context')
                    key = [stage, probe, row['id'], pos]
                    record_id = identity(json_bytes(key))
                    if record_id in seen:
                        raise ContractError('duplicate position')
                    seen.add(record_id)
                    target = targets[pos] if pos < len(targets) else None
                    result.append({'schema': SCHEMA+'.position', 'record_id': record_id,
                        'stage': stage, 'probe': probe, 'row_id': row['id'],
                        'family': row['family'], 'task': row['observation']['task'],
                        'visited': row['id'] in visited, 'scalar_label': row['answer'] if row['observation']['task']=='check' else None,
                        'position': pos, 'phase': 'residue' if pos<len(answer) else 'terminal' if pos==len(answer) else 'beyond-terminal',
                        'prefix_tokens': prefix, 'answer_prefix_is_gold': sequence[:pos]==answer[:pos] and pos<=len(answer),
                        'expected_next_token': target if probe=='gold-prefix' else None,
                        'positional_reference_target': target,
                        'historical_greedy_token': generated[pos] if probe=='greedy-prefix' else None,
                        'offset_bytes': len(result)*ar.VECTOR.size, 'length_bytes': ar.VECTOR.size,
                        'dtype': 'little-endian-ieee754-float32', 'vocabulary': 259,
                        'numerical_contract': SCHEMA+'.fp64-temperature-one'})
    counts = Counter((x['stage'], x['probe']) for x in result)
    if counts != {('initial','gold-prefix'):3402, ('final','gold-prefix'):3402,
                  ('initial','greedy-prefix'):4860, ('final','greedy-prefix'):2025}:
        raise ContractError('position inventory differs from frozen scope')
    return result


def request_for(dataset, teacher, previous):
    rows, visited, predictions = historical(dataset, teacher, previous)
    positions = inventory(rows, visited, predictions)
    plan = parse_json(lab.source_bytes(PLAN), canonical=True)
    if plan != PLAN_VALUE:
        raise ContractError('versioned logit plan changed')
    prior = parse_json(diag.read_member(previous,'request.json'), canonical=True)
    paths = sorted(set(CLOSURE) | {s['path'] for s in prior['diagnostic_source_closure']})
    closure = [{'path':p,'identity':identity(lab.source_bytes(p))} for p in paths]
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=lab.ROOT,text=True).strip()
    for item in closure:
        committed = subprocess.check_output(['git','show',head+':'+item['path']],cwd=lab.ROOT)
        if identity(committed) != item['identity']:
            raise ContractError('uncommitted source in frozen closure')
    return {'schema':SCHEMA+'.request','source_git_commit':head,'source_closure':closure,
        'plan':plan,'protocol_identity':identity(lab.source_bytes('docs/TEACHER-LOGIT-001-V2.md')),
        'previous_request_identity':PREVIOUS_REQUEST,'previous_result_identity':PREVIOUS_RESULT,
        'historical_pins':diag.PIN,'rows':972,'visited':128,'unvisited':844,
        'position_inventory_identity':identity(b''.join(json_bytes(x) for x in positions)),
        'gold_positions':6804,'greedy_positions':6885,'positions':len(positions),
        'logit_bytes':len(positions)*ar.VECTOR.size,'optimizer_updates':0,'heldout_scored':False}


PLAN_VALUE = {'schema':SCHEMA+'.plan','max_execution_seconds':1800,'max_replay_seconds':1800,
    'max_result_bytes':128*1024*1024,'max_positions':13689,'vocabulary':259,'optimizer_updates':0,
    'dtype':'native-cpu-float32','derived_dtype':'python-binary64','temperature':1,
    'normalization_absolute_tolerance':'1e-12','consistency_absolute_relative_tolerance':'1e-12',
    'replay_policy':'exact-fp32-bytes','failure_policy':'retain-partials-no-retry',
    'rank_ties':'lower-token-id-first','summary_quantiles':'nearest-rank-p05-p50-p95',
    'training_authorized':False,'heldout_scored':False}


def freeze(out, dataset, teacher, previous):
    request = request_for(dataset, teacher, previous)
    fresh(out,dataset,teacher,previous)
    raw = json_bytes(request); write(out,'request.json',raw)
    return {'request_identity':identity(raw),'positions':request['positions'],'optimizer_updates':0}


def verify_request(folder, anchor, dataset, teacher, previous):
    diag.check_members(folder,('request.json',))
    raw=read(folder,'request.json')
    if identity(raw)!=anchor:
        raise ContractError('external request anchor differs')
    req=parse_json(raw,canonical=True)
    if req!=request_for(dataset,teacher,previous):
        raise ContractError('frozen source/input/inventory changed')
    return req


def distribution(values):
    vals=sorted(values)
    if not vals:return None
    return {k:v.hex() for k,v in {'min':vals[0],'p05':vals[max(0,math.ceil(.05*len(vals))-1)],
        'median':vals[math.ceil(.5*len(vals))-1],'p95':vals[math.ceil(.95*len(vals))-1],
        'max':vals[-1],'mean':math.fsum(vals)/len(vals)}.items()}


def summarize(index, measures):
    if len(index) != len(measures) or len({i['record_id'] for i in index}) != len(index):
        raise ContractError('summary inventory mismatch')
    groups=defaultdict(list)
    for i,m in zip(index,measures):
        base=i['stage']+'/'+i['probe'];cohort='visited' if i['visited'] else 'unvisited'
        keys=['all','cohort:'+cohort,'task:'+i['task'],'circuit:'+i['family'],
            'phase:'+i['phase'],'task:'+i['task']+'/phase:'+i['phase'],
            'cohort:'+cohort+'/task:'+i['task']+'/phase:'+i['phase'],
            'circuit:'+i['family']+'/task:'+i['task']+'/phase:'+i['phase'],
            'circuit:'+i['family']+'/cohort:'+cohort+'/task:'+i['task']+'/phase:'+i['phase'],
            'task:'+i['task']+'/position:'+str(i['position']),
            'task:'+i['task']+'/cohort:'+cohort+'/position:'+str(i['position'])]
        if i['scalar_label'] is not None:keys.append('scalar_label:'+i['scalar_label']+'/phase:'+i['phase'])
        for k in keys:groups[base+'/'+k].append(m)
    out={}
    for key,ms in sorted(groups.items()):
        out[key]={'count':len(ms),'argmax_counts':dict(sorted(Counter(str(m['argmax']) for m in ms).items())),
            'argmax_categories':dict(sorted(Counter(m['argmax_category'] for m in ms).items())),
            'eos_beats_target':sum(m['eos_beats_target'] is True for m in ms),
            'eos_beats_target_loses_third':sum(m['eos_beats_target_loses_third'] is True for m in ms),
            'target_best_non_eos':sum(m['target_is_best_non_eos'] is True for m in ms),
            'target_best_residue':sum(m['target_residue_rank']==1 for m in ms),
            'target_rank_counts':dict(sorted(Counter(str(m['target_rank']) for m in ms if m['target_rank'] is not None).items())),
            'distributions':{k:distribution([float.fromhex(m[k]) for m in ms if m[k] is not None]) for k in (
                'full_eos_probability_hex','eos_rest_log_odds_hex','eos_target_margin_hex',
                'eos_best_residue_margin_hex','target_probability_hex','target_log_probability_hex','target_nll_hex')}}
    return {'schema':SCHEMA+'.summary','positions':len(index),'groups':out,
        'optimizer_updates':0,'heldout_scored':False,
        'greedy_target_scope':'positional reference; off-gold prefixes do not define a correct conditional continuation'}
