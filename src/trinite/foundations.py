"""Explicit foundations stages, verified train gate and transactional summaries."""
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import resource
import time

import torch

from .contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json, require_identity
from .foundations_data import dataset, audit_bytes
from .foundations_training import LANES, SEEDS, WORKLOADS, PROTOCOL_IDENTITY, PROTOCOL_COMMIT, protocol, sources, state_for
from .foundations_plan import FoundationsPlan
from .foundations_checkpoint import snapshot, restore
from .comparison import greedy, weight_diagnostics
from .training import require_environment, model_identity, run_steps
from .tokenizer import EOS
from .observation import BundleObserver, verify_observation


def read_bytes(path,limit=32*1024*1024):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ContractError('missing/unsafe foundations artifact')
    with path.open('rb') as stream:raw=stream.read(limit+1)
    if len(raw)>limit:raise ContractError('foundations artifact exceeds budget')
    return raw


def write(root,name,value):
    raw=json_bytes(value)
    with (Path(root)/name).open('xb') as stream:stream.write(raw)
    return raw


def request_core():
    return {'schema':'trinite.foundations-request.v1','protocol':protocol(),'protocol_identity':PROTOCOL_IDENTITY,
        'protocol_commit':PROTOCOL_COMMIT,'source':sources(),'environment':require_environment(),
        'corpora':{w:{'dataset_identity':identity(dataset(w)[0]),'manifest_identity':identity(dataset(w)[1])} for w in WORKLOADS}}


def freeze_request(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json_bytes({**request_core(),'created_at':datetime.now(timezone.utc).isoformat()})
    with path.open('xb') as stream:stream.write(raw)
    return {'request_identity':identity(raw),'protocol_identity':PROTOCOL_IDENTITY,'planned_cells':27}


def read_request(path,expected_identity):
    require_identity(expected_identity)
    raw=read_bytes(path,128*1024)
    if identity(raw)!=expected_identity:raise ContractError('foundations request identity mismatch')
    record=parse_json(raw,canonical=True);expected=request_core()
    exact_keys(record,set(expected)|{'created_at'},'foundations request')
    if any(json_bytes(record[k])!=json_bytes(v) for k,v in expected.items()):
        raise ContractError('foundations request source/data/environment/protocol mismatch')
    try:
        if datetime.fromisoformat(record['created_at']).utcoffset() is None:raise ValueError('timezone required')
    except (TypeError,ValueError) as error:raise ContractError('invalid request clock') from error
    return record,raw


def majority_answers(examples):
    counts=defaultdict(Counter)
    for e in examples:
        if e['split']=='train':counts[e['task']][e['answer']]+=1
    return {task:min(values,key=lambda answer:(-values[answer],answer)) for task,values in counts.items()}


def metrics(rows):
    if not rows:raise ContractError('empty foundations scoring set')
    tasks={};families=defaultdict(list)
    for row in rows:families[row['family_id']].append(int(row['correct']))
    family_values={f:(sum(values)/len(values)).hex() for f,values in sorted(families.items())}
    for task in sorted({r['task'] for r in rows}):
        selected=[r for r in rows if r['task']==task]
        tasks[task]={'examples':len(selected),'correct':sum(r['correct'] for r in selected),
                     'baseline_correct':sum(r['baseline_correct'] for r in selected)}
    return {'examples':len(rows),'correct':sum(r['correct'] for r in rows),'tasks':tasks,
        'families':family_values,'family_macro_accuracy_hex':(math.fsum(float.fromhex(v) for v in family_values.values())/len(family_values)).hex()}


def score(model,workload,split):
    if split not in ('train','test'):raise ContractError('unknown scoring split')
    examples=parse_json(dataset(workload)[0],canonical=True)['examples'];baseline=majority_answers(examples)
    rows=[];before=model_identity(model)
    for item in examples:
        if item['split']!=split:continue
        expected=[*item['answer'].encode(),EOS]
        emitted=greedy(model,item['prompt'],protocol()['generation_caps'][workload])
        constant=[*baseline[item['task']].encode(),EOS]
        rows.append({'example_identity':item['example_identity'],'family_id':item['family_id'],
            'task':item['task'],'split':split,'expected_tokens':expected,'generated_tokens':emitted,
            'correct':emitted==expected,'baseline_tokens':constant,'baseline_correct':constant==expected})
    if model_identity(model)!=before or any(p.grad is not None for p in model.parameters()):
        raise ContractError('foundations scoring mutated the model')
    return metrics(rows),rows


def check_predictions(workload,split,raw):
    examples=parse_json(dataset(workload)[0],canonical=True)['examples'];baselines=majority_answers(examples)
    selected={e['example_identity']:e for e in examples if e['split']==split}
    record=parse_json(raw,canonical=True);exact_keys(record,{'predictions'},'foundations predictions')
    rows=record['predictions']
    if type(rows) is not list or len(rows)!=len(selected):raise ContractError('incomplete foundations predictions')
    seen=set()
    for row in rows:
        exact_keys(row,{'example_identity','family_id','task','split','expected_tokens','generated_tokens',
                        'correct','baseline_tokens','baseline_correct'},'prediction')
        key=row['example_identity']
        if type(key) is not str or key not in selected or key in seen:raise ContractError('unknown/duplicate prediction')
        seen.add(key);item=selected[key];expected=[*item['answer'].encode(),EOS]
        constant=[*baselines[item['task']].encode(),EOS];emitted=row['generated_tokens']
        if type(emitted) is not list or not 1<=len(emitted)<=protocol()['generation_caps'][workload]:
            raise ContractError('invalid generated token count')
        for token in emitted:integer(token,'generated token',0,258)
        expected_fields={'family_id':item['family_id'],'task':item['task'],'split':split,'expected_tokens':expected,
            'correct':emitted==expected,'baseline_tokens':constant,'baseline_correct':constant==expected}
        if any(json_bytes(row[k])!=json_bytes(v) for k,v in expected_fields.items()):
            raise ContractError('prediction/scorer context mismatch')
    return metrics(rows)


def task_failures(scores,stage):
    if stage not in ('train','test') or type(scores) is not dict:
        raise ContractError('invalid foundations scoring gate')
    exact_keys(scores,{'examples','correct','tasks','families','family_macro_accuracy_hex'},'scores')
    if type(scores['tasks']) is not dict or not scores['tasks']:
        raise ContractError('learning gate requires nonempty task scores')
    integer(scores['examples'],'total score count',1,2000)
    integer(scores['correct'],'total correct count',0,scores['examples'])
    if type(scores['families']) is not dict:
        raise ContractError('learning gate requires family scores')
    try:
        values=[float.fromhex(v) for v in scores['families'].values()]
        macro=float.fromhex(scores['family_macro_accuracy_hex'])
        if (not math.isfinite(macro) or not 0<=macro<=1
                or macro.hex()!=scores['family_macro_accuracy_hex']
                or any(not math.isfinite(v) or not 0<=v<=1 or v.hex()!=raw
                       for v,raw in zip(values,scores['families'].values()))
                or macro!=(math.fsum(values)/len(values) if values else 0)):
            raise ValueError('inconsistent family macro')
    except (TypeError,ValueError,OverflowError) as error:
        raise ContractError('learning gate requires canonical finite family metrics') from error
    threshold=protocol()['gates']['train_each_task' if stage=='train' else 'test_each_task']
    failures=[];total=correct=0
    for task,value in scores['tasks'].items():
        exact_keys(value,{'examples','correct','baseline_correct'},'task score')
        n,c,b=value['examples'],value['correct'],value['baseline_correct']
        integer(n,'score count',1,2000);integer(c,'correct count',0,n);integer(b,'baseline count',0,n)
        total+=n;correct+=c
        if c*threshold['denominator']<n*threshold['numerator']:failures.append(task+': accuracy below gate')
        if stage=='test':
            margin=protocol()['gates']['test_over_training_majority_each_task']
            if (c-b)*margin['denominator']<n*margin['numerator']:failures.append(task+': below baseline margin')
    if (total,correct)!=(scores['examples'],scores['correct']):
        raise ContractError('learning gate task totals contradict aggregate counts')
    return failures


def retain(root,stage,request_raw,workload,outputs,inputs=None):
    data,manifest=dataset(workload)
    collector=BundleObserver(root/(stage+'-provenance'))
    collector.record('foundations-'+stage,inputs={'request.json':request_raw,'dataset.json':data,
        'dataset-manifest.json':manifest,**(inputs or {})},outputs={n:read_bytes(root/n) for n in outputs})
    report=collector.finalize();write(root,stage+'-verification.json',report)
    if not report['integrity_verified'] or report['errors'] or report['known_missing_artifacts']:
        raise ContractError('foundations evidence failed upstream verification')


def failure(root,stage,error):
    path=Path(root)/(stage+'-failure.json')
    if not path.exists():write(root,path.name,{'outcome':'failed','stage':stage,'error':type(error).__name__+': '+str(error)})


def train_cell(root,workload,lane,seed,request,request_identity):
    _,request_raw=read_request(request,request_identity)
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    try:
        start=time.perf_counter();state=state_for(workload,lane,seed);run_steps(state)
        seconds=time.perf_counter()-start
        payload,metadata=snapshot(state,workload,request_identity)
        (root/'tensors.safetensors').write_bytes(payload);(root/'metadata.json').write_bytes(metadata)
        write(root,'steps.json',{'steps':state.history,'validation':state.validation})
        scores,predictions=score(state.model,workload,'train')
        write(root,'training-predictions.json',{'predictions':predictions})
        result={'schema':'trinite.foundations-training.v1','outcome':'completed','workload':workload,'lane':lane,
            'seed':seed,'request_identity':request_identity,'dataset_identity':state.dataset_identity,
            'manifest_identity':state.manifest_identity,'initialization_identity':state.initialization_identity,
            'model_identity':model_identity(state.model),'plan_identity':state.config.content_identity,
            'parameter_count':state.config.model.parameter_count,'steps':state.step,'target_tokens':state.target_tokens,
            'schedule_identity':identity(json_bytes([h['examples'] for h in state.history])),
            'tensors_identity':identity(payload),'metadata_identity':identity(metadata),
            'training_wall_seconds_hex':seconds.hex(),'latent_float32_bytes':state.config.model.parameter_count*4,
            'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'initial_train_loss_hex':state.initial_train_loss,'final_validation_loss_hex':state.validation[-1]['loss_hex'],
            'scores':scores,'training_gate_failures':task_failures(scores,'train'),'scope':protocol()['scope']}
        write(root,'training.json',result);read_request(request,request_identity)
        retain(root,'train',request_raw,workload,('training.json','training-predictions.json','steps.json',
                                                'tensors.safetensors','metadata.json'))
        return result
    except Exception as error:failure(root,'train',error);raise


def verified_stage(root,workload,lane,seed,request_identity,stage):
    if stage not in ('train','test'):raise ContractError('unknown evidence stage')
    root=Path(root);bundle=root/(stage+'-provenance');verification=verify_observation(bundle)
    manifest=parse_json(read_bytes(bundle/'manifest.json'),canonical=True)
    members={a['content_identity'] for a in manifest['core']['artifacts']};indexes=[]
    for key in members:
        try:record=parse_json(read_bytes(bundle/'artifacts/sha256'/key[7:]))
        except ContractError:continue
        if type(record) is dict and record.get('schema')=='trinite.observer-index.v1':indexes.append(record)
    if len(indexes)!=1:raise ContractError('foundations bundle requires one name index')
    names=indexes[0]['artifacts']
    if names.get('request.json')!=request_identity:raise ContractError('foundations evidence/request mismatch')
    def bound(name):
        raw=read_bytes(root/name)
        if identity(raw)!=names.get(name) or names[name] not in members:
            raise ContractError('foundations result copy differs from verified retained evidence')
        return raw
    name='training.json' if stage=='train' else 'evaluation.json'
    report=parse_json(bound(name),canonical=True)
    for k,v in {'workload':workload,'lane':lane,'seed':seed,'request_identity':request_identity,
                'outcome':'completed','schema':'trinite.foundations-'+('training' if stage=='train' else 'evaluation')+'.v1'}.items():
        if json_bytes(report.get(k))!=json_bytes(v):raise ContractError('foundations result context mismatch')
    scores=check_predictions(workload,'train' if stage=='train' else 'test',bound('training-predictions.json' if stage=='train' else 'test-predictions.json'))
    if json_bytes(report['scores'])!=json_bytes(scores):raise ContractError('foundations scores differ from retained predictions')
    if stage=='train':
        state=restore(bound('tensors.safetensors'),bound('metadata.json'),workload,lane,seed,request_identity,
            (report['tensors_identity'],report['metadata_identity']))
        expected={'model_identity':model_identity(state.model),'plan_identity':state.config.content_identity,
            'dataset_identity':state.dataset_identity,'manifest_identity':state.manifest_identity,
            'initialization_identity':state.initialization_identity,'parameter_count':state.config.model.parameter_count,
            'steps':state.config.steps,'target_tokens':state.target_tokens,
            'schedule_identity':identity(json_bytes([h['examples'] for h in state.history])),
            'training_gate_failures':task_failures(scores,'train')}
        if state.step!=state.config.steps or any(json_bytes(report.get(k))!=json_bytes(v) for k,v in expected.items()):
            raise ContractError('foundations training receipt differs from elected checkpoint')
        history=parse_json(bound('steps.json'),canonical=True)
        if history!={'steps':state.history,'validation':state.validation}:raise ContractError('history copy mismatch')
        wall=float.fromhex(report['training_wall_seconds_hex'])
        if (not math.isfinite(wall) or wall<=0 or wall.hex()!=report['training_wall_seconds_hex']
                or type(report['latent_float32_bytes']) is not int
                or report['latent_float32_bytes']!=state.config.model.parameter_count*4):
            raise ContractError('invalid foundations resource receipt')
    else:
        if (names.get('test-decision.json')!=report['decision_identity']
                or names.get('training.json')!=report['training_identity']
                or names['test-decision.json'] not in members or names['training.json'] not in members):
            raise ContractError('held-out bundle input bindings mismatch')
    return {'report':report,'report_identity':identity(bound(name)),'verification':verification}


def cells():return list(itertools.product(WORKLOADS,SEEDS,LANES))


def training_inventory(root,request_identity,worker_errors):
    records=[]
    for workload,seed,lane in cells():
        name=f'{workload}-{seed}-{lane}'
        try:
            if name in worker_errors:raise ContractError(worker_errors[name])
            value=verified_stage(Path(root)/name,workload,lane,seed,request_identity,'train')
            records.append({'workload':workload,'seed':seed,'lane':lane,'outcome':'completed',**value})
        except (ContractError,OSError,KeyError,TypeError,ValueError) as error:
            records.append({'workload':workload,'seed':seed,'lane':lane,'outcome':'failed','error':type(error).__name__+': '+str(error)})
    return records


def decision_record(root,request_identity,worker_errors):
    records=training_inventory(root,request_identity,worker_errors)
    failures=[]
    for row in records:
        if row['lane']=='dense':
            if row['outcome']!='completed':failures.append({'workload':row['workload'],'seed':row['seed'],'errors':['missing/failed verified dense cell']})
            elif task_failures(row['report']['scores'],'train'):
                failures.append({'workload':row['workload'],'seed':row['seed'],'errors':task_failures(row['report']['scores'],'train')})
    pairing_errors=[]
    if all(r['outcome']=='completed' for r in records):
        try:validate_pairs(records)
        except (ContractError,KeyError,TypeError,ValueError) as error:
            pairing_errors.append(type(error).__name__+': '+str(error))
    eligible=not worker_errors and all(r['outcome']=='completed' for r in records) and not failures and not pairing_errors
    return {'schema':'trinite.foundations-test-decision.v1','request_identity':request_identity,
        'protocol_identity':PROTOCOL_IDENTITY,'eligible':eligible,'dense_training_failures':failures,
        'pairing_errors':pairing_errors,'worker_errors':dict(worker_errors),'training_receipts':{f"{r['workload']}-{r['seed']}-{r['lane']}":
            {'report_identity':r['report_identity'],'manifest_identity':r['verification']['manifest_identity']}
            for r in records if r['outcome']=='completed'}},records


def decide(root,request,request_identity,worker_errors):
    read_request(request,request_identity)
    record,_=decision_record(root,request_identity,worker_errors)
    return {'decision_identity':identity(write(root,'test-decision.json',record)),**record}


def validate_decision(root,request_identity,decision_identity):
    raw=read_bytes(Path(root)/'test-decision.json',128*1024)
    if identity(raw)!=decision_identity:raise ContractError('test decision identity mismatch')
    record=parse_json(raw,canonical=True)
    if type(record) is not dict or type(record.get('worker_errors')) is not dict:raise ContractError('invalid test decision')
    expected,_=decision_record(root,request_identity,record['worker_errors'])
    if raw!=json_bytes(expected) or not expected['eligible']:
        raise ContractError('held-out scoring blocked by verified dense learning gate')
    return raw


def test_cell(root,workload,lane,seed,request,request_identity,decision_identity):
    _,request_raw=read_request(request,request_identity);root=Path(root)
    try:
        decision=validate_decision(root.parent,request_identity,decision_identity)
        trained=verified_stage(root,workload,lane,seed,request_identity,'train')
        info=trained['report']
        state=restore(read_bytes(root/'tensors.safetensors'),read_bytes(root/'metadata.json'),workload,lane,seed,
            request_identity,(info['tensors_identity'],info['metadata_identity']))
        scores,predictions=score(state.model,workload,'test')
        write(root,'test-predictions.json',{'predictions':predictions})
        result={'schema':'trinite.foundations-evaluation.v1','outcome':'completed','workload':workload,'lane':lane,
            'seed':seed,'request_identity':request_identity,'decision_identity':decision_identity,
            'training_identity':trained['report_identity'],'model_identity':model_identity(state.model),
            'scores':scores,'test_gate_failures':task_failures(scores,'test'),
            'learning_adequate':not info['training_gate_failures'] and not task_failures(scores,'test'),
            'weights':weight_diagnostics(state.model),'scope':protocol()['scope']}
        write(root,'evaluation.json',result);read_request(request,request_identity)
        retain(root,'test',request_raw,workload,('evaluation.json','test-predictions.json'),
               {'test-decision.json':decision,'training.json':read_bytes(root/'training.json')})
        return result
    except Exception as error:failure(root,'test',error);raise


def validate_pairs(records):
    by_key={(r['workload'],r['seed'],r['lane']):r for r in records}
    for workload,seed in itertools.product(WORKLOADS,SEEDS):
        runs=[by_key[workload,seed,lane]['report'] for lane in LANES]
        for name in ('initialization_identity','dataset_identity','manifest_identity','plan_identity',
                     'parameter_count','steps','target_tokens','schedule_identity'):
            if len({r[name] for r in runs})!=1:raise ContractError('unmatched foundations '+name)


def comparison_groups(records):
    validate_pairs(records)
    by_key={(r['workload'],r['seed'],r['lane']):r for r in records}
    groups={}
    for workload in WORKLOADS:
        groups[workload]={}
        for lane in LANES:
            values=[];diffs=[];wall=[];latent=[];adequate=[]
            for seed in SEEDS:
                row,dense=by_key[workload,seed,lane],by_key[workload,seed,'dense']
                value=float.fromhex(row['evaluation']['scores']['family_macro_accuracy_hex'])
                baseline=float.fromhex(dense['evaluation']['scores']['family_macro_accuracy_hex'])
                values.append(value);diffs.append(value-baseline)
                wall.append(float.fromhex(row['report']['training_wall_seconds_hex'])/float.fromhex(dense['report']['training_wall_seconds_hex']))
                latent.append(row['report']['latent_float32_bytes']/dense['report']['latent_float32_bytes'])
                if (not 0<=value<=1 or not 0<=baseline<=1 or not math.isfinite(wall[-1]) or wall[-1]<=0
                        or not math.isfinite(latent[-1]) or latent[-1]<=0):
                    raise ContractError('invalid foundations aggregate metric')
                adequate.append(row['evaluation']['learning_adequate'])
            groups[workload][lane]={'seed_accuracy_hex':[v.hex() for v in values],
                'mean_hex':(math.fsum(values)/len(SEEDS)).hex(),'min_hex':min(values).hex(),'max_hex':max(values).hex(),
                'paired_accuracy_difference_hex':[v.hex() for v in diffs],'quality_margin_each_seed':[v>=-.05 for v in diffs],
                'paired_training_wall_ratio_hex':[v.hex() for v in wall],'wall_margin_each_seed':[v<=1.25 for v in wall],
                'paired_latent_byte_ratio_hex':[v.hex() for v in latent],'latent_byte_margin_each_seed':[v<=1 for v in latent],
                'learning_adequate_each_seed':adequate,'interpretation':'descriptive only; finite visible tasks; no superiority claim'}
    return groups


def summarize(root,request,request_identity,worker_errors):
    read_request(request,request_identity);root=Path(root);errors=[];groups={}
    try:
        raw=read_bytes(root/'test-decision.json',128*1024);decision=parse_json(raw,canonical=True)
        expected,records=decision_record(root,request_identity,decision['worker_errors'])
        if raw!=json_bytes(expected):raise ContractError('retained test decision contradicts verified training')
        errors.extend(decision['pairing_errors'])
    except (ContractError,OSError,KeyError,TypeError,ValueError) as error:
        errors.append(type(error).__name__+': '+str(error));decision={'eligible':False}
        records=training_inventory(root,request_identity,worker_errors)
    if decision['eligible']:
        for row in records:
            workload,seed,lane=row['workload'],row['seed'],row['lane'];name=f'{workload}-{seed}-{lane}'
            try:
                if name in worker_errors:raise ContractError(worker_errors[name])
                tested=verified_stage(root/name,workload,lane,seed,request_identity,'test')
                report=tested['report']
                if (report['training_identity']!=row['report_identity'] or report['decision_identity']!=identity(raw)
                        or report['model_identity']!=row['report']['model_identity']
                        or report['test_gate_failures']!=task_failures(report['scores'],'test')
                        or json_bytes(report['learning_adequate'])!=json_bytes(not row['report']['training_gate_failures'] and not report['test_gate_failures'])):
                    raise ContractError('evaluation/training/gate receipt mismatch')
                row.update(evaluation=report,test_verification=tested['verification'])
            except (ContractError,OSError,KeyError,TypeError,ValueError) as error:
                row.update(outcome='failed',error=type(error).__name__+': '+str(error))
        if not worker_errors and not errors and all(r['outcome']=='completed' for r in records):
            try:groups=comparison_groups(records)
            except (ContractError,KeyError,TypeError,ValueError,OverflowError,ZeroDivisionError) as error:
                errors.append(type(error).__name__+': '+str(error))
    else:
        if any((root/f'{w}-{s}-{l}'/name).exists() for w,s,l in cells()
               for name in ('evaluation.json','test-predictions.json','test-provenance')):
            errors.append('held-out artifacts exist under a blocked decision')
    failed=bool(errors or worker_errors or any(r['outcome']=='failed' for r in records))
    completed=bool(groups)
    dense_adequate=completed and all(all(groups[w]['dense']['learning_adequate_each_seed']) for w in WORKLOADS)
    result={'schema':'trinite.foundations-summary.v1','request_identity':request_identity,
        'protocol_identity':PROTOCOL_IDENTITY,'protocol_commit':PROTOCOL_COMMIT,
        'outcome':'failed' if failed else 'completed' if completed else 'blocked',
        'test_decision':decision,'cells':records,'groups':groups,'aggregate_errors':errors,
        'worker_errors':dict(worker_errors),'dense_learning_adequate':dense_adequate,
        'format_selection_eligible':dense_adequate,'phase5_ready':False,'release_ready':False,
        'result_status':'inconclusive','scope':protocol()['scope'],'original_comparison_preserved':True}
    write(root,'summary.json',result)
    return result
