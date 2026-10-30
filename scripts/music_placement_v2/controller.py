"""Fixed four-cell replacement study controller; one separately authorized attempt."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
from time import monotonic

from trinite.contracts import ContractError, identity, json_bytes, parse_json, exact_keys
from native_diagnostic_v1.artifacts import Store
from frozen_music_eval_v1.artifacts import safe_path, disjoint, publish
from .request import ROOT, read, core, validate, freeze
from .spec import CELLS, PROTOCOL, cell_name, dataset, schedule_receipt
from .workflow import (CLAIMS, SCOPE, bounded, audit_cell, gate_value, admit_cells,
                       train_worker, development_worker, audit_development, finite)


def configure():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def launch(action, root, request, request_id, expected_directory, *, arm=None, seed=None, study=None, gate_id=None):
    """Fixed isolated worker, no shell, bounded pipe and one whole-worker deadline."""
    root, request = safe_path(root), safe_path(request)
    argv = [sys.executable, '-I', '-B', str(ROOT/'scripts/run_music_placement_v2.py'), action,
            '--request', str(request), '--request-identity', request_id, '--output', str(root),
            '--expected-directory', *map(str,expected_directory), '--authorize-run']
    if action == '_train':
        cell_name(arm,seed)
        argv += ['--arm',arm,'--seed',str(seed)]
    elif action == '_development':
        argv += ['--study',str(safe_path(study)),'--gate-identity',gate_id]
    else:
        raise ContractError('unknown fixed worker')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    store = Store(root,expected_directory)
    try:
        fd = os.open('worker.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=store.fd)
        with os.fdopen(fd,'wb') as log:
            # Fixed trusted launcher and bounded arguments, never a caller-selected executable.
            # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            with subprocess.Popen(argv,cwd=ROOT,env=env,shell=False,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as process:
                errors=[]
                def drain():
                    count=0
                    try:
                        while True:
                            chunk=process.stdout.read(4096)
                            if not chunk: break
                            remaining=1048576-count
                            log.write(chunk[:remaining]);count+=min(len(chunk),remaining)
                            if len(chunk)>remaining:
                                errors.append('worker log bound exceeded');process.kill();break
                        log.flush();os.fsync(log.fileno())
                    except Exception as error:
                        errors.append(type(error).__name__);process.kill()
                thread=threading.Thread(target=drain,daemon=True);thread.start()
                try: code=process.wait(timeout=900)
                except BaseException:
                    process.kill();process.wait();thread.join();raise
                thread.join();store.check()
                if errors: raise ContractError('worker log failure: '+','.join(errors))
                return code
    finally:store.close()


def attempt(root, action, request_path, raw, request_id, *, arm=None, seed=None, study=None, gate_id=None):
    root.mkdir(mode=0o700,exist_ok=False)
    store=Store(root); started=monotonic()
    outcome,code,completion,detail='failed',None,None,'worker did not complete'
    try:
        code=launch(action,root,request_path,request_id,(store.stat.st_dev,store.stat.st_ino),arm=arm,seed=seed,study=study,gate_id=gate_id)
        if code:
            detail=f'worker exit {code}; retained log'
        else:
            completion=identity(read(root/'completion.json'))
            if action=='_train':audit_cell(root,raw,request_id,arm,seed,completion)
            # Development full audit follows publication of its attempt receipt.
            outcome,detail='completed','worker completed; retained artifacts bound'
    except subprocess.TimeoutExpired:
        outcome,detail='timed_out','fixed worker deadline exceeded'
    except KeyboardInterrupt:
        outcome,detail='interrupted','controller interrupted'
    except Exception as error:
        detail=type(error).__name__+': '+str(error)
    if outcome!='completed':completion=None
    try:
        receipt={'schema':'trinite.music-placement-attempt.v2','action':action,
                 'cell':cell_name(arm,seed) if action=='_train' else 'development',
                 'request_identity':request_id,'outcome':outcome,'worker_exit':code,
                 'completion_identity':completion,'detail':detail,'wall_seconds_hex':(monotonic()-started).hex()}
        store.write('attempt.json',json_bytes(receipt))
    finally:store.close()
    return {'outcome':outcome,'completion_identity':completion,'attempt_identity':identity(json_bytes(receipt)),
            'files':bounded(root)}


def run_matrix(output, request_path, request_id):
    request_path=safe_path(request_path)
    raw=read(request_path,4*1024*1024);validate(raw,request_id)
    output=disjoint(output,ROOT,request_path)
    output.mkdir(mode=0o700,exist_ok=False)
    store=Store(output)
    cells={};development={'outcome':'not_attempted','reason':'all-four completion gate not admitted'}
    gate_id=None;failure=None;summary=None;failure_error=None;awaiting_record=None
    try:
        store.write('request.json',raw)
        interrupted=False
        for arm,seed in CELLS:
            name=cell_name(arm,seed)
            if interrupted:
                cells[name]={'outcome':'not_attempted','reason':'earlier interruption'}
            else:
                awaiting_record=name
                cells[name]=attempt(output/name,'_train',request_path,raw,request_id,arm=arm,seed=seed)
                awaiting_record=None
                interrupted=cells[name]['outcome']=='interrupted'
        if all(cells[cell_name(a,s)]['outcome']=='completed' for a,s in CELLS):
            gate=gate_value(request_id,{cell_name(a,s):cells[cell_name(a,s)]['completion_identity'] for a,s in CELLS})
            admit_cells(output,raw,request_id,gate)
            gate_raw=json_bytes(gate);store.write('development-gate.json',gate_raw);gate_id=identity(gate_raw)
            awaiting_record='development'
            development=attempt(output/'development','_development',request_path,raw,request_id,study=output,gate_id=gate_id)
            awaiting_record=None
            if development['outcome']=='completed':
                summary=audit_development(output,raw,request_id,gate_id,development['completion_identity'])['summary']
        validate(raw,request_id)
    except BaseException as error:
        failure_error=error
        failure={'type':type(error).__name__,'message':str(error)}
    finally:
        # A return record is the custody boundary. Setup/receipt-write failures
        # can escape attempt() before that boundary. Preserve raw partial bytes,
        # but never seal an unauditable matrix or print an invented identity.
        try:
            try:
                require_sealable(output, cells, development, gate_id, request_id, awaiting_record, raw)
            except (ValueError, OSError) as error:
                raise ContractError('unsealed partial study: '+str(error)) from (failure_error or error)
            # Inventory and request revalidation failures cannot certify completion.
            complete=failure is None and development['outcome']=='completed' and summary is not None
            matrix={'schema':'trinite.music-placement-matrix.v2','request_identity':request_id,
                    'outcome':'completed' if complete else 'incomplete','cells':cells,
                    'development':development,'gate_identity':gate_id,'failure':failure,
                    'summary':summary if complete else None,'verification_scope':SCOPE,**CLAIMS}
            matrix['files']=bounded(output)
            raw_matrix=json_bytes(matrix)
            store.write('matrix.json',raw_matrix)
        finally:
            store.close()
    return {'outcome':matrix['outcome'],'matrix_identity':identity(raw_matrix)}



def require_sealable(root, cells, development, gate_id, request_id, awaiting_record, raw):
    """Require returned receipts and their exact inventory before matrix publication."""
    if awaiting_record is not None or set(cells) != {cell_name(a,s) for a,s in CELLS}:
        raise ContractError('a required attempt has no returned record')
    if read(root/'request.json', 4*1024*1024) != raw:
        raise ContractError('retained root request changed before seal')
    expected_roots = {'request.json'}
    interrupted = False
    for arm, seed in CELLS:
        name = cell_name(arm, seed)
        record = cells[name]
        if record.get('outcome') == 'not_attempted':
            if not interrupted or record != {'outcome':'not_attempted','reason':'earlier interruption'}:
                raise ContractError('unjustified absent fixed cell')
            continue
        if interrupted:
            raise ContractError('attempt after recorded interruption')
        check_attempt(root/name, record, '_train', name, request_id)
        expected_roots.add(name)
        interrupted = record['outcome'] == 'interrupted'
    if gate_id is not None:
        expected_roots.add('development-gate.json')
        gate_raw = read(root/'development-gate.json')
        if identity(gate_raw) != gate_id:
            raise ContractError('development gate changed before seal')
    if development.get('outcome') != 'not_attempted':
        check_attempt(root/'development', development, '_development', 'development', request_id)
        expected_roots.add('development')
    elif development != {'outcome':'not_attempted','reason':'all-four completion gate not admitted'}:
        raise ContractError('unjustified absent development')
    if {p.name for p in root.iterdir()} != expected_roots:
        raise ContractError('unrecorded root or partial artifact before seal')


def check_attempt(root, record, action, name, request_id):
    exact_keys(record,{'outcome','completion_identity','attempt_identity','files'},'attempt record')
    if bounded(root)!=record['files']:raise ContractError('attempt inventory differs')
    raw=read(Path(root)/'attempt.json')
    if identity(raw)!=record['attempt_identity']:raise ContractError('attempt identity differs')
    value=parse_json(raw,canonical=True)
    exact_keys(value,{'schema','action','cell','request_identity','outcome','worker_exit','completion_identity','detail','wall_seconds_hex'},'attempt')
    if (value['schema']!='trinite.music-placement-attempt.v2' or value['action']!=action or value['cell']!=name
            or value['request_identity']!=request_id or value['outcome']!=record['outcome']
            or value['completion_identity']!=record['completion_identity'] or type(value['detail']) is not str
            or value['outcome'] not in ('completed','failed','timed_out','interrupted')
            or (value['worker_exit'] is not None and type(value['worker_exit']) is not int)):
        raise ContractError('attempt context differs')
    finite(value['wall_seconds_hex'])
    if value['outcome']=='completed':
        if value['worker_exit']!=0 or record['completion_identity'] is None:raise ContractError('invalid completed attempt')
    elif record['completion_identity'] is not None:raise ContractError('failed attempt claims completion')


def audit_matrix(root, raw, request_id, expected_id):
    root=safe_path(root);before=bounded(root);validate(raw,request_id)
    matrix_raw=read(root/'matrix.json',4*1024*1024)
    if identity(matrix_raw)!=expected_id:raise ContractError('independent matrix identity differs')
    matrix=parse_json(matrix_raw,canonical=True)
    exact_keys(matrix,{'schema','request_identity','outcome','cells','development','gate_identity','failure','summary','verification_scope','files',*CLAIMS},'matrix')
    if (matrix['schema']!='trinite.music-placement-matrix.v2' or matrix['request_identity']!=request_id
            or read(root/'request.json',4*1024*1024)!=raw or matrix['files']!={k:v for k,v in before.items() if k!='matrix.json'}
            or matrix['verification_scope']!=SCOPE or any(matrix[k] is not False for k in CLAIMS)):
        raise ContractError('matrix context/inventory differs')
    exact_keys(matrix['cells'],{cell_name(a,s) for a,s in CELLS},'matrix cells')
    interrupted=False;completed=True;expected_roots={'request.json','matrix.json'}
    for arm,seed in CELLS:
        name=cell_name(arm,seed);record=matrix['cells'][name]
        if record.get('outcome')=='not_attempted':
            if not interrupted or record!={'outcome':'not_attempted','reason':'earlier interruption'}:raise ContractError('unjustified missing cell')
            completed=False;continue
        if interrupted:raise ContractError('cell ran after interruption')
        expected_roots.add(name);check_attempt(root/name,record,'_train',name,request_id)
        interrupted=record['outcome']=='interrupted'
        if record['outcome']=='completed':audit_cell(root/name,raw,request_id,arm,seed,record['completion_identity'])
        else:completed=False
    dev=matrix['development'];summary=None
    if matrix['gate_identity'] is not None:
        if not completed:raise ContractError('development gate without all four completions')
        expected_roots.add('development-gate.json')
        gate=parse_json(read(root/'development-gate.json'),canonical=True)
        if identity(json_bytes(gate))!=matrix['gate_identity']:raise ContractError('gate link differs')
        if gate!=gate_value(request_id,{n:r['completion_identity'] for n,r in matrix['cells'].items()}):raise ContractError('gate completion links differ')
        admit_cells(root,raw,request_id,gate)
    if dev.get('outcome')!='not_attempted':
        if matrix['gate_identity'] is None:raise ContractError('development without gate')
        expected_roots.add('development');check_attempt(root/'development',dev,'_development','development',request_id)
        if dev['outcome']=='completed':summary=audit_development(root,raw,request_id,matrix['gate_identity'],dev['completion_identity'])['summary']
    elif dev!={'outcome':'not_attempted','reason':'all-four completion gate not admitted'}:
        raise ContractError('invalid development omission')
    if {p.name for p in root.iterdir()}!=expected_roots:raise ContractError('matrix root membership differs')
    complete=matrix['failure'] is None and dev.get('outcome')=='completed' and summary is not None
    if matrix['failure'] is not None:
        exact_keys(matrix['failure'],{'type','message'},'failure')
        if any(type(v) is not str for v in matrix['failure'].values()):raise ContractError('invalid failure')
    if matrix['outcome']!=('completed' if complete else 'incomplete') or json_bytes(matrix['summary'])!=json_bytes(summary if complete else None):
        raise ContractError('matrix outcome/summary differs')
    validate(raw,request_id)
    if bounded(root)!=before:raise ContractError('audit mutated matrix')
    return {'verified':True,'outcome':matrix['outcome'],'matrix_identity':expected_id,
            'scope':SCOPE,'inference_replayed':False,'optimizer_replayed':False,**CLAIMS}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('preflight','freeze','run','audit','_train','_development'))
    for flag in ('request','request-identity','output','expected-identity','arm','study','gate-identity'):
        p.add_argument('--'+flag)
    p.add_argument('--seed',type=int);p.add_argument('--expected-directory',type=int,nargs=2)
    p.add_argument('--authorize-run',action='store_true')
    args=p.parse_args(argv)
    try:
        common={'request','request_identity','output'}
        permitted={'preflight':set(), 'freeze':{'request'}, 'run':common|{'authorize_run'},
            'audit':common|{'expected_identity'}, '_train':common|{'authorize_run','arm','seed','expected_directory'},
            '_development':common|{'authorize_run','study','gate_identity','expected_directory'}}[args.action]
        if any(k!='action' and v is not None and v is not False and k not in permitted for k,v in vars(args).items()):
            raise ContractError('irrelevant selector/authorization flags forbidden')
        if args.action=='preflight':
            raw,receipt=dataset()
            print(json_bytes({'protocol':PROTOCOL,'dataset_identity':identity(raw),'admission_identity':identity(receipt),
                'schedules':{a:schedule_receipt(a) for a in ('control','swap')},'model_executed':False}).decode(),end='');return 0
        if args.action in ('run','_train','_development') and not args.authorize_run:
            raise ContractError('separate explicit owner authorization and --authorize-run required')
        if args.request is None:raise ContractError('request required')
        configure()
        if args.action=='freeze':
            print(json_bytes(freeze(args.request)).decode(),end='');return 0
        if args.output is None or args.request_identity is None:raise ContractError('output and request identity required')
        raw=read(args.request,4*1024*1024);validate(raw,args.request_identity)
        if args.action=='run':result=run_matrix(args.output,args.request,args.request_identity)
        elif args.action=='audit':
            if not args.expected_identity:raise ContractError('independent expected identity required')
            result=audit_matrix(args.output,raw,args.request_identity,args.expected_identity)
        else:
            if args.expected_directory is None:raise ContractError('internal worker directory identity required')
            disjoint(args.output,ROOT,args.request)
            if args.action=='_train':result={'completion_identity':train_worker(args.output,raw,args.request_identity,args.arm,args.seed,args.expected_directory)}
            else:
                if not args.study or not args.gate_identity:raise ContractError('study and independent gate identity required')
                if safe_path(args.output)!=safe_path(args.study)/'development':raise ContractError('development output must be fixed study child')
                result={'completion_identity':development_worker(args.study,args.output,raw,args.request_identity,args.gate_identity,args.expected_directory)}
        print(json_bytes(result).decode(),end='')
        return 1 if result.get('outcome')=='incomplete' else 0
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr);return 1
