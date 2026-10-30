"""Fixed paired CPU/CUDA trial controller; one separately authorized attempt."""
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
from .spec import BACKENDS, PROTOCOL, backend_name, dataset, schedule_receipt
from .workflow import (CLAIMS, SCOPE, bounded, audit_cell, gate_value, admit_cells,
                       train_worker, development_worker, audit_development, finite, probe_worker, audit_probe)


from .runtime import configure

def launch(action, root, request, request_id, expected_directory, *, backend=None, study=None, gate_id=None, probe_id=None):
    """Fixed isolated worker, no shell, bounded pipe and one whole-worker deadline."""
    root, request = safe_path(root), safe_path(request)
    argv = [sys.executable, '-I', '-B', str(ROOT/'scripts/run_music_device_trial_v2.py'), action,
            '--request', str(request), '--request-identity', request_id, '--output', str(root),
            '--expected-directory', *map(str,expected_directory), '--authorize-run']
    if action == '_train':
        backend_name(backend)
        argv += ['--backend',backend,'--probe-identity',probe_id]
    elif action == '_development':
        argv += ['--study',str(safe_path(study)),'--gate-identity',gate_id]
    elif action != '_probe':
        raise ContractError('unknown fixed worker')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
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


def attempt(root, action, request_path, raw, request_id, *, backend=None, study=None, gate_id=None, probe_id=None):
    root.mkdir(mode=0o700,exist_ok=False)
    store=Store(root); started=monotonic()
    outcome,code,completion,detail='failed',None,None,'worker did not complete'
    worker_finished=None; admission_started=None
    try:
        code=launch(action,root,request_path,request_id,(store.stat.st_dev,store.stat.st_ino),backend=backend,study=study,gate_id=gate_id,probe_id=probe_id)
        worker_finished=monotonic(); admission_started=worker_finished
        if code:
            if code == 2: outcome='unsupported'
            detail=f'worker exit {code}; retained log'
        else:
            completion=identity(read(root/'completion.json'))
            if action=='_train':audit_cell(root,raw,request_id,backend,completion)
            if action=='_probe':audit_probe(root,raw,request_id,completion)
            # Development full audit follows publication of its attempt receipt.
            outcome,detail='completed','worker completed; retained artifacts bound'
    except subprocess.TimeoutExpired:
        outcome,detail='timed_out','fixed worker deadline exceeded'
    except KeyboardInterrupt:
        outcome,detail='interrupted','controller interrupted'
    except Exception as error:
        detail=type(error).__name__+': '+str(error)
    ended=monotonic()
    if worker_finished is None:worker_finished=ended
    if admission_started is None:admission_started=ended
    if outcome!='completed':completion=None
    try:
        receipt={'schema':'trinite.music-device-attempt.v2','action':action,
                 'cell':backend if action=='_train' else action.removeprefix('_'),
                 'request_identity':request_id,'outcome':outcome,'worker_exit':code,
                 'completion_identity':completion,'detail':detail,'wall_seconds_hex':(ended-started).hex(),
                 'worker_wall_seconds_hex':(worker_finished-started).hex(),
                 'postworker_admission_seconds_hex':(ended-admission_started).hex()}
        store.write('attempt.json',json_bytes(receipt))
    finally:store.close()
    return {'outcome':outcome,'completion_identity':completion,'attempt_identity':identity(json_bytes(receipt)),
            'files':bounded(root)}


def run_matrix(output,request_path,request_id):
    started=monotonic();request_path=safe_path(request_path)
    raw=read(request_path,4194304);validate(raw,request_id)
    output=disjoint(output,ROOT,request_path);output.mkdir(mode=0o700,exist_ok=False);store=Store(output)
    missing={'outcome':'not_attempted','reason':'parity gate not admitted'}
    cells={b:dict(missing) for b in BACKENDS};probe=dict(missing)
    development={'outcome':'not_attempted','reason':'both-cell completion gate not admitted'}
    gate_id=None;failure=None;summary=None;parity=None
    pending_attempt=None;escaping_error=None
    try:
        store.write('request.json',raw)
        pending_attempt='probe'
        probe=attempt(output/'probe','_probe',request_path,raw,request_id)
        pending_attempt=None
        if probe['outcome']=='completed':
            parity=audit_probe(output/'probe',raw,request_id,probe['completion_identity'])
            if parity['passed']:
                interrupted=False
                for backend in BACKENDS:
                    if interrupted:cells[backend]={'outcome':'not_attempted','reason':'earlier interruption'}
                    else:
                        pending_attempt=backend
                        cells[backend]=attempt(output/backend,'_train',request_path,raw,request_id,backend=backend,probe_id=probe['completion_identity'])
                        pending_attempt=None
                        interrupted=cells[backend]['outcome']=='interrupted'
                if all(cells[b]['outcome']=='completed' for b in BACKENDS):
                    gate=gate_value(request_id,{b:cells[b]['completion_identity'] for b in BACKENDS},probe['completion_identity'])
                    admit_cells(output,raw,request_id,gate)
                    gate_raw=json_bytes(gate);store.write('development-gate.json',gate_raw);gate_id=identity(gate_raw)
                    pending_attempt='development'
                    development=attempt(output/'development','_development',request_path,raw,request_id,study=output,gate_id=gate_id)
                    pending_attempt=None
                    if development['outcome']=='completed':summary=audit_development(output,raw,request_id,gate_id,development['completion_identity'])['summary']
        validate(raw,request_id)
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)}
        escaping_error=error
    finally:
        try:
            # A failed setup/receipt publication has no returned attempt record.
            # Keep all partial files; never invent a receipt or seal an unauditable matrix.
            if pending_attempt is not None:
                if escaping_error is not None:raise escaping_error
                raise ContractError('unreturned device attempt; partial artifacts unsealed')
            files=sealable_inventory(output,raw,request_id,cells,probe,development,gate_id)
            complete=failure is None and development['outcome']=='completed' and summary is not None
            matrix={'schema':'trinite.music-device-matrix.v2','request_identity':request_id,
                'outcome':'completed' if complete else 'incomplete','probe':probe,'parity':parity,'cells':cells,
                'development':development,'gate_identity':gate_id,'failure':failure,'summary':summary if complete else None,'timing_comparison':timing_comparison(output) if complete else None,
                'controller_wall_seconds_hex':(monotonic()-started).hex(),
                'timing_scope':'controller run including workers, admission, serialization and inline audits; excludes Python controller startup and later standalone audit',
                'verification_scope':SCOPE,**CLAIMS}
            matrix['files']=files;raw_matrix=json_bytes(matrix)
            store.write('matrix.json',raw_matrix)
        finally:store.close()
    return {'outcome':matrix['outcome'],'matrix_identity':identity(raw_matrix)}



def sealable_inventory(root,raw,request_id,cells,probe,development,gate_id):
    """No sealing over unexplained partial directories or missing returned receipts."""
    exact_keys(cells,set(BACKENDS),'matrix cells')
    expected={'request.json'}
    if read(root/'request.json',4194304)!=raw:raise ContractError('root request differs before seal')
    for name,record in [('probe',probe),*cells.items(),('development',development)]:
        if record.get('outcome')=='not_attempted':
            exact_keys(record,{'outcome','reason'},'not-attempted record')
            continue
        expected.add(name)
        action='_'+name if name in ('probe','development') else '_train'
        check_attempt(root/name,record,action,name,request_id)
    if gate_id is not None:
        expected.add('development-gate.json')
        if identity(read(root/'development-gate.json'))!=gate_id:raise ContractError('gate changed before seal')
    if {p.name for p in root.iterdir()}!=expected:
        raise ContractError('partial or unexplained artifact directory; matrix unsealed')
    return bounded(root)


def timing_comparison(root):
    """Descriptive retained clocks; no performance gate or hidden warmup."""
    result={'by_backend':{},'scope':'fixed CPU then CUDA pair; cold process wall includes spawn/import/runtime/context and output handling; phase clocks synchronize CUDA; no significance claim'}
    for backend in BACKENDS:
        report=parse_json(read(Path(root)/backend/'report.json'),canonical=True)
        attempt_record=parse_json(read(Path(root)/backend/'attempt.json'),canonical=True)
        result['by_backend'][backend]={'worker_wall_seconds_hex':attempt_record['worker_wall_seconds_hex'],
            'postworker_admission_seconds_hex':attempt_record['postworker_admission_seconds_hex'],
            'phase_seconds_hex':report['timing_hex'],
            'diagnostic_greedy_seconds_hex_nested':report['diagnostic_greedy_seconds_hex']}
        worker=finite(attempt_record['worker_wall_seconds_hex'])
        phases=sum(finite(v) for v in report['timing_hex'].values())
        if phases>worker+1e-6:raise ContractError('named phases exceed cold worker wall')
        result['by_backend'][backend]['unattributed_startup_admission_handling_seconds_hex']=max(0.,worker-phases).hex()
    cpu=finite(result['by_backend']['cpu']['worker_wall_seconds_hex'])
    cuda=finite(result['by_backend']['cuda']['worker_wall_seconds_hex'])
    result['cuda_over_cpu_worker_wall_ratio_hex']=(cuda/cpu).hex() if cpu>0 else None
    result['overhead_attempt_seconds_hex']={name:parse_json(read(Path(root)/name/'attempt.json'),canonical=True)['wall_seconds_hex'] for name in ('probe','development')}
    result['development_decode_seconds_hex']=parse_json(read(Path(root)/'development/report.json'),canonical=True)['timing_hex']
    result['development_greedy_seconds_hex_nested']=parse_json(read(Path(root)/'development/report.json'),canonical=True)['greedy_seconds_hex']
    return result

def check_attempt(root,record,action,name,request_id):
    exact_keys(record,{'outcome','completion_identity','attempt_identity','files'},'attempt record')
    if bounded(root)!=record['files']:raise ContractError('attempt inventory differs')
    raw=read(Path(root)/'attempt.json')
    if identity(raw)!=record['attempt_identity']:raise ContractError('attempt identity differs')
    value=parse_json(raw,canonical=True)
    exact_keys(value,{'schema','action','cell','request_identity','outcome','worker_exit','completion_identity','detail','wall_seconds_hex','worker_wall_seconds_hex','postworker_admission_seconds_hex'},'attempt')
    if (value['schema']!='trinite.music-device-attempt.v2' or value['action']!=action or value['cell']!=name
        or value['request_identity']!=request_id or value['outcome']!=record['outcome']
        or value['completion_identity']!=record['completion_identity'] or type(value['detail']) is not str
        or value['outcome'] not in ('completed','failed','timed_out','interrupted','unsupported')
        or (value['worker_exit'] is not None and type(value['worker_exit']) is not int)):
        raise ContractError('attempt context differs')
    total=finite(value['wall_seconds_hex'])
    worker=finite(value['worker_wall_seconds_hex']);admission=finite(value['postworker_admission_seconds_hex'])
    if abs(total-worker-admission)>1e-8:raise ContractError('attempt timing partition differs')
    if value['outcome']=='completed':
        if value['worker_exit']!=0 or record['completion_identity'] is None:raise ContractError('invalid completed attempt')
    elif record['completion_identity'] is not None:raise ContractError('failed attempt claims completion')
    if value['outcome']=='unsupported' and value['worker_exit']!=2:raise ContractError('unsupported exit differs')


def audit_matrix(root,raw,request_id,expected_id):
    started=monotonic();root=safe_path(root);before=bounded(root);validate(raw,request_id)
    matrix_raw=read(root/'matrix.json',4194304)
    if identity(matrix_raw)!=expected_id:raise ContractError('independent matrix identity differs')
    matrix=parse_json(matrix_raw,canonical=True)
    exact_keys(matrix,{'schema','request_identity','outcome','probe','parity','cells','development','gate_identity','failure','summary','timing_comparison','controller_wall_seconds_hex','timing_scope','verification_scope','files',*CLAIMS},'matrix')
    if (matrix['schema']!='trinite.music-device-matrix.v2' or matrix['request_identity']!=request_id
        or read(root/'request.json',4194304)!=raw or matrix['files']!={k:v for k,v in before.items() if k!='matrix.json'}
        or matrix['verification_scope']!=SCOPE or any(matrix[k] is not False for k in CLAIMS)
        or matrix['timing_scope']!='controller run including workers, admission, serialization and inline audits; excludes Python controller startup and later standalone audit'):
        raise ContractError('matrix context/inventory differs')
    finite(matrix['controller_wall_seconds_hex']);exact_keys(matrix['cells'],set(BACKENDS),'matrix cells')
    expected_roots={'request.json','matrix.json'};parity=None
    if matrix['probe'].get('outcome')!='not_attempted':
        expected_roots.add('probe');check_attempt(root/'probe',matrix['probe'],'_probe','probe',request_id)
        if matrix['probe']['outcome']=='completed':parity=audit_probe(root/'probe',raw,request_id,matrix['probe']['completion_identity'])
    elif matrix['probe']!={'outcome':'not_attempted','reason':'parity gate not admitted'} or matrix['failure'] is None:
        raise ContractError('unjustified missing probe')
    if json_bytes(matrix['parity'])!=json_bytes(parity):raise ContractError('matrix parity differs')
    admitted=parity is not None and parity['passed'];interrupted=False;completed=admitted
    for backend in BACKENDS:
        record=matrix['cells'][backend]
        if not admitted:
            if record!={'outcome':'not_attempted','reason':'parity gate not admitted'}:raise ContractError('target ran before parity gate')
            continue
        if interrupted:
            if record!={'outcome':'not_attempted','reason':'earlier interruption'}:raise ContractError('target ran after interruption')
            completed=False;continue
        if record.get('outcome')=='not_attempted':
            # A controller failure between parity and spawning is retained, not completed.
            if matrix['failure'] is None or record!={'outcome':'not_attempted','reason':'parity gate not admitted'}:raise ContractError('unjustified missing cell')
            completed=False;continue
        expected_roots.add(backend);check_attempt(root/backend,record,'_train',backend,request_id)
        interrupted=record['outcome']=='interrupted'
        if record['outcome']=='completed':audit_cell(root/backend,raw,request_id,backend,record['completion_identity'])
        else:completed=False
    if matrix['gate_identity'] is not None:
        if not completed:raise ContractError('development gate lacks both completions')
        expected_roots.add('development-gate.json');gate_raw=read(root/'development-gate.json')
        if identity(gate_raw)!=matrix['gate_identity']:raise ContractError('gate identity differs')
        gate=parse_json(gate_raw,canonical=True)
        if gate!=gate_value(request_id,{b:matrix['cells'][b]['completion_identity'] for b in BACKENDS},matrix['probe']['completion_identity']):raise ContractError('gate links differ')
        admit_cells(root,raw,request_id,gate)
    dev=matrix['development'];summary=None
    if dev.get('outcome')!='not_attempted':
        if matrix['gate_identity'] is None:raise ContractError('development without gate')
        expected_roots.add('development');check_attempt(root/'development',dev,'_development','development',request_id)
        if dev['outcome']=='completed':summary=audit_development(root,raw,request_id,matrix['gate_identity'],dev['completion_identity'])['summary']
    elif dev!={'outcome':'not_attempted','reason':'both-cell completion gate not admitted'}:raise ContractError('invalid development omission')
    if {p.name for p in root.iterdir()}!=expected_roots:raise ContractError('matrix root membership differs')
    complete=matrix['failure'] is None and dev.get('outcome')=='completed' and summary is not None
    if matrix['failure'] is not None:
        exact_keys(matrix['failure'],{'type','message'},'failure')
        if any(type(v) is not str for v in matrix['failure'].values()):raise ContractError('invalid failure')
    if matrix['outcome']!=('completed' if complete else 'incomplete') or json_bytes(matrix['summary'])!=json_bytes(summary if complete else None):raise ContractError('matrix outcome/summary differs')
    if json_bytes(matrix['timing_comparison'])!=json_bytes(timing_comparison(root) if complete else None):raise ContractError('timing comparison differs')
    validate(raw,request_id)
    if bounded(root)!=before:raise ContractError('audit changed retained matrix')
    return {'verified':True,'outcome':matrix['outcome'],'matrix_identity':expected_id,
        'scope':SCOPE,'inference_replayed':False,'optimizer_replayed':False,
        'audit_wall_seconds_hex':(monotonic()-started).hex(),**CLAIMS}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('preflight','freeze','run','audit','_probe','_train','_development'))
    for flag in ('request','request-identity','output','expected-identity','backend','study','gate-identity','probe-identity'):p.add_argument('--'+flag)
    p.add_argument('--expected-directory',type=int,nargs=2);p.add_argument('--authorize-run',action='store_true')
    args=p.parse_args(argv)
    try:
        common={'request','request_identity','output'}
        permitted={'preflight':set(),'freeze':{'request'},'run':common|{'authorize_run'},'audit':common|{'expected_identity'},
            '_probe':common|{'authorize_run','expected_directory'},'_train':common|{'authorize_run','backend','expected_directory','probe_identity'},
            '_development':common|{'authorize_run','study','gate_identity','expected_directory'}}[args.action]
        if any(k!='action' and v is not None and v is not False and k not in permitted for k,v in vars(args).items()):raise ContractError('irrelevant selector/authorization flags forbidden')
        if args.action=='preflight':
            raw,receipt=dataset()
            print(json_bytes({'protocol':PROTOCOL,'dataset_identity':identity(raw),'admission_identity':identity(receipt),
                'schedule':schedule_receipt('control'),'model_executed':False,'hardware_tested':False}).decode(),end='');return 0
        if args.action in ('run','_probe','_train','_development') and not args.authorize_run:raise ContractError('separate owner authorization and --authorize-run required')
        if args.request is None:raise ContractError('request required')
        configure()
        if args.action=='freeze':print(json_bytes(freeze(args.request)).decode(),end='');return 0
        if args.output is None or args.request_identity is None:raise ContractError('output and request identity required')
        raw=read(args.request,4194304);validate(raw,args.request_identity)
        if args.action=='run':result=run_matrix(args.output,args.request,args.request_identity)
        elif args.action=='audit':
            if not args.expected_identity:raise ContractError('independent matrix identity required')
            result=audit_matrix(args.output,raw,args.request_identity,args.expected_identity)
        else:
            if args.expected_directory is None:raise ContractError('internal worker directory identity required')
            disjoint(args.output,ROOT,args.request)
            if args.action=='_probe':value=probe_worker(args.output,raw,args.request_identity,args.expected_directory)
            elif args.action=='_train':value=train_worker(args.output,raw,args.request_identity,args.backend,args.expected_directory,args.probe_identity)
            else:
                if not args.study or not args.gate_identity or safe_path(args.output)!=safe_path(args.study)/'development':raise ContractError('fixed study child and gate required')
                value=development_worker(args.study,args.output,raw,args.request_identity,args.gate_identity,args.expected_directory)
            result={'completion_identity':value}
        print(json_bytes(result).decode(),end='');return 1 if result.get('outcome')=='incomplete' else 0
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        from .runtime import UnsupportedRuntime
        unsupported=isinstance(error,UnsupportedRuntime)
        print(json_bytes({'outcome':'unsupported' if unsupported else 'failed','type':type(error).__name__,'detail':str(error)}).decode(),file=sys.stderr,end='')
        return 2 if unsupported else 1
