"""Fixed six-cell subprocess controller; every attempted outcome is retained."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from trinite.contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from native_diagnostic_v1.artifacts import Store, publish
from unseen_evaluation_v1.packet import inventory, unchanged
from .request import ROOT, read, core, validate, cells, cell_name
from .data import dataset, audit_bytes, validate_matching
from .protocol import PROTOCOL, SEEDS, STRATA


def configure():
    import torch
    import trinite
    if Path(trinite.__file__).resolve().parent != ROOT/'src/trinite':
        raise ContractError('addition runtime must use this checkout')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


def protect(output, request):
    output=Path(output).absolute();request=Path(request).resolve()
    resolved=output.resolve()
    if resolved.is_relative_to(ROOT) or ROOT.is_relative_to(resolved) or request.is_relative_to(resolved):
        raise ContractError('output overlaps source or request; use a separate fresh directory')
    for parent in output.parents:
        if parent.is_symlink() or not parent.is_dir():
            raise ContractError('output parent must exist without symlink ancestors')
    return output,request


def launch(root, request, request_identity, arm, seed, expected_directory):
    store=Store(root,expected_directory)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        fd=os.open('worker.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=store.fd)
        with os.fdopen(fd,'wb') as log:
            argv=[sys.executable,'-B',str(ROOT/'scripts/run_balanced_addition.py'),'_worker',
                '--request',str(request),'--request-identity',request_identity,'--output',str(root),
                '--arm',arm,'--seed',str(seed),'--expected-directory',str(store.stat.st_dev),str(store.stat.st_ino)]
            # Fixed source-bound interpreter/launcher, no shell or caller-selected executable.
            # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            with subprocess.Popen(argv,cwd=ROOT,env=env,shell=False,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as process:
                errors=[]
                def drain():
                    count=0
                    try:
                        while True:
                            block=process.stdout.read(4096)
                            if not block:break
                            remain=PROTOCOL['max_worker_log_bytes']-count
                            log.write(block[:remain]);count+=min(len(block),remain)
                            if len(block)>remain:
                                errors.append('worker log exceeds bound');process.kill();break
                        log.flush()
                    except Exception as error:
                        errors.append('worker log failure: '+type(error).__name__);process.kill()
                thread=threading.Thread(target=drain,daemon=True);thread.start()
                try:code=process.wait(timeout=PROTOCOL['worker_seconds'])
                except BaseException:
                    process.kill();process.wait();thread.join();raise
                thread.join();store.check()
                if errors:raise ContractError('; '.join(errors))
                return code
    finally:store.close()


def bound_files(root):
    before=inventory(root)
    result={name:identity(read(Path(root)/name)) for name,meta in before['members'].items()
            if name!='.' and not (meta[-1]&0o170000)==0o040000}
    unchanged(root,before)
    return result


def run_matrix(output, request_path, request_identity):
    output,request_path=protect(output,request_path)
    raw=read(request_path,1024*1024);validate(raw,request_identity)
    output.mkdir(mode=0o700,exist_ok=False);store=Store(output)
    records={};interrupted=False
    try:
        store.write('request.json',raw)
        for arm,seed in cells():
            name=cell_name(arm,seed)
            if interrupted:
                records[name]={'outcome':'not_attempted','reason':'earlier controller interruption'}
                continue
            store.check();os.mkdir(name,0o700,dir_fd=store.fd);store.check()
            child=Store(output/name);start=time.monotonic();outcome='failed';detail='worker not completed';code=None;completion=None
            try:
                code=launch(output/name,request_path,request_identity,arm,seed,(child.stat.st_dev,child.stat.st_ino))
                if code:
                    detail=f'worker exit {code}; see retained log'
                else:
                    from .workflow import audit
                    completion=identity(read(output/name/'completion.json'))
                    audit(output/name,raw,request_identity,arm,seed,completion)
                    outcome='completed';detail='structural audit passed; no optimizer or prediction replay'
            except subprocess.TimeoutExpired:outcome='timed_out';detail='fixed whole-worker deadline exceeded'
            except KeyboardInterrupt:outcome='interrupted';detail='controller interrupted';interrupted=True
            except Exception as error:detail=type(error).__name__+': '+str(error)
            finally:
                attempt={'schema':'trinite.balanced-addition-attempt.v1','cell':name,
                    'request_identity':request_identity,'outcome':outcome,'worker_exit':code,
                    'detail':detail,'completion_identity':completion if outcome=='completed' else None,
                    'wall_seconds_hex':(time.monotonic()-start).hex()}
                try:child.write('attempt.json',json_bytes(attempt))
                finally:child.close()
            records[name]={'outcome':outcome,'attempt_identity':identity(json_bytes(attempt)),
                          'completion_identity':attempt['completion_identity'],
                          'retained_files':bound_files(output/name)}
        validate(raw,request_identity);store.check()
        matrix={'schema':'trinite.balanced-addition-matrix.v1','request_identity':request_identity,
            'cells':records,'outcome':'completed' if all(r['outcome']=='completed' for r in records.values()) else 'incomplete',
            'fresh_confirmation':False,'learning_adequate':False,'phase5_ready':False,'release_ready':False}
        encoded=json_bytes(matrix);store.write('matrix.json',encoded)
        return {'outcome':matrix['outcome'],'matrix_identity':identity(encoded),'cells':records}
    finally:store.close()


def matrix_layout(root):
    import stat
    root=Path(root)
    for parent in (root,*root.parents):
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ContractError('matrix path requires non-link directories')
    result={}
    with os.scandir(root) as entries:
        for entry in entries:
            if len(result)>=8:raise ContractError('matrix root member count exceeded')
            info=entry.stat(follow_symlinks=False)
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ContractError('matrix root links/special members forbidden')
            result[entry.name]=(info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
    return result


def audit_matrix(root, request_raw, request_identity, expected_identity):
    root=Path(root);validate(request_raw,request_identity);require_identity(expected_identity)
    # Per-cell inventory bounds avoid pretending six cells share a32MiB payload allowance.
    if root.is_symlink() or not root.is_dir():raise ContractError('unsafe matrix root')
    root_stat=root.stat();before_layout=matrix_layout(root);before_names=set(before_layout)
    raw=read(root/'matrix.json')
    if identity(raw)!=expected_identity:raise ContractError('matrix expected identity differs')
    matrix=parse_json(raw,canonical=True)
    exact_keys(matrix,{'schema','request_identity','cells','outcome','fresh_confirmation','learning_adequate','phase5_ready','release_ready'},'matrix')
    if matrix['schema']!='trinite.balanced-addition-matrix.v1' or matrix['request_identity']!=request_identity or read(root/'request.json')!=request_raw:
        raise ContractError('matrix request/schema differs')
    expected_cells={cell_name(a,s) for a,s in cells()}
    exact_keys(matrix['cells'],expected_cells,'matrix cells')
    reports={};attempted=set();interrupted=False
    for arm,seed in cells():
        name=cell_name(arm,seed);record=matrix['cells'][name]
        if record.get('outcome')=='not_attempted':
            if not interrupted or record!={'outcome':'not_attempted','reason':'earlier controller interruption'}:
                raise ContractError('invalid unattempted cell')
            continue
        if interrupted:raise ContractError('cell attempted after controller interruption')
        attempted.add(name)
        exact_keys(record,{'outcome','attempt_identity','completion_identity','retained_files'},'matrix cell')
        if bound_files(root/name)!=record['retained_files']:
            raise ContractError('retained cell inventory differs')
        attempt_raw=read(root/name/'attempt.json')
        if identity(attempt_raw)!=record['attempt_identity']:raise ContractError('cell attempt identity differs')
        attempt=parse_json(attempt_raw,canonical=True)
        exact_keys(attempt,{'schema','cell','request_identity','outcome','worker_exit','detail','completion_identity','wall_seconds_hex'},'attempt')
        if (attempt['schema']!='trinite.balanced-addition-attempt.v1' or attempt['cell']!=name or
            attempt['request_identity']!=request_identity or attempt['outcome']!=record['outcome'] or
            attempt['completion_identity']!=record['completion_identity'] or
            record['outcome'] not in ('completed','failed','timed_out','interrupted')):
            raise ContractError('matrix/attempt fields differ')
        import math
        try:wall=float.fromhex(attempt['wall_seconds_hex'])
        except (ValueError,TypeError) as error:raise ContractError('invalid attempt timing') from error
        if not math.isfinite(wall) or wall<0 or wall.hex()!=attempt['wall_seconds_hex'] or type(attempt['detail'])is not str:
            raise ContractError('invalid attempt observations')
        if attempt['worker_exit'] is not None and type(attempt['worker_exit'])is not int:
            raise ContractError('invalid worker exit')
        if record['outcome']=='interrupted':interrupted=True
        if record['outcome']=='completed':
            if attempt['worker_exit']!=0:raise ContractError('completed worker did not exit successfully')
            from .workflow import audit
            reports[name]=audit(root/name,request_raw,request_identity,arm,seed,record['completion_identity'])['report']
        elif record['completion_identity']is not None:raise ContractError('failed cell cannot certify completion')
    if before_names!=attempted|{'request.json','matrix.json'}:raise ContractError('matrix physical membership differs')
    complete=len(reports)==len(expected_cells)
    if matrix['outcome']!=('completed' if complete else 'incomplete') or any(matrix[k]is not False for k in ('fresh_confirmation','learning_adequate','phase5_ready','release_ready')):
        raise ContractError('matrix outcome or claims differ')
    pairs={}
    for seed in SEEDS:
        left=reports.get(cell_name('control',seed));right=reports.get(cell_name('candidate',seed))
        if left is not None and right is not None:
            if left['initialization_identity']!=right['initialization_identity'] or left['target_tokens']!=right['target_tokens']:
                raise ContractError('paired initialization/exposure differs')
            pairs[str(seed)]={s:{'control_correct':left['development'][s]['correct'],
                'candidate_correct':right['development'][s]['correct'],
                'difference':right['development'][s]['correct']-left['development'][s]['correct'],
                'examples':left['development'][s]['examples']} for s in STRATA}
    validate(request_raw,request_identity)
    if root.stat().st_ino!=root_stat.st_ino or root.stat().st_dev!=root_stat.st_dev or matrix_layout(root)!=before_layout:
        raise ContractError('matrix root changed during audit')
    if read(root/'matrix.json')!=raw or read(root/'request.json')!=request_raw:
        raise ContractError('matrix inputs changed during audit')
    for name in attempted:
        if bound_files(root/name)!=matrix['cells'][name]['retained_files']:
            raise ContractError('cell changed after verification')
    return {'verified':True,'matrix_identity':expected_identity,'outcome':matrix['outcome'],
            'reports':reports,'paired_descriptive_counts':pairs,'optimizer_replayed':False,
            'predictions_replayed':False,'fresh_confirmation':False,'learning_adequate':False,
            'phase5_ready':False,'release_ready':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('preflight','freeze','run','audit','summarize','_worker'))
    p.add_argument('--request',type=Path);p.add_argument('--request-identity');p.add_argument('--output',type=Path)
    p.add_argument('--expected-identity');p.add_argument('--authorize-run',action='store_true')
    p.add_argument('--arm',help=argparse.SUPPRESS);p.add_argument('--seed',type=int,help=argparse.SUPPRESS)
    p.add_argument('--expected-directory',type=int,nargs=2,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    try:
        if args.action=='preflight':
            raw,manifest=dataset();audit_bytes(raw,manifest)
            print(json_bytes({'dataset_identity':identity(raw),'admission_identity':identity(manifest),'matching':validate_matching(),'model_executed':False}).decode(),end='');return 0
        configure()
        if args.request is None:raise ContractError('--request required')
        if args.action=='freeze':
            if args.request.resolve().is_relative_to(ROOT):raise ContractError('request must be outside immutable source checkout')
            raw=json_bytes(core());publish(args.request,raw);print(identity(raw));return 0
        if args.output is None or args.request_identity is None:raise ContractError('request identity and output required')
        if args.action=='run':
            if not args.authorize_run:raise ContractError('target run requires separate reviewed authorization and --authorize-run')
            result=run_matrix(args.output,args.request,args.request_identity)
            print(json_bytes(result).decode(),end='');return 0 if result['outcome']=='completed' else 1
        raw=read(args.request,1024*1024);validate(raw,args.request_identity)
        if args.action=='_worker':
            if args.expected_directory is None:raise ContractError('internal worker needs pinned controller directory')
            from .workflow import run
            print(run(args.output,raw,args.request_identity,args.arm,args.seed,args.expected_directory));return 0
        if not args.expected_identity:raise ContractError('external expected matrix identity required')
        result=audit_matrix(args.output,raw,args.request_identity,args.expected_identity)
        print(json_bytes(result).decode(),end='');return 0
    except (ContractError,OSError,ValueError) as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr);return 1
