"""Bounded two-seed train-only exposure; no held-out execution path."""
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
from .request import ROOT, read, core, validate, cell_name
from .spec import PROTOCOL, SEEDS, CHECKPOINTS, dataset, schedule_receipt
from .workflow import finite, CLAIMS, missing_checkpoint_files

def configure():
    import torch
    import trinite
    if Path(trinite.__file__).resolve().parent != ROOT/'src/trinite':
        raise ContractError('music runtime must use this checkout')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


def protect(output, request):
    output=Path(output).absolute();request=Path(request).absolute()
    resolved=output.resolve()
    if resolved.is_relative_to(ROOT) or ROOT.is_relative_to(resolved) or request.is_relative_to(resolved):
        raise ContractError('output overlaps source or request; use a separate fresh directory')
    for parent in output.parents:
        if parent.is_symlink() or not parent.is_dir():
            raise ContractError('output parent must exist without symlink ancestors')
    return output,request


def launch(root, request, request_identity, seed, expected_directory):
    store=Store(root,expected_directory)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        fd=os.open('worker.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=store.fd)
        with os.fdopen(fd,'wb') as log:
            argv=[sys.executable,'-I','-B',str(ROOT/'scripts/run_music_exposure.py'),'_worker',
                '--request',str(request),'--request-identity',request_identity,'--output',str(root),
                '--authorize-run','--seed',str(seed),'--expected-directory',str(store.stat.st_dev),str(store.stat.st_ino)]
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
    if before['bytes'] > 32*1024*1024:
        raise ContractError('exposure cell byte bound exceeded')
    if 'worker.log' in before['members']:
        read(Path(root)/'worker.log', PROTOCOL['max_worker_log_bytes'])
    result={name:identity(read(Path(root)/name)) for name,meta in before['members'].items()
            if name!='.' and not (meta[-1]&0o170000)==0o040000}
    unchanged(root,before)
    return result


def _cell(output, request_path, request_raw, request_identity, seed):
    name = cell_name(seed)
    root = output/name
    root.mkdir(mode=0o700, exist_ok=False)
    child = Store(root)
    started = time.monotonic()
    code, completion, outcome, detail = None, None, 'failed', 'worker did not complete'
    try:
        code = launch(root, request_path, request_identity, seed,
                      (child.stat.st_dev, child.stat.st_ino))
        if code:
            detail = f'worker exit {code}; see retained log'
        else:
            from .workflow import audit
            completion = identity(read(root/'completion.json'))
            audit(root, request_raw, request_identity, seed, completion)
            outcome, detail = 'completed', 'retained byte/structure/score checks passed; no numerical replay'
    except subprocess.TimeoutExpired:
        outcome, detail = 'timed_out', 'fixed whole-worker deadline exceeded'
    except KeyboardInterrupt:
        outcome, detail = 'interrupted', 'controller interrupted'
    except Exception as error:
        detail = type(error).__name__+': '+str(error)
    finally:
        attempt = {'schema': 'trinite.music-exposure-attempt.v1', 'cell': name,
            'request_identity': request_identity, 'outcome': outcome, 'worker_exit': code,
            'detail': detail, 'completion_identity': completion if outcome == 'completed' else None,
            'wall_seconds_hex': (time.monotonic()-started).hex(),
            'missing_checkpoint_files': missing_checkpoint_files(root)}
        try:
            child.write('attempt.json', json_bytes(attempt))
        finally:
            child.close()
    return {'outcome': outcome, 'attempt_identity': identity(json_bytes(attempt)),
            'completion_identity': attempt['completion_identity'], 'retained_files': bound_files(root)}


def _fit(root, records, request_raw, request_identity):
    from .workflow import audit
    fit = True
    for seed in SEEDS:
        record = records[cell_name(seed)]
        if record['outcome'] != 'completed':
            fit = False
        else:
            report = audit(root/cell_name(seed), request_raw, request_identity, seed,
                           record['completion_identity'])['report']
            fit = fit and report['training_fit'] is True
    return fit


def _outcome(records, fit):
    if not all(records[cell_name(s)]['outcome'] == 'completed' for s in SEEDS):
        return 'incomplete'
    return 'training_fit_passed' if fit else 'training_fit_failed'


def run_matrix(output, request_path, request_identity):
    output, request_path = protect(output, request_path)
    raw = read(request_path, 1024*1024)
    validate(raw, request_identity)
    output.mkdir(mode=0o700, exist_ok=False)
    store = Store(output)
    records, interrupted = {}, False
    try:
        store.write('request.json', raw)
        for seed in SEEDS:
            name = cell_name(seed)
            if interrupted:
                records[name] = {'outcome': 'not_attempted', 'reason': 'earlier controller interruption'}
            else:
                records[name] = _cell(output, request_path, raw, request_identity, seed)
                interrupted = records[name]['outcome'] == 'interrupted'
        fit = _fit(output, records, raw, request_identity)
        validate(raw, request_identity)
        matrix = {'schema': 'trinite.music-exposure-matrix.v1', 'request_identity': request_identity,
            'cells': records, 'outcome': _outcome(records, fit), 'both_seeds_fit': fit, **CLAIMS}
        encoded = json_bytes(matrix)
        store.write('matrix.json', encoded)
        if inventory(output)['bytes'] > PROTOCOL['max_total_payload_bytes']:
            raise ContractError('exposure aggregate retained payload exceeds bound')
        return {'outcome': matrix['outcome'], 'matrix_identity': identity(encoded),
                'both_seeds_fit': fit, 'cells': records}
    finally:
        store.close()


def audit_matrix(root, request_raw, request_identity, expected_identity):
    root = Path(root)
    validate(request_raw, request_identity)
    require_identity(expected_identity)
    before = inventory(root)
    if before['bytes'] > PROTOCOL['max_total_payload_bytes']:
        raise ContractError('exposure aggregate retained payload exceeds bound')
    raw = read(root/'matrix.json')
    if identity(raw) != expected_identity:
        raise ContractError('matrix expected identity differs')
    matrix = parse_json(raw, canonical=True)
    exact_keys(matrix, {'schema', 'request_identity', 'cells', 'outcome', 'both_seeds_fit', *CLAIMS}, 'exposure matrix')
    if matrix['schema'] != 'trinite.music-exposure-matrix.v1' or matrix['request_identity'] != request_identity \
            or read(root/'request.json') != request_raw or any(matrix[k] is not False for k in CLAIMS):
        raise ContractError('exposure matrix request/schema/claims differ')
    exact_keys(matrix['cells'], {cell_name(s) for s in SEEDS}, 'exposure matrix cells')
    reports, attempted, interrupted = {}, set(), False
    from .workflow import audit
    for seed in SEEDS:
        name = cell_name(seed)
        record = matrix['cells'][name]
        if record.get('outcome') == 'not_attempted':
            if not interrupted or record != {'outcome': 'not_attempted', 'reason': 'earlier controller interruption'}:
                raise ContractError('unjustified unattempted exposure cell')
            continue
        if interrupted:
            raise ContractError('exposure cell executed after interruption')
        attempted.add(name)
        exact_keys(record, {'outcome', 'attempt_identity', 'completion_identity', 'retained_files'}, 'exposure cell')
        if bound_files(root/name) != record['retained_files']:
            raise ContractError('exposure retained cell membership differs')
        attempt_raw = read(root/name/'attempt.json')
        if identity(attempt_raw) != record['attempt_identity']:
            raise ContractError('exposure attempt identity differs')
        attempt = parse_json(attempt_raw, canonical=True)
        exact_keys(attempt, {'schema', 'cell', 'request_identity', 'outcome', 'worker_exit', 'detail',
            'completion_identity', 'wall_seconds_hex', 'missing_checkpoint_files'}, 'exposure attempt')
        if attempt['schema'] != 'trinite.music-exposure-attempt.v1' or attempt['cell'] != name \
                or attempt['request_identity'] != request_identity or attempt['outcome'] != record['outcome'] \
                or attempt['completion_identity'] != record['completion_identity'] \
                or record['outcome'] not in ('completed','failed','timed_out','interrupted') \
                or type(attempt['detail']) is not str \
                or attempt['missing_checkpoint_files'] != missing_checkpoint_files(root/name):
            raise ContractError('exposure attempt context differs')
        finite(attempt['wall_seconds_hex'])
        if attempt['worker_exit'] is not None and type(attempt['worker_exit']) is not int:
            raise ContractError('exposure worker exit invalid')
        if record['outcome'] == 'interrupted':
            interrupted = True
        if record['outcome'] == 'completed':
            if attempt['worker_exit'] != 0:
                raise ContractError('completed exposure worker did not exit successfully')
            reports[name] = audit(root/name, request_raw, request_identity, seed,
                                 record['completion_identity'])['report']
        elif record['completion_identity'] is not None:
            raise ContractError('failed exposure cell cannot certify completion')
    if {p.name for p in root.iterdir()} != {'request.json', 'matrix.json'} | attempted:
        raise ContractError('exposure matrix root membership differs')
    fit = len(reports) == len(SEEDS) and all(r['training_fit'] is True for r in reports.values())
    if matrix['both_seeds_fit'] is not fit or matrix['outcome'] != _outcome(matrix['cells'], fit):
        raise ContractError('exposure matrix outcome differs')
    validate(request_raw, request_identity)
    unchanged(root, before)
    return {'verified': True, 'matrix_identity': expected_identity, 'outcome': matrix['outcome'],
            'both_seeds_fit': fit, 'reports': reports,
            'predictions_replayed': False, 'optimizer_replayed': False, **CLAIMS}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('preflight','freeze','run','audit','summarize','_worker'))
    p.add_argument('--request', type=Path); p.add_argument('--request-identity'); p.add_argument('--output', type=Path)
    p.add_argument('--expected-identity'); p.add_argument('--authorize-run', action='store_true')
    p.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    p.add_argument('--expected-directory', type=int, nargs=2, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    try:
        if args.action == 'preflight':
            raw, receipt = dataset()
            print(json_bytes({'dataset_identity': identity(raw), 'admission_identity': identity(receipt),
                'schedules': {str(step): schedule_receipt(step) for step in CHECKPOINTS}, 'model_executed': False}).decode(), end='')
            return 0
        if args.action in ('run','_worker') and not args.authorize_run:
            raise ContractError('target run requires separate explicit owner authorization and --authorize-run')
        configure()
        if args.request is None:
            raise ContractError('--request required')
        if args.action == 'freeze':
            if args.request.resolve().is_relative_to(ROOT):
                raise ContractError('request must be outside immutable source checkout')
            for parent in args.request.absolute().parents:
                if parent.is_symlink() or not parent.is_dir():
                    raise ContractError('request parent must exist without symlink ancestors')
            raw = json_bytes(core()); publish(args.request, raw); print(identity(raw)); return 0
        if args.output is None or args.request_identity is None:
            raise ContractError('request identity and output required')
        if args.action == 'run':
            result = run_matrix(args.output, args.request, args.request_identity)
            print(json_bytes(result).decode(), end='')
            return 0 if result['outcome'] in ('training_fit_passed','training_fit_failed') else 1
        raw = read(args.request, 1024*1024); validate(raw, args.request_identity)
        if args.action == '_worker':
            if args.expected_directory is None:
                raise ContractError('internal worker requires pinned controller directory')
            protect(args.output, args.request)
            from .workflow import run
            print(run(args.output, raw, args.request_identity, args.seed, args.expected_directory)); return 0
        if not args.expected_identity:
            raise ContractError('external expected matrix identity required')
        print(json_bytes(audit_matrix(args.output, raw, args.request_identity, args.expected_identity)).decode(), end='')
        return 0
    except (ContractError, OSError, ValueError) as error:
        print(type(error).__name__+': '+str(error), file=sys.stderr)
        return 1
