"""Bounded two-seed training then receipt-gated heldout execution."""
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
from .spec import PROTOCOL, SEEDS, dataset, schedule_receipt
from .workflow import finite, CLAIMS

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


def launch(root, request, request_identity, seed, phase, expected_directory, gate_identity=None):
    store=Store(root,expected_directory)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        fd=os.open('worker.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=store.fd)
        with os.fdopen(fd,'wb') as log:
            argv=[sys.executable,'-I','-B',str(ROOT/'scripts/run_music_contrast.py'),'_worker',
                '--request',str(request),'--request-identity',request_identity,'--output',str(root),
                '--authorize-run','--seed',str(seed),'--phase',phase,'--expected-directory',str(store.stat.st_dev),str(store.stat.st_ino)]
            if gate_identity is not None:
                argv += ['--gate-identity', gate_identity]
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
        raise ContractError('contrast cell byte bound exceeded')
    if 'worker.log' in before['members']:
        read(Path(root)/'worker.log', PROTOCOL['max_worker_log_bytes'])
    result={name:identity(read(Path(root)/name)) for name,meta in before['members'].items()
            if name!='.' and not (meta[-1]&0o170000)==0o040000}
    unchanged(root,before)
    return result


def _cell(output, request_path, request_raw, request_identity, seed, phase, gate_identity):
    name = cell_name(seed, phase)
    root = output/name
    root.mkdir(mode=0o700, exist_ok=False)
    child = Store(root)
    started = time.monotonic()
    code, completion, outcome, detail = None, None, 'failed', 'worker did not complete'
    try:
        code = launch(root, request_path, request_identity, seed, phase,
                      (child.stat.st_dev, child.stat.st_ino), gate_identity)
        if code:
            detail = f'worker exit {code}; see retained log'
        else:
            from .workflow import audit
            completion = identity(read(root/'completion.json'))
            audit(root, request_raw, request_identity, seed, phase, completion, gate_identity)
            outcome, detail = 'completed', 'retained byte/structure/score checks passed; no numerical replay'
    except subprocess.TimeoutExpired:
        outcome, detail = 'timed_out', 'fixed whole-worker deadline exceeded'
    except KeyboardInterrupt:
        outcome, detail = 'interrupted', 'controller interrupted'
    except Exception as error:
        detail = type(error).__name__+': '+str(error)
    finally:
        attempt = {'schema': 'trinite.music-contrast-attempt.v1', 'cell': name,
            'request_identity': request_identity, 'outcome': outcome, 'worker_exit': code,
            'detail': detail, 'completion_identity': completion if outcome == 'completed' else None,
            'wall_seconds_hex': (time.monotonic()-started).hex()}
        try:
            child.write('attempt.json', json_bytes(attempt))
        finally:
            child.close()
    return {'outcome': outcome, 'attempt_identity': identity(json_bytes(attempt)),
            'completion_identity': attempt['completion_identity'], 'retained_files': bound_files(root)}


def _gate(root, records, request_raw, request_identity):
    from .workflow import audit
    completions, fit = {}, True
    for seed in SEEDS:
        name = cell_name(seed, 'train')
        record = records[name]
        completion = record.get('completion_identity') if record['outcome'] == 'completed' else None
        completions[name] = completion
        if completion is None:
            fit = False
        else:
            report = audit(root/name, request_raw, request_identity, seed, 'train', completion)['report']
            fit = fit and report['training_fit'] is True
    return {'schema': 'trinite.music-contrast-gate.v1', 'request_identity': request_identity,
            'passed': fit, 'training_completions': completions}


def _outcome(records, passed):
    train_complete = all(records[cell_name(s, 'train')]['outcome'] == 'completed' for s in SEEDS)
    heldout_complete = all(records[cell_name(s, 'heldout')]['outcome'] == 'completed' for s in SEEDS)
    return 'completed' if passed and heldout_complete else ('training_fit_failed' if train_complete and not passed else 'incomplete')


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
            name = cell_name(seed, 'train')
            if interrupted:
                records[name] = {'outcome': 'not_attempted', 'reason': 'earlier controller interruption'}
            else:
                records[name] = _cell(output, request_path, raw, request_identity, seed, 'train', None)
                interrupted = records[name]['outcome'] == 'interrupted'
        gate = _gate(output, records, raw, request_identity)
        gate_raw = json_bytes(gate)
        store.write('gate.json', gate_raw)
        gate_identity = identity(gate_raw)
        for seed in SEEDS:
            name = cell_name(seed, 'heldout')
            if not gate['passed']:
                records[name] = {'outcome': 'not_attempted', 'reason': 'both-seed training-fit gate not met'}
            elif interrupted:
                records[name] = {'outcome': 'not_attempted', 'reason': 'earlier controller interruption'}
            else:
                records[name] = _cell(output, request_path, raw, request_identity, seed, 'heldout', gate_identity)
                interrupted = records[name]['outcome'] == 'interrupted'
        validate(raw, request_identity)
        matrix = {'schema': 'trinite.music-contrast-matrix.v1', 'request_identity': request_identity,
            'cells': records, 'gate_identity': gate_identity,
            'outcome': _outcome(records, gate['passed']), **CLAIMS}
        encoded = json_bytes(matrix)
        store.write('matrix.json', encoded)
        if inventory(output)['bytes'] > PROTOCOL['max_total_payload_bytes']:
            raise ContractError('contrast aggregate retained payload exceeds bound')
        return {'outcome': matrix['outcome'], 'matrix_identity': identity(encoded),
                'training_gate_passed': gate['passed'], 'cells': records}
    finally:
        store.close()


def audit_matrix(root, request_raw, request_identity, expected_identity):
    root = Path(root)
    validate(request_raw, request_identity)
    require_identity(expected_identity)
    before = inventory(root)
    if before['bytes'] > PROTOCOL['max_total_payload_bytes']:
        raise ContractError('contrast aggregate retained payload exceeds bound')
    raw = read(root/'matrix.json')
    if identity(raw) != expected_identity:
        raise ContractError('matrix expected identity differs')
    matrix = parse_json(raw, canonical=True)
    exact_keys(matrix, {'schema', 'request_identity', 'cells', 'gate_identity', 'outcome', *CLAIMS}, 'contrast matrix')
    if matrix['schema'] != 'trinite.music-contrast-matrix.v1' or matrix['request_identity'] != request_identity \
            or read(root/'request.json') != request_raw or any(matrix[k] is not False for k in CLAIMS):
        raise ContractError('contrast matrix request/schema/claims differ')
    expected_cells = {cell_name(s, p) for p in ('train', 'heldout') for s in SEEDS}
    exact_keys(matrix['cells'], expected_cells, 'contrast matrix cells')
    reports, attempted, interrupted = {}, set(), False
    from .workflow import audit
    gate = None
    for phase in ('train', 'heldout'):
        if phase == 'heldout':
            expected_gate = _gate(root, matrix['cells'], request_raw, request_identity)
            gate_raw = read(root/'gate.json', 1024*1024)
            if identity(gate_raw) != matrix['gate_identity'] or gate_raw != json_bytes(expected_gate):
                raise ContractError('contrast matrix training gate differs')
            gate = expected_gate
        for seed in SEEDS:
            name = cell_name(seed, phase)
            record = matrix['cells'][name]
            if record.get('outcome') == 'not_attempted':
                reason = 'both-seed training-fit gate not met' if phase == 'heldout' and not gate['passed'] else 'earlier controller interruption'
                if (reason == 'earlier controller interruption' and not interrupted) or record != {'outcome':'not_attempted', 'reason':reason}:
                    raise ContractError('unjustified unattempted contrast cell')
                continue
            if interrupted or (phase == 'heldout' and not gate['passed']):
                raise ContractError('contrast cell executed after interruption or failed gate')
            attempted.add(name)
            exact_keys(record, {'outcome', 'attempt_identity', 'completion_identity', 'retained_files'}, 'contrast cell')
            if bound_files(root/name) != record['retained_files']:
                raise ContractError('contrast retained cell membership differs')
            attempt_raw = read(root/name/'attempt.json')
            if identity(attempt_raw) != record['attempt_identity']:
                raise ContractError('contrast attempt identity differs')
            attempt = parse_json(attempt_raw, canonical=True)
            exact_keys(attempt, {'schema','cell','request_identity','outcome','worker_exit','detail','completion_identity','wall_seconds_hex'}, 'contrast attempt')
            if attempt['schema'] != 'trinite.music-contrast-attempt.v1' or attempt['cell'] != name \
                    or attempt['request_identity'] != request_identity or attempt['outcome'] != record['outcome'] \
                    or attempt['completion_identity'] != record['completion_identity'] \
                    or record['outcome'] not in ('completed','failed','timed_out','interrupted') \
                    or type(attempt['detail']) is not str:
                raise ContractError('contrast attempt context differs')
            finite(attempt['wall_seconds_hex'])
            if attempt['worker_exit'] is not None and type(attempt['worker_exit']) is not int:
                raise ContractError('contrast worker exit invalid')
            if record['outcome'] == 'interrupted':
                interrupted = True
            if record['outcome'] == 'completed':
                if attempt['worker_exit'] != 0:
                    raise ContractError('completed contrast worker did not exit successfully')
                reports[name] = audit(root/name, request_raw, request_identity, seed, phase,
                    record['completion_identity'], matrix['gate_identity'] if phase == 'heldout' else None)['report']
            elif record['completion_identity'] is not None:
                raise ContractError('failed contrast cell cannot certify completion')
    expected_paths = {'request.json', 'gate.json', 'matrix.json'} | attempted
    if {p.name for p in root.iterdir()} != expected_paths:
        raise ContractError('contrast matrix root membership differs')
    if matrix['outcome'] != _outcome(matrix['cells'], gate['passed']):
        raise ContractError('contrast matrix outcome differs')
    validate(request_raw, request_identity)
    unchanged(root, before)
    return {'verified': True, 'matrix_identity': expected_identity, 'outcome': matrix['outcome'],
            'training_gate_passed': gate['passed'], 'reports': reports,
            'predictions_replayed': False, 'optimizer_replayed': False, **CLAIMS}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('preflight','freeze','run','audit','summarize','_worker'))
    p.add_argument('--request', type=Path); p.add_argument('--request-identity'); p.add_argument('--output', type=Path)
    p.add_argument('--expected-identity'); p.add_argument('--authorize-run', action='store_true')
    p.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    p.add_argument('--phase', choices=('train','heldout'), help=argparse.SUPPRESS)
    p.add_argument('--expected-directory', type=int, nargs=2, help=argparse.SUPPRESS)
    p.add_argument('--gate-identity', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    try:
        if args.action == 'preflight':
            raw, receipt = dataset()
            print(json_bytes({'dataset_identity': identity(raw), 'admission_identity': identity(receipt),
                'schedule': schedule_receipt(), 'model_executed': False}).decode(), end='')
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
            return 0 if result['outcome'] in ('completed','training_fit_failed') else 1
        raw = read(args.request, 1024*1024); validate(raw, args.request_identity)
        if args.action == '_worker':
            if args.expected_directory is None:
                raise ContractError('internal worker requires pinned controller directory')
            protect(args.output, args.request)
            from .workflow import run
            print(run(args.output, raw, args.request_identity, args.seed, args.phase,
                      args.expected_directory, args.gate_identity)); return 0
        if not args.expected_identity:
            raise ContractError('external expected matrix identity required')
        print(json_bytes(audit_matrix(args.output, raw, args.request_identity, args.expected_identity)).decode(), end='')
        return 0
    except (ContractError, OSError, ValueError) as error:
        print(type(error).__name__+': '+str(error), file=sys.stderr)
        return 1
