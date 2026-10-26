"""Explicit preflight/freeze/run/audit/summarize controller with one worker deadline."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import threading

import torch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from .artifacts import Store, publish
from .request import ROOT, PROTOCOL, core, read, validate
from .mechanics import preflight


def configure():
    import trinite
    if Path(trinite.__file__).resolve().parent != ROOT/'src/trinite':
        raise ContractError('diagnostic runtime must use this source-bound checkout')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def launch(output, request, request_identity, expected_directory=None):
    """Only our fixed interpreter/script is executable, with validated selectors."""
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    store = Store(output, expected_directory)
    try:
        return _launch_pinned(store, output, request, request_identity, env)
    finally:
        store.close()


def _launch_pinned(store, output, request, request_identity, env):
    store.check()
    fd = os.open('worker.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=store.fd)
    with os.fdopen(fd, 'wb') as log:
        argv = [sys.executable, str(ROOT/'scripts/run_native_diagnostic.py'),
            '_worker', '--request', str(request), '--request-identity', request_identity,
            '--output', str(output), '--expected-directory', str(store.stat.st_dev), str(store.stat.st_ino)]
        # Fixed source-bound launcher, no shell; caller values never executable code.
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        with subprocess.Popen(argv, cwd=ROOT, env=env, shell=False,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            errors = []
            def drain():
                retained = 0
                try:
                    while True:
                        chunk = process.stdout.read(4096)
                        if not chunk: break
                        available = PROTOCOL['max_worker_log_bytes'] - retained
                        log.write(chunk[:available]); retained += min(len(chunk), available)
                        if len(chunk) > available:
                            errors.append('worker log exceeded bound'); process.kill(); break
                    log.flush()
                except Exception as error:
                    errors.append('worker log failure: '+type(error).__name__)
                    process.kill()
            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            try:
                code = process.wait(timeout=PROTOCOL['worker_seconds'])
            except BaseException:
                process.kill(); process.wait(); reader.join(); raise
            reader.join()
            store.check()
            if errors: raise ContractError('; '.join(errors))
            return subprocess.CompletedProcess(argv, code)


def run_controller(output, request, request_identity):
    output, request = Path(output).absolute(), Path(request).resolve()
    request_raw = read(request, 1024*1024)
    validate(request_raw, request_identity)
    # Output parent must exist. Never overwrite an earlier complete or failed attempt.
    output.mkdir(mode=0o700, exist_ok=False)
    store = Store(output); started = time.monotonic(); outcome = 'failed'; code = None
    detail = 'worker not completed'; completion = None
    try:
        result = launch(output, request, request_identity, (store.stat.st_dev, store.stat.st_ino))
        code = result.returncode
        if code:
            detail = 'worker exit '+str(code)+'; see retained worker.log'
        else:
            from .workflow import audit
            completion = identity(read(output/'completion.json'))
            audit(output, request_raw, request_identity, completion)
            outcome, detail = 'completed', 'closed and structurally audited; not full optimizer replay'
    except subprocess.TimeoutExpired:
        outcome, detail = 'timed_out', 'whole worker deadline exceeded; partial files retained'
    except KeyboardInterrupt:
        outcome, detail = 'interrupted', 'controller interrupted; partial files retained'
    except Exception as error:
        detail = type(error).__name__+': '+str(error)
    finally:
        record = {'schema':'trinite.native-diagnostic-attempt.v1', 'outcome':outcome,
                  'request_identity':request_identity, 'worker_exit':code, 'detail':detail,
                  'completion_identity':completion if outcome=='completed' else None,
                  'wall_seconds_hex':(time.monotonic()-started).hex()}
        try: store.write('attempt.json', json_bytes(record))
        finally: store.close()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight','freeze','run','audit','summarize','_worker'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--request-identity')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-identity')
    parser.add_argument('--expected-directory', type=int, nargs=2, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--authorize-run', action='store_true',
                        help='acknowledge separately reviewed target-run authorization; never set by tests')
    args = parser.parse_args(argv)
    try:
        configure()
        if args.action == 'preflight':
            print(json_bytes(preflight()).decode(), end=''); return 0
        if not args.request:
            raise ContractError('--request is required')
        if args.action == 'freeze':
            encoded = json_bytes(core(args.seed))
            publish(args.request, encoded)
            print(identity(encoded)); return 0
        if not args.output or not args.request_identity:
            raise ContractError('bound request and output required')
        request_raw = read(args.request, 1024*1024)
        validate(request_raw, args.request_identity)
        if args.action == 'run':
            if not args.authorize_run:
                raise ContractError('target run requires explicit --authorize-run and prior reviewed authorization')
            result = run_controller(args.output, args.request, args.request_identity)
            print(json_bytes(result).decode(), end='')
            return 0 if result['outcome']=='completed' else 1
        if args.action == '_worker':
            from .workflow import run
            if args.expected_directory is None:
                raise ContractError('internal worker requires controller directory identity')
            print(run(args.output, request_raw, args.request_identity, args.expected_directory)); return 0
        if not args.expected_identity:
            raise ContractError('audit/summarize requires independently retained --expected-identity')
        from .workflow import audit
        result = audit(args.output, request_raw, args.request_identity, args.expected_identity)
        if args.action == 'summarize':
            result['milestones'] = [{ 'step':step,
                'summary':parse_json(read(args.output/f'diagnostics-{step}.json'),canonical=True)['summary']}
                for step in PROTOCOL['milestones']]
        print(json_bytes(result).decode(), end=''); return 0
    except (ContractError, OSError, ValueError) as error:
        if args.action == '_worker' and args.output and args.output.is_dir():
            try:
                store = Store(args.output, args.expected_directory)
                try: store.write('failure.json', json_bytes({'schema':'trinite.native-diagnostic-failure.v1',
                    'outcome':'failed', 'error_type':type(error).__name__, 'detail':str(error)}))
                finally: store.close()
            except (OSError, ContractError): pass  # Primary error and worker.log remain authoritative.
        print(type(error).__name__+': '+str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
