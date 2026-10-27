"""One fresh worker, fixed launcher, bounded logs and retained failure receipts."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from native_diagnostic_v1.artifacts import Store, publish
from .request import ROOT, LIMITS, core, plan, read, validate


def configure():
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def protect(path, *inputs):
    path = Path(path).absolute(); resolved = path.resolve()
    if resolved.is_relative_to(ROOT) or ROOT.is_relative_to(resolved):
        raise ContractError('destination must be separate from source checkout')
    for parent in path.parents:
        if parent.is_symlink() or not parent.is_dir():
            raise ContractError('destination ancestors must be existing non-link directories')
    for source in inputs:
        source = Path(source).resolve()
        if source.is_relative_to(resolved) or resolved.is_relative_to(source):
            raise ContractError('destination overlaps immutable inputs')
    return path


def launch(root, request, expected, checkpoint, seconds, directory_identity):
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    store = Store(root, directory_identity)
    try:
        fd = os.open('worker.log', os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600, dir_fd=store.fd)
        with os.fdopen(fd, 'wb') as log:
            argv = [sys.executable, '-B', str(ROOT/'scripts/run_cpu_cost_ledger.py'), '_worker',
                    '--request', str(request), '--request-identity', expected, '--checkpoint', str(checkpoint),
                    '--output', str(root), '--authorize-run', '--expected-directory', *map(str, directory_identity)]
            # Fixed checkout launcher/interpreter, no shell or caller-selected executable.
            # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            with subprocess.Popen(argv, cwd=ROOT, env=env, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
                errors = []
                def drain():
                    count = 0
                    try:
                        while True:
                            block = process.stdout.read(4096)
                            if not block: break
                            remaining = LIMITS['worker_log_bytes']-count
                            log.write(block[:remaining]); count += min(len(block), remaining)
                            if len(block) > remaining:
                                errors.append('worker log exceeds frozen bound'); process.kill(); break
                        log.flush()
                    except Exception as error:
                        errors.append(type(error).__name__); process.kill()
                thread = threading.Thread(target=drain, daemon=True); thread.start()
                try:
                    code = process.wait(timeout=seconds)
                except BaseException:
                    process.kill(); process.wait(); thread.join(); raise
                thread.join(); store.check()
                if errors: raise ContractError('; '.join(errors))
                return code
    finally:
        store.close()


def run(output, request_path, expected, checkpoint):
    output = protect(output, request_path, checkpoint)
    request_path, checkpoint = Path(request_path).absolute(), Path(checkpoint).absolute()
    raw = read(request_path, LIMITS['request_bytes']); request = validate(raw, expected)
    if identity(read(checkpoint, LIMITS['checkpoint_bytes'])) != request['plan']['checkpoint_identity']:
        raise ContractError('checkpoint identity differs before worker admission')
    output.mkdir(mode=0o700, exist_ok=False); store = Store(output)
    outcome = 'failed'; detail = 'worker did not complete'; code = None; completion = None
    start = time.monotonic_ns()
    try:
        store.write('request.json', raw)
        try:
            code = launch(output, request_path, expected, checkpoint, request['plan']['worker_seconds'],
                          (store.stat.st_dev, store.stat.st_ino))
            if code:
                detail = f'worker exit {code}; see retained log'
            else:
                from .workflow import audit
                completion = identity(read(output/'completion.json'))
                audit(output, raw, expected, completion)
                outcome = 'completed'; detail = 'source/input binding and closed observation integrity checked; no numerical replay'
        except subprocess.TimeoutExpired:
            outcome = 'timed_out'; detail = 'whole-worker deadline enforced; partial artifacts retained'
        except KeyboardInterrupt:
            outcome = 'interrupted'; detail = 'worker terminated; partial artifacts retained'
        except Exception as error:
            detail = type(error).__name__+': '+str(error)
        attempt = {'schema': 'trinite.cpu-cost-ledger-attempt.v1', 'request_identity': expected,
                   'outcome': outcome, 'worker_exit': code, 'detail': detail,
                   'completion_identity': completion if outcome == 'completed' else None,
                   'controller_wall_ns': time.monotonic_ns()-start,
                   'scope': 'controller wall includes worker imports/loading/inference/audit and parent verification; process CPU measured inside worker only'}
        store.write('attempt.json', json_bytes(attempt))
        return attempt
    finally:
        store.close()


def main(argv=None, startup_start=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'freeze', 'run', 'audit', '_worker'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--request', type=Path); parser.add_argument('--request-identity')
    parser.add_argument('--checkpoint', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-identity'); parser.add_argument('--authorize-run', action='store_true')
    parser.add_argument('--expected-directory', type=int, nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.action in ('run', '_worker') and not args.authorize_run:
            raise ContractError('target execution requires separate owner authorization and --authorize-run')
        if args.action == 'preflight':
            if args.plan is None: raise ContractError('--plan required')
            print(json_bytes({'bounds': plan(parse_json(read(args.plan))), 'model_executed': False}).decode(), end=''); return 0
        configure()
        if args.request is None: raise ContractError('--request required')
        if args.action == 'freeze':
            if args.plan is None or args.checkpoint is None: raise ContractError('--plan and --checkpoint required')
            value = parse_json(read(args.plan)); plan(value)
            # Hash/size binding only, no model construction or forward at freeze.
            if identity(read(args.checkpoint, LIMITS['checkpoint_bytes'])) != value['checkpoint_identity']:
                raise ContractError('checkpoint identity differs')
            request = json_bytes(core(value))
            if len(request) > LIMITS['request_bytes']: raise ContractError('request exceeds bound')
            publish(protect(args.request, args.plan, args.checkpoint), request)
            print(identity(request)); return 0
        if args.output is None or args.request_identity is None:
            raise ContractError('--output and --request-identity required')
        if args.action == 'run':
            if args.checkpoint is None: raise ContractError('--checkpoint required')
            result = run(args.output, args.request, args.request_identity, args.checkpoint)
            print(json_bytes(result).decode(), end=''); return 0 if result['outcome'] == 'completed' else 1
        raw = read(args.request, LIMITS['request_bytes'])
        if args.action == '_worker':
            if args.expected_directory is None or args.checkpoint is None:
                raise ContractError('worker requires checkpoint and pinned controller directory')
            protect(args.output, args.request, args.checkpoint)
            from .measure import clock, elapsed
            from .workflow import run as worker_run
            if startup_start is None: startup_start = clock()
            startup = elapsed(startup_start)
            print(worker_run(args.output, raw, args.request_identity, args.checkpoint, args.expected_directory, startup, startup_start)); return 0
        if args.expected_identity is None: raise ContractError('external --expected-identity required')
        from .workflow import audit
        result = audit(args.output, raw, args.request_identity, args.expected_identity)
        print(json_bytes(result).decode(), end=''); return 0
    except (ContractError, OSError, ValueError) as error:
        print(type(error).__name__+': '+str(error), file=sys.stderr); return 1
