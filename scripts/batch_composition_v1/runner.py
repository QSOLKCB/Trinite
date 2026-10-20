"""Fixed fresh-worker launch, hard deadline and no-clobber evidence publication."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time
import uuid
from trinite.contracts import ContractError, identity, json_bytes, parse_json, require_identity
from frozen_decoding_transport_v2.artifacts import publish, write_all
from .spec import ROOT, MAX_SECONDS, MAX_REPORT, MAX_REQUEST, read
from .validation import validate_report


def prepare():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def worker(args):
    # The parent deadline starts before process creation, so imports/loading are charged.
    prepare()
    import resource
    from .spec import request
    from .experiment import report
    started = time.monotonic()
    binding = json_bytes(request())
    raw = read(args.request, MAX_REQUEST)
    if identity(raw) != require_identity(args.request_identity) or raw != binding:
        raise ContractError('current source/input/environment request mismatch')
    expected = None
    if args.worker_action == 'verify':
        expected = read(args.output, MAX_REPORT)
        if identity(expected) != require_identity(args.expected_identity):
            raise ContractError('externally bound expected report identity mismatch')
        old = parse_json(expected, canonical=True)
        if old.get('schema') != 'trinite.batch-composition-report.v2' or old.get('request') != parse_json(raw, canonical=True):
            raise ContractError('expected report schema/request mismatch')
        validate_report(old, parse_json(raw, canonical=True))
    elif args.output.exists() or args.output.is_symlink():
        raise FileExistsError('final output already exists')
    attempt = args.attempt
    partial_path = attempt/'completed-cells.jsonl'
    layout_path = attempt/'completed-layouts.jsonl'
    with partial_path.open('xb') as partial, layout_path.open('xb') as layouts:
        def retain(stream, value):
            data = json_bytes(value)
            if stream.tell() + len(data) > MAX_REPORT:
                raise ContractError('partial records exceed byte bound')
            write_all(stream, data)
            stream.flush()
        value = report(on_cell=lambda cell: retain(partial, cell),
                       on_layout=lambda layout: retain(layouts, layout))
    validate_report(value, parse_json(binding, canonical=True))
    actual = json_bytes(value)
    if len(actual) > MAX_REPORT:
        raise ContractError('report exceeds byte bound')
    # Always retain recomputed bytes before comparison, including negative/mismatch evidence.
    publish(attempt/'computed-report.json', actual)
    metrics = {'wall_seconds_hex': (time.monotonic()-started).hex(),
               'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'scope': 'whole worker process observation, not a speed or whole-system memory claim'}
    publish(attempt/'measurements.json', json_bytes(metrics))
    if args.worker_action == 'verify':
        matched = actual == expected
        receipt = {'schema': 'trinite.batch-composition-replay-receipt.v2',
                   'request_identity': identity(raw), 'expected_identity': args.expected_identity,
                   'actual_identity': identity(actual), 'exact_match': matched,
                   'conforms': value['conforms'], 'scope': 'same-environment complete inference replay, not optimizer replay'}
        publish(attempt/'replay-receipt.json', json_bytes(receipt))
        if not matched:
            raise ContractError('exact report replay mismatch; both results and false-match receipt retained')
    else:
        publish(args.output, actual)
    print(identity(actual), flush=True)
    print('conforms=' + str(value['conforms']).lower(), flush=True)
    return 0 if value['conforms'] else 2


def launch(args):
    if not args.output or not args.request_identity:
        raise ContractError('run/verify requires output and externally retained request identity')
    require_identity(args.request_identity)
    if args.action == 'verify':
        require_identity(args.expected_identity)
    elif args.expected_identity is not None:
        raise ContractError('expected identity is only valid for replay')
    elif args.output.exists() or args.output.is_symlink():
        raise FileExistsError('final output already exists')
    # Validate accessible, bounded bytes before launching; worker additionally compares current bindings.
    if identity(read(args.request, MAX_REQUEST)) != args.request_identity:
        raise ContractError('request identity mismatch')
    if args.action == 'verify' and identity(read(args.output, MAX_REPORT)) != args.expected_identity:
        raise ContractError('expected report identity mismatch')
    attempt = args.output.with_name(args.output.name+'.attempt-'+uuid.uuid4().hex)
    attempt.mkdir()
    print('attempt_directory=' + str(attempt), flush=True)
    publish(attempt/'invocation.json', json_bytes({'schema': 'trinite.batch-composition-invocation.v2',
        'action': args.action, 'request_identity': args.request_identity, 'expected_identity': args.expected_identity,
        'deadline_seconds': MAX_SECONDS, 'final_name': args.output.name}))
    command = [sys.executable, '-B', str(ROOT/'scripts/run_batch_composition.py'), '_worker',
               '--worker-action', args.action, '--request', str(args.request.resolve()),
               '--request-identity', args.request_identity, '--output', str(args.output.absolute()),
               '--attempt', str(attempt.resolve())]
    if args.expected_identity:
        command += ['--expected-identity', args.expected_identity]
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'scripts'),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    start = time.monotonic()
    outcome, code = 'failed', 1
    with (attempt/'worker.log').open('xb') as log:
        process = None
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait(timeout=max(0.0, MAX_SECONDS-(time.monotonic()-start)))
            outcome = 'completed' if code == 0 else 'completed_nonconformant' if code == 2 else 'failed'
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            outcome, code = 'timeout', 124
        except OSError as error:
            if process is not None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait()
            outcome, code = ('launch_failed' if process is None else 'controller_failed'), 1
            write_all(log, (type(error).__name__ + ': ' + str(error) + '\n').encode())
        except BaseException:
            if process is not None:
                process.kill()
                process.wait()
            publish(attempt/'status.json', json_bytes({'schema': 'trinite.batch-composition-status.v2',
                'outcome': 'interrupted', 'exit_code': 130, 'wall_seconds_hex': (time.monotonic()-start).hex()}))
            raise
    publish(attempt/'status.json', json_bytes({'schema': 'trinite.batch-composition-status.v2',
        'outcome': outcome, 'exit_code': code, 'wall_seconds_hex': (time.monotonic()-start).hex()}))
    print('worker_outcome=' + outcome, flush=True)
    if code in (0, 2):
        computed = read(attempt/'computed-report.json', MAX_REPORT)
        print(identity(computed), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'verify', '_worker'))
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-identity')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-identity')
    parser.add_argument('--worker-action', choices=('run', 'verify'), help=argparse.SUPPRESS)
    parser.add_argument('--attempt', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.action == 'freeze':
            if any((args.output, args.request_identity, args.expected_identity, args.worker_action, args.attempt)):
                raise ContractError('freeze takes only a new request path')
            prepare()
            from .spec import request
            raw = json_bytes(request())
            if len(raw) > MAX_REQUEST:
                raise ContractError('request exceeds byte bound')
            publish(args.request, raw)
            print(identity(raw))
            return 0
        if args.action == '_worker':
            if not args.worker_action or not args.attempt or not args.output:
                raise ContractError('internal worker requires fixed launch arguments')
            return worker(args)
        if args.worker_action or args.attempt:
            raise ContractError('internal arguments are not public run options')
        return launch(args)
    except (ContractError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr, flush=True)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
