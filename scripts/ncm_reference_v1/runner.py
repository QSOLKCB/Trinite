"""Fixed subprocess deadline, retained partials, atomic evidence and exact replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from .core import ROOT, digest, json_bytes, publish, read_bytes, request, source_bytes, verify_weights
from .evidence import LIMIT, pack, unpack, verify


def worker(directory, checkpoints):
    directory = Path(directory)
    req = json.loads(read_bytes(directory/'request.json', LIMIT))
    if json_bytes(req) != json_bytes(request()):
        raise ValueError('worker request/source/input mismatch')
    from .runtime import execute
    counter = 0
    def progress(value):
        nonlocal counter
        publish(directory/f'progress-{counter:03d}.json', json_bytes(value))
        counter += 1
    result, timings = execute(req, checkpoints, progress)
    publish(directory/'result.json', json_bytes(result))
    publish(directory/'timings.json', json_bytes(timings))
    return 0


def run(output, checkpoints, request_path, request_sha256):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    request_data = read_bytes(request_path, LIMIT)
    if digest(request_data) != request_sha256:
        raise ValueError('external request SHA-256 mismatch')
    req = json.loads(request_data)
    if json_bytes(req) != json_bytes(request()):
        raise ValueError('frozen request mismatch')
    sources = source_bytes()
    attempt = Path(tempfile.mkdtemp(prefix=output.name+'.attempt-', dir=output.parent))
    publish(attempt/'request.json', json_bytes(req))
    print('Retained attempt directory: ' + str(attempt), flush=True)
    env = dict(os.environ,
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    launcher = ROOT.parent/'run_ncm_reference.py'
    state, code = 'failed', None
    with (attempt/'worker.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-I', '-B', str(launcher), '_worker',
                                    '--attempt', str(attempt), '--checkpoints', str(Path(checkpoints).resolve())],
                                   stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            code = process.wait(timeout=req['protocol']['worker_timeout_seconds'])
            state = 'completed' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
            state = 'timeout'
        except KeyboardInterrupt:
            process.kill()
            code = process.wait()
            state = 'interrupted'
    if json_bytes(request()) != json_bytes(req):
        state = 'failed'
    members = {'request.json': json_bytes(req), 'status.json': json_bytes(dict(state=state, returncode=code)),
               'worker.log': read_bytes(attempt/'worker.log', 1024*1024)}
    members.update({'source/'+name: data for name,data in sources.items()})
    for path in sorted(attempt.glob('progress-*.json')):
        members['progress/'+path.name] = read_bytes(path, 65536)
    for name in ('result.json','timings.json'):
        if (attempt/name).exists():
            members[name] = read_bytes(attempt/name, LIMIT)
    data = pack(members)
    if state == 'completed':
        verify(data, digest(data))
    publish(output, data)
    print(digest(data), flush=True)
    return data, state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--output', required=True)
    for command in ('run', 'replay'):
        p = sub.add_parser(command)
        p.add_argument('--output', required=True)
        p.add_argument('--checkpoints', required=True)
        p.add_argument('--request', required=True)
        p.add_argument('--request-sha256', required=True)
        if command == 'replay':
            p.add_argument('--expected', required=True)
            p.add_argument('--expected-sha256', required=True)
    p = sub.add_parser('check-assets')
    p.add_argument('--checkpoints', required=True)
    p = sub.add_parser('verify')
    p.add_argument('--archive', required=True)
    p.add_argument('--expected-sha256', required=True)
    p = sub.add_parser('_worker')
    p.add_argument('--attempt', required=True)
    p.add_argument('--checkpoints', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'check-assets':
            verify_weights(args.checkpoints)
            print('Pinned local asset hashes verified; no inference executed')
            return 0
        if args.command == 'freeze':
            data = json_bytes(request())
            publish(args.output, data)
            print(digest(data))
            return 0
        if args.command == '_worker':
            return worker(args.attempt, args.checkpoints)
        if args.command == 'verify':
            print(json.dumps(verify(read_bytes(args.archive, LIMIT), args.expected_sha256), sort_keys=True))
            return 0
        expected = None
        if args.command == 'replay':
            receipt_path = Path(str(args.output)+'.replay.json')
            if receipt_path.exists() or receipt_path.is_symlink():
                raise FileExistsError(receipt_path)
            data = read_bytes(args.expected, LIMIT)
            if verify(data, args.expected_sha256)['state'] != 'completed':
                raise ValueError('cannot replay a failed run')
            expected = unpack(data, args.expected_sha256)['result.json']
        data, state = run(args.output, args.checkpoints, args.request, args.request_sha256)
        if state != 'completed':
            if expected is not None:
                publish(str(args.output)+'.replay.json', json_bytes(dict(
                    schema='ncm-reference-replay-v1', expected_archive=args.expected_sha256,
                    replay_archive=digest(data), exact_result_match=None, state=state,
                    timing_comparison=False, scope='inference replay failed; no equality established')))
            return 1
        if expected is not None:
            actual = unpack(data, digest(data))['result.json']
            equal = actual == expected
            receipt = dict(schema='ncm-reference-replay-v1', expected_archive=args.expected_sha256,
                           replay_archive=digest(data), exact_result_match=equal,
                           timing_comparison=False, scope='same-environment inference, not training replay')
            publish(str(args.output)+'.replay.json', json_bytes(receipt))
            if not equal:
                raise ValueError('exact inference/environment replay mismatch; both archives retained')
        return 0
    except Exception:
        traceback.print_exc()
        return 1
