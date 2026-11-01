#!/usr/bin/env python3
"""Checkout-only, one-episode queue evaluation. No model or training imports."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import selectors
import signal
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CRATE = 'external/queue-evaluator-v1'
BINARY = CRATE + '/target/release/trinite-queue-evaluator-v1'
SOURCE_FILES = (
    'scripts/queue_evaluation_v1.py',
    CRATE + '/Cargo.toml', CRATE + '/Cargo.lock', CRATE + '/rust-toolchain.toml',
    CRATE + '/src/main.rs',
    'external/queue-sim-v1/Cargo.toml', 'external/queue-sim-v1/src/lib.rs',
    'external/queue-sim-v1/src/json.rs',
)
PROTOCOL = {'schema': 'trinite.queue-protocol.v1', 'target': 'forecast.dropped',
            'prediction': 'integer_or_abstain', 'max_per_tick': 32,
            'scoring': 'absolute_error_and_exact_match', 'episodes': 1}
LIMIT = 262144
PROCESS_LIMIT = 65536
TIMEOUT = 5.0

class GateError(ValueError):
    pass

class ProcessError(GateError):
    def __init__(self, evidence):
        super().__init__('runner failed; inspect retained process evidence')
        self.evidence = evidence


def require(condition, message):
    if not condition:
        raise GateError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('ascii')


def decode(data):
    require(len(data) <= LIMIT, 'JSON exceeds bound')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def bad(_):
        raise GateError('only integer JSON numbers are admitted')
    try:
        value = json.loads(data.decode('ascii'), object_pairs_hook=pairs,
                           parse_float=bad, parse_constant=bad)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GateError('invalid bounded JSON') from error
    def walk(item, depth=0):
        require(depth <= 12, 'JSON nesting exceeds bound')
        if type(item) is int:
            require(-(2**63) <= item < 2**63, 'integer overflow')
        elif isinstance(item, dict):
            for key, val in item.items():
                require(isinstance(key, str) and key.isascii(), 'non-ASCII key')
                walk(val, depth + 1)
        elif isinstance(item, list):
            for val in item:
                walk(val, depth + 1)
        elif isinstance(item, str):
            require(item.isascii(), 'non-ASCII string')
        else:
            require(type(item) in (bool, type(None)), 'invalid JSON value')
    walk(value)
    return value


def keys(obj, expected):
    require(type(obj) is dict and set(obj) == set(expected), 'unexpected object fields')


def integer(value, lo, hi):
    require(type(value) is int and lo <= value <= hi, 'integer outside bounds')
    return value


def hash_arg(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'expected SHA-256')
    return value


def safe_path(path):
    path = Path(os.path.abspath(path))
    for part in reversed((path,) + tuple(path.parents)):
        if part.exists() or part.is_symlink():
            require(not stat.S_ISLNK(part.lstat().st_mode), 'symlink path rejected')
    return path


def read(path, limit=LIMIT, allow_links=False):
    path = safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and (allow_links or info.st_nlink == 1), 'expected single-link regular file')
        require(info.st_size <= limit, 'file exceeds bound')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(limit + 1)
        require(len(data) <= limit, 'file exceeds bound')
        return data
    finally:
        os.close(fd)


def write(path, data):
    path = safe_path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def disjoint(*paths):
    paths = [safe_path(p) for p in paths]
    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            require(a != b and a not in b.parents and b not in a.parents, 'overlapping paths')
    return paths


def fresh(path):
    path = safe_path(path)
    require(ROOT != path and ROOT not in path.parents, 'artifacts must be outside checkout')
    path.mkdir(mode=0o700)  # Parent must exist; existing/partial attempts are never overwritten.
    return path


def bundle(path, names):
    path = safe_path(path)
    require(path.is_dir(), 'expected artifact directory')
    require(set(p.name for p in path.iterdir()) == set(names), 'incomplete or extra artifact set')
    return {name: read(path / name) for name in names}


def failure(path, error):
    record = {'schema': 'trinite.queue-attempt-failure.v1', 'success': False,
              'reason': 'operation_failed'}
    if isinstance(error, ProcessError):
        record['process'] = error.evidence
    try:
        write(path / 'failure.json', encode(record))
    except OSError:
        pass


def source_identity():
    return {name: digest(read(ROOT / name)) for name in SOURCE_FILES}


def binary_identity():
    return digest(read(ROOT / BINARY, 64 * 1024 * 1024, allow_links=True))


def bounded_process(argv, data, timeout=TIMEOUT, cap=PROCESS_LIMIT):
    """Internal process primitive; public CLI cannot supply argv or a runner path."""
    started = time.monotonic()
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=ROOT, start_new_session=True,
                            env={'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C'})
    output = {'stdout': bytearray(), 'stderr': bytearray()}
    reason = None
    sel = selectors.DefaultSelector()
    for stream, name, mode in ((proc.stdin, 'stdin', selectors.EVENT_WRITE),
                              (proc.stdout, 'stdout', selectors.EVENT_READ),
                              (proc.stderr, 'stderr', selectors.EVENT_READ)):
        os.set_blocking(stream.fileno(), False)
        sel.register(stream, mode, name)
    offset = 0
    try:
        while sel.get_map():
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                reason = 'timeout'
                break
            for key, _ in sel.select(min(remaining, 0.1)):
                stream, name = key.fileobj, key.data
                if name == 'stdin':
                    try:
                        count = os.write(stream.fileno(), data[offset:offset + 4096])
                        offset += count
                    except BrokenPipeError:
                        offset = len(data)
                    if offset == len(data):
                        sel.unregister(stream)
                        stream.close()
                else:
                    chunk = os.read(stream.fileno(), 4096)
                    if not chunk:
                        sel.unregister(stream)
                        stream.close()
                    else:
                        allowed = cap - len(output[name])
                        output[name].extend(chunk[:allowed])
                        if len(chunk) > allowed:
                            reason = 'output_limit'
                            break
            if reason:
                break
        if not reason:
            try:
                proc.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                reason = 'timeout'
    finally:
        if proc.poll() is None or reason:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait()
        sel.close()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if not stream.closed:
                stream.close()
    evidence = {'exit_code': proc.returncode, 'failure': reason,
                'stdout_b64': base64.b64encode(output['stdout']).decode(),
                'stderr_b64': base64.b64encode(output['stderr']).decode()}
    if reason or proc.returncode != 0:
        raise ProcessError(evidence)
    return bytes(output['stdout']), evidence


def run(operation, data):
    require(operation in ('observe', 'target'), 'unknown runner operation')
    safe_path(ROOT / BINARY)
    return bounded_process([str(ROOT / BINARY), operation], data)


def observation(data):
    obj = decode(data)
    keys(obj, ('schema', 'capacity', 'observe_ticks', 'horizon', 'rows'))
    require(obj['schema'] == 'trinite.queue-observation.v1', 'wrong observation schema')
    integer(obj['capacity'], 1, 128)
    n = integer(obj['observe_ticks'], 1, 255)
    horizon = integer(obj['horizon'], 1, 255)
    require(n + horizon <= 256 and type(obj['rows']) is list and len(obj['rows']) == n, 'invalid observation length')
    for tick, row in enumerate(obj['rows']):
        keys(row, ('tick', 'outage', 'values', 'observed'))
        require(type(row['tick']) is int and row['tick'] == tick and type(row['outage']) is bool, 'invalid tick')
        values, mask = row['values'], row['observed']
        require(type(values) is list and type(mask) is list and len(values) == len(mask) == 6, 'invalid measurements')
        for value, known, maximum in zip(values, mask, (obj['capacity'], 32, 32, 32, 32, tick + 1)):
            integer(value, 0, maximum)
            require(type(known) is bool and (known or value == 0), 'invalid missing measurement')
    return obj


def prediction(data, observation_bytes):
    obs = observation(observation_bytes)
    pred = decode(data)
    require(type(pred) is dict, 'expected prediction object')
    status = pred.get('status')
    require(status in ('prediction', 'abstain'), 'invalid prediction status')
    keys(pred, ('schema', 'observation_sha256', 'status', 'value') if status == 'prediction'
         else ('schema', 'observation_sha256', 'status'))
    require(pred['schema'] == 'trinite.queue-prediction.v1' and
            pred['observation_sha256'] == digest(observation_bytes), 'prediction identity mismatch')
    if status == 'prediction':
        integer(pred['value'], 0, 32 * obs['horizon'])
    return pred


PUBLIC = ('observation.json', 'protocol.json')
PREPARED = ('request.json', 'observation.json', 'protocol.json', 'process.json', 'manifest.json')
COMMITTED = ('observation.json', 'protocol.json', 'prediction.json', 'manifest.json')
RESULT = ('evaluation.json', 'process.json')


def prepare(request_file, out, public_out):
    request_file, out, public_out = disjoint(request_file, out, public_out)
    private = fresh(out)
    try:
        public = fresh(public_out)
        request_bytes = read(request_file, 16384)
        write(private / 'request.json', request_bytes)
        sources, binary = source_identity(), binary_identity()
        observed, evidence = run('observe', request_bytes)
        write(private / 'process.json', encode(evidence))
        observation(observed)
        protocol = encode(PROTOCOL)
        for directory in (private, public):
            write(directory / 'observation.json', observed)
            write(directory / 'protocol.json', protocol)
        manifest = {'schema': 'trinite.queue-preparation.v1',
                    'files': {name: digest(read(private / name)) for name in PREPARED if name != 'manifest.json'},
                    'sources': sources, 'binary_sha256': binary,
                    'environment': {'python': platform.python_version(), 'system': platform.system(),
                                    'machine': platform.machine(), 'toolchain': 'declared Rust 1.90.0'}}
        raw = encode(manifest)
        write(private / 'manifest.json', raw)
        return digest(raw)
    except Exception as error:
        failure(private, error)
        raise


def commit(public_dir, prediction_file, out):
    public_dir, prediction_file, out = disjoint(public_dir, prediction_file, out)
    destination = fresh(out)
    try:
        files = bundle(public_dir, PUBLIC)
        require(files['protocol.json'] == encode(PROTOCOL), 'protocol mismatch')
        pred = read(prediction_file)
        prediction(pred, files['observation.json'])
        files['prediction.json'] = pred
        for name, raw in files.items():
            write(destination / name, raw)
        raw = encode({'schema': 'trinite.queue-commitment.v1',
                      'files': {name: digest(raw) for name, raw in files.items()}})
        write(destination / 'manifest.json', raw)
        return digest(raw)
    except Exception as error:
        failure(destination, error)
        raise


def committed(directory, expected):
    hash_arg(expected)
    files = bundle(directory, COMMITTED)
    require(digest(files['manifest.json']) == expected, 'commitment anchor mismatch')
    manifest = decode(files['manifest.json'])
    keys(manifest, ('schema', 'files'))
    require(manifest['schema'] == 'trinite.queue-commitment.v1', 'wrong commitment schema')
    require(manifest['files'] == {name: digest(raw) for name, raw in files.items() if name != 'manifest.json'}, 'commitment bytes changed')
    require(files['protocol.json'] == encode(PROTOCOL), 'protocol mismatch')
    pred = prediction(files['prediction.json'], files['observation.json'])
    return files, pred


def evaluate(prepared_dir, commit_dir, prepare_sha, commit_sha):
    # Keep this ordering: malformed public commitment must fail before private file access.
    files, pred = committed(commit_dir, commit_sha)
    hash_arg(prepare_sha)
    private = bundle(prepared_dir, PREPARED)
    require(digest(private['manifest.json']) == prepare_sha, 'preparation anchor mismatch')
    manifest = decode(private['manifest.json'])
    keys(manifest, ('schema', 'files', 'sources', 'binary_sha256', 'environment'))
    require(manifest['schema'] == 'trinite.queue-preparation.v1', 'wrong preparation schema')
    require(manifest['files'] == {name: digest(raw) for name, raw in private.items() if name != 'manifest.json'}, 'preparation bytes changed')
    require(manifest['sources'] == source_identity() and manifest['binary_sha256'] == binary_identity(), 'source or binary changed')
    require(private['observation.json'] == files['observation.json'] and private['protocol.json'] == files['protocol.json'], 'public/private mismatch')
    observed, observe_evidence = run('observe', private['request.json'])
    try:
        require(observed == files['observation.json'], 'observation replay mismatch')
        require(encode(observe_evidence) == private['process.json'], 'preparation process replay mismatch')
    except GateError as error:
        raise ProcessError({'observe': observe_evidence, 'failure': 'invalid_observation'}) from error
    target, target_evidence = run('target', private['request.json'])
    try:
        actual = decode(target)
        keys(actual, ('schema', 'dropped'))
        require(actual['schema'] == 'trinite.queue-target.v1', 'wrong target schema')
        bound = 32 * observation(observed)['horizon']
        integer(actual['dropped'], 0, bound)
    except GateError as error:
        raise ProcessError({'observe': observe_evidence, 'target': target_evidence,
                            'failure': 'invalid_target'}) from error
    result = {'schema': 'trinite.queue-evaluation.v1', 'prepare_sha256': prepare_sha,
              'commit_sha256': commit_sha, 'observation_sha256': digest(observed),
              'target': 'forecast.dropped', 'actual': actual['dropped'], 'status': pred['status']}
    if pred['status'] == 'prediction':
        result.update(predicted=pred['value'], absolute_error=abs(pred['value'] - actual['dropped']),
                      exact_match=pred['value'] == actual['dropped'])
    evidence = {'observe': observe_evidence, 'target': target_evidence}
    return {'evaluation.json': encode(result), 'process.json': encode(evidence)}


def score(prepared_dir, commit_dir, prepare_sha, commit_sha, out):
    # Do not inspect private paths until the public commitment is checked.
    committed(commit_dir, commit_sha)
    prepared_dir, commit_dir, out = disjoint(prepared_dir, commit_dir, out)
    destination = fresh(out)
    try:
        files = evaluate(prepared_dir, commit_dir, prepare_sha, commit_sha)
        for name, raw in files.items():
            write(destination / name, raw)
    except Exception as error:
        failure(destination, error)
        raise


def verify(prepared_dir, commit_dir, prepare_sha, commit_sha, result_dir):
    committed(commit_dir, commit_sha)
    disjoint(prepared_dir, commit_dir, result_dir)
    expected = evaluate(prepared_dir, commit_dir, prepare_sha, commit_sha)
    require(bundle(result_dir, RESULT) == expected, 'evaluation replay mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--request', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--public-out', required=True)
    p = sub.add_parser('commit')
    p.add_argument('--public', required=True)
    p.add_argument('--prediction', required=True)
    p.add_argument('--out', required=True)
    for operation in ('score', 'verify'):
        p = sub.add_parser(operation)
        p.add_argument('--prepared', required=True)
        p.add_argument('--commit', required=True)
        p.add_argument('--prepare-sha', required=True)
        p.add_argument('--commit-sha', required=True)
        p.add_argument('--out' if operation == 'score' else '--result', required=True)
    args = parser.parse_args()
    try:
        require(os.name == 'posix', 'POSIX checkout required')
        if args.operation == 'prepare':
            print(prepare(args.request, args.out, args.public_out))
        elif args.operation == 'commit':
            print(commit(args.public, args.prediction, args.out))
        elif args.operation == 'score':
            score(args.prepared, args.commit, args.prepare_sha, args.commit_sha, args.out)
            print('scored one engineering episode')
        else:
            verify(args.prepared, args.commit, args.prepare_sha, args.commit_sha, args.result)
            print('verified one engineering episode')
    except (GateError, OSError, RecursionError) as error:
        print('queue-evaluation-v1: operation rejected; retained attempts are not successful results', file=sys.stderr)
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())
