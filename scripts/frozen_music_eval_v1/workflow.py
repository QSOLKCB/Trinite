"""Fresh source-bound requests and bounded, failure-retaining local attempts."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

from .artifacts import (ROOT, digest, disjoint, encoded, identity, inventory,
                        parse, publish, read, safe_path, sources)
from .protocol import (PROTOCOL, input_rows, data_contract, score_records)
from .receipt import validate_original, validate_attempt

REQUEST_SCHEMA = 'trinite.frozen-music-eval-request.v1'
REPORT_SCHEMA = 'trinite.frozen-music-eval-report.v1'


def original_args(args):
    result = []
    for name in ('original_source', 'original_run', 'original_request_identity', 'original_matrix_identity'):
        value = getattr(args, name, None)
        if not value:
            raise ValueError('original producer/run and independent identities required')
        # Resolve filesystem arguments in the caller's cwd, not the worker's
        # fixed checkout cwd. Do not follow a symlink into an unchecked alias.
        if name in ('original_source', 'original_run'):
            value = safe_path(value)
        result.extend(['--' + name.replace('_', '-'), str(value)])
    return result


def launch(action, args, log, extra=()):
    """No shell, fixed worker/interpreter, bounded drain, kill and retain on error."""
    argv = [sys.executable, '-I', '-B', str(ROOT / 'scripts/frozen_music_eval_v1/worker.py'),
            action, *original_args(args), *extra]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    fd = os.open(safe_path(log), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        # Fixed source-bound local worker; never a user-selected executable.
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        with subprocess.Popen(argv, cwd=ROOT, env=env, shell=False,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            errors = []
            def drain():
                count = 0
                try:
                    while True:
                        block = process.stdout.read(4096)
                        if not block:
                            break
                        remaining = 1048576 - count
                        stream.write(block[:remaining]); count += min(len(block), remaining)
                        if len(block) > remaining:
                            errors.append('worker log exceeded bound'); process.kill(); break
                    stream.flush(); os.fsync(stream.fileno())
                except BaseException as error:
                    errors.append('worker log failure: ' + type(error).__name__); process.kill()
            thread = threading.Thread(target=drain, daemon=True); thread.start()
            try:
                code = process.wait(timeout=600 if action == 'inspect' else 900)
            except BaseException:
                process.kill(); process.wait(); thread.join(); raise
            thread.join()
            if errors:
                raise ValueError('; '.join(errors))
            if code != 0:
                raise ValueError('worker failed; retained log records details')


def validate_request(raw, expected):
    if type(raw) is not bytes or len(raw) > 4 * 1024 * 1024 or digest(raw) != identity(expected):
        raise ValueError('evaluation request type/size/identity differs')
    value = parse(raw)
    if (type(value) is not dict or set(value) != {'schema', 'probe_source', 'protocol',
            'input_rows', 'data_contract', 'original'} or value['schema'] != REQUEST_SCHEMA
            or encoded(value['probe_source']) != encoded(sources())
            or encoded(value['protocol']) != encoded(PROTOCOL)
            or encoded(value['input_rows']) != encoded(input_rows())
            or encoded(value['data_contract']) != encoded(data_contract())):
        raise ValueError('evaluation request source/protocol/inputs/data contract differs')
    # Native original admission is separately recomputed by the isolated worker.
    validate_original(value['original'])
    return value


def freeze(args):
    original_args(args)
    request_path = disjoint(args.request, ROOT, args.original_source, args.original_run)
    if request_path.exists() or request_path.is_symlink():
        raise ValueError('request already exists')
    # Keep failed admission logs; do not hide an unsuccessful freeze attempt.
    attempt = Path(tempfile.mkdtemp(prefix=request_path.name + '.freeze-', dir=request_path.parent))
    launch('inspect', args, attempt / 'admission.log')
    original = parse(read(attempt / 'admission.log', 1048576))
    validate_original(original)
    value = {'schema': REQUEST_SCHEMA, 'probe_source': sources(), 'protocol': PROTOCOL,
             'input_rows': input_rows(), 'data_contract': data_contract(),
             'original': original}
    raw = encoded(value)
    publish(request_path, raw)
    return {'request_identity': digest(raw), 'inference_performed': False,
            'admission_log': str(attempt / 'admission.log')}


def predictions(output):
    raw = read(Path(output) / 'predictions.jsonl', 1048576)
    lines = raw.splitlines(keepends=True)
    if len(lines) > 320:
        raise ValueError('prediction coverage exceeds budget')
    return [parse(line) for line in lines]


def replay_gate():
    return {'schema': 'trinite.frozen-music-eval-replay-gate.v1', 'seeds': [17, 29],
            'step': 640, 'replay_decodes': 80, 'exact_retained_predictions': True,
            'both_training_fit': True}


def validate_completion(completion, records):
    keys = {'decodes', 'forward_calls', 'model_identities_unchanged',
            'original_inventory_unchanged', 'training_performed', 'train_replay_decodes',
            'heldout_decodes', 'replay_gate_identity'}
    if (type(completion) is not dict or set(completion) != keys
            or type(completion['decodes']) is not int or completion['decodes'] != 320
            or type(completion['forward_calls']) is not int
            or not 320 <= completion['forward_calls'] <= 20480
            or type(completion['train_replay_decodes']) is not int or completion['train_replay_decodes'] != 80
            or type(completion['heldout_decodes']) is not int or completion['heldout_decodes'] != 240
            or completion['replay_gate_identity'] != digest(encoded(replay_gate()))
            or completion['training_performed'] is not False
            or completion['model_identities_unchanged'] is not True
            or completion['original_inventory_unchanged'] is not True
            or completion['forward_calls'] != sum(len(r['generated_tokens']) for r in records)):
        raise ValueError('worker completion/forward accounting differs')


def run(args):
    if not args.authorize_inference:
        raise ValueError('separate explicit target inference authorization required')
    original_args(args)
    request_path = safe_path(args.request)
    raw = read(request_path, 4 * 1024 * 1024)
    request = validate_request(raw, args.request_identity)
    output = disjoint(args.output, ROOT, args.original_source, args.original_run, request_path)
    output.mkdir(mode=0o700, exist_ok=False)
    output_stat = output.stat()
    publish(output / 'request.json', raw)
    publish(output / 'attempt.json', encoded({'schema': 'trinite.frozen-music-eval-attempt.v1',
        'request_identity': args.request_identity, 'authorization': 'explicit-inference-flag',
        'selection': 'fixed final640 seeds17/29; both-seed replay barrier; one attempt; no training'}))
    failure = None
    summary = None
    completion = None
    unchanged = False
    try:
        if inventory(args.original_run) != request['original']['inventory']:
            raise ValueError('original inventory differs before inference')
        launch('infer', args, output / 'worker.log', (
            '--request', str(request_path), '--request-identity', args.request_identity,
            '--output', str(output), '--output-identity', str(output_stat.st_dev),
            str(output_stat.st_ino), '--authorize-inference'))
        completion = parse(read(output / 'worker-completion.json'))
        records = predictions(output)
        summary = score_records(records)
        validate_completion(completion, records)
        if read(output / 'replay-gate.json') != encoded(replay_gate()):
            raise ValueError('both-seed replay gate differs')
        validate_request(raw, args.request_identity)
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
    finally:
        try:
            unchanged = inventory(args.original_run) == request['original']['inventory']
        except Exception:
            unchanged = False
        if not unchanged and failure is None:
            failure = {'type': 'ValueError', 'message': 'original preservation failed'}
    if (output.stat().st_dev, output.stat().st_ino) != (output_stat.st_dev, output_stat.st_ino):
        raise ValueError('attempt directory was replaced; retain evidence without resealing')
    files = inventory(output, max_files=8, max_bytes=8 * 1024 * 1024)
    report = {'schema': REPORT_SCHEMA, 'request_identity': args.request_identity,
        'outcome': 'completed' if failure is None else 'failed', 'failure': failure,
        'original_inventory_unchanged': unchanged, 'files': files,
        'summary': summary if failure is None else None,
        'worker_completion': completion if failure is None else None,
        'fresh_confirmation': False, 'causal_transfer_established': False,
        'optimizer_replayed': False, 'learning_adequate': False, 'release_ready': False}
    report_raw = encoded(report)
    publish(output / 'report.json', report_raw)
    return {'outcome': report['outcome'], 'report_identity': digest(report_raw)}


def audit(args):
    output = safe_path(args.output)
    before = inventory(output, max_files=9, max_bytes=8 * 1024 * 1024)
    # Failed partial journals remain opaque, but never escape frozen byte bounds.
    for name in ('worker.log', 'predictions.jsonl'):
        if name in before:
            read(output / name, 1048576)
    raw = read(output / 'report.json')
    if digest(raw) != identity(args.expected_identity):
        raise ValueError('external report identity differs')
    report = parse(raw)
    if (type(report) is not dict or set(report) != {'schema', 'request_identity', 'outcome',
            'failure', 'original_inventory_unchanged', 'files', 'summary', 'worker_completion',
            'fresh_confirmation', 'causal_transfer_established', 'optimizer_replayed',
            'learning_adequate', 'release_ready'}):
        raise ValueError('report fields differ')
    request_raw = read(args.request, 4 * 1024 * 1024)
    validate_request(request_raw, args.request_identity)
    if (read(output / 'request.json') != request_raw or report['schema'] != REPORT_SCHEMA
            or report['request_identity'] != args.request_identity
            or report['files'] != {k: v for k, v in before.items() if k != 'report.json'}):
        raise ValueError('report/request/retained inventory differs')
    flags = ('fresh_confirmation', 'causal_transfer_established', 'optimizer_replayed',
             'learning_adequate', 'release_ready')
    if any(report[k] is not False for k in flags):
        raise ValueError('unsupported claim in report')
    validate_attempt(parse(read(output / 'attempt.json')), args.request_identity)
    allowed = {'request.json', 'attempt.json', 'worker.log', 'predictions.jsonl',
               'worker-completion.json', 'replay-gate.json', 'report.json'}
    if not {'request.json', 'attempt.json', 'report.json'} <= set(before) <= allowed:
        raise ValueError('attempt artifact inventory differs')
    if type(report['original_inventory_unchanged']) is not bool:
        raise ValueError('preservation result must be boolean')
    if report['outcome'] == 'completed':
        if set(before) != {'request.json', 'attempt.json', 'worker.log', 'predictions.jsonl',
                           'worker-completion.json', 'replay-gate.json', 'report.json'}:
            raise ValueError('completed artifact inventory differs')
        validate_completion(report['worker_completion'], predictions(output))
        if read(output / 'replay-gate.json') != encoded(replay_gate()):
            raise ValueError('both-seed replay gate differs')
        if (report['failure'] is not None or report['original_inventory_unchanged'] is not True
                or encoded(report['summary']) != encoded(score_records(predictions(output)))
                or encoded(report['worker_completion']) != read(output / 'worker-completion.json')):
            raise ValueError('completed report recomputation differs')
    elif (report['outcome'] != 'failed' or type(report['failure']) is not dict
            or set(report['failure']) != {'type', 'message'}
            or any(type(v) is not str for v in report['failure'].values())
            or report['summary'] is not None or report['worker_completion'] is not None):
        raise ValueError('invalid failed outcome')
    if inventory(output, max_files=9, max_bytes=8 * 1024 * 1024) != before:
        raise ValueError('probe evidence changed during audit')
    return {'verified': True, 'outcome': report['outcome'], 'report_identity': args.expected_identity,
            'scope': 'read-only report/source/score consistency; no tensor or inference replay',
            'target_inference_performed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'freeze', 'run', 'audit'))
    for name in ('original-source', 'original-run', 'original-request-identity',
                 'original-matrix-identity', 'request', 'request-identity', 'output', 'expected-identity'):
        parser.add_argument('--' + name)
    parser.add_argument('--authorize-inference', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'preflight':
            result = {'protocol': PROTOCOL, 'input_rows': input_rows(),
                      'data_contract': data_contract(), 'model_executed': False}
        else:
            required = {'freeze': ('request',), 'run': ('request', 'request_identity', 'output'),
                        'audit': ('request', 'request_identity', 'output', 'expected_identity')}[args.command]
            if any(getattr(args, key) is None for key in required):
                raise ValueError('missing required request/output identities')
            result = {'freeze': freeze, 'run': run, 'audit': audit}[args.command](args)
        print(encoded(result).decode(), end='')
        return 1 if result.get('outcome') == 'failed' else 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1
