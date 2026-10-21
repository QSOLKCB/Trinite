"""Read-only, standard-library audit of one archived CPU run and full replay.

Run with python -I -S -B scripts/check_cyclic_learning_evidence.py.
Hashes establish retained-byte integrity, not hardware or execution attestation.
"""
import sys

if not (sys.flags.isolated and sys.flags.no_site and sys.flags.ignore_environment
        and sys.flags.no_user_site and sys.flags.safe_path):
    sys.stderr.write('requires python -I -S -B scripts/check_cyclic_learning_evidence.py\n')
    raise SystemExit(1)

import hashlib
import json
import lzma
import math
from pathlib import Path
import tempfile
import unittest
import copy

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'fixtures/cyclic-learning-cpu-20261021'
COMMITMENT = 'sha256:6140ab4427dede60b24f29da6d160d8d95722692faa2178f6bd4ad6b14b00613'
LOADER = 'sha256:3bfcfa9817d0fc387f647652fba683552ae55f29c59df36418334f4dc5e3f05a'
LIMIT = 8 * 1024 * 1024
TOTAL = 32 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def parse(raw):
    value = json.loads(raw)
    require(canonical(value) == raw, 'noncanonical JSON')
    return value


def read(path):
    require(path.is_file() and not path.is_symlink(), 'regular file required')
    with path.open('rb') as stream:
        raw = stream.read(LIMIT + 1)
    require(len(raw) <= LIMIT, 'file exceeds bound')
    return raw


def decode(raw):
    decoder = lzma.LZMADecompressor(memlimit=64 * 1024 * 1024)
    value = decoder.decompress(raw, max_length=LIMIT + 1)
    require(len(value) <= LIMIT and decoder.eof and not decoder.unused_data,
            'invalid or oversized XZ object')
    return value


def safe_path(name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix() == name,
            'unsafe member path')
    return p


def finite_hex(value):
    require(type(value) is str, 'hex float required')
    number = float.fromhex(value)
    require(math.isfinite(number) and number >= 0, 'nonfinite or negative measurement')
    return number


def inventory(folder):
    raw = read(folder / 'commitment.json')
    require(digest(raw) == COMMITMENT, 'commitment identity mismatch')
    packet = parse(raw)
    require(packet['schema'] == 'trinite.cyclic-learning-cpu-evidence.v1', 'schema mismatch')
    require(packet['producer_commit'] == 'b4341c5a150847eaf4158574423ea5fa153c9486'
            and packet['producer_tree'] == 'e5650b2869673801d45ee3c4a31349958730944a', 'producer pin mismatch')
    require(len(packet['members']) == 150, 'member inventory mismatch')
    members, total = {}, 0
    for name, record in packet['members'].items():
        safe_path(name)
        stored = safe_path(record['stored_path'])
        require(stored.as_posix() == 'objects/' + record['public_identity'][7:] + '.xz', 'object path mismatch')
        compressed = read(folder / stored)
        require(len(compressed) == record['stored_bytes'] and digest(compressed) == record['stored_identity'], 'stored identity mismatch')
        data = decode(compressed)
        total += len(data)
        require(total <= TOTAL, 'expanded archive exceeds bound')
        require(len(data) == record['public_bytes'] and digest(data) == record['public_identity'], 'public identity mismatch')
        if not record['redactions']:
            require(record['original_bytes'] == len(data) and record['original_identity'] == digest(data), 'original identity mismatch')
        else:
            require(name == 'artifacts/pip-check.log' and record['redactions'] == [{'rule': 'cloud-home', 'count': 1}], 'unexpected redaction')
        members[name] = data
    expected = {r['stored_path'] for r in packet['members'].values()} | {'commitment.json', '.gitattributes'}
    require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()} == expected, 'extra or missing archive files')
    require(read(folder / '.gitattributes') == b'objects/*.xz binary\n', 'checkout byte preservation missing')
    return packet, members


def score_check(scores, rows):
    require(len(scores['records']) == len(rows), 'score row count')
    groups = {s: {'rows': 0, 'canonical': 0, 'exact': 0} for s in ('train', 'validation', 'test')}
    for record, row in zip(scores['records'], rows):
        tokens = record['tokens']
        require(type(tokens) is list and len(tokens) <= 4 and
                all(type(t) is int and 0 <= t < 259 for t in tokens), 'invalid raw tokens')
        good = len(tokens) == 4 and tokens[-1] == 257 and all(48 <= t <= 57 for t in tokens[:3])
        exact = good and bytes(tokens[:3]).decode('ascii') == row['answer']
        require(record == {'identity': row['identity'], 'family': row['family'], 'split': row['split'],
                           'tokens': tokens, 'canonical': good, 'exact': exact}, 'score claim mismatch')
        group = groups[row['split']]
        group['rows'] += 1
        group['canonical'] += good
        group['exact'] += exact
    require(scores['groups'] == groups, 'score aggregate mismatch')
    return groups


def report_check(report, request):
    require(report['schema'] == 'trinite.cyclic-learning-report.v2' and
            report['request_identity'] == digest(canonical(request)), 'report request binding')
    cells = report['cells']
    require([(c['seed'], c['arm']) for c in cells] ==
            [(s, a) for s in (0, 1) for a in ('decimal', 'sexagesimal')], 'four-cell order')
    for cell in cells:
        require((cell['steps'], cell['target_tokens'], cell['scheduled_visits'], cell['unique_train_rows']) == (32, 512, 128, 192), 'training budget mismatch')
        require(cell['model_id'] == request['protocol']['model_id'] and cell['parameters'] == 7264, 'model mismatch')
        require(cell['untrained']['model_identity'] == cell['initial_model_identity'], 'baseline identity mismatch')
        require(len(cell['history']) == 32, 'history count')
        visits = request['schedules'][str(cell['seed'])]['scheduled_visits']
        for i, entry in enumerate(cell['history']):
            require(entry['step'] == i + 1 and entry['target_tokens'] == 16 and entry['examples'] == visits[i * 4:(i + 1) * 4], 'history schedule mismatch')
            finite_hex(entry['loss_hex']); finite_hex(entry['gradient_norm_hex'])
        finite_hex(cell['initial_train_loss'])
        require(len(cell['train_only_diagnostics']) == 1 and cell['train_only_diagnostics'][0]['step'] == 32, 'diagnostic schedule')
        finite_hex(cell['train_only_diagnostics'][0]['loss_hex'])
        initial = score_check(cell['untrained']['scores'], request['items'])
        final = score_check(cell['scores'], request['items'])
        require(cell['final_minus_untrained_counts'] ==
                {s: {k: final[s][k] - initial[s][k] for k in ('exact', 'canonical')} for s in final}, 'delta mismatch')
    require(cells[0]['initial_model_identity'] == cells[1]['initial_model_identity'] and
            cells[2]['initial_model_identity'] == cells[3]['initial_model_identity'], 'paired initialization')


def attempts_check(members, request_raw, report_raw, report):
    attempts = sorted(n.rsplit('/', 1)[0] for n in members if n.endswith('/status.json'))
    require(len(attempts) == 2, 'attempt count')
    actions = []
    partial = [dict(phase='training_completed', seed=c['seed'], arm=c['arm'],
                    request_identity=report['request_identity'], final_model_identity=c['final_model_identity'],
                    steps=c['steps'], target_tokens=c['target_tokens'], history=c['history']) for c in report['cells']]
    partial += [{'phase': 'scoring_completed', 'cell': c} for c in report['cells']]
    for prefix in attempts:
        status = parse(members[prefix + '/status.json'])
        action = status['action']; actions.append(action)
        require(status['exit_code'] == 0 and status['controller_exit_code'] == 0 and
                status['timed_out'] is False and status['publication'] == 'published', 'unsuccessful attempt')
        require(status['report_identity'] == digest(report_raw), 'status report identity')
        expected = digest(report_raw) if action == 'replay' else None
        require(status['expected_report_identity'] == expected and status['exact_replay'] is (True if action == 'replay' else None), 'replay receipt')
        require(finite_hex(status['elapsed_seconds_hex']) <= 900, 'controller deadline')
        for usage in status['worker_resource_usage'].values():
            require(type(usage['peak_rss_kib']) is int and usage['peak_rss_kib'] > 0 and
                    usage['platform'] == 'linux' and usage['units'] == 'KiB' and
                    usage['scope'] == 'controller cumulative reaped-child high-water mark; fresh CLI normally launches one worker; not a per-child delta or memory cap', 'resource scope')
        require(set(status['worker_resource_usage']) == {'before', 'after'}, 'usage endpoints')
        require(members[prefix + '/report.json'] == report_raw and members[prefix + '/request.json'] == request_raw, 'attempt bytes mismatch')
        require(members[prefix + '/report.json.cells.jsonl'] == b''.join(canonical(c) for c in partial), 'partial records mismatch')
        require(members[prefix + '/stderr.log'] == members[prefix + '/stdout.log'] == b'', 'unexpected worker log')
        journal = [parse(line + b'\n') for line in members[prefix + '/controller.jsonl'].splitlines()]
        destination = 'replay.json' if action == 'replay' else 'report.json'
        require(journal == [dict(phase='attempt_started', request_identity=digest(request_raw), action=action,
                                destination_sibling_name=destination, expected_report_identity=expected),
                            dict(phase='publication_intent', report_identity=digest(report_raw), destination_sibling_name=destination),
                            {'phase': 'publication_outcome', **{k: v for k, v in status.items() if k != 'controller_exit_code'}}], 'journal mismatch')
        if action == 'replay':
            require(members[prefix + '/expected-report.json'] == report_raw, 'expected replay bytes')
    require(sorted(actions) == ['replay', 'run'], 'run/replay actions')


def audit():
    packet, members = inventory(FOLDER)
    request_raw = members['artifacts/request.json']; report_raw = members['artifacts/report.json']
    require(digest(request_raw) == packet['request_identity'] and digest(report_raw) == packet['report_identity'], 'artifact binding')
    require(members['artifacts/replay.json'] == report_raw, 'full deterministic replay mismatch')
    for action in ('freeze', 'run', 'replay'):
        require(members['artifacts/' + action + '.exit'] == b'0\n', 'process failure')
    request = parse(request_raw); report = parse(report_raw)
    source = request['source']
    require(len(source) == 124, 'source closure count')
    loader_path = 'scripts/audit_batch_composition_v5.py'
    require(set(packet['source_git_blobs']) == set(source) | {loader_path}, 'source Git inventory')
    with tempfile.TemporaryDirectory(prefix='cyclic-evidence-') as temp:
        root = Path(temp)
        for name in packet['source_git_blobs']:
            raw = members['source/' + name]
            require(hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == packet['source_git_blobs'][name], 'Git blob mismatch')
            require(digest(raw) == (LOADER if name == loader_path else source[name]), 'source digest mismatch')
            path = root / safe_path(name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        loader_raw = members['source/' + loader_path]
        namespace = {'__name__': 'captured_verified_loader', '__file__': str(root / loader_path)}
        exec(compile(loader_raw, str(root / loader_path), 'exec', dont_inherit=True, optimize=0), namespace)
        with namespace['VerifiedSourceImports'](root, canonical({'source': source})) as imports:
            from cyclic_learning_v1.spec import request as regenerate
            require(canonical(regenerate(request['environment'])) == request_raw, 'regenerated request/admission mismatch')
            imports.checked_executed_sources()
            require('torch' not in sys.modules, 'offline checker imported Torch')
    report_check(report, request)
    attempts_check(members, request_raw, report_raw, report)
    return request, report


class FunctionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request, cls.report = audit()
        _, cls.members = inventory(FOLDER)

    def test_nonfinite_measurements_rejected(self):
        for value in ('inf', 'nan', '-0x1p0'):
            with self.assertRaises(ValueError): finite_hex(value)

    def test_invalid_and_trailing_xz_rejected(self):
        for raw in (lzma.compress(b'a')[:-1], lzma.compress(b'a') + b'trailing'):
            with self.assertRaises(ValueError): decode(raw)

    def test_paths_rejected(self):
        for path in ('../outside', '/absolute', 'a/../b'):
            with self.assertRaises(ValueError): safe_path(path)

    def test_metric_and_schedule_mutations_rejected(self):
        for field, value in [('scheduled_visits', 129), ('steps', 31), ('target_tokens', 511)]:
            report = copy.deepcopy(self.report); report['cells'][0][field] = value
            with self.assertRaises(ValueError): report_check(report, self.request)
        report = copy.deepcopy(self.report); report['cells'][0]['scores']['records'][0]['exact'] = True
        with self.assertRaises(ValueError): report_check(report, self.request)
        report = copy.deepcopy(self.report); report['cells'][0]['final_minus_untrained_counts']['test']['exact'] = 1
        with self.assertRaises(ValueError): report_check(report, self.request)


    def test_request_history_and_raw_token_mutations_rejected(self):
        report = copy.deepcopy(self.report)
        report['request_identity'] = 'sha256:' + '0' * 64
        with self.assertRaises(ValueError): report_check(report, self.request)
        report = copy.deepcopy(self.report)
        report['cells'][0]['history'][0]['examples'].reverse()
        with self.assertRaises(ValueError): report_check(report, self.request)
        report = copy.deepcopy(self.report)
        report['cells'][0]['scores']['records'][0]['tokens'] = [50, 50, 50, 999]
        with self.assertRaises(ValueError): report_check(report, self.request)

    def test_failed_publication_and_nonfinite_receipts_rejected(self):
        status_name = next(n for n in self.members if n.endswith('/status.json'))
        for field, value in [('publication', 'failed'), ('controller_exit_code', 1),
                             ('elapsed_seconds_hex', 'inf'), ('exact_replay', False)]:
            members = dict(self.members)
            status = parse(members[status_name]); status[field] = value
            members[status_name] = canonical(status)
            with self.assertRaises(ValueError):
                attempts_check(members, members['artifacts/request.json'],
                               members['artifacts/report.json'], self.report)

    def test_commitment_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'commitment.json').write_bytes(b'{}\n')
            with self.assertRaises(ValueError): inventory(root)


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        unittest.main(argv=[sys.argv[0]])
    else:
        require(not sys.argv[1:], 'unknown arguments')
        audit()
        print('Retained CPU evidence verified: 124 sources, four cells, exact full-report replay; no training executed.')
