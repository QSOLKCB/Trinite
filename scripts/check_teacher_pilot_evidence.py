"""Offline audit of retained teacher intake and CPU training replay, not a rerun.

Use python -I -S -B scripts/check_teacher_pilot_evidence.py.
"""
import sys
if not (sys.flags.isolated and sys.flags.no_site and sys.flags.ignore_environment
        and sys.flags.no_user_site and sys.flags.safe_path):
    sys.stderr.write('requires python -I -S -B scripts/check_teacher_pilot_evidence.py\n')
    raise SystemExit(1)

import base64
import collections
import copy
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'fixtures/teacher-pilot-local-20261021'
COMMITMENT = 'sha256:4e3a4758ebbc1de178080c3e22732acf4ee28f8927f060293d98d29be27c88cb'
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


def same(left, right, message):
    require(canonical(left) == canonical(right), message)


def safe_path(name):
    require(type(name) is str and bool(name), 'member name required')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and
            path.as_posix() == name and '\\' not in name, 'unsafe member path')
    return Path(*path.parts)


def read(path, limit=LIMIT):
    require(path.is_file() and not path.is_symlink(), 'regular file required')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, 'file exceeds bound')
    return raw


def finite_hex(value):
    require(type(value) is str, 'hex measurement required')
    number = float.fromhex(value)
    require(math.isfinite(number) and number >= 0, 'invalid measurement')
    return number


def sha(value):
    require(type(value) is str and re.fullmatch('sha256:[0-9a-f]{64}', value), 'invalid identity')


def inventory(folder):
    raw = read(folder / 'commitment.json')
    require(digest(raw) == COMMITMENT, 'commitment identity mismatch')
    packet = parse(raw)
    require(packet['schema'] == 'trinite.teacher-pilot-local-evidence.v1', 'schema mismatch')
    require(packet['producer_commit'] == '60a1797124a8390c1183a130846a82e06dea94ab' and
            packet['producer_tree'] == '63febaf4046f64b67bff78e97748810668a58d3f', 'producer mismatch')
    archive = read(folder / 'objects.zip', TOTAL)
    require(digest(archive) == packet['objects_archive']['identity'] and
            len(archive) == packet['objects_archive']['bytes'], 'object archive mismatch')
    require(read(folder / '.gitattributes') == b'objects.zip binary\ncommitment.json text eol=lf\n', 'binary checkout rule required')
    require({p.name for p in folder.iterdir()} == {'commitment.json', 'objects.zip', '.gitattributes'}, 'unexpected fixture files')
    expected = {record['stored_path'] for record in packet['members'].values()}
    objects = {}; total = 0
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = bundle.namelist()
        require(len(names) == len(set(names)) and set(names) == expected, 'object inventory mismatch')
        for item in bundle.infolist():
            safe_path(item.filename)
            require(not item.is_dir() and item.file_size <= LIMIT and item.flag_bits & 1 == 0,
                    'invalid object type or size')
            total += item.file_size
            require(total <= TOTAL, 'expanded objects exceed bound')
            with bundle.open(item) as stream:
                data = stream.read(LIMIT + 1)
            require(len(data) == item.file_size and len(data) <= LIMIT, 'object size mismatch')
            objects[item.filename] = data
    members = {}; total = 0
    for name, record in packet['members'].items():
        safe_path(name); sha(record['identity'])
        require(record['stored_path'] == 'objects/' + record['identity'][7:] + '.bin', 'object path mismatch')
        data = objects[record['stored_path']]
        require(len(data) == record['bytes'] and digest(data) == record['identity'], 'member identity mismatch')
        total += len(data)
        require(total <= TOTAL, 'logical members exceed bound')
        members[name] = data
    for name, record in packet['excluded'].items():
        safe_path(name); sha(record['identity'])
        require(name not in members and type(record['bytes']) is int and record['bytes'] > 0
                and type(record['reason']) is str and record['reason'], 'invalid exclusion')
    require(packet['original_zip'] == {'identity': 'sha256:354829e6f7b6a0312397d8b40e43b61ead198f269e6596d0d8053adb7cbc4045',
                                       'bytes': 11041296, 'members': 853}, 'original archive pin')
    require(len(members) == 691 and len(packet['excluded']) == 164, 'retention inventory count')
    require(packet['public_transformations'] == [], 'unexpected publication transformation')
    added = {'verification/verified_source.py', 'attribution/gpt-oss-20b-LICENSE.txt'}
    require(set(packet['added_members']) == added and added <= set(members), 'added member inventory')
    original = {}
    for line in members['SHA256SUMS'].decode('ascii').splitlines():
        checksum, name = line.split('  ', 1)
        safe_path(name)
        require(name not in original, 'duplicate original manifest entry')
        sha('sha256:' + checksum); original[name] = 'sha256:' + checksum
    require(len(original) == 852 and set(original) | {'SHA256SUMS'} ==
            (set(members) - added) | set(packet['excluded']), 'original manifest coverage')
    for name, identity in original.items():
        retained = packet['members'].get(name, packet['excluded'].get(name))
        require(retained['identity'] == identity, 'original manifest identity mismatch')
    return packet, members


def score_check(scores, rows):
    require(len(scores['records']) == len(rows), 'score row count')
    counts = {s: {'rows': 0, 'exact': 0} for s in ('train', 'validation', 'test')}
    for record, row in zip(scores['records'], rows):
        tokens = record['tokens']
        require(type(tokens) is list and len(tokens) <= len(row['answer']) + 1 and
                all(type(t) is int and 0 <= t < 259 for t in tokens), 'invalid raw tokens')
        exact = tokens == list(row['answer'].encode('ascii')) + [257]
        same(record, {'identity': row['identity'], 'split': row['split'], 'tokens': tokens, 'exact': exact}, 'score record mismatch')
        counts[row['split']]['rows'] += 1
        counts[row['split']]['exact'] += exact
    same(scores['counts'], counts, 'score counts mismatch')
    return counts


def state_check(state, visits, diagnostic=False):
    same([state['steps'], state['target_tokens'], state['scheduled_visits']],
         [256, 2048 if diagnostic else 3072, 1024], 'training budget mismatch')
    require(len(state['history']) == 256, 'history count')
    for i, entry in enumerate(state['history']):
        same([entry['step'], entry['target_tokens'], entry['examples']],
             [i + 1, 8 if diagnostic else 12, visits[i * 4:(i + 1) * 4]], 'history schedule mismatch')
        finite_hex(entry['loss_hex']); finite_hex(entry['gradient_norm_hex'])
    finite_hex(state['initial_train_loss'])
    require(len(state['train_only_diagnostics']) == 1 and
            state['train_only_diagnostics'][0]['step'] == 256, 'diagnostic count')
    finite_hex(state['train_only_diagnostics'][0]['loss_hex'])
    sha(state['initial_model_identity']); sha(state['final_model_identity'])
    require(type(state['optimizer']) is dict and bool(state['optimizer']), 'optimizer records required')
    for values in state['optimizer'].values():
        require(set(values) == {'step', 'exp_avg', 'exp_avg_sq'}, 'optimizer state fields')
        for value in values.values(): sha(value)


def report_check(report, request):
    require(report['schema'] == 'trinite.teacher-student-report.v1' and
            report['request_identity'] == digest(canonical(request)), 'report binding mismatch')
    diagnostic = report['diagnostic']; rows = request['diagnostic_rows']
    ordered = sorted(rows, key=lambda r: digest(canonical({'seed': 0, 'identity': r['identity']})))
    visits = [ordered[i % len(ordered)]['identity'] for i in range(1024)]
    state_check(diagnostic, visits, True)
    scores = score_check(diagnostic['scores'], rows)
    require(diagnostic['passed'] is True and scores['train']['exact'] == 4 and report['outcome'] == 'completed', 'diagnostic outcome mismatch')
    cells = report['cells']
    same([(c['seed'], c['arm']) for c in cells],
         [(s, a) for s in (0, 1) for a in ('teacher_selected', 'oracle_only')], 'four-cell order')
    for cell in cells:
        seed, arm = cell['seed'], cell['arm']
        state_check(cell, request['scheduled_visits'][str(seed)][arm])
        rows = request['pools'][arm] + request['pools']['heldout']
        score_check(cell['initial_scores'], rows); score_check(cell['final_scores'], rows)
        cfg = cell['config']
        same([cfg['seed'], cfg['steps'], cfg['batch_size'], cfg['max_target_tokens']], [seed, 256, 4, 3072], 'configuration budget mismatch')
    require(cells[0]['initial_model_identity'] == cells[1]['initial_model_identity'] and
            cells[2]['initial_model_identity'] == cells[3]['initial_model_identity'], 'paired initialization mismatch')
    for row in request['pools']['heldout'] + request['pools']['teacher_selected'] + request['pools']['oracle_only']:
        f = row['formal_spec']
        require(row['answer'] == f"{(f['a'] + f['b']) % f['n']:02d}", 'independent modulo label mismatch')


def teacher_check(members, packet):
    capture = parse(members['reports/capture.json'])
    same(capture, packet['capture'], 'capture binding mismatch')
    require(len(capture['attempts']) == 32, 'teacher attempt count')
    total = 0
    for i, attempt in enumerate(capture['attempts']):
        prefix = 'reports/teacher-attempt-20261021-v2/' + f'{i:02d}'
        raw = members[prefix + '.raw']
        require(base64.b64decode(attempt['raw_base64'], validate=True) == raw, 'raw capture mismatch')
        events = [parse(line + b'\n') for line in members[prefix + '.events.jsonl'].splitlines()]
        require(events[0]['event'] == 'call_started' and events[0]['fresh_kv'] is True and
                events[-1]['event'] == 'call_completed' and events[-1]['stop'] == 'eog', 'call lifecycle mismatch')
        require(all(type(e['index']) is int and e['index'] == i for e in events), 'event index mismatch')
        tokens = events[1:-1]
        require(all(e['event'] == 'token' and type(e['token_id']) is int for e in tokens), 'token event mismatch')
        require(b''.join(bytes.fromhex(e['piece_hex']) for e in tokens) == raw, 'token bytes mismatch')
        require(events[-1]['generated_tokens'] == len(tokens) <= 512, 'generated token count')
        total += len(tokens)
        receipt = parse(members[prefix + '.receipt.json'])
        same([receipt['index'], receipt['outcome'], receipt['raw_identity'], receipt['raw_bytes'],
              receipt['final_start'], receipt['final_end'], receipt['generated_tokens_available']],
             [i, 'completed', digest(raw), len(raw), attempt['final_start'], attempt['final_end'], len(tokens)], 'call receipt mismatch')
        require(receipt['deadline_seconds'] == 120 and type(receipt['elapsed_seconds']) in (int, float)
                and math.isfinite(receipt['elapsed_seconds']) and 0 <= receipt['elapsed_seconds'] <= 120
                and receipt['error'] is None and receipt['extraction_error'] is None, 'call timing/error mismatch')
        settings = packet['request']['protocol']['sampling']
        same(receipt['settings'], settings, 'receipt sampling mismatch')
        same(events[0]['settings'], settings, 'event sampling mismatch')
        inp = parse(members[prefix + '.input.json'])
        same({k: inp[k] for k in settings}, settings, 'input sampling mismatch')
        require(inp['index'] == i and packet['request']['slots'][i]['prompt'] in inp['prompt'], 'slot prompt mismatch')
        marker = b'<|channel|>final<|message|>'
        require(raw.count(marker) == 1 and raw.endswith(b'<|return|>'), 'final channel structure')
        require(attempt['final_start'] == raw.index(marker) + len(marker) and
                attempt['final_end'] == len(raw) - len(b'<|return|>'), 'final channel byte offsets')
    require(total == 1523, 'retained total token count')
    old = parse(members['reports/teacher-request.json']); new = packet['request']
    same({k: v for k, v in old.items() if k != 'runtime_manifest'},
         {k: v for k, v in new.items() if k != 'runtime_manifest'}, 'pre-generation request changed protocol')
    failure = json.loads(members['setup/capture-startup-failure.json'])
    require(failure['teacher_calls'] == 0 and failure['native_worker_launched'] is False,
            'startup failure call count')
    require(b'ModuleNotFoundError' in members['logs/teacher-capture.stderr'] and
            b'_colorize' in members['logs/teacher-capture.stderr'], 'missing startup failure log')


def attempts_check(members, request_raw, report_raw, report):
    prefixes = sorted(n.rsplit('/', 1)[0] for n in members
                      if n.startswith('reports/student-') and n.endswith('/status.json'))
    require(len(prefixes) == 2, 'student attempt count')
    actions = []
    partial = [{'phase': 'diagnostic_completed', 'request_identity': report['request_identity'], 'diagnostic': report['diagnostic']}]
    snapshot_keys = set(report['diagnostic']) - {'scores', 'passed'}
    partial += [{'phase': 'training_completed', 'seed': c['seed'], 'arm': c['arm'],
                 'state': {k: c[k] for k in snapshot_keys}} for c in report['cells']]
    partial += [{'phase': 'scoring_completed', 'cell': c} for c in report['cells']]
    for prefix in prefixes:
        status = parse(members[prefix + '/status.json']); action = status['action']; actions.append(action)
        require(status['exit_code'] == 0 and status['controller_exit_code'] == 0 and
                status['timed_out'] is False and status['publication'] == 'published', 'failed student attempt')
        require(status['report_identity'] == digest(report_raw), 'status report binding')
        expected = digest(report_raw) if action == 'replay' else None
        same(status['expected_report_identity'], expected, 'expected replay identity')
        require(status['exact_replay'] is (True if action == 'replay' else None), 'replay equality receipt')
        require(finite_hex(status['elapsed_seconds_hex']) <= 3600, 'controller deadline')
        require(set(status['worker_resource_usage']) == {'before', 'after'}, 'usage endpoints')
        for usage in status['worker_resource_usage'].values():
            require(type(usage['peak_rss_kib']) is int and usage['peak_rss_kib'] > 0 and
                    usage['platform'] == 'linux' and usage['units'] == 'KiB' and usage['scope'] ==
                    'controller cumulative reaped-child high-water mark; fresh CLI normally launches one worker; not a per-child delta or memory cap', 'resource scope')
        require(members[prefix + '/request.json'] == request_raw and members[prefix + '/report.json'] == report_raw, 'attempt bytes mismatch')
        require(members[prefix + '/report.json.cells.jsonl'] == b''.join(canonical(x) for x in partial), 'partial history mismatch')
        require(members[prefix + '/stdout.log'] == members[prefix + '/stderr.log'] == b'', 'unexpected worker logs')
        destination = 'student-replay.json' if action == 'replay' else 'student-report.json'
        journal = [parse(line + b'\n') for line in members[prefix + '/controller.jsonl'].splitlines()]
        same(journal, [dict(phase='attempt_started', request_identity=digest(request_raw), action=action,
                            destination_sibling_name=destination, expected_report_identity=expected),
                       dict(phase='publication_intent', report_identity=digest(report_raw), destination_sibling_name=destination),
                       {'phase': 'publication_outcome', **{k: v for k, v in status.items() if k != 'controller_exit_code'}}], 'publication journal mismatch')
        if action == 'replay': require(members[prefix + '/expected-report.json'] == report_raw, 'retained expected bytes mismatch')
    require(sorted(actions) == ['replay', 'run'], 'student actions mismatch')


def audit():
    commitment, members = inventory(FOLDER)
    request_raw = members['reports/student-request.json']; report_raw = members['reports/student-report.json']
    request = parse(request_raw); report = parse(report_raw)
    packet_raw = members['reports/quarantine.json']; packet = parse(packet_raw)
    require(members['reports/student-replay.json'] == report_raw, 'numerical replay bytes differ')
    require(canonical(request['packet']) == packet_raw and request['packet_identity'] == digest(packet_raw), 'student packet binding')
    require(canonical(packet['request']) == members['reports/teacher-request-v2.json'], 'teacher request binding')
    identities = {'student-original-report': report_raw, 'student-replay-report': report_raw,
                  'student-request': request_raw, 'teacher-packet': packet_raw,
                  'teacher-request-v2': members['reports/teacher-request-v2.json'],
                  'teacher-request': members['reports/teacher-request.json']}
    for name, raw in identities.items():
        require(members['setup/identities/' + name + '.identity'].decode().strip() == digest(raw), 'independently retained identity mismatch')
    source = request['source']; require(len(source) == 126, 'source closure count')
    same(source, packet['request']['source'], 'teacher/student source mismatch')
    with tempfile.TemporaryDirectory(prefix='teacher-evidence-') as temp:
        root = Path(temp)
        for name, expected in source.items():
            raw = members['setup/source/' + name]
            require(digest(raw) == expected, 'source digest mismatch')
            path = root / safe_path(name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        loader_raw = members['verification/verified_source.py']
        require(digest(loader_raw) == LOADER, 'verified loader mismatch')
        loader_path = root / 'verified_source.py'
        namespace = {'__name__': 'captured_verified_loader', '__file__': str(loader_path)}
        exec(compile(loader_raw, str(loader_path), 'exec', dont_inherit=True, optimize=0), namespace)
        with namespace['VerifiedSourceImports'](root, canonical({'source': source})) as imports:
            from teacher_intake_v1.intake import verify
            from teacher_intake_v1.student import request as regenerate
            require(canonical(verify(packet)) == packet_raw, 'actual intake verification mismatch')
            require(canonical(regenerate(packet, request['environment'])) == request_raw, 'request regeneration mismatch')
            imports.checked_executed_sources()
            require('torch' not in sys.modules, 'offline audit imported Torch')
    teacher_check(members, packet); report_check(report, request)
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

    def test_paths_rejected(self):
        for name in ('../outside', '/absolute', 'a/../b', 'a\\b'):
            with self.assertRaises(ValueError): safe_path(name)

    def test_metric_schedule_and_bool_substitutions_rejected(self):
        for field, value in [('steps', 255), ('target_tokens', 3071), ('scheduled_visits', 1023)]:
            report = copy.deepcopy(self.report); report['cells'][0][field] = value
            with self.assertRaises(ValueError): report_check(report, self.request)
        report = copy.deepcopy(self.report); report['cells'][0]['final_scores']['records'][0]['exact'] = 0
        with self.assertRaises(ValueError): report_check(report, self.request)
        report = copy.deepcopy(self.report); report['cells'][0]['history'][0]['examples'].reverse()
        with self.assertRaises(ValueError): report_check(report, self.request)

    def test_report_binding_and_raw_token_rejected(self):
        report = copy.deepcopy(self.report); report['request_identity'] = 'sha256:' + '0' * 64
        with self.assertRaises(ValueError): report_check(report, self.request)
        report = copy.deepcopy(self.report); report['cells'][0]['final_scores']['records'][0]['tokens'] = [48, 48, True]
        with self.assertRaises(ValueError): report_check(report, self.request)

    def test_failed_publication_and_nonfinite_receipts_rejected(self):
        name = next(n for n in self.members if n.startswith('reports/student-') and n.endswith('/status.json'))
        for field, value in [('publication', 'failed'), ('controller_exit_code', 1), ('elapsed_seconds_hex', 'inf')]:
            members = dict(self.members); status = parse(members[name]); status[field] = value; members[name] = canonical(status)
            with self.assertRaises(ValueError):
                attempts_check(members, members['reports/student-request.json'], members['reports/student-report.json'], self.report)

    def test_teacher_capture_and_offsets_rejected(self):
        packet = parse(self.members['reports/quarantine.json'])
        for changed in ('raw', 'receipt.json'):
            members = dict(self.members); name = 'reports/teacher-attempt-20261021-v2/00.' + changed
            if changed == 'raw': members[name] += b'x'
            else:
                receipt = parse(members[name]); receipt['final_start'] += 1; members[name] = canonical(receipt)
            with self.assertRaises(ValueError): teacher_check(members, packet)

    def test_commitment_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); (folder / 'commitment.json').write_bytes(b'{}\n')
            with self.assertRaises(ValueError): inventory(folder)


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        unittest.main(argv=[sys.argv[0]])
    else:
        require(not sys.argv[1:], 'unknown arguments')
        audit()
        print('Retained teacher pilot verified: 32 captures, 126 sources, four CPU cells and exact report replay; no inference or training executed.')
