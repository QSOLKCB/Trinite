"""Read-only retained Needle compatibility audit; never execute captured adapters or models."""
import sys
if not (sys.flags.isolated and sys.flags.no_site and sys.flags.ignore_environment
        and sys.flags.no_user_site and sys.flags.safe_path):
    raise SystemExit('requires python -I -S -B scripts/check_needle_compatibility_evidence.py')
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import types
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'fixtures/needle-compatibility-local-20261021'
COMMITMENT = 'sha256:2e38d7a9153c7a95ce205c48d2927bce2ad184861b55a7aad6af9a4d00b3c2c3'
LIMIT = 4 * 1024 * 1024
TOTAL = 24 * 1024 * 1024
FIXTURE_FILES = ('objects.zip', 'commitment.json', '.gitattributes')
PREFIX = 'assets/.trinite-needle-compatibility-v2-attempt/'
PACKET = 'sha256:3a4384848c8d2ef99d44051662134be364fac3f7b3d46c8704972f3a90627081'
REQUEST = 'sha256:71d51bae151c124b0de3ba6dd2e1efab704bc61a2a8285e22b84ac3a17121716'
SOURCE = {
    'scripts/run_tiny_compatibility_v2.py': '22bff960435f0102ecc690e0283012529ee6267788432b3b8b5e9ef82f79723b',
    'scripts/tiny_compatibility_v2.py': '031cbc3dee9b2044b31d5b19a1b7e29f2e2f542243f8369ec7d2228c319c44aa',
    'scripts/tiny_needle_worker_v2.py': 'c6902dce8df2f99798c3e91acdbb82bd9c0a18e4c8568b55826d1f65f7218edf',
    'scripts/tiny_output_v1.py': '2830070999d296be909e8729ab0a62008501df93093626fc40bba5db7fd6ca55',
    'src/trinite/contracts.py': '36562611c6d60b4426a28e0974cda77b81a305186768f69d85bbbaf12a355d24',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode()


def same(a, b, message):
    require(canonical(a) == canonical(b), message)


def safe_path(name):
    require(type(name) is str and bool(name), 'member name required')
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix() == name
            and '\\' not in name, 'unsafe path')
    return Path(*p.parts)


def read(path, limit=LIMIT):
    require(path.is_file() and not path.is_symlink(), 'regular file required')
    with path.open('rb') as f:
        raw = f.read(limit + 1)
    require(len(raw) <= limit, 'size bound exceeded')
    return raw


def snapshot(folder):
    require({x.name for x in folder.iterdir()} == set(FIXTURE_FILES), 'fixture inventory')
    return {name: read(folder / name) for name in FIXTURE_FILES}


def inventory(captured):
    raw = captured['commitment.json']
    require(digest(raw) == COMMITMENT, 'commitment mismatch')
    p = json.loads(raw)
    require(canonical(p) == raw, 'noncanonical commitment')
    require(p['schema'] == 'trinite.needle-compatibility-local-evidence.v1', 'schema')
    same(p['original_zip'], {'identity': 'sha256:ee4a48806be2307af22ab2d0cdeb75f29e6dd8bd12125943d9c504f0e83fabe0',
                            'bytes': 3773058, 'members': 802}, 'original archive pin')
    require(p['producer_commit'] == '8823a92cdaf28139a852f896af457b3b0a013535'
            and p['producer_tree'] == '29942d2ff759cb4cefcc512308932f12d933d361', 'producer pin')
    require(p['added_members'] == p['public_transformations'] == [], 'unexpected transformations')
    require(captured['.gitattributes'] == b'objects.zip binary\ncommitment.json text eol=lf\n', 'binary rule')
    data = captured['objects.zip']
    require(digest(data) == p['objects_archive']['identity'] and len(data) == p['objects_archive']['bytes'], 'archive identity')
    expected = {r['stored_path'] for r in p['members'].values()}
    objects = {}; total = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        require(len(z.namelist()) == len(set(z.namelist())) == p['objects_archive']['unique_objects']
                and set(z.namelist()) == expected, 'object inventory')
        for info in z.infolist():
            safe_path(info.filename)
            require(not info.is_dir() and info.file_size <= LIMIT and not info.flag_bits & 1
                    and info.external_attr >> 16 == 0o100644, 'object type/size')
            total += info.file_size
            require(total <= TOTAL, 'expanded size bound')
            with z.open(info) as stream:
                b = stream.read(LIMIT + 1)
            require(len(b) == info.file_size, 'expanded size mismatch')
            objects[info.filename] = b
    members = {}; total = 0
    for name, rec in p['members'].items():
        safe_path(name)
        require(rec['stored_path'] == 'objects/' + rec['identity'][7:] + '.bin', 'object path')
        b = objects[rec['stored_path']]
        require(digest(b) == rec['identity'] and len(b) == rec['bytes'], 'member identity')
        total += len(b); require(total <= TOTAL, 'logical size bound')
        members[name] = b
    require(len(members) == 802 and p['excluded'] == {}, 'retention counts')
    for name, rec in p['excluded'].items():
        safe_path(name)
        require(name not in members and re.fullmatch('sha256:[0-9a-f]{64}', rec['identity'])
                and type(rec['bytes']) is int and rec['bytes'] > 0 and rec['reason'], 'exclusion')
    original = {}
    for line in members['SHA256SUMS'].decode('ascii').splitlines():
        h, name = line.split('  ', 1); safe_path(name)
        require(name not in original and re.fullmatch('[0-9a-f]{64}', h), 'original manifest record')
        original[name] = 'sha256:' + h
    require(len(original) == 801 and set(original) | {'SHA256SUMS'} == set(members) | set(p['excluded']), 'original manifest coverage')
    for name, identity in original.items():
        require(p['members'].get(name, p['excluded'].get(name))['identity'] == identity, 'original digest mismatch')
    return p, members


def source_module(members, root, source_root=ROOT):
    # Check the entire dependency closure before compiling anything. Only these
    # independently pinned repository files execute; archived SDKs stay inert.
    verified = {}
    for name, expected in SOURCE.items():
        raw = read(source_root / name)
        require(digest(raw) == 'sha256:' + expected
                and raw == members['checkout/' + name], 'authoritative source mismatch')
        verified[name] = raw
    for name, raw in verified.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    module = types.ModuleType('pinned_needle_compatibility_evidence')
    module.__file__ = str(root / 'scripts/tiny_compatibility_v2.py')
    exec(compile(verified['scripts/tiny_compatibility_v2.py'], module.__file__,
                 'exec', dont_inherit=True), module.__dict__)
    return module


def mapping_check(m, packet):
    ledger = json.loads(m['reports/omission-ledger.json'])['excluded_or_identical']
    rows = {}
    for row in ledger:
        safe_path(row['path'])
        require(row['path'] not in rows and row['path'] not in m, 'omission inventory')
        rows[row['path']] = row
        equivalent = row['retained_equivalent']
        if equivalent is not None:
            safe_path(equivalent)
            require(equivalent in m and digest(m[equivalent]) == row['identity']
                    and len(m[equivalent]) == row['bytes'], 'staged mapping identity')
    expected = {PREFIX + 'prepared-result.json': 'reports/result.json',
                PREFIX + 'request.json': 'reports/request.json'}
    expected.update({PREFIX + 'source/' + n: 'checkout/' + n for n in SOURCE})
    for record in packet['request']['assets']:
        name = record['path']
        original = 'assets/' + name
        staged = PREFIX + 'assets/' + name
        if name in ('needle3.cact', 'libneedle3.so'):
            for omitted in (original, staged):
                row = rows[omitted]
                require(row['retained_equivalent'] is None and omitted not in m
                        and row['identity'] == record['identity']
                        and row['bytes'] == record['bytes'], 'omitted binary declaration')
        else:
            require(digest(m[original]) == record['identity']
                    and len(m[original]) == record['bytes'], 'retained asset identity')
            expected[staged] = original
    for staged, original in expected.items():
        require(rows[staged]['retained_equivalent'] == original, 'staged mapping target')
    require({r['path']: r['identity'] for r in packet['request']['sources']}
            == {n: 'sha256:' + h for n, h in SOURCE.items()}, 'source closure')


def evidence_check(m, module):
    require(digest(m['reports/result.json']) == PACKET, 'packet identity')
    require(digest(m['reports/request.json']) == REQUEST, 'request identity')
    packet = module.parse(m['reports/result.json'])
    request = module.parse(m['reports/request.json'])
    same(packet['request'], request, 'request copy')
    same(packet['capture'], module.parse(m[PREFIX + 'capture.json']), 'capture copy')
    same(packet['invocation'], module.parse(m[PREFIX + 'invocation.json']), 'invocation copy')
    for key, name in [('response_base64', 'sdk_serialization.bin'),
                      ('stdout_base64', 'worker.stdout.bin'),
                      ('diagnostics_base64', 'runtime.stderr.bin')]:
        require(base64.b64decode(packet['capture'][key], validate=True) == m[PREFIX + name],
                'raw capture copy')
    raw = m[PREFIX + 'sdk_serialization.bin']
    require(len(raw) == 450 and digest(raw) ==
            'sha256:cc5217af1a9954a4406a369047dad363d17445a59d3123ae7db171dc76f1988c',
            'complete negative serialization')
    payload = json.loads(raw)
    same([payload['success'], payload['error_code'], payload['error'],
          payload['function_calls'], payload['confidence'], payload['peak_ram_mb'], payload['reason']],
         [False, 'truncated', 'tool call truncated: token budget exhausted', [],
          0.0558, 103.5, 'runtime_failure'], 'negative metadata')
    reservation = module.parse(m[PREFIX + 'reservation.json'])
    require(reservation['request_identity'] == REQUEST
            and reservation['native_execution_opt_in'] is True
            and reservation['target_readiness'] is False, 'reservation')
    require(module.record('python-isolated', m[PREFIX + 'python-isolated'])
            == packet['invocation']['shim'], 'shim bytes')
    mapping_check(m, packet)
    control = module.assess(b'{"type":"call","function_calls":[{"name":"answer","arguments":{"answer":1234}}]}', True)
    require(control['attempt_accepted'] is True and control['attempt_correct'] is True,
            'actual scorer positive control')
    result = module.verify(packet)  # Actual v2 verifier, which invokes actual v1 scorer.
    same(result, {'verified': True, 'target_readiness': False,
         'scope': 'read_only_source_and_capture_consistency_not_inference_replay_or_hardware_attestation'},
         'verification scope')
    same(packet['assessment']['payload_diagnostic'], {'accepted': False, 'answer': None,
         'correct': False, 'reason': 'unknown Needle envelope fields require prospective contract review'},
         'strict rejection')
    require(packet['assessment']['attempt_accepted'] is False
            and packet['assessment']['attempt_correct'] is False, 'negative attempt')
    same([packet['capture'][k] for k in ('status', 'load_completed', 'call_started',
          'capture_completed', 'returncode')], ['completed', True, True, True, 0], 'capture lifecycle')
    outcome = json.loads(m['reports/outcome.json'])
    same(outcome['verification'], result, 'retained verification')
    require(outcome['native_probe_command_count'] == outcome['native_attempt_reservations'] == 1
            and outcome['target_readiness'] is False, 'one non-target attempt')
    same(json.loads(m['logs/packet-verification.stdout.log']), result, 'verification log')
    for kind, identity in [('packet', PACKET), ('request', REQUEST)]:
        require(m['reports/' + kind + '-identity.txt'].decode().strip() == identity, 'printed identity')
        receipt = json.loads(m['reports/' + kind + '-identity-verification.json'])
        require(receipt['printed_identity'] == receipt['independent_identity'] == identity
                and receipt['printed_identity_matches'] is True, 'independent identity receipt')
    bundle = json.loads(m['reports/bundle-members.json'])
    require(len(bundle['files']) == 800 and len(bundle['additional_generated_members']) == 2,
            'bundle inventory counts')
    names = set()
    for row in bundle['files']:
        name = row['path']; safe_path(name)
        require(name not in names and name in m and digest(m[name]) == row['identity']
                and len(m[name]) == row['bytes'], 'bundle member identity')
        names.add(name)
    require(names | {'SHA256SUMS', 'reports/bundle-members.json'} == set(m), 'bundle coverage')
    # Claims about omitted assets remain declarations, not independently checked binaries.
    for row in json.loads(m['reports/artifact-hashes.json'])['files']:
        name = row['path']; safe_path(name)
        if name in m:
            require(digest(m[name]) == row['identity'] and len(m[name]) == row['bytes'],
                    'original artifact identity')
    before = json.loads(m['reports/source-before.json'])
    after = json.loads(m['reports/source-after.json'])
    same(before['tracked_files'], after['tracked_files'], 'producer source unchanged')
    require(len(before['tracked_files']) == 1255, 'producer tracked inventory')
    for row in before['tracked_files']:
        name = 'checkout/' + row['path']; safe_path(name)
        if name in m:
            require(digest(m[name]) == row['identity'] and len(m[name]) == row['size'],
                    'captured producer source identity')
    for receipt in json.loads(m['reports/test-results.json']):
        require(receipt['exit_status'] == 0 and
                int(m['logs/' + receipt['label'] + '.exit-status.txt']) == 0, 'historical tests exit')
        log = m[receipt['stderr']].decode()
        require(re.search(r'Ran ' + str(receipt['count']) + r' tests? in ', log)
                and log.rstrip().endswith('OK'), 'historical tests receipt')
    return result


def audit(folder=FOLDER, source_root=ROOT):
    captured = snapshot(folder)
    before = {name: digest(raw) for name, raw in captured.items()}
    _, members = inventory(captured)
    with tempfile.TemporaryDirectory(prefix='needle-evidence-') as temp:
        module = source_module(members, Path(temp), source_root)
        result = evidence_check(members, module)
    require(before == {name: digest(raw) for name, raw in snapshot(folder).items()}, 'fixture changed')
    require(not {'torch', 'needle', 'transformers'} & set(sys.modules), 'runtime imported')
    return result


class FunctionalTests(unittest.TestCase):
    def test_archive_and_actual_verifier(self):
        audit()

    def test_paths(self):
        for name in ('../x', '/x', 'a/../b', 'a\\b', '', 'a//b', './a'):
            with self.assertRaises(ValueError): safe_path(name)

    def test_commitment_and_object_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in ('commitment.json', 'objects.zip', '.gitattributes'):
                (folder / name).write_bytes(read(FOLDER / name))
            (folder / 'objects.zip').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'archive identity'): inventory(snapshot(folder))
            (folder / 'commitment.json').write_bytes(b'{}\n')
            with self.assertRaisesRegex(ValueError, 'commitment mismatch'): inventory(snapshot(folder))

    def test_snapshot_replacement_during_audit(self):
        original_inventory = inventory
        for changed in FIXTURE_FILES:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                for name, raw in snapshot(FOLDER).items():
                    (folder / name).write_bytes(raw)
                def replace_after_validation(captured):
                    result = original_inventory(captured)
                    (folder / changed).write_bytes(b'replaced after validation')
                    return result
                with mock.patch.dict(globals(), inventory=replace_after_validation):
                    with self.assertRaisesRegex(ValueError, 'fixture changed'):
                        audit(folder)

    def test_snapshot_inventory_change_during_audit(self):
        original_inventory = inventory
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name, raw in snapshot(FOLDER).items():
                (folder / name).write_bytes(raw)
            def add_after_validation(captured):
                result = original_inventory(captured)
                (folder / 'unexpected.txt').write_bytes(b'new file')
                return result
            with mock.patch.dict(globals(), inventory=add_after_validation):
                with self.assertRaisesRegex(ValueError, 'fixture inventory'):
                    audit(folder)

    def test_dependency_closure_before_import(self):
        _, original = inventory(snapshot(FOLDER))
        for missing in SOURCE:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / 'sources'; staged = Path(temp) / 'staged'
                for name in SOURCE:
                    if name == missing: continue
                    p = root / name; p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(read(ROOT / name))
                with mock.patch('builtins.compile') as compiler, mock.patch('builtins.exec') as executor:
                    with self.assertRaises(ValueError): source_module(original, staged, root)
                    compiler.assert_not_called(); executor.assert_not_called()
                self.assertFalse(staged.exists())
        with tempfile.TemporaryDirectory() as temp:
            for name in SOURCE:
                m = dict(original); m['checkout/' + name] += b'\n'
                with mock.patch('builtins.compile') as compiler, mock.patch('builtins.exec') as executor:
                    with self.assertRaisesRegex(ValueError, 'authoritative source mismatch'):
                        source_module(m, Path(temp))
                    compiler.assert_not_called(); executor.assert_not_called()
        # Positive control demonstrates both spies observe the valid load path.
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch('builtins.compile', wraps=compile) as compiler, mock.patch('builtins.exec', wraps=exec) as executor:
                source_module(original, Path(temp))
                compiler.assert_called_once(); executor.assert_called_once()

    def test_raw_packet_and_mapping_mutations(self):
        _, original = inventory(snapshot(FOLDER))
        with tempfile.TemporaryDirectory() as temp:
            module = source_module(original, Path(temp))
            for name in ('reports/result.json', 'reports/request.json', PREFIX + 'sdk_serialization.bin',
                         PREFIX + 'worker.stdout.bin', PREFIX + 'python-isolated'):
                m = dict(original); m[name] += b'\n'
                with self.subTest(name=name), self.assertRaises(ValueError): evidence_check(m, module)
            m = dict(original); ledger = json.loads(m['reports/omission-ledger.json'])
            ledger['excluded_or_identical'][0]['retained_equivalent'] = 'assets/tokenizer-embedded.raw'
            m['reports/omission-ledger.json'] = canonical(ledger)
            with self.assertRaisesRegex(ValueError, 'staged mapping'): evidence_check(m, module)

    def test_actual_verifier_semantic_mutations(self):
        _, m = inventory(snapshot(FOLDER))
        with tempfile.TemporaryDirectory() as temp:
            module = source_module(m, Path(temp))
            mutations = [lambda p: p.update(target_readiness=True),
                lambda p: p['capture'].update(response_base64='eA=='),
                lambda p: p['invocation']['materialized_sources'].pop(),
                lambda p: p['assessment'].update(attempt_accepted=True),
                lambda p: p['capture'].update(memory_bytes={'value': 0, 'reason': None})]
            for mutate in mutations:
                packet = module.parse(m['reports/result.json']); mutate(packet)
                with self.assertRaises(ValueError): module.verify(packet)


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        unittest.main(argv=[sys.argv[0]])
    else:
        require(not sys.argv[1:], 'unknown arguments')
        print(json.dumps(audit(), sort_keys=True))
