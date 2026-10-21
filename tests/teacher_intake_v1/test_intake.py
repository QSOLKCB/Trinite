"""Synthetic contract tests only: no teacher download, inference, or training."""
import base64
import copy
from pathlib import Path
import tempfile
import unittest

from trinite.contracts import ContractError, identity, json_bytes
from scripts.teacher_intake_v1 import intake, spec


def attempt(index, raw, outcome='completed', start=0, end=None):
    return {'index': index, 'outcome': outcome,
            'raw_base64': base64.b64encode(raw).decode('ascii'),
            'final_start': start, 'final_end': len(raw) if end is None else end}


class IntakeContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        artifacts = []
        for role in sorted(spec.ROLES):
            raw = ('synthetic, non-executable ' + role).encode()
            path = role + '.fixture'
            (self.root / path).write_bytes(raw)
            artifacts.append({'path': path, 'role': role, 'bytes': len(raw),
                              'identity': identity(raw)})
        self.manifest = {'model': 'openai/gpt-oss-20b', 'revision': 'a' * 40,
                         'runtime': 'synthetic-test-only', 'quantization': 'not-loaded',
                         'artifacts': artifacts}
        self.frozen = spec.request(self.manifest)

    def capture(self, count=32):
        seen = set()
        attempts = []
        for slot in self.frozen['slots']:
            n, a = slot['n'], slot['a']
            b = (a + slot['index'] // (3 * n)) % n
            self.assertNotIn((n, min(a,b), max(a,b)), seen)
            seen.add((n, min(a,b), max(a,b)))
            raw = json_bytes({'n': n, 'a': a, 'b': b})
            attempts.append(attempt(slot['index'], raw))
        for i in range(count, 32):
            attempts[i] = attempt(i, b'synthetic timeout trace', 'timeout')
        return {'schema': 'trinite.teacher-capture.v1',
                'request_identity': identity(json_bytes(self.frozen)), 'attempts': attempts}

    def test_fake_assets_all_four_roles_checked_without_execution(self):
        spec.check_assets(self.manifest, self.root)
        entry = self.manifest['artifacts'][0]
        path = self.root / entry['path']
        path.write_bytes(b'x' * entry['bytes'])
        with self.assertRaises(ContractError):
            spec.check_assets(self.manifest, self.root)

    def test_manifest_requires_roles_revision_and_contained_paths(self):
        mutations = [('revision', 'main'), ('model', 'different/model')]
        for key, value in mutations:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.manifest)
                bad[key] = value
                with self.assertRaises(ContractError):
                    spec.validate_manifest(bad)
        for path in ('../escape', '/absolute', 'dir/../escape', 'a\\b'):
            bad = copy.deepcopy(self.manifest)
            bad['artifacts'][0]['path'] = path
            with self.subTest(path=path), self.assertRaises(ContractError):
                spec.validate_manifest(bad)
        bad = copy.deepcopy(self.manifest)
        bad['artifacts'][0]['role'] = bad['artifacts'][1]['role']
        with self.assertRaises(ContractError):
            spec.validate_manifest(bad)

    def test_request_binds_all_32_prescribed_slots_and_sources(self):
        self.assertEqual(len(self.frozen['slots']), 32)
        for i, slot in enumerate(self.frozen['slots']):
            self.assertEqual((slot['index'], slot['n'], slot['a']),
                             (i, spec.MODULI[i % 3], (i // 3) % spec.MODULI[i % 3]))
        for path in ('docs/DATA.md', 'AGENTS.md', 'tests/teacher_intake_v1/test_intake.py'):
            self.assertIn(path, self.frozen['source'])
        for key in ('n', 'a', 'prompt'):
            bad = copy.deepcopy(self.frozen)
            bad['slots'][0][key] = 'tampered'
            with self.subTest(key=key), self.assertRaises(ContractError):
                spec.validate_request(bad)

    def test_strict_proposals_reject_types_keys_prose_utf8_and_bounds(self):
        cases = [b'{"n":5,"a":0,"b":true}', b'{"n":5,"a":false,"b":1}',
                 b'{"n":5.0,"a":0,"b":1}', b'{"n":5,"a":0,"b":1.0}',
                 b'{"n":5,"a":0,"b":1,"b":2}', b'{"n":5,"a":0,"b":NaN}',
                 b'{"n":5,"a":0,"b":Infinity}', b'{"n":5,"a":0,"b":-Infinity}',
                 b'{"n":5,"a":0,"b":9007199254740992}',
                 b'{"n":5,"a":0,"b":1e999}', b'\xff', b'\xef\xbb\xbf{}',
                 b'{"n":5,"a":0,"b":-1}', b'{"n":5,"a":0,"b":5}',
                 b'{"n":7,"a":0,"b":1}', b'{"n":5,"a":1,"b":1}',
                 b'{"n":5,"a":0,"b":1,"answer":1}',
                 b'Here is the answer: {"n":5,"a":0,"b":1}',
                 b'```json\n{"n":5,"a":0,"b":1}\n```', b'[]', b'null']
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                intake.proposal(raw, 5, 0)
        valid = b'{"n":5,"a":0,"b":4}'
        self.assertEqual(intake.proposal(valid, 5, 0), {'n':5, 'a':0, 'b':4})
        self.assertEqual(intake.proposal(valid + b' ' * (4096-len(valid)), 5, 0)['b'], 4)
        with self.assertRaises(ContractError):
            intake.proposal(valid + b' ' * (4097-len(valid)), 5, 0)

    def test_counter_independently_matches_modulo_exhaustively(self):
        for n in spec.MODULI:
            for a in range(n):
                for b in range(n):
                    self.assertEqual(intake.counter(n, a, b), (a+b) % n)

    def test_complete_packet_formal_only_with_exact_oracle_labels(self):
        packet = intake.seal(self.frozen, self.capture())
        self.assertEqual(packet['counts'], {'attempts':32, 'valid_unique':32, 'rejected':0})
        self.assertEqual(packet['admission'], 'ADMITTED_FORMAL_ONLY')
        self.assertEqual(packet['raw_admission'], 'QUARANTINED_NOT_TRAINING_DATA')
        self.assertEqual(intake.verify(packet), packet)
        for item in packet['items']:
            n,a,b = (item['formal_spec'][k] for k in ('n','a','b'))
            self.assertEqual(item['answer'], f'{(a+b)%n:02d}')
            self.assertEqual(item['prompt'], f'addmod[{n:02d},{a:02d},{b:02d}]=')
            self.assertEqual(item['family'], [n,min(a,b),max(a,b)])
            self.assertIn('teacher_origin', item['lineage'])
            self.assertEqual(item['admission'], 'ADMITTED_FORMAL_ONLY')

    def test_threshold_and_failures_retained(self):
        for count in (0, 15, 16):
            with self.subTest(count=count):
                capture = self.capture(count)
                capture['attempts'][-1] = attempt(31, b'error trace\xff', 'runtime_error')
                packet = intake.seal(self.frozen, capture)
                self.assertEqual(packet['counts']['valid_unique'], count)
                self.assertEqual(len(packet['receipts']), 32)
                self.assertEqual(packet['capture'], capture)
                self.assertEqual(packet['receipts'][-1]['reason'], 'runtime_error')
                self.assertEqual(packet['admission'], 'ADMITTED_FORMAL_ONLY' if count >= 16 else 'BLOCKED_UNDERFILLED')
                self.assertEqual(packet['raw_admission'], 'QUARANTINED_NOT_TRAINING_DATA')

    def test_rejected_final_keeps_raw_bytes_and_receipt(self):
        capture = self.capture()
        capture['attempts'][0] = attempt(0, b'not JSON\xff')
        packet = intake.seal(self.frozen, capture)
        self.assertEqual(packet['receipts'][0]['decision'], 'rejected')
        self.assertEqual(packet['receipts'][0]['raw_identity'], identity(b'not JSON\xff'))
        self.assertEqual(packet['capture'], capture)
        self.assertEqual(packet['counts']['valid_unique'], 31)

    def test_final_byte_offsets_exclude_analysis_but_preserve_raw(self):
        capture = self.capture()
        final = base64.b64decode(capture['attempts'][0]['raw_base64'])
        prefix = 'synthetic analysis: π\n'.encode()
        raw = prefix + final + b'\nsynthetic trailer'
        capture['attempts'][0] = attempt(0, raw, start=len(prefix), end=len(prefix)+len(final))
        packet = intake.seal(self.frozen, capture)
        receipt = packet['receipts'][0]
        self.assertEqual(receipt['raw_identity'], identity(raw))
        self.assertEqual(receipt['final_identity'], identity(final))
        self.assertEqual(receipt['raw_bytes'], len(raw))
        self.assertEqual(packet['counts']['valid_unique'], 32)
        capture['attempts'][0]['final_start'] = 0
        self.assertEqual(intake.seal(self.frozen, capture)['receipts'][0]['decision'], 'rejected')

    def test_commutative_duplicates_keep_first_valid_attempt(self):
        capture = self.capture()
        capture['attempts'][0] = attempt(0, json_bytes({'n':5,'a':0,'b':1}))
        capture['attempts'][3] = attempt(3, json_bytes({'n':5,'a':1,'b':0}))
        packet = intake.seal(self.frozen, capture)
        self.assertNotEqual(packet['receipts'][0]['decision'], 'rejected')
        self.assertEqual(packet['receipts'][3]['decision'], 'rejected')
        self.assertIn('duplicate', packet['receipts'][3]['reason'])
        family = [item for item in packet['items'] if item['family'] == [5,0,1]]
        self.assertEqual(len(family), 1)
        self.assertEqual(family[0]['lineage']['attempt'], 0)

    def test_all_32_required_in_fixed_order_with_valid_metadata(self):
        for count in (0, 31, 33):
            capture = self.capture()
            capture['attempts'] = (capture['attempts'] + [capture['attempts'][0]])[:count]
            with self.subTest(count=count), self.assertRaises(ContractError):
                intake.seal(self.frozen, capture)
        for field, value in [('index', True), ('index', 1), ('outcome', 'skipped'),
                             ('final_start', True), ('final_start', -1),
                             ('final_start', 99999), ('final_end', -1), ('final_end', 99999),
                             ('raw_base64', '!'), ('raw_base64', 'eA==='), ('raw_base64', 7)]:
            capture = self.capture()
            capture['attempts'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ContractError):
                intake.seal(self.frozen, capture)
        capture = self.capture()
        capture['request_identity'] = identity(b'wrong request')
        with self.assertRaises(ContractError):
            intake.seal(self.frozen, capture)

    def test_retention_bound_includes_nonfinal_bytes(self):
        capture = self.capture()
        capture['attempts'][0] = attempt(0, b'x' * 65536, 'timeout', end=0)
        packet = intake.seal(self.frozen, capture)
        self.assertEqual(packet['receipts'][0]['raw_bytes'], 65536)
        capture['attempts'][0] = attempt(0, b'x' * 65537, 'timeout', end=0)
        with self.assertRaises(ContractError):
            intake.seal(self.frozen, capture)

    def test_verify_exact_regeneration_rejects_tampering(self):
        packet = intake.seal(self.frozen, self.capture())
        mutations = [lambda p: p['items'][0].update(answer='99'),
                     lambda p: p['receipts'][0].update(reason='tampered'),
                     lambda p: p['counts'].update(valid_unique=31),
                     lambda p: p.update(raw_admission='ADMITTED_FORMAL_ONLY'),
                     lambda p: p['capture']['attempts'][0].update(final_end=0),
                     lambda p: p['request']['source'].update({'AGENTS.md':identity(b'changed')}),
                     lambda p: p.update(extra='unexpected')]
        for index, mutate in enumerate(mutations):
            bad = copy.deepcopy(packet)
            mutate(bad)
            with self.subTest(index=index), self.assertRaises(ContractError):
                intake.verify(bad)


    def test_boolean_integer_equality_cannot_forge_regeneration(self):
        packet=intake.seal(self.frozen,self.capture())
        bad=copy.deepcopy(packet)
        bad['counts']['attempts']=True
        with self.assertRaises(ContractError):intake.verify(bad)
        bad=copy.deepcopy(packet)
        row=next(r for r in bad['items'] if r['formal_spec']['a']==1)
        row['formal_spec']['a']=True
        with self.assertRaises(ContractError):intake.verify(bad)
        bad=copy.deepcopy(self.frozen)
        bad['slots'][1]['index']=True
        with self.assertRaises(ContractError):spec.validate_request(bad)


if __name__ == '__main__':
    unittest.main()
