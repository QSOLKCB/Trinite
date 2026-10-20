"""Offline inspection of retained local evidence, never a fresh inference run."""
import hashlib
import json
import lzma
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from trinite.contracts import identity, json_bytes, parse_json
from trinite.data import audit_bytes
from batch_composition_v1.validation import validate_report
from batch_composition_v1.spec import LAYOUTS

FOLDER = ROOT / 'fixtures/batch-composition-ryzen-20261020'
MAX = 64 * 1024 * 1024
COMMITMENT = 'sha256:f47c29d47cfd91eecaecf8e5c432e93b59b30f380a1f945ae1775c8e29f971d9'


def read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular file required')
    with path.open('rb') as stream:
        raw = stream.read(MAX + 1)
    if len(raw) > MAX:
        raise ValueError('file exceeds bound')
    return raw


def decode(raw):
    reader = lzma.LZMADecompressor(memlimit=MAX)
    data = reader.decompress(raw, max_length=MAX + 1)
    if len(data) > MAX or not reader.eof or reader.unused_data:
        raise ValueError('invalid or oversized XZ object')
    return data


class RyzenEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = read(FOLDER / 'commitment.json')
        if identity(raw) != COMMITMENT:
            raise ValueError('commitment identity mismatch')
        cls.packet = parse_json(raw, canonical=True)

    def member(self, name):
        record = self.packet['members'][name]
        stored = record['stored_path']
        self.assertEqual(Path(stored).parts[0], 'objects')
        self.assertNotIn('..', Path(stored).parts)
        raw = read(FOLDER / stored)
        self.assertEqual(identity(raw), self.packet['files'][stored])
        if stored.endswith('.xz'):
            raw = decode(raw)
            self.assertEqual(self.packet['compressed'][stored], {
                'uncompressed_identity': identity(raw), 'uncompressed_bytes': len(raw)})
        self.assertEqual(len(raw), record['public_bytes'])
        self.assertEqual(identity(raw), record['public_identity'])
        if not record['redactions']:
            self.assertEqual(record['original_identity'], identity(raw))
            self.assertEqual(record['original_bytes'], len(raw))
        else:
            self.assertNotEqual(record['original_identity'], identity(raw))
            for edit in record['redactions']:
                self.assertGreater(edit['count'], 0)
                self.assertLess(edit['rule'], len(self.packet['redaction_rules']))
        return raw

    def test_complete_inventory_integrity_and_privacy_derivation(self):
        actual = {p.relative_to(FOLDER).as_posix() for p in FOLDER.rglob('*') if p.is_file()}
        self.assertEqual(actual, set(self.packet['files']) | {'commitment.json'})
        for name, digest in self.packet['files'].items():
            self.assertNotIn('..', Path(name).parts)
            self.assertFalse(Path(name).is_absolute())
            self.assertEqual(identity(read(FOLDER / name)), digest)
        inventory = json.loads(read(FOLDER / 'inventory.json'))
        members = self.packet['members']
        self.assertEqual(len(members), 105)
        self.assertEqual({r['path']: r['size'] for r in inventory},
                         {p: r['original_bytes'] for p, r in members.items()})
        manifest = json.loads(self.member('evidence/file-identities.json'))
        self.assertEqual(len(manifest), 104)
        self.assertEqual(set(manifest), set(members) - {'evidence/file-identities.json'})
        for name, record in members.items():
            raw = self.member(name)
            self.assertNotIn(b'/home/', raw, name)
            self.assertNotIn(b'/tmp/', raw, name)
            self.assertNotIn(b'Linux ubuntu ', raw, name)
            if name in manifest:
                self.assertEqual(record['original_identity'], manifest[name])
        used = {r['stored_path'] for r in members.values()}
        self.assertEqual(used, {p for p in self.packet['files'] if p.startswith('objects/')})

    def test_recorded_conformance_source_binding_and_exact_replay(self):
        prefix = 'evidence/local/'
        request_raw = self.member(prefix + 'batch-composition-request.json')
        report_raw = self.member(prefix + 'batch-composition-report.json')
        request = parse_json(request_raw, canonical=True)
        report = parse_json(report_raw, canonical=True)
        self.assertEqual(identity(request_raw), 'sha256:aa4313375461b6cebfe2b73b4f20d71ccca1b9a38e0bc15ab676fe1238a2b335')
        self.assertEqual(identity(report_raw), 'sha256:92b7ad27e9e10f980ab28899028f02d0f21167b92df854102cd554e78f2e2f1a')
        self.assertEqual(validate_report(report, request), {'conforms': True, 'budget': {
            'physical_forward_calls': 2500, 'row_evaluations': 6720, 'active_decisions': 6720}})
        self.assertTrue(audit_bytes(read(ROOT/'fixtures/formal-v1/dataset.json'),
                                   read(ROOT/'fixtures/formal-v1/manifest.json'))['verified'])
        approved = parse_json(read(ROOT/'fixtures/batch-composition-v2/prepublication-review.json'), canonical=True)
        self.assertEqual(request['source'], approved['source'])
        self.assertEqual(len(request['source']), 149)
        blobs = json.loads(read(FOLDER/'source-git-blobs.json'))
        for path, digest in request['source'].items():
            raw = read(ROOT/path)
            self.assertEqual(identity(raw), digest, path)
            self.assertEqual(hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest(), blobs[path])
        cells = b''.join(json_bytes(c) for c in report['cells'])
        layouts = b''.join(json_bytes({'seed': c['seed'], 'arm': c['arm'], 'layout': layout,
                                     'result': c['layouts'][layout]})
                           for c in report['cells'] for layout in LAYOUTS)
        attempts = sorted({p.rsplit('/', 1)[0] for p in self.packet['members']
                           if p.startswith(prefix) and p.endswith('/invocation.json')})
        actions = []
        for attempt in attempts:
            invocation = parse_json(self.member(attempt+'/invocation.json'), canonical=True)
            actions.append(invocation['action'])
            self.assertEqual(invocation['request_identity'], identity(request_raw))
            status = parse_json(self.member(attempt+'/status.json'), canonical=True)
            self.assertEqual((status['outcome'], status['exit_code']), ('completed', 0))
            self.assertEqual(self.member(attempt+'/computed-report.json'), report_raw)
            self.assertEqual(self.member(attempt+'/completed-cells.jsonl'), cells)
            self.assertEqual(self.member(attempt+'/completed-layouts.jsonl'), layouts)
            if invocation['action'] == 'verify':
                receipt = parse_json(self.member(attempt+'/replay-receipt.json'), canonical=True)
                self.assertIs(receipt['exact_match'], True)
                self.assertIs(receipt['conforms'], True)
                self.assertEqual(receipt['request_identity'], identity(request_raw))
                self.assertEqual(receipt['actual_identity'], identity(report_raw))
                self.assertEqual(receipt['expected_identity'], identity(report_raw))
        self.assertEqual(actions, ['run', 'verify'])
        summary = json.loads(self.member('evidence/summary.json'))
        self.assertEqual(sum(r['test_count'] for r in summary['test_results']), 50)
        self.assertTrue(all(r['exit_code'] == 0 for r in summary['test_results']))
        audit = json.loads(read(FOLDER/'independent-audit.json'))
        self.assertEqual(audit['vectors'], 10920)
        self.assertEqual(audit['violations'], 0)
        self.assertEqual(audit['report_identity'], identity(report_raw))


if __name__ == '__main__':
    unittest.main()
