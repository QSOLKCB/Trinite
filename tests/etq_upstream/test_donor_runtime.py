"""Real upstream execution and adversarial evidence checks; Node is required."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_etq_donor import FIXTURE, DONOR_BLOBS, compare, compare_output, verify
from trinite.contracts import json_bytes


class DonorRuntimeTests(unittest.TestCase):
    def copy_fixture(self, temporary):
        target = Path(temporary) / 'fixture'
        shutil.copytree(FIXTURE, target)
        return target

    def test_real_runtime_and_retained_output_match_without_writes(self):
        before = {str(p): p.read_bytes() for p in FIXTURE.rglob('*') if p.is_file()}
        result = verify()
        self.assertTrue(all(value is True for value in result['checks'].values()))
        self.assertEqual(len(result['commitments']), 7)
        self.assertEqual(before, {str(p): p.read_bytes() for p in FIXTURE.rglob('*') if p.is_file()})

    def test_each_copied_donor_blob_is_checked_before_execution(self):
        for name in DONOR_BLOBS:
            with self.subTest(path=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self.copy_fixture(temporary)
                path = fixture / 'donor' / name
                path.write_bytes(path.read_bytes() + b' ')
                # Missing executable proves bad source is rejected before Node starts.
                with self.assertRaisesRegex(ValueError, 'donor Git blob mismatch'):
                    verify(fixture, node=str(Path(temporary) / 'missing-node'))

    def test_separately_entered_commitment_cannot_replace_published_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.copy_fixture(temporary)
            path = fixture / 'donor-commitments.json'
            record = json.loads(path.read_bytes())
            record['commitments']['eventsSha256'] = '0' * 64
            path.write_bytes(json_bytes(record))
            with self.assertRaisesRegex(ValueError, 'published commitments mismatch'):
                verify(fixture)

    def test_each_historical_check_is_consumed_and_requires_boolean_true(self):
        old = json.loads((FIXTURE / 'donor-runtime-comparison.json').read_bytes())
        for name in old['checks']:
            with self.subTest(check=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self.copy_fixture(temporary)
                record = copy.deepcopy(old)
                record['checks'][name] = 1  # Python equality alone would accept 1 == True.
                (fixture / 'donor-runtime-comparison.json').write_bytes(json_bytes(record))
                with self.assertRaisesRegex(ValueError, 'historical checks mismatch'):
                    verify(fixture)

    def test_changed_historical_metadata_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.copy_fixture(temporary)
            path = fixture / 'donor-runtime-comparison.json'
            record = json.loads(path.read_bytes())
            record['node_version'] = 'v0.0.0'
            path.write_bytes(json_bytes(record))
            with self.assertRaisesRegex(ValueError, 'retained runtime evidence mismatch'):
                verify(fixture)

    def test_retained_output_is_compared_to_fresh_stdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.copy_fixture(temporary)
            path = fixture / 'donor-runtime-output.json'
            path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'retained donor stdout mismatch'):
                verify(fixture)

    def test_evidence_cannot_change_output_identity_producer_or_checks(self):
        original = json.loads((FIXTURE / 'donor-runtime-evidence-v2.json').read_bytes())
        for field in ('stdout', 'producer_files', 'checks'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                fixture = self.copy_fixture(temporary)
                record = copy.deepcopy(original)
                record[field] = {}
                (fixture / 'donor-runtime-evidence-v2.json').write_bytes(json_bytes(record))
                with self.assertRaisesRegex(ValueError, 'retained runtime evidence mismatch'):
                    verify(fixture)

    def test_complete_live_payload_is_checked_even_with_valid_commitment_labels(self):
        record, raw, _ = compare()
        original = json.loads(raw)
        for field in ('eventDocumentCanonical', 'graph', 'registry'):
            with self.subTest(field=field):
                output = copy.deepcopy(original)
                if field == 'eventDocumentCanonical':
                    output[field] += '\n'
                elif field == 'graph':
                    output[field]['edges'][0]['target'] += 1
                else:
                    output[field]['sites'][0]['v2MidiNote'] += 1
                with self.assertRaisesRegex(ValueError, 'donor comparison failed'):
                    compare_output(json_bytes(output)[:-1], record['commitments'])

    def test_missing_node_is_a_failure_not_a_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                verify(node=str(Path(temporary) / 'missing-node'))


if __name__ == '__main__':
    unittest.main()
