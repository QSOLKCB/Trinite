"""Real upstream execution and adversarial evidence checks; Node is required."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_etq_donor import FIXTURE, DONOR_BLOBS, compare, compare_output, verify
import check_etq_donor as donor
from trinite.contracts import json_bytes


class DonorRuntimeTests(unittest.TestCase):
    def copy_fixture(self, temporary):
        target = Path(temporary) / 'fixture'
        shutil.copytree(FIXTURE, target)
        return target

    def test_real_runtime_and_retained_output_match_without_writes(self):
        before = {str(p): p.read_bytes() for p in FIXTURE.rglob('*') if p.is_file()}
        result = verify()
        self.assertEqual(result['node_version'], subprocess.check_output(['node', '--version']).decode().strip())
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

    def controller_node(self, directory, body):
        """Executable fakes are only used to assert controller failures."""
        path = Path(directory) / 'node'
        path.write_text('#!' + sys.executable + '\n' + body, encoding='utf-8')
        path.chmod(0o700)
        return str(path)

    def test_foreign_live_node_version_cannot_verify_same_evidence(self):
        real_node = shutil.which('node')
        self.assertIsNotNone(real_node, 'real Node is required')
        with tempfile.TemporaryDirectory() as temporary:
            node = self.controller_node(temporary,
                'import os, sys\n'
                'if sys.argv[1:] == ["--version"]:\n'
                '    print("v24.15.0")\n'
                'else:\n'
                f'    os.execv({real_node!r}, [{real_node!r}, *sys.argv[1:]])\n')
            with self.assertRaisesRegex(ValueError, 'Node version mismatch.*v24.15.0'):
                verify(node=node)

    def test_v2_evidence_extra_lf_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.copy_fixture(temporary)
            path = fixture / 'donor-runtime-evidence-v2.json'
            path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'v2 evidence is not canonical'):
                verify(fixture)

    def test_v3_evidence_requires_exact_bytes_and_bound_fields(self):
        original = (FIXTURE / 'donor-runtime-evidence-v3.json').read_bytes()
        record = json.loads(original)
        variants = [original + b'\n', b' ' + original, original.replace(b'\n', b'\r\n'),
                    json.dumps(record, indent=2).encode() + b'\n']
        for field in ('stdout', 'producer_files', 'checks', 'historical_evidence_v2'):
            changed = copy.deepcopy(record)
            changed[field] = {}
            variants.append(json_bytes(changed))
        for index, raw in enumerate(variants):
            with self.subTest(variant=index), tempfile.TemporaryDirectory() as temporary:
                fixture = self.copy_fixture(temporary)
                (fixture / 'donor-runtime-evidence-v3.json').write_bytes(raw)
                with self.assertRaisesRegex(ValueError, 'retained runtime evidence mismatch'):
                    verify(fixture)

    def test_cli_preserves_version_and_driver_failure_diagnostics(self):
        for stage in ('version', 'driver'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                self.controller_node(temporary,
                    'import sys\n'
                    f'if sys.argv[1:] == ["--version"] and {stage!r} == "driver":\n'
                    '    print("v24.19.0")\n'
                    '    sys.exit(0)\n'
                    'print("DONOR-PARTIAL-STDOUT", flush=True)\n'
                    'sys.stderr.buffer.write(b"DONOR-SPECIFIC-DIAGNOSTIC\\xff\\n")\n'
                    'sys.stderr.flush()\n'
                    'sys.exit(7)\n')
                env = dict(os.environ, PATH=temporary + os.pathsep + os.environ['PATH'],
                           PYTHONPATH=str(ROOT / 'src'))
                result = subprocess.run([sys.executable, '-S', str(ROOT / 'scripts/check_etq_donor.py')],
                                        env=env, capture_output=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b'DONOR-SPECIFIC-DIAGNOSTIC', result.stderr)
                self.assertIn(b'DONOR-PARTIAL-STDOUT', result.stderr)
                self.assertIn(b'\\xff', result.stderr)
                self.assertIn(b'returncode=7', result.stderr)

    def test_timeout_preserves_partial_streams_and_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            node = self.controller_node(temporary,
                'import sys, time\n'
                'print("DONOR-TIMEOUT-STDOUT", flush=True)\n'
                'print("DONOR-TIMEOUT-STDERR", file=sys.stderr, flush=True)\n'
                'time.sleep(10)\n')
            with self.assertRaises(RuntimeError) as caught:
                donor.run_node([node, '--version'], env=os.environ.copy(),
                               cwd=temporary, timeout=0.5)
            self.assertIn('DONOR-TIMEOUT-STDOUT', str(caught.exception))
            self.assertIn('DONOR-TIMEOUT-STDERR', str(caught.exception))
            self.assertIn('timeout=0.5', str(caught.exception))
            self.assertEqual(caught.exception.stdout, b'DONOR-TIMEOUT-STDOUT\n')
            self.assertEqual(caught.exception.stderr, b'DONOR-TIMEOUT-STDERR\n')


if __name__ == '__main__':
    unittest.main()
