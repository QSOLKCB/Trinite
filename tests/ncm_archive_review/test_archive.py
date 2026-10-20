"""Frozen public evidence regression tests; no model, network, GPU or writes."""
import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from verify_ncm_rtx_archive import DEFAULT_FIXTURE, FIXTURE_SHA256, parse, verify_fixture


class PublicArchiveTests(unittest.TestCase):
    def test_frozen_summary_and_separate_gold_diagnostic(self):
        result = verify_fixture()
        self.assertEqual(result['failed']['state'], 'failed-before-inference')
        self.assertIsNone(result['failed']['warm_solved'])
        self.assertFalse(result['failed']['rejected_tensor_identified'])
        for label in ('run', 'replay'):
            self.assertEqual(result[label]['warm_boards'], 4)
            self.assertEqual(result[label]['warm_solved'], 4)
            self.assertEqual(result[label]['gold_codec_exact_target'], 4)
            self.assertEqual(result[label]['peak_allocated_bytes'], 256471552)
            self.assertEqual(result[label]['peak_reserved_bytes'], 281018368)
        self.assertAlmostEqual(result['run']['warm_mean_ms'], 63.31726825010264)
        self.assertAlmostEqual(result['replay']['warm_mean_ms'], 61.810458499167)
        self.assertTrue(result['exact_recorded_result_match'])
        self.assertFalse(result['hardware_attestation'])
        self.assertFalse(result['new_inference_executed'])
        self.assertFalse(result['original_private_handoffs_reverified'])

    def test_read_only_bytes(self):
        before = {p.name: p.read_bytes() for p in DEFAULT_FIXTURE.iterdir()}
        verify_fixture()
        self.assertEqual(before, {p.name: p.read_bytes() for p in DEFAULT_FIXTURE.iterdir()})

    def test_manifest_external_commitment(self):
        self.assertEqual(hashlib.sha256((DEFAULT_FIXTURE / 'manifest.json').read_bytes()).hexdigest(),
                         FIXTURE_SHA256)
        with self.assertRaisesRegex(ValueError, 'external manifest'):
            verify_fixture(expected_sha256='0' * 64)

    def test_payload_byte_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'fixture'
            shutil.copytree(DEFAULT_FIXTURE, root)
            with (root / 'run.zip').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'digest/size'):
                verify_fixture(root)

    def test_unexpected_payload_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'fixture'
            shutil.copytree(DEFAULT_FIXTURE, root)
            (root / 'extra.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'unexpected or missing'):
                verify_fixture(root)

    def test_missing_payload_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'fixture'
            shutil.copytree(DEFAULT_FIXTURE, root)
            (root / 'replay.zip').unlink()
            with self.assertRaisesRegex(ValueError, 'unexpected or missing'):
                verify_fixture(root)

    def test_symlink_payload_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'fixture'
            shutil.copytree(DEFAULT_FIXTURE, root)
            (root / 'run.zip').unlink()
            (root / 'run.zip').symlink_to(DEFAULT_FIXTURE / 'run.zip')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                verify_fixture(root)

    def test_json_duplicate_and_nonfinite_rejected(self):
        for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse(data)

    def test_verification_does_not_import_torch(self):
        self.assertNotIn('torch', sys.modules)
        verify_fixture()
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
