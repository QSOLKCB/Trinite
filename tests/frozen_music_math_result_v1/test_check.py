"""Adversarial derivative-integrity and frozen-score checks, without models."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from frozen_music_math_result_v1 import check


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Path(self.tmp.name) / 'fixture'
        shutil.copytree(check.FIXTURE, self.fixture)

    def test_all_records_and_scope(self):
        before = check.snapshot(self.fixture)
        receipt = check.verify(self.fixture)
        self.assertEqual(receipt['prediction_records'], 256)
        self.assertEqual(receipt['initial_exact'], 0)
        self.assertEqual(receipt['final_exact'], 3)
        for name in ('original_raw_custody_verified', 'inference_replayed',
                     'fresh_confirmation', 'learning_adequate', 'phase5_ready', 'release_ready'):
            self.assertIs(receipt[name], False)
        self.assertEqual(before, check.snapshot(self.fixture))
        self.assertNotIn('torch', sys.modules)

    def test_each_member_is_bound(self):
        for name in check.FIXTURE_HASHES:
            with self.subTest(name=name):
                path = self.fixture / name
                old = path.read_bytes()
                path.write_bytes(old + b' ')
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    check.verify(self.fixture)
                path.write_bytes(old)

    def test_missing_and_extra(self):
        path = self.fixture / 'README.md'
        path.rename(self.fixture / 'extra.md')
        with self.assertRaisesRegex(ValueError, 'membership'):
            check.verify(self.fixture)

    def test_extra_directory(self):
        (self.fixture / 'nested').mkdir()
        with self.assertRaisesRegex(ValueError, 'membership'):
            check.verify(self.fixture)

    def test_linked_member(self):
        path = self.fixture / 'predictions.json'
        path.unlink()
        path.symlink_to(check.FIXTURE / 'predictions.json')
        with self.assertRaisesRegex(ValueError, 'regular'):
            check.verify(self.fixture)

    def test_linked_root(self):
        link = Path(self.tmp.name) / 'link'
        link.symlink_to(self.fixture, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'non-link'):
            check.verify(link)

    def test_linked_ancestor(self):
        parent = Path(self.tmp.name) / 'parent'
        parent.symlink_to(self.fixture.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'ancestor'):
            check.verify(parent / 'fixture')

    def test_nonregular_member(self):
        path = self.fixture / 'summary.json'
        path.unlink()
        path.mkdir()
        with self.assertRaisesRegex(ValueError, 'regular'):
            check.verify(self.fixture)

    def test_frozen_source_change(self):
        original = check.read
        def changed(path):
            raw = original(path)
            return raw + b' ' if str(path).endswith('/protocol.py') else raw
        with mock.patch.object(check, 'read', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                check.verify(self.fixture)

    def test_foreign_import(self):
        foreign = types.SimpleNamespace(__file__='/foreign/protocol.py')
        with mock.patch.dict(sys.modules, {'frozen_music_math_v1.protocol': foreign}):
            with self.assertRaisesRegex(ValueError, 'foreign'):
                check.verify(self.fixture)

    def test_cache_rejected_before_import(self):
        root = Path(self.tmp.name) / 'checkout'
        package = root / 'scripts/frozen_music_math_v1'
        package.mkdir(parents=True)
        (package / '__pycache__').mkdir()
        with mock.patch.object(check, 'ROOT', root):
            with self.assertRaisesRegex(ValueError, 'cache'):
                check.sources()

    def test_package_shadow_rejected_without_execution(self):
        root = Path(self.tmp.name) / 'checkout'
        package = root / 'scripts/frozen_music_math_v1'
        shutil.copytree(check.ROOT / 'scripts/frozen_music_math_v1', package)
        shadow = package / 'protocol'
        shadow.mkdir()
        marker = Path(self.tmp.name) / 'executed'
        (shadow / '__init__.py').write_text(
            "from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n")
        with mock.patch.object(check, 'ROOT', root):
            with self.assertRaisesRegex(ValueError, 'shadow'):
                check.sources()
        self.assertFalse(marker.exists())

    def test_foreign_package_resolution_cannot_execute(self):
        foreign = Path(self.tmp.name) / 'foreign'
        package = foreign / 'frozen_music_math_v1'
        package.mkdir(parents=True)
        marker = foreign / 'executed'
        (package / '__init__.py').write_text(
            "from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n")
        with mock.patch.dict(sys.modules):
            for name in list(sys.modules):
                if name == 'frozen_music_math_v1' or name.startswith('frozen_music_math_v1.'):
                    del sys.modules[name]
            with mock.patch.object(sys, 'path', [str(foreign), *sys.path]):
                with self.assertRaisesRegex(ValueError, 'resolution'):
                    check.sources()
        self.assertFalse(marker.exists())

    def test_exact_scoring_not_just_eos(self):
        check.sources()
        from frozen_music_math_v1.protocol import score_records
        records = json.loads((self.fixture / 'predictions.json').read_text())
        summary = score_records(records)
        self.assertEqual(summary['cells']['one-29-064']['eos_terminated'], 32)
        self.assertEqual(summary['cells']['one-29-064']['correct'], 2)
        self.assertEqual(summary['cells']['one-29-064']['unique_raw_outputs'], 1)
        self.assertEqual(summary['max_constant_baseline']['correct'], 2)
        records[0]['generated_tokens'] = [257]
        self.assertNotEqual(score_records(records), summary)
        with self.assertRaises(ValueError):
            score_records(records[:-1])

    def test_mutation_during_scoring(self):
        check.sources()
        from frozen_music_math_v1 import protocol
        scorer = protocol.score_records
        def changed(records):
            result = scorer(records)
            (self.fixture / 'README.md').write_text('changed')
            return result
        with mock.patch.object(protocol, 'score_records', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'during verification'):
                check.verify(self.fixture)


if __name__ == '__main__':
    unittest.main()
