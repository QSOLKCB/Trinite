from pathlib import Path
import shutil
import tempfile
import unittest

from trinite.contracts import identity, parse_json
from retention_source_history import (
    ORIGINAL_AGENTS_ID, PRODUCER_ARCHIVE, assert_preserved_sources, historical_agents,
)


ROOT = Path(__file__).resolve().parents[1]


class RetentionInstructionHistoryTests(unittest.TestCase):
    def test_original_instructions_match_both_unchanged_historical_manifests(self):
        original = historical_agents(ROOT)
        self.assertEqual(identity(original), ORIGINAL_AGENTS_ID)
        for name in ('retention-budget-v2/prior-assurance-source.json',
                     'retention-task-balance-v1/prior-budget-source.json'):
            expected = parse_json((ROOT / 'fixtures' / name).read_bytes(), canonical=True)
            assert_preserved_sources(self, ROOT, expected)

    def temporary_root(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        target = root / PRODUCER_ARCHIVE
        target.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / PRODUCER_ARCHIVE, target)
        return root

    def test_only_root_instructions_use_history_and_runtime_mutation_still_fails(self):
        root = self.temporary_root()
        (root / 'AGENTS.md').write_bytes(b'current operating guidance\n')
        runtime = root / 'src/trinite/training.py'
        runtime.parent.mkdir(parents=True)
        runtime.write_bytes(b'original runtime\n')
        expected = {'AGENTS.md': ORIGINAL_AGENTS_ID,
                    'src/trinite/training.py': identity(runtime.read_bytes())}
        assert_preserved_sources(self, root, expected)
        runtime.write_bytes(b'changed runtime\n')
        with self.assertRaisesRegex(AssertionError, 'src/trinite/training.py'):
            assert_preserved_sources(self, root, expected)

    def test_repinning_historical_instructions_to_current_bytes_is_rejected(self):
        root = self.temporary_root()
        (root / 'AGENTS.md').write_bytes(b'new instructions\n')
        with self.assertRaisesRegex(AssertionError, 'historical AGENTS pin'):
            assert_preserved_sources(self, root, {
                'AGENTS.md': identity((root / 'AGENTS.md').read_bytes()),
            })

    def test_changed_archive_is_rejected_before_reading_members(self):
        root = self.temporary_root()
        path = root / PRODUCER_ARCHIVE
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(AssertionError, 'archive identity mismatch'):
            historical_agents(root)
