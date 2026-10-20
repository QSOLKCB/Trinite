"""Test-only preservation of historical instructions, never a runtime fallback."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from trinite.contracts import identity


# Independently retained identities from the unchanged budget evidence packet.
PRODUCER_ARCHIVE = 'fixtures/retention-budget-v2/producer-source.zip'
PRODUCER_ARCHIVE_ID = 'sha256:011deae1cd927958d658695f2b192115cb3e40fb0d5000ce189ae78f1f2fd0ef'
PRODUCER_ARCHIVE_BYTES = 356701
ORIGINAL_AGENTS_ID = 'sha256:44734f6d051e6c8b1b6fcc230c2488042176b5b83111c9b39fd24439cf1d3d6f'
ORIGINAL_AGENTS_BYTES = 3563


def historical_agents(root):
    """Read only the known bounded member, after verifying the complete ZIP."""
    with (Path(root) / PRODUCER_ARCHIVE).open('rb') as stream:
        raw = stream.read(PRODUCER_ARCHIVE_BYTES + 1)
    if len(raw) != PRODUCER_ARCHIVE_BYTES or identity(raw) != PRODUCER_ARCHIVE_ID:
        raise AssertionError('historical producer source archive identity mismatch')
    with ZipFile(BytesIO(raw)) as archive:
        members = [item for item in archive.infolist() if item.filename == 'AGENTS.md']
        if len(members) != 1 or members[0].is_dir() or members[0].file_size != ORIGINAL_AGENTS_BYTES:
            raise AssertionError('historical AGENTS member inventory mismatch')
        with archive.open(members[0]) as stream:
            data = stream.read(ORIGINAL_AGENTS_BYTES + 1)
    if len(data) != ORIGINAL_AGENTS_BYTES or identity(data) != ORIGINAL_AGENTS_ID:
        raise AssertionError('historical AGENTS identity mismatch')
    return data


def assert_preserved_sources(test, root, expected):
    """Only operating guidance uses history; all runtime files stay live-pinned."""
    test.assertEqual(expected.get('AGENTS.md'), ORIGINAL_AGENTS_ID, 'historical AGENTS pin')
    original = historical_agents(root)
    for name, key in expected.items():
        data = original if name == 'AGENTS.md' else (Path(root) / name).read_bytes()
        test.assertEqual(identity(data), key, name)
