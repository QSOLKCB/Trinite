"""Historical source supplement must bind Git tree, not only a resealed ZIP."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import supplement_procedural_qec_source as supplement
from supplement_procedural_qec_source import (EXPECTED_FILES, EXPECTED_GIT_TREE,
    SourceError, reconstruct_git_tree, sha_manifest, sha, read_supplement)


class HistoricalGitTreeTests(unittest.TestCase):
    def test_standalone_manifest_cannot_claim_an_arbitrary_git_tree(self):
        values = {
            "schema": "trinite.procedural-qec-full-source-supplement.v1",
            "git_commit": "93522bf8f18fe38970ceafdd2ace792f8922c382",
            "git_tree": "0" * 40, "run_id": 38355170260,
            "negative_summary_sha256":
                "a6d53d4adff35f806226333891a38dd472506752c3d22d70cc96c76e8e2e0a68",
            "original_request_sha256":
                "35dfbaafd2f85e5c6836bab1837bdc99c18a8b003ca7c46273f2f4d265ce37b1",
            "original_research_zip_sha256":
                "ac3b3627a27bcf6d43f89b31eeabaf1859949ae939784da06ba49916b495fdca",
            "expected_dataset_sha256":
                "120b95ac68825b50b92dc0210d1f142a30980bfe5acedcfafe6d8f3cdadf83c4",
            "scope": "full Git tracked source tree; not a new trained model or independent numerical replication",
            "members": {f"source/{i:04}.py": {} for i in range(EXPECTED_FILES)}}
        with self.assertRaisesRegex(SourceError, "invalid historical Git tree identifier"):
            sha_manifest(values)
        values["git_tree"] = EXPECTED_GIT_TREE
        self.assertIs(sha_manifest(values), values)

    def test_complete_count_is_not_enough_and_blob_tampering_changes_git_root(self):
        original = {f"source/pkg/{i:04}.py": ("print("+str(i)+")\n").encode()
                    for i in range(EXPECTED_FILES)}
        digest = reconstruct_git_tree(original)
        self.assertNotEqual(digest, EXPECTED_GIT_TREE)
        changed = dict(original)
        changed["source/pkg/0000.py"] = b"print('tampered')\n"
        self.assertNotEqual(reconstruct_git_tree(changed), digest)
        del changed["source/pkg/0100.py"]
        with self.assertRaisesRegex(SourceError, "missing tracked source"):
            reconstruct_git_tree(changed)


class BoundedSupplementInputTests(unittest.TestCase):
    def test_small_regular_packet_is_read_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packet.zip"
            path.write_bytes(b"bounded")
            self.assertEqual(read_supplement(path), b"bounded")

    def test_oversize_rejected_from_stat_before_payload_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packet.zip"
            path.write_bytes(b"too large")
            with mock.patch.object(supplement, "LIMIT", 4):
                with mock.patch.object(supplement.os, "fdopen",
                                       side_effect=AssertionError("payload was read")):
                    with self.assertRaisesRegex(SourceError, "bounded regular file"):
                        read_supplement(path)

    def test_symlinked_file_and_ancestor_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "actual"
            real.mkdir()
            (real / "packet.zip").write_bytes(b"content")
            link = root / "packet.zip"
            link.symlink_to(real / "packet.zip")
            with self.assertRaisesRegex(SourceError, "symlinks"):
                read_supplement(link)
            parent_link = root / "linked"
            parent_link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(SourceError, "symlinks"):
                read_supplement(parent_link / "packet.zip")

    def test_nonregular_directory_is_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SourceError, "bounded regular file"):
                read_supplement(directory)


class QECV3WorkflowTriggerTests(unittest.TestCase):
    def test_shared_training_changes_cannot_skip_native_smoke(self):
        workflow = (ROOT / ".github/workflows/procedural-qec-v3.yml").read_text()
        self.assertIn("on:\n", workflow)
        trigger = workflow.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
        self.assertIn("\n  pull_request:\n", trigger)
        self.assertNotIn("\n    paths:", trigger)
        self.assertIn("\n  workflow_dispatch:", trigger)
        self.assertIn("tests/procedural_qec_v3", workflow)


if __name__ == "__main__":
    unittest.main()
