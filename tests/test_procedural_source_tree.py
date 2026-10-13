"""Historical source supplement must bind Git tree, not only a resealed ZIP."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from supplement_procedural_qec_source import (EXPECTED_FILES, EXPECTED_GIT_TREE,
    SourceError, reconstruct_git_tree, sha_manifest, sha)


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
        with self.assertRaisesRegex(SourceError, "exact historical result"):
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


if __name__ == "__main__":
    unittest.main()
