"""Integrity checks for supplied reports; no claim of rerunning their Torch tests."""
import hashlib
import json
from pathlib import Path
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1] / "fixtures/curriculum-review-c9f4399"
PACKETS = {
    "Trinite-reports-c9f4399.zip": "3d9ffd2f8cbd589e85b404251764b2d21360af0e3b7b432fca814ffcd381acc9",
    "Trinite-reports-c9f4399-torch-validated.zip": "6ae046378a4762b46ea10a4011e228fcdf60fc999535192b853686e094384a08",
}


class CurriculumReviewEvidence(unittest.TestCase):
    def test_original_packets_and_all_manifest_members_are_unchanged(self):
        prefix = "Trinite-reports-c9f4399/"
        for name, expected in PACKETS.items():
            with self.subTest(packet=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)
                self.assertEqual((ROOT / (name + ".sha256")).read_text().split(), [expected, name])
                with zipfile.ZipFile(ROOT / name) as archive:
                    names = archive.namelist()
                    self.assertEqual(len(names), len(set(names)))
                    manifest = json.loads(archive.read(prefix + "MANIFEST.json"))
                    self.assertEqual(set(names), {prefix + n for n in manifest["members"]}
                                     | {prefix + "MANIFEST.json"})
                    for member, record in manifest["members"].items():
                        raw = archive.read(prefix + member)
                        self.assertEqual(len(raw), record["bytes"])
                        self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"])

    def test_reproduced_audit_matches_every_non_environment_field(self):
        receipt = json.loads((ROOT / "reproduction.json").read_bytes())
        raw = (ROOT / "reproduced-dataset-audit.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), receipt["reproduced_dataset_audit_sha256"])
        with zipfile.ZipFile(ROOT / "Trinite-reports-c9f4399-torch-validated.zip") as archive:
            original = archive.read("Trinite-reports-c9f4399/dataset-audit.json")
        self.assertEqual(hashlib.sha256(original).hexdigest(), receipt["original_dataset_audit_sha256"])
        expected, reproduced = json.loads(original), json.loads(raw)
        self.assertEqual(receipt["comparison_excludes"], ["environment"])
        self.assertEqual({k: v for k, v in expected.items() if k != "environment"},
                         {k: v for k, v in reproduced.items() if k != "environment"})
        self.assertEqual(expected["commit"], receipt["reviewed_commit"])


if __name__ == "__main__":
    unittest.main()
