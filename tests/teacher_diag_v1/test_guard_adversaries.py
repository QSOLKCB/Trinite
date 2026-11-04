# SPDX-License-Identifier: MPL-2.0
"""Fail-closed retarget and symlink guards for the frozen diagnostic."""
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from trinite.contracts import ContractError, json_bytes
from teacher_diag_v1 import core

class OriginalCheckpointGuards(unittest.TestCase):
    def test_canonical_retargeted_checkpoint_is_rejected(self):
        read = core.lab.source_bytes
        changed = core.plan()
        changed["teacher_tensor_identity"] = "sha256:" + "0"*64
        raw = json_bytes(changed)
        with mock.patch.object(core.lab, "source_bytes", side_effect=lambda p: raw if p == core.PLAN_PATH else read(p)):
            with self.assertRaises(ContractError):
                core.plan()

    def test_directory_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'packet'
            folder.mkdir()
            (folder / "request.json").write_bytes(b"{}\n")
            core.check_members(folder, ("request.json",))
            (folder / "extra").write_bytes(b"\n")
            with self.assertRaises(ContractError):
                core.check_members(folder, ("request.json",))
            (folder / "extra").unlink()
            path = folder / "request.json"
            outside = Path(tmp) / 'outside'
            path.rename(outside)
            path.symlink_to(outside)
            with self.assertRaises(ContractError):
                core.read_member(folder, "request.json")

    def test_new_output_cannot_overlap_original_evidence_or_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, run, request = root / "data", root / "run", root / "request"
            for folder in (dataset, run, request):
                folder.mkdir()
            before = {folder.name: list(folder.iterdir())
                      for folder in (dataset, run, request)}
            for candidate in (dataset / "new-result", run / "new-result",
                              request / "new-result",
                              core.lab.ROOT / "diag-should-not-exist"):
                with self.subTest(candidate=str(candidate)):
                    with self.assertRaises(ContractError):
                        core.require_safe_output(candidate, request, dataset, run)
            self.assertEqual(before, {folder.name: list(folder.iterdir())
                                      for folder in (dataset, run, request)})

    def test_unknown_member_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractError):
                core.read_member(Path(tmp), "../outside")

if __name__ == "__main__":
    unittest.main()
