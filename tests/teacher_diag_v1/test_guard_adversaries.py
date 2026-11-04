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

    def test_unknown_member_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractError):
                core.read_member(Path(tmp), "../outside")

if __name__ == "__main__":
    unittest.main()
