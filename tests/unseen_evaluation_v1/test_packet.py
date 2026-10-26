"""Bounded, quiescent local packet guards; no hostile-race guarantee."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError
from unseen_evaluation_v1 import packet


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'packet'; self.root.mkdir()
        (self.root / 'item').write_bytes(b'abc')

    def test_readonly_inventory_and_unchanged(self):
        before = (self.root / 'item').read_bytes()
        snap = packet.inventory(self.root)
        packet.unchanged(self.root, snap)
        self.assertEqual(before, (self.root / 'item').read_bytes())
        self.assertEqual(snap['bytes'], 3)

    def test_member_change_and_root_replacement(self):
        snap = packet.inventory(self.root)
        (self.root / 'item').write_bytes(b'def')
        with self.assertRaises(ContractError): packet.unchanged(self.root, snap)
        snap = packet.inventory(self.root)
        moved = self.root.with_name('old'); self.root.rename(moved)
        self.root.mkdir(); (self.root / 'item').write_bytes(b'def')
        with self.assertRaises(ContractError): packet.unchanged(self.root, snap)

    def test_symlink_member_root_and_ancestor(self):
        linked = Path(self.temp.name) / 'linked'; linked.symlink_to(self.root)
        for root in (linked, linked / 'sub'):
            with self.assertRaises((ContractError, FileNotFoundError)): packet.inventory(root)
        (self.root / 'link').symlink_to(self.root / 'item')
        with self.assertRaises(ContractError): packet.inventory(self.root)

    def test_member_and_byte_bounds(self):
        for key, value in [('MAX_MEMBERS', 1), ('MAX_FILE_BYTES', 2), ('MAX_BYTES', 2)]:
            with self.subTest(key=key), patch.object(packet, key, value):
                with self.assertRaises(ContractError): packet.inventory(self.root)

    def test_special_member_rejected(self):
        import os
        os.mkfifo(self.root / 'pipe')
        with self.assertRaises(ContractError): packet.inventory(self.root)
