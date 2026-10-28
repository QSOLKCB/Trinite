"""Local evidence safety contracts; no native imports or target checkpoint data."""
import tempfile
import unittest
from pathlib import Path

from frozen_music_math_v1 import artifacts as a


class ArtifactContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_canonical_json_and_identity(self):
        raw = a.encoded({'b': 2, 'a': 1})
        self.assertEqual(raw, b'{"a":1,"b":2}\n')
        self.assertEqual(a.parse(raw), {'a': 1, 'b': 2})
        self.assertEqual(a.identity(a.digest(raw)), a.digest(raw))
        for value in ('0' * 64, 'sha256:' + 'A' * 64, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                a.identity(value)
        for raw in (b'{"a":1,"a":2}\n', b'{"a": 1}\n', b'{"n":NaN}\n'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                a.parse(raw)

    def test_atomic_no_clobber(self):
        target = self.root / 'report.json'
        a.publish(target, b'first')
        with self.assertRaises(FileExistsError):
            a.publish(target, b'second')
        self.assertEqual(a.read(target), b'first')
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['report.json'])

    def test_symlink_file_and_ancestor_rejected(self):
        original = self.root / 'original'
        original.write_bytes(b'bound')
        link = self.root / 'link'
        link.symlink_to(original)
        with self.assertRaises(ValueError):
            a.read(link)
        nested = self.root / 'nested'
        nested.mkdir()
        alias = self.root / 'alias'
        alias.symlink_to(nested, target_is_directory=True)
        with self.assertRaises(ValueError):
            a.publish(alias / 'new', b'not allowed')
        self.assertEqual(list(nested.iterdir()), [])

    def test_file_inventory_and_byte_bounds(self):
        (self.root / 'first').write_bytes(b'abc')
        (self.root / 'second').write_bytes(b'def')
        self.assertEqual(a.inventory(self.root), {
            'first': a.digest(b'abc'), 'second': a.digest(b'def')})
        with self.assertRaises(ValueError):
            a.read(self.root / 'first', limit=2)
        with self.assertRaises(ValueError):
            a.inventory(self.root, max_files=1)
        with self.assertRaises(ValueError):
            a.inventory(self.root, max_bytes=5)
        (self.root / 'link').symlink_to(self.root / 'first')
        with self.assertRaises(ValueError):
            a.inventory(self.root)

    def test_disjoint_rejects_child_parent_and_equal(self):
        protected = self.root / 'input'
        protected.mkdir()
        for path in (protected, protected / 'new', self.root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                a.disjoint(path, protected)
        target = self.root / 'new'
        self.assertEqual(a.disjoint(target, protected), target)


if __name__ == '__main__':
    unittest.main()
