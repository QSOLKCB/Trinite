"""Adversarial descriptor-read snapshot checks; no claim of atomic snapshots."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from symbolic_music_v1 import corpus


@unittest.skipUnless(os.name == "posix", "POSIX descriptor-specific tests")
class BundleSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.bundle = self.base / "bundle"
        self.bundle.mkdir()
        self.files = {"first.json": b"first\n", "later.json": b"later\n", "manifest.json": b"manifest\n"}
        for name, data in self.files.items():
            (self.bundle / name).write_bytes(data)
        self.commitment = corpus.digest(self.files["manifest.json"])
        self.artifacts = patch.object(corpus, "artifacts", return_value=self.files)
        self.artifacts.start()
        self.addCleanup(self.artifacts.stop)

    def verify(self):
        return corpus.verify_bundle(Path(__file__).resolve().parents[2], self.bundle, self.commitment)

    def test_observed_snapshot_is_read_only(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                  for p in self.bundle.iterdir()}
        receipt = self.verify()
        self.assertTrue(receipt["verified"])
        self.assertIn("observed snapshot bytes", receipt["scope"])
        self.assertIn("not an atomic snapshot", receipt["scope"])
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                 for p in self.bundle.iterdir()}
        self.assertEqual(before, after)

    def test_symlink_replacement_after_descriptor_stat_is_rejected(self):
        outside = self.base / "outside"
        outside.write_bytes(b"foreign private bytes must not become bundle data")
        original_fstat = os.fstat
        calls = 0

        def replace_after_stat(fd):
            nonlocal calls
            result = original_fstat(fd)
            calls += 1
            if calls == 2:  # directory stat, then first member's pre-read stat
                target = self.bundle / "first.json"
                target.unlink()
                target.symlink_to(outside)
            return result

        with patch.object(os, "fstat", side_effect=replace_after_stat):
            with self.assertRaises(ValueError):
                self.verify()
        self.assertEqual(outside.read_bytes(), b"foreign private bytes must not become bundle data")

    def test_earlier_file_mutation_during_later_read_is_rejected(self):
        original_read = os.read
        later_inode = (self.bundle / "later.json").stat().st_ino
        changed = False

        def mutate_earlier(fd, size):
            nonlocal changed
            if not changed and os.fstat(fd).st_ino == later_inode:
                changed = True
                # Same byte length: size-only checks cannot detect this change.
                (self.bundle / "first.json").write_bytes(b"other\n")
            return original_read(fd, size)

        with patch.object(os, "read", side_effect=mutate_earlier):
            with self.assertRaisesRegex(ValueError, "changed across snapshot reads"):
                self.verify()
        self.assertTrue(changed)

    def test_fifo_rejected_without_waiting_for_writer(self):
        target = self.bundle / "first.json"
        target.unlink()
        os.mkfifo(target)
        started = time.monotonic()
        with self.assertRaisesRegex(ValueError, "type/size"):
            self.verify()
        self.assertLess(time.monotonic() - started, 2)

    def test_root_replacement_during_read_is_rejected(self):
        original_read = os.read
        replaced = False

        def move_directory(fd, size):
            nonlocal replaced
            if not replaced:
                replaced = True
                self.bundle.rename(self.base / "original-bundle")
                self.bundle.mkdir()
                for name, data in self.files.items():
                    (self.bundle / name).write_bytes(data)
            return original_read(fd, size)

        with patch.object(os, "read", side_effect=move_directory):
            with self.assertRaisesRegex(ValueError, "directory changed"):
                self.verify()

    def test_growing_file_read_is_bounded_and_rejected(self):
        original_read = os.read
        first_inode = (self.bundle / "first.json").stat().st_ino
        requests = []
        changed = False

        def grow_after_stat(fd, size):
            nonlocal changed
            requests.append(size)
            if not changed and os.fstat(fd).st_ino == first_inode:
                changed = True
                with (self.bundle / "first.json").open("ab") as stream:
                    stream.write(b"x" * 100000)
            return original_read(fd, size)

        with patch.object(os, "read", side_effect=grow_after_stat):
            with self.assertRaisesRegex(ValueError, "changed during read"):
                self.verify()
        self.assertEqual(requests, [len(self.files["first.json"]) + 1])

    def test_descriptors_close_on_success_and_failure(self):
        original_close = os.close
        for fail in (False, True):
            with self.subTest(failure=fail):
                closed = []

                def track_close(fd):
                    original_close(fd)
                    closed.append(fd)

                if fail:
                    (self.bundle / "first.json").write_bytes(b"wrong\n")
                with patch.object(os, "close", side_effect=track_close):
                    if fail:
                        with self.assertRaises(ValueError):
                            self.verify()
                    else:
                        self.verify()
                self.assertEqual(len(closed), 2 if fail else len(self.files) + 1)
                for fd in closed:
                    with self.assertRaises(OSError):
                        os.fstat(fd)

    def test_unsupported_primitives_fail_explicitly(self):
        with patch.object(os, "supports_dir_fd", set()):
            with self.assertRaisesRegex(ValueError, "requires POSIX"):
                self.verify()


if __name__ == "__main__":
    unittest.main()
