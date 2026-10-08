from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("git"), "Git is required for checkout conformance")
class CheckoutTests(unittest.TestCase):
    def test_upgrade_refreshes_legacy_byte_bound_sources(self):
        # Reconstruct the exact two unchanged source blobs from ccda937. Other
        # legacy files only need to differ: Git's two-tree checkout must rewrite
        # every byte-bound source and fixture, not merely introduce attributes.
        marker = b"# Byte-bound source receipt requires LF; this revision refreshes legacy CRLF checkouts.\n"
        legacy_hashes = {
            "contracts.py": "0285a21a2ba3b9bf34c49a9ab8364025fe1a12d77049c8c131f7a4cda19e379d",
            "tokenizer.py": "28bf8f63c1708db26e3b11f7de6c27dcdcdf86f18993095f8a36640ee45d5d6a",
        }
        with tempfile.TemporaryDirectory() as temp:
            checkout, target = Path(temp)/"checkout", Path(temp)/"target"
            checkout.mkdir()
            target.mkdir()
            def git(*args, work_tree=None):
                command = ["git", "-C", str(checkout)]
                if work_tree is not None:
                    command.append(f"--work-tree={work_tree}")
                result = subprocess.run(command + list(args), capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                return result.stdout.decode().strip()
            git("init", "--quiet")
            git("config", "core.autocrlf", "true")
            for directory in ("src/trinite", "configs", "fixtures"):
                for original in (ROOT/directory).rglob("*"):
                    if not original.is_file() or original.suffix not in (".py", ".json"):
                        continue
                    relative = original.relative_to(ROOT)
                    content = original.read_bytes()
                    new = target/relative
                    new.parent.mkdir(parents=True, exist_ok=True)
                    new.write_bytes(content)
                    if original.name in legacy_hashes:
                        self.assertEqual(content.count(marker), 1)
                        content = content.replace(marker, b"", 1)
                        self.assertEqual(hashlib.sha256(content).hexdigest(),
                                         legacy_hashes[original.name])
                    elif relative == Path("src/trinite/data.py"):
                        content += b"# Legacy generator revision.\n"
                    elif relative.parts[0] == "fixtures":
                        content = b"{}\n"
                    old = checkout/relative
                    old.parent.mkdir(parents=True, exist_ok=True)
                    old.write_bytes(content)
            (checkout/"control.txt").write_bytes(b"conversion control\n")
            (target/"control.txt").write_bytes(b"conversion control\n")
            shutil.copyfile(ROOT/".gitattributes", target/".gitattributes")
            git("add", ".")
            old_tree = git("write-tree")
            # Materialize a genuine pre-attributes CRLF working tree.
            for original in checkout.rglob("*"):
                if original.is_file() and ".git" not in original.parts:
                    original.unlink()
            git("checkout-index", "--all", "--force")
            for name in ("contracts.py", "data.py", "tokenizer.py"):
                self.assertIn(b"\r\n", (checkout/"src/trinite"/name).read_bytes())
            env = dict(os.environ, PYTHONPATH=str(checkout/"src"))
            def audit():
                return subprocess.run([sys.executable, "-m", "trinite", "audit-fixture",
                                       "fixtures/formal-v1"], cwd=checkout, env=env,
                                      capture_output=True, text=True, timeout=15)
            # Independently exercise the stale-source diagnostic with valid
            # target artifacts, before Git refreshes the working tree.
            for name in ("dataset.json", "manifest.json"):
                shutil.copyfile(target/"fixtures/formal-v1"/name,
                                checkout/"fixtures/formal-v1"/name)
            result = audit()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires LF", result.stderr)
            self.assertIn("docs/GETTING_STARTED.md", result.stderr)
            # Restore the old fixture bytes to permit Git's normal transition.
            git("checkout-index", "--all", "--force")
            git("add", ".", work_tree=target)
            new_tree = git("write-tree")
            git("read-tree", old_tree)
            git("update-index", "--refresh")
            git("read-tree", "-m", "-u", old_tree, new_tree)
            self.assertEqual((checkout/"control.txt").read_bytes(), b"conversion control\r\n")
            for name in ("contracts.py", "data.py", "tokenizer.py"):
                self.assertEqual((checkout/"src/trinite"/name).read_bytes(),
                                 (target/"src/trinite"/name).read_bytes())
            result = audit()
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_autocrlf_checkout_preserves_source_and_fixture_bytes(self):
        # Exercise real Git checkout conversion, without making any commit.
        with tempfile.TemporaryDirectory() as temp:
            source, checkout = Path(temp)/"source", Path(temp)/"checkout"
            source.mkdir()
            checkout.mkdir()
            shutil.copyfile(ROOT/".gitattributes", source/".gitattributes")
            for directory in ("src/trinite", "tests", "configs", "fixtures"):
                for original in (ROOT/directory).rglob("*"):
                    if (original.is_file() and "__pycache__" not in original.parts
                            and original.suffix in (".py", ".json")):
                        dest = source/original.relative_to(ROOT)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(original.read_bytes())
            (source/"control.txt").write_bytes(b"conversion control\n")
            for args in (("init", "--quiet"), ("config", "core.autocrlf", "true"),
                         ("add", "."), ("checkout-index", "--all", f"--prefix={checkout}/")):
                subprocess.run(["git", "-C", str(source), *args], check=True,
                               capture_output=True, timeout=15)
            self.assertEqual((checkout/"control.txt").read_bytes(), b"conversion control\r\n")
            for original in source.rglob("*"):
                if original.is_file() and original.suffix in (".py", ".json") and ".git" not in original.parts:
                    target = checkout/original.relative_to(source)
                    self.assertEqual(target.read_bytes(), original.read_bytes(), str(target))
                    self.assertNotIn(b"\r\n", target.read_bytes(), str(target))
            env = dict(os.environ, PYTHONPATH=str(checkout/"src"))
            result = subprocess.run([sys.executable, "-m", "trinite", "audit-fixture",
                                     str(checkout/"fixtures/formal-v1")], cwd=checkout,
                                    env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, "-m", "trinite", "generate-fixture",
                                     str(Path(temp)/"replayed")], cwd=checkout, env=env,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("dataset.json", "manifest.json"):
                self.assertEqual((checkout/"fixtures/formal-v1"/name).read_bytes(),
                                 (Path(temp)/"replayed"/name).read_bytes())


if __name__ == "__main__":
    unittest.main()
