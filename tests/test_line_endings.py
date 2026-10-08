from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("git"), "Git is required for checkout conformance")
class CheckoutTests(unittest.TestCase):
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
