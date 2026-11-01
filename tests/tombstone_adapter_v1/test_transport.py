"""Failure doubles test transport, never certify native splice authority."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("adapter", ROOT / "scripts/tombstone_adapter_v1.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class TransportTests(unittest.TestCase):
    def fake(self, body, timeout=0.2):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "fake"
        path.write_text("#!/usr/bin/python3\n" + body)
        path.chmod(0o700)
        return path, timeout

    def run_fake(self, body, timeout=0.2):
        binary, timeout = self.fake(body, timeout)
        with patch.object(adapter, "BINARY", binary), patch.object(adapter, "verify_build"), patch.object(adapter, "TIMEOUT", timeout):
            return adapter.execute("observe", b"{}")

    def test_success_and_schema_failure_are_distinct(self):
        self.assertEqual(self.run_fake("print('{}')"), (0, b"{}\n", b""))
        code, out, _ = self.run_fake("import sys\nprint('invalid request', file=sys.stderr)\nsys.exit(2)")
        self.assertEqual((code, out), (2, b""))

    def test_failures_are_not_negative_labels(self):
        bodies = ["import sys\nsys.exit(3)", "import sys\nsys.exit(1)",
                  "import time\ntime.sleep(10)", "print('x'*5000)",
                  "import sys\nprint('{}')\nsys.exit(2)",
                  "import sys\nprint('bad', file=sys.stderr)"]
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(adapter.InfrastructureError):
                self.run_fake(body)

    def test_timeout_reaps_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            pid = Path(directory) / "pid"
            body = f"import os,time\nopen({str(pid)!r},'w').write(str(os.getpid()))\ntime.sleep(10)"
            with self.assertRaises(adapter.InfrastructureError):
                self.run_fake(body)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid.read_text()), 0)

    def test_runtime_strips_preload_and_environment(self):
        with patch.dict(os.environ, {"LD_PRELOAD": "/not/loaded", "SECRET_TEST": "secret"}):
            result = self.run_fake("import os\nassert 'LD_PRELOAD' not in os.environ\nassert 'SECRET_TEST' not in os.environ\nprint('{}')")
        self.assertEqual(result[0], 0)

    def test_operation_and_input_limits(self):
        with self.assertRaises(ValueError):
            adapter.execute("/bin/sh", b"")
        with self.assertRaises(ValueError):
            adapter.execute("observe", b"x" * 1025)

    def test_source_binary_record_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crate = root / "external/tombstone-adapter-v1"
            shutil.copytree(adapter.CRATE, crate, ignore=shutil.ignore_patterns("target", "tests"))
            (root / "scripts").mkdir()
            shutil.copyfile(ROOT / "scripts/tombstone_adapter_v1.py", root / "scripts/tombstone_adapter_v1.py")
            binary = crate / "target/release/trinite-tombstone-adapter-v1"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"synthetic non-executable build fixture")
            record = crate / "target/build-record.json"
            with patch.multiple(adapter, ROOT=root, CRATE=crate, BINARY=binary, RECORD=record):
                baseline = {"schema": "trinite.tombstone-build.v1", "sources": adapter.source_hashes(), "binary_sha256": adapter.digest(binary.read_bytes())}
                record.write_text(json.dumps(baseline))
                adapter.verify_build()
                binary.write_bytes(b"changed")
                with self.assertRaises(adapter.InfrastructureError):
                    adapter.verify_build()
                record.write_text("bad json")
                with self.assertRaises(adapter.InfrastructureError):
                    adapter.verify_build()
                manifest = crate / "upstream-pin.json"
                original = manifest.read_bytes()
                for invalid in ([], None, {}, {"files": []}):
                    manifest.write_text(json.dumps(invalid))
                    with self.assertRaises(adapter.InfrastructureError):
                        adapter.verify_upstream()
                manifest.write_bytes(original)
                upstream = crate / "upstream/src/core/deletion.c"
                upstream.write_bytes(upstream.read_bytes() + b"\n")
                with self.assertRaises(adapter.InfrastructureError):
                    adapter.verify_upstream()

    def test_non_utf8_native_argument_is_sanitized(self):
        result = subprocess.run([bytes(adapter.BINARY), b"\xff"], input=b"", capture_output=True, timeout=5)
        self.assertEqual((result.returncode, result.stdout, result.stderr), (2, b"", b"invalid request\n"))

    def test_stalled_stdin_deadline(self):
        process = subprocess.Popen(["python3", str(ROOT / "scripts/tombstone_adapter_v1.py"), "observe"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            process.wait(timeout=7)
            self.assertEqual(process.returncode, 3)
            self.assertEqual(process.stdout.read(), b"")
            self.assertEqual(process.stderr.read(), b"infrastructure failure\n")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()


if __name__ == "__main__":
    unittest.main()
