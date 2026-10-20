"""Independent controller-failure tests, not evidence of numerical conformance."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trinite.contracts import identity, json_bytes, parse_json
from batch_composition_v1 import runner


class RunnerReviewTests(unittest.TestCase):
    def args(self, directory):
        request = directory / "request.json"
        raw = json_bytes({"synthetic_controller_test": True})
        request.write_bytes(raw)
        return SimpleNamespace(action="run", request=request, request_identity=identity(raw),
                               output=directory / "report.json", expected_identity=None)

    def test_start_failure_retains_machine_readable_attempt_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args = self.args(directory)
            with patch.object(runner.subprocess, "Popen", side_effect=OSError("synthetic launch failure")), \
                    redirect_stdout(io.StringIO()):
                try:
                    code = runner.launch(args)
                    self.assertNotEqual(code, 0)
                except OSError:
                    pass
            attempts = list(directory.glob("report.json.attempt-*"))
            self.assertEqual(len(attempts), 1)
            self.assertTrue((attempts[0] / "invocation.json").is_file())
            self.assertTrue((attempts[0] / "status.json").is_file(),
                            "Process-start failures must retain status, not just an empty log.")
            status = parse_json((attempts[0] / "status.json").read_bytes(), canonical=True)
            self.assertNotEqual(status["exit_code"], 0)
            self.assertNotEqual(status["outcome"], "completed")
            self.assertFalse(args.output.exists())

    def test_timeout_kills_worker_and_preserves_attempt(self):
        class Process:
            def __init__(self):
                self.killed, self.waits = False, []

            def wait(self, timeout=None):
                self.waits.append(timeout)
                if timeout is not None:
                    raise subprocess.TimeoutExpired("synthetic fixed worker", timeout)
                return -9

            def kill(self):
                self.killed = True

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args, process = self.args(directory), Process()
            with patch.object(runner.subprocess, "Popen", return_value=process), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(runner.launch(args), 124)
            self.assertTrue(process.killed)
            self.assertEqual(len(process.waits), 2)
            self.assertLessEqual(process.waits[0], 300)
            attempt = next(directory.glob("report.json.attempt-*"))
            status = parse_json((attempt / "status.json").read_bytes(), canonical=True)
            self.assertEqual(status["outcome"], "timeout")
            self.assertEqual(status["exit_code"], 124)
            self.assertTrue((attempt / "worker.log").is_file())
            self.assertFalse(args.output.exists())

    def test_wait_error_does_not_orphan_a_started_worker(self):
        class Process:
            def __init__(self):
                self.killed, self.wait_count = False, 0

            def wait(self, timeout=None):
                self.wait_count += 1
                if timeout is not None:
                    raise OSError("synthetic wait failure")
                return -9

            def kill(self):
                self.killed = True

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args, process = self.args(directory), Process()
            with patch.object(runner.subprocess, "Popen", return_value=process), \
                    redirect_stdout(io.StringIO()):
                self.assertNotEqual(runner.launch(args), 0)
            self.assertTrue(process.killed, "A started worker must be terminated after a controller error.")
            self.assertEqual(process.wait_count, 2)
            attempt = next(directory.glob("report.json.attempt-*"))
            status = parse_json((attempt / "status.json").read_bytes(), canonical=True)
            self.assertNotEqual(status["exit_code"], 0)
            self.assertFalse(args.output.exists())

    def test_existing_files_and_symlinks_are_never_replaced_or_launched(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args = self.args(directory)
            args.output.write_bytes(b"historical report")
            with patch.object(runner.subprocess, "Popen") as popen:
                with self.assertRaises(FileExistsError):
                    runner.launch(args)
                popen.assert_not_called()
            self.assertEqual(args.output.read_bytes(), b"historical report")
            args.output = directory / "dangling.json"
            args.output.symlink_to(directory / "absent-target")
            with patch.object(runner.subprocess, "Popen") as popen:
                with self.assertRaises(FileExistsError):
                    runner.launch(args)
                popen.assert_not_called()
            self.assertTrue(args.output.is_symlink())
            self.assertEqual(list(directory.glob("*.attempt-*")), [])


if __name__ == "__main__":
    unittest.main()
