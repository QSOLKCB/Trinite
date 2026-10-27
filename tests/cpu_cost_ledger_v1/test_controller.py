"""Process doubles test controller failures, not custody or numerical results."""
import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from cpu_cost_ledger_v1.controller import launch, main, protect, run
from cpu_cost_ledger_v1.request import LIMITS, ROOT


class ProcessDouble:
    def __init__(self, output=b'', first_wait_error=None):
        self.stdout = io.BytesIO(output)
        self.first_wait_error = first_wait_error
        self.killed = 0
        self.waits = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stdout.close()

    def wait(self, timeout=None):
        self.waits += 1
        if self.waits == 1 and self.first_wait_error is not None:
            raise self.first_wait_error
        return -9 if self.killed else 0

    def kill(self):
        self.killed += 1


class ControllerFailures(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='cpu-ledger-controller-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.request = self.root/'request.json'
        self.checkpoint = self.root/'synthetic.bin'
        self.raw = json_bytes({'synthetic_controller_double': True})
        self.request.write_bytes(self.raw)
        self.checkpoint.write_bytes(b'synthetic; never loaded')
        self.expected = identity(self.raw)
        self.validated_double = {'plan': {'worker_seconds': 1,
                                        'checkpoint_identity': identity(self.checkpoint.read_bytes())}}

    def run_double(self, name, **launch_kwargs):
        output = self.root/name
        with (patch('cpu_cost_ledger_v1.controller.validate', return_value=self.validated_double),
              patch('cpu_cost_ledger_v1.controller.launch', **launch_kwargs)):
            result = run(output, self.request, self.expected, self.checkpoint)
        self.assertEqual(parse_json((output/'attempt.json').read_bytes(), canonical=True), result)
        self.assertEqual((output/'request.json').read_bytes(), self.raw)
        self.assertEqual(self.request.read_bytes(), self.raw)
        self.assertIsNone(result['completion_identity'])
        self.assertGreaterEqual(result['controller_wall_ns'], 0)
        return result

    def test_timeout_retains_non_success_receipt(self):
        result = self.run_double('timeout', side_effect=subprocess.TimeoutExpired('fixed-worker', 1))
        self.assertEqual(result['outcome'], 'timed_out')
        self.assertIsNone(result['worker_exit'])

    def test_cancellation_retains_non_success_receipt(self):
        result = self.run_double('cancelled', side_effect=KeyboardInterrupt())
        self.assertEqual(result['outcome'], 'interrupted')

    def test_worker_failure_or_log_error_never_becomes_success(self):
        result = self.run_double('exit', return_value=7)
        self.assertEqual(result['outcome'], 'failed')
        self.assertEqual(result['worker_exit'], 7)
        result = self.run_double('log-error', side_effect=ContractError('worker log exceeds frozen bound'))
        self.assertEqual(result['outcome'], 'failed')
        self.assertIn('log exceeds', result['detail'])

    def test_success_exit_without_completion_is_failure(self):
        result = self.run_double('missing-completion', return_value=0)
        self.assertEqual(result['outcome'], 'failed')
        self.assertIn('FileNotFoundError', result['detail'])

    def test_existing_output_never_clobbered_or_worker_started(self):
        output = self.root/'existing'
        output.mkdir()
        (output/'historical').write_bytes(b'keep')
        with (patch('cpu_cost_ledger_v1.controller.validate', return_value=self.validated_double),
              patch('cpu_cost_ledger_v1.controller.launch') as worker):
            with self.assertRaises(FileExistsError):
                run(output, self.request, self.expected, self.checkpoint)
            worker.assert_not_called()
        self.assertEqual(list(output.iterdir()), [output/'historical'])
        self.assertEqual((output/'historical').read_bytes(), b'keep')

    def test_wrong_checkpoint_rejected_before_output_or_worker(self):
        output = self.root/'wrong-checkpoint'
        request = {'plan': {**self.validated_double['plan'], 'checkpoint_identity': identity(b'wrong')}}
        with (patch('cpu_cost_ledger_v1.controller.validate', return_value=request),
              patch('cpu_cost_ledger_v1.controller.launch') as worker):
            with self.assertRaisesRegex(ContractError, 'checkpoint identity'):
                run(output, self.request, self.expected, self.checkpoint)
            worker.assert_not_called()
        self.assertFalse(output.exists())

    def test_execution_requires_authorization_before_backend_configuration(self):
        for action in ('run', '_worker'):
            with self.subTest(action=action), contextlib.redirect_stderr(io.StringIO()) as error:
                with patch('cpu_cost_ledger_v1.controller.configure') as configure:
                    self.assertEqual(main([action]), 1)
                    configure.assert_not_called()
                self.assertIn('separate owner authorization', error.getvalue())

    def test_source_inputs_and_link_ancestors_are_protected(self):
        for output in (ROOT/'new-output', ROOT.parent, self.request, self.root):
            with self.subTest(output=output):
                with self.assertRaises(ContractError):
                    protect(output, self.request, self.checkpoint)
        target = self.root/'real'
        target.mkdir()
        (self.root/'alias').symlink_to(target, target_is_directory=True)
        with self.assertRaises(ContractError):
            protect(self.root/'alias'/'output', self.request, self.checkpoint)


class LauncherFailures(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='cpu-ledger-launcher-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        status = self.root.stat()
        self.directory_identity = (status.st_dev, status.st_ino)

    def invoke(self, process):
        with patch('cpu_cost_ledger_v1.controller.subprocess.Popen', return_value=process) as popen:
            result = launch(self.root, self.root.parent/'request.json', identity(b'double'),
                            self.root.parent/'checkpoint', 1, self.directory_identity)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[2], str(ROOT/'scripts/run_cpu_cost_ledger.py'))
        self.assertFalse(popen.call_args.kwargs['shell'])
        self.assertEqual(popen.call_args.kwargs['cwd'], ROOT)
        return result

    def test_bounded_log_overflow_kills_worker_and_fails(self):
        process = ProcessDouble(b'x'*(LIMITS['worker_log_bytes']+1))
        with self.assertRaisesRegex(ContractError, 'log exceeds'):
            self.invoke(process)
        self.assertGreaterEqual(process.killed, 1)
        self.assertEqual((self.root/'worker.log').stat().st_size, LIMITS['worker_log_bytes'])

    def test_timeout_kills_and_waits_before_raising(self):
        process = ProcessDouble(b'partial output\n', subprocess.TimeoutExpired('double', 1))
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke(process)
        self.assertEqual(process.killed, 1)
        self.assertEqual(process.waits, 2)
        self.assertEqual((self.root/'worker.log').read_bytes(), b'partial output\n')

    def test_cancel_kills_and_waits_before_raising(self):
        process = ProcessDouble(b'partial\n', KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(process)
        self.assertEqual(process.killed, 1)
        self.assertEqual(process.waits, 2)

    def test_fixed_launcher_success_and_log_no_clobber(self):
        self.assertEqual(self.invoke(ProcessDouble(b'only software-double output\n')), 0)
        with patch('cpu_cost_ledger_v1.controller.subprocess.Popen') as popen:
            with self.assertRaises(FileExistsError):
                launch(self.root, self.root.parent/'request', identity(b'double'),
                       self.root.parent/'checkpoint', 1, self.directory_identity)
            popen.assert_not_called()
        self.assertEqual((self.root/'worker.log').read_bytes(), b'only software-double output\n')


if __name__ == '__main__':
    unittest.main()
