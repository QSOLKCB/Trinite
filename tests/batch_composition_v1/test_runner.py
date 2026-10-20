import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from trinite.contracts import ContractError, identity, json_bytes
from batch_composition_v1 import runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        validator = patch.object(runner, 'validate_report')
        validator.start()
        self.addCleanup(validator.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = self.root/'request.json'
        self.request.write_bytes(json_bytes({'fixed': True}))
        self.output = self.root/'report.json'
        self.args = argparse.Namespace(action='run', request=self.request,
            request_identity=identity(self.request.read_bytes()), output=self.output, expected_identity=None)

    def attempts(self):
        return list(self.root.glob('report.json.attempt-*'))

    def worker_args(self, action='run'):
        attempt = self.root/'worker-attempt'
        attempt.mkdir()
        return argparse.Namespace(request=self.request, request_identity=self.args.request_identity,
            output=self.output, expected_identity=None, attempt=attempt, worker_action=action)

    def test_timeout_kills_fixed_worker_and_retains_status_without_output(self):
        class Process:
            killed = False
            calls = 0
            def wait(self, timeout=None):
                self.calls += 1
                if timeout is not None:
                    self.timeout = timeout
                    raise subprocess.TimeoutExpired('fixed worker', timeout)
                return -9
            def kill(self): self.killed = True
        process = Process()
        with patch.object(runner.subprocess, 'Popen', return_value=process) as create:
            self.assertEqual(runner.launch(self.args), 124)
        self.assertTrue(process.killed)
        self.assertLessEqual(process.timeout, 300)
        self.assertGreater(process.timeout, 299)
        self.assertNotIn('shell', create.call_args.kwargs)
        self.assertEqual(create.call_args.args[0][2], str(runner.ROOT/'scripts/run_batch_composition.py'))
        attempt, = self.attempts()
        self.assertEqual(json.loads((attempt/'status.json').read_bytes())['outcome'], 'timeout')
        self.assertFalse(self.output.exists())
        self.assertTrue((attempt/'invocation.json').exists())
        self.assertTrue((attempt/'worker.log').exists())

    def test_interruption_retains_status_and_kills_worker(self):
        class Process:
            killed = False
            def wait(self, timeout=None):
                if timeout is not None: raise KeyboardInterrupt()
            def kill(self): self.killed = True
        process = Process()
        with patch.object(runner.subprocess, 'Popen', return_value=process), self.assertRaises(KeyboardInterrupt):
            runner.launch(self.args)
        self.assertTrue(process.killed)
        attempt, = self.attempts()
        self.assertEqual(json.loads((attempt/'status.json').read_bytes())['outcome'], 'interrupted')

    def test_existing_file_or_symlink_never_clobbered_or_launched(self):
        for symlink in (False, True):
            if symlink: self.output.symlink_to(self.root/'missing')
            else: self.output.write_bytes(b'old')
            with patch.object(runner.subprocess, 'Popen', side_effect=AssertionError('no launch')):
                with self.assertRaises(FileExistsError): runner.launch(self.args)
            if not symlink: self.assertEqual(self.output.read_bytes(), b'old')
            self.output.unlink()
        self.assertEqual(self.attempts(), [])

    def test_request_mismatch_rejected_before_worker(self):
        self.args.request_identity = 'sha256:'+'0'*64
        with patch.object(runner.subprocess, 'Popen', side_effect=AssertionError('no launch')):
            with self.assertRaises(ContractError): runner.launch(self.args)
        self.assertEqual(self.attempts(), [])

    def test_negative_report_is_published_and_partial_records_remain(self):
        args = self.worker_args()
        def report(**callbacks):
            callbacks['on_layout']({'partial_layout': True})
            callbacks['on_cell']({'partial_cell': True})
            return {'conforms': False}
        with patch.object(runner, 'prepare'), patch('batch_composition_v1.spec.request', return_value={'fixed': True}), \
             patch('batch_composition_v1.experiment.report', side_effect=report):
            self.assertEqual(runner.worker(args), 2)
        self.assertEqual(json.loads(self.output.read_bytes()), {'conforms': False})
        self.assertEqual((args.attempt/'computed-report.json').read_bytes(), self.output.read_bytes())
        self.assertEqual(len((args.attempt/'completed-layouts.jsonl').read_text().splitlines()), 1)
        self.assertEqual(len((args.attempt/'completed-cells.jsonl').read_text().splitlines()), 1)

    def test_mid_computation_failure_keeps_partial_layout_and_no_final(self):
        args = self.worker_args()
        def fail(**callbacks):
            callbacks['on_layout']({'partial_layout': True})
            raise ContractError('injected computation failure')
        with patch.object(runner, 'prepare'), patch('batch_composition_v1.spec.request', return_value={'fixed': True}), \
             patch('batch_composition_v1.experiment.report', side_effect=fail), self.assertRaises(ContractError):
            runner.worker(args)
        self.assertTrue((args.attempt/'completed-layouts.jsonl').read_bytes())
        self.assertFalse(self.output.exists())

    def test_replay_mismatch_keeps_expected_unchanged_and_false_receipt(self):
        args = self.worker_args('verify')
        old = json_bytes({'schema': 'trinite.batch-composition-report.v1', 'request': {'fixed': True}, 'conforms': True})
        self.output.write_bytes(old)
        args.expected_identity = identity(old)
        with patch.object(runner, 'prepare'), patch('batch_composition_v1.spec.request', return_value={'fixed': True}), \
             patch('batch_composition_v1.experiment.report', return_value={'conforms': False}), self.assertRaisesRegex(ContractError, 'replay mismatch'):
            runner.worker(args)
        self.assertEqual(self.output.read_bytes(), old)
        receipt = json.loads((args.attempt/'replay-receipt.json').read_bytes())
        self.assertFalse(receipt['exact_match'])
        self.assertTrue((args.attempt/'computed-report.json').exists())

    def test_report_bound_is_enforced_before_publication(self):
        args = self.worker_args()
        with patch.object(runner, 'prepare'), patch('batch_composition_v1.spec.request', return_value={'fixed': True}), \
             patch('batch_composition_v1.experiment.report', return_value={'conforms': True, 'large': 'x'*80}), \
             patch.object(runner, 'MAX_REPORT', 64), self.assertRaisesRegex(ContractError, 'report exceeds'):
            runner.worker(args)
        self.assertFalse(self.output.exists())

if __name__ == '__main__':
    unittest.main()
