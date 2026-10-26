"""Real custody with a tiny test-only plan; no target experiment outcomes."""
from contextlib import ExitStack
from dataclasses import replace
import copy
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from trinite.contracts import ContractError, identity, json_bytes, parse_json, ModelConfig
from trinite.budget_training import plan_for
from trinite.observation import BundleObserver
from native_diagnostic_v1 import request, workflow, runner
from native_diagnostic_v1.artifacts import publish, Store


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)/'base'; cls.base.mkdir()
        cls.stack = ExitStack()
        cls.plan = replace(plan_for(0, 'extended'), steps=2, validation_every=1,
            model=ModelConfig(context_length=64, blocks=1, width=16, heads=2, feed_forward_width=32))
        cls.stack.enter_context(patch('native_diagnostic_v1.request.plan_for', return_value=cls.plan))
        cls.stack.enter_context(patch.dict(request.PROTOCOL, {'milestones':[0,1,2]}))
        cls.stack.enter_context(patch('native_diagnostic_v1.workflow.MILESTONES', (0,1,2)))
        cls.raw = json_bytes(request.core(0)); cls.rid = identity(cls.raw)
        cls.cid = workflow.run(cls.base, cls.raw, cls.rid)

    @classmethod
    def tearDownClass(cls):
        cls.stack.close(); cls.temp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'copy'; shutil.copytree(self.base, self.root)

    def contents(self):
        return {p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def reseal_test_packet(self):
        """Test-only forged fresh packet; never touches historical/public evidence."""
        completion = parse_json((self.root/'completion.json').read_bytes(), canonical=True)
        outputs = {name:(self.root/name).read_bytes() for name in completion['artifacts']}
        shutil.rmtree(self.root/'provenance')
        observer = BundleObserver(self.root/'provenance')
        raw, manifest = request.corpus()
        observer.record('native-learning-diagnostic', inputs={'request.json':self.raw,
            'dataset.json':raw,'dataset-manifest.json':manifest}, outputs=outputs)
        verification = json_bytes(observer.finalize()); (self.root/'verification.json').write_bytes(verification)
        completion['artifacts'] = {n:identity(b) for n,b in outputs.items()}
        completion['verification_identity'] = identity(verification)
        completion['manifest_identity'] = identity((self.root/'provenance/manifest.json').read_bytes())
        encoded = json_bytes(completion); (self.root/'completion.json').write_bytes(encoded)
        return identity(encoded)

    def mutate(self, name, change):
        value = parse_json((self.root/name).read_bytes(), canonical=True)
        change(value); (self.root/name).write_bytes(json_bytes(value))
        return self.reseal_test_packet()

    def test_actual_custody_safe_checkpoints_and_readonly_audit(self):
        before = self.contents()
        result = workflow.audit(self.root, self.raw, self.rid, self.cid)
        self.assertTrue(result['verified'])
        self.assertFalse(result['optimizer_replayed'])
        self.assertFalse(result['diagnostic_predictions_replayed'])
        self.assertEqual(before,self.contents())

    def test_source_and_external_identity_rejection(self):
        for raw,rid,cid in ((self.raw,self.rid,'sha256:'+'0'*64),
                            (self.raw,self.rid.replace('a','b'),self.cid)):
            with self.assertRaises(ContractError): workflow.audit(self.root,raw,rid,cid)
        with patch('native_diagnostic_v1.request.sources', return_value={}):
            with self.assertRaises(ContractError): workflow.audit(self.root,self.raw,self.rid,self.cid)

    def test_unbound_root_copy_rejected(self):
        self.mutate('report.json',lambda v:v.update(seed=True))
        with self.assertRaises(ContractError): workflow.audit(self.root,self.raw,self.rid,self.cid)

    def test_report_boolean_alias_rejected_even_with_fresh_custody(self):
        cid = self.mutate('report.json',lambda v:v.update(seed=False))
        with self.assertRaises(ContractError): workflow.audit(self.root,self.raw,self.rid,cid)

    def test_measurement_checkpoint_identity_rejected(self):
        cid = self.mutate('diagnostics-1.json',lambda v:v.update(model_identity='sha256:'+'0'*64))
        with self.assertRaises(ContractError): workflow.audit(self.root,self.raw,self.rid,cid)

    def test_skeletal_preflight_rejected(self):
        (self.root/'preflight.json').write_bytes(json_bytes({'passed':True}))
        cid = self.reseal_test_packet()
        with self.assertRaises(ContractError): workflow.audit(self.root,self.raw,self.rid,cid)

    def test_resources_negative_or_false_cache_rejected(self):
        cid = self.mutate('resources.json',lambda v:v.update(kv_payload_bytes=0))
        with self.assertRaises(ContractError): workflow.audit(self.root,self.raw,self.rid,cid)

    def test_audit_enforces_aggregate_retained_payload_bound(self):
        completion = parse_json((self.root/'completion.json').read_bytes(), canonical=True)
        names = list(completion['artifacts']) + ['request.json','completion.json','verification.json']
        total = sum((self.root/name).stat().st_size for name in names)
        # Test-only smaller verifier ceiling, no producer/request mutation or large allocation.
        with patch('native_diagnostic_v1.workflow.PROTOCOL',
                   {**request.PROTOCOL, 'max_retained_payload_bytes':total-1}):
            with self.assertRaisesRegex(ContractError, 'payload budget'):
                workflow.audit(self.root,self.raw,self.rid,self.cid)

    def test_upstream_bundle_tamper_is_not_refreshed(self):
        artifact = next((self.root/'provenance/artifacts/sha256').iterdir())
        artifact.write_bytes(b'corrupt')
        before = self.contents()
        with self.assertRaises(ContractError): workflow.audit(self.root,self.raw,self.rid,self.cid)
        self.assertEqual(before,self.contents())

    def test_request_extra_fields_seed_bool_and_environment_rejected(self):
        value = parse_json(self.raw)
        for mutation in ({'extra':0},{'seed':True},{'environment':{}}):
            altered = json_bytes({**value,**mutation})
            with self.assertRaises(ContractError): request.validate(altered,identity(altered))

    def test_controller_preserves_failures_and_timeout_no_retry(self):
        req = Path(self.tmp.name)/'request.json';req.write_bytes(self.raw)
        for i,result in enumerate((subprocess.CompletedProcess([],2),subprocess.TimeoutExpired('fixed',900))):
            target = Path(self.tmp.name)/('failed'+str(i))
            with patch('native_diagnostic_v1.runner.launch', side_effect=result if isinstance(result,Exception) else None,
                       return_value=result) as launched:
                report=runner.run_controller(target,req,self.rid)
            self.assertEqual(launched.call_count,1)
            self.assertIn(report['outcome'],('failed','timed_out'))
            self.assertIsNone(report['completion_identity'])
            self.assertTrue((target/'attempt.json').is_file())
            with self.assertRaises(FileExistsError): runner.run_controller(target,req,self.rid)

    def test_source_closure_contains_new_launcher_and_transitive_packages(self):
        closure=request.sources()
        for name in ('scripts/run_native_diagnostic.py','scripts/native_diagnostic_v1/measurements.py',
                     'scripts/native_diagnostic_v1/mechanics.py','src/trinite/training.py',
                     'src/trinite/foundations_oracle.py','src/trinite/budget_checkpoint.py',
                     'src/provenance_verify/__init__.py','requirements.lock'):
            self.assertIn(name,closure)


class ArtifactTests(unittest.TestCase):
    def fake_process(self, payload=b'', waits=None):
        process = MagicMock()
        process.__enter__.return_value = process
        process.__exit__.return_value = False
        process.stdout = io.BytesIO(payload)
        if waits is None:
            process.wait.return_value = 0
        else:
            process.wait.side_effect = waits
        return process

    def test_launch_fixed_argv_no_shell_and_directory_handoff(self):
        with tempfile.TemporaryDirectory() as t:
            output = Path(t)
            stat = output.stat()
            request_path = output/'request with spaces;echo-not-code.json'
            rid = 'sha256:'+'a'*64
            process = self.fake_process(b'bounded output\n')
            with patch.object(runner.subprocess, 'Popen', return_value=process) as popen:
                result = runner.launch(output, request_path, rid, (stat.st_dev, stat.st_ino))
            expected = [runner.sys.executable, str(runner.ROOT/'scripts/run_native_diagnostic.py'),
                        '_worker', '--request', str(request_path), '--request-identity', rid,
                        '--output', str(output), '--expected-directory',
                        str(stat.st_dev), str(stat.st_ino)]
            self.assertEqual(popen.call_count, 1)
            args, kwargs = popen.call_args
            self.assertEqual(args, (expected,))
            self.assertIs(kwargs['shell'], False)
            self.assertEqual(kwargs['cwd'], runner.ROOT)
            self.assertEqual(kwargs['stdout'], subprocess.PIPE)
            self.assertEqual(kwargs['stderr'], subprocess.STDOUT)
            self.assertEqual(kwargs['env']['PYTHONPATH'], str(runner.ROOT/'src'))
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                self.assertEqual(kwargs['env'][key], '1')
            process.wait.assert_called_once_with(timeout=runner.PROTOCOL['worker_seconds'])
            process.kill.assert_not_called()
            self.assertEqual(result.args, expected)
            self.assertEqual(result.returncode, 0)
            self.assertEqual((output/'worker.log').read_bytes(), b'bounded output\n')

    def test_launch_rejects_wrong_directory_before_spawning(self):
        with tempfile.TemporaryDirectory() as t:
            output = Path(t); stat = output.stat()
            with patch.object(runner.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(ContractError, 'identity differs'):
                    runner.launch(output, output/'request.json', 'bound-id',
                                  (stat.st_dev, stat.st_ino+1))
            popen.assert_not_called()
            self.assertEqual(list(output.iterdir()), [])

    def test_launch_log_overflow_kills_rejects_and_retains_bounded_prefix(self):
        with tempfile.TemporaryDirectory() as t:
            output = Path(t)
            payload = b'x'*129
            process = self.fake_process(payload)
            original_limit = runner.PROTOCOL['max_worker_log_bytes']
            # A separate protocol dictionary avoids changing request/source identity.
            bounded = {**runner.PROTOCOL, 'max_worker_log_bytes':64}
            with patch.object(runner, 'PROTOCOL', bounded), \
                 patch.object(runner.subprocess, 'Popen', return_value=process):
                with self.assertRaisesRegex(ContractError, 'worker log exceeded bound'):
                    runner.launch(output, output/'request.json', 'bound-id')
            process.kill.assert_called_once_with()
            process.wait.assert_called_once_with(timeout=bounded['worker_seconds'])
            retained = (output/'worker.log').read_bytes()
            self.assertEqual(retained, payload[:64])
            self.assertLessEqual(len(retained), 1024*1024)
            self.assertEqual(runner.PROTOCOL['max_worker_log_bytes'], original_limit)
            process.__exit__.assert_called_once()

    def test_launch_timeout_kills_and_reaps_before_propagating(self):
        with tempfile.TemporaryDirectory() as t:
            output = Path(t)
            timeout = subprocess.TimeoutExpired('fixed launcher', runner.PROTOCOL['worker_seconds'])
            process = self.fake_process(b'partial output\n', waits=[timeout, -9])
            with patch.object(runner.subprocess, 'Popen', return_value=process):
                with self.assertRaises(subprocess.TimeoutExpired) as raised:
                    runner.launch(output, output/'request.json', 'bound-id')
            self.assertIs(raised.exception, timeout)
            process.kill.assert_called_once_with()
            self.assertEqual(process.wait.call_count, 2)
            self.assertEqual(process.wait.call_args_list[0].kwargs,
                             {'timeout':runner.PROTOCOL['worker_seconds']})
            self.assertEqual(process.wait.call_args_list[1].args, ())
            self.assertEqual(process.wait.call_args_list[1].kwargs, {})
            self.assertEqual((output/'worker.log').read_bytes(), b'partial output\n')
            process.__exit__.assert_called_once()

    def test_atomic_publication_refuses_existing_and_symlink(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x';publish(p,b'one')
            with self.assertRaises(FileExistsError):publish(p,b'two')
            self.assertEqual(p.read_bytes(),b'one')
            q=Path(t)/'link';q.symlink_to(p)
            with self.assertRaises(FileExistsError):publish(q,b'two')
            self.assertEqual(p.read_bytes(),b'one')

    def test_store_rejects_wrong_controller_directory_identity(self):
        with tempfile.TemporaryDirectory() as t:
            stat = Path(t).stat()
            with self.assertRaisesRegex(ContractError, 'identity differs'):
                Store(t, (stat.st_dev, stat.st_ino+1))
            self.assertEqual(list(Path(t).iterdir()), [])

    def test_store_rejects_path_escape_and_existing(self):
        with tempfile.TemporaryDirectory() as t:
            store=Store(t)
            try:
                with self.assertRaises(ContractError):store.write('../outside',b'a')
                store.write('a',b'1')
                with self.assertRaises(FileExistsError):store.write('a',b'2')
            finally:store.close()


if __name__ == '__main__':unittest.main()
