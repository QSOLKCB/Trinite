"""Real tiny-custody integration; process fakes test failures only, never success."""
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from unseen_evaluation_v1 import request, runner, workflow
import test_producer as producer_fixtures


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A real two-update width-16 engineering producer, never the user model.
        producer_fixtures.ProducerTests.setUpClass()
        cls.parent = producer_fixtures.ProducerTests
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.producer = cls.parent.base
        cls.raw = json_bytes(request.core(cls.producer, cls.parent.rid, cls.parent.cid, cls.parent.mid))
        cls.rid = identity(cls.raw)
        cls.base = cls.root/'completed'; cls.base.mkdir()
        stat = cls.base.stat()
        try:
            with patch('torch.optim.AdamW', side_effect=AssertionError('optimizer construction')), \
                 patch('trinite.training.run_steps', side_effect=AssertionError('training')):
                cls.cid = workflow.run(cls.base, cls.raw, cls.rid, cls.producer, (stat.st_dev, stat.st_ino))
        except BaseException:
            cls.temp.cleanup()
            producer_fixtures.ProducerTests.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        producer_fixtures.ProducerTests.tearDownClass()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.output = self.path/'completed'; shutil.copytree(self.base, self.output)
        self.request_path = self.path/'request.json'; self.request_path.write_bytes(self.raw)

    def audit(self, expected=None):
        return workflow.audit(self.output, self.raw, self.rid, self.producer, expected or self.cid)

    @staticmethod
    def inventory(root):
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}

    def reseal_artifact_for_negative_test(self, name, mutate):
        """Only tests reseal altered bytes to reach deeper independent validators."""
        path = self.output/name
        record = parse_json(path.read_bytes(), canonical=True); mutate(record)
        raw = json_bytes(record); path.write_bytes(raw)
        completion = parse_json((self.output/'completion.json').read_bytes(), canonical=True)
        completion['artifacts'][name] = identity(raw)
        encoded = json_bytes(completion); (self.output/'completion.json').write_bytes(encoded)
        return identity(encoded)

    def test_real_custody_audit_and_summary_are_readonly_without_inference(self):
        before = self.inventory(self.output); producer_before = self.inventory(self.producer)
        out = io.StringIO()
        original_read = runner.read
        def no_summary_reread(path, *args, **kwargs):
            if Path(path).name == 'measurements.json':
                raise AssertionError('summarize must use audited retained bytes, not reread')
            return original_read(path, *args, **kwargs)
        with patch.object(runner, 'read', side_effect=no_summary_reread), \
             patch('torch.optim.AdamW', side_effect=AssertionError('optimizer')), \
             patch('trinite.training.run_steps', side_effect=AssertionError('training')), \
             patch('trinite.model.Decoder.forward', side_effect=AssertionError('forward')):
            receipt = self.audit()
            with redirect_stdout(out):
                code = runner.main(['summarize','--producer',str(self.producer),
                    '--request',str(self.request_path),'--request-identity',self.rid,
                    '--output',str(self.output),'--expected-identity',self.cid])
        self.assertEqual(code, 0)
        result = parse_json(out.getvalue().encode(), canonical=True)
        self.assertEqual(set(result['by_split']), {'validation','test'})
        self.assertTrue(receipt['verified'])
        for field in workflow.CLAIMS: self.assertIs(receipt[field], False)
        self.assertEqual(before, self.inventory(self.output))
        self.assertEqual(producer_before, self.inventory(self.producer))

    def test_summary_uses_verified_snapshot_after_disk_changes(self):
        original_audit = workflow.audit
        observed = {}
        def mutate_after_audit(*args, **kwargs):
            result = original_audit(*args, **kwargs)
            observed['verified'] = result[1]
            path = self.output/'measurements.json'
            changed = parse_json(path.read_bytes(), canonical=True)
            changed['summary'] = {'unverified': True}
            path.write_bytes(json_bytes(changed))
            return result
        output = io.StringIO()
        with patch.object(workflow, 'audit', side_effect=mutate_after_audit), redirect_stdout(output):
            code = runner.main(['summarize','--producer',str(self.producer),
                '--request',str(self.request_path),'--request-identity',self.rid,
                '--output',str(self.output),'--expected-identity',self.cid])
        self.assertEqual(code, 0)
        actual = parse_json(output.getvalue().encode(), canonical=True)
        self.assertEqual(actual['aggregate_descriptive_only'], observed['verified']['summary'])
        self.assertNotIn('unverified', actual['aggregate_descriptive_only'])

    def test_external_completion_and_retained_request_required(self):
        with self.assertRaises(ContractError): self.audit('sha256:'+'0'*64)
        (self.output/'request.json').write_bytes(b'{}\n')
        with self.assertRaises(ContractError): self.audit()

    def test_missing_artifact_and_tamper_rejected(self):
        path = self.output/'row-0000.json'; original = path.read_bytes()
        path.unlink()
        with self.assertRaises(ContractError): self.audit()
        path.write_bytes(original+b' ')
        with self.assertRaises(ContractError): self.audit()

    def test_report_model_and_claim_context_rejected(self):
        cid = self.reseal_artifact_for_negative_test('report.json',lambda r:r.update(training_updates=True))
        with self.assertRaises(ContractError): self.audit(cid)

    def test_progress_row_mismatch_rejected(self):
        cid = self.reseal_artifact_for_negative_test('row-0000.json',lambda r:r.update(targets=[]))
        with self.assertRaises(ContractError): self.audit(cid)

    def test_row_index_cannot_omit_progress_receipts(self):
        cid = self.reseal_artifact_for_negative_test('row-index.json',lambda r:r.pop('row-0000.json'))
        with self.assertRaises(ContractError): self.audit(cid)

    def test_resource_bool_noncanonical_and_bound_rejected(self):
        original = (self.output/'resources.json').read_bytes()
        completion = (self.output/'completion.json').read_bytes()
        for value in (True, 0, -1):
            with self.subTest(rss=value):
                cid = self.reseal_artifact_for_negative_test('resources.json',lambda r:r.update(peak_rss_kib=value))
                with self.assertRaises(ContractError): self.audit(cid)
                (self.output/'resources.json').write_bytes(original)
                (self.output/'completion.json').write_bytes(completion)
        cid = self.reseal_artifact_for_negative_test('resources.json',lambda r:r.update(wall_seconds_hex='nan'))
        with self.assertRaises(ContractError): self.audit(cid)

    def test_request_source_environment_protocol_and_identity_rejected(self):
        for key, value in [('source',{}),('environment',{}),('protocol',{}),('unknown',True)]:
            altered = parse_json(self.raw,canonical=True); altered[key]=value
            raw=json_bytes(altered)
            with self.subTest(key=key), self.assertRaises(ContractError):
                request.validate(raw,identity(raw),self.producer)
        with self.assertRaises(ContractError): request.validate(self.raw,'sha256:'+'0'*64,self.producer)

    def test_source_and_environment_live_guard_rejected(self):
        with patch('unseen_evaluation_v1.request.sources', return_value={}):
            with self.assertRaises(ContractError): self.audit()
        with patch('unseen_evaluation_v1.request.require_environment', return_value={}):
            with self.assertRaises(ContractError): self.audit()

    def test_disjoint_paths_and_symlink_aliases_rejected(self):
        for path in (self.producer, self.producer/'new-request.json',self.producer.parent):
            with self.subTest(path=path), self.assertRaises(ContractError): request.separate(path,self.producer)
        alias=self.path/'alias'; alias.symlink_to(self.producer,target_is_directory=True)
        with self.assertRaises(ContractError): request.separate(alias/'new',self.producer)
        alias.unlink(); alias.symlink_to(self.output,target_is_directory=True)
        with self.assertRaises(ContractError): workflow.audit(alias,self.raw,self.rid,self.producer,self.cid)
        target=self.output/'row-0000.json'; original=self.path/'original.json'; target.rename(original);target.symlink_to(original)
        with self.assertRaises(ContractError): self.audit()

    def test_controller_never_overwrites_existing_attempt(self):
        before=self.inventory(self.output)
        with patch.object(runner,'launch',side_effect=AssertionError('must not launch')):
            with self.assertRaises(FileExistsError): runner.run_controller(self.output,self.request_path,self.rid,self.producer)
        self.assertEqual(before,self.inventory(self.output))

    def test_controller_failed_timeout_interrupt_retains_outcomes(self):
        cases=[('failed',subprocess.CompletedProcess([],7)),
               ('failed',subprocess.CompletedProcess([],0)),
               ('timed_out',subprocess.TimeoutExpired('fixed-worker',900)),
               ('interrupted',KeyboardInterrupt()), ('failed',RuntimeError('worker failed'))]
        for index,(expected,result) in enumerate(cases):
            output=self.path/f'attempt-{index}'
            kwargs={'side_effect':result} if isinstance(result,BaseException) else {'return_value':result}
            with self.subTest(expected=expected,index=index), patch.object(runner,'launch',**kwargs):
                record=runner.run_controller(output,self.request_path,self.rid,self.producer)
                self.assertEqual(record['outcome'],expected)
                self.assertIsNone(record['completion_identity'])
                self.assertEqual(parse_json((output/'attempt.json').read_bytes(),canonical=True),record)
                self.assertFalse((output/'completion.json').exists())

    def test_cli_requires_authorization_and_external_audit_identity(self):
        args=['--producer',str(self.producer),'--request',str(self.request_path),
              '--request-identity',self.rid,'--output',str(self.path/'not-created')]
        with redirect_stderr(io.StringIO()), patch.object(runner,'run_controller',side_effect=AssertionError('must not run')):
            self.assertEqual(runner.main(['evaluate',*args]),1)
            self.assertEqual(runner.main(['audit',*args]),1)
        self.assertFalse((self.path/'not-created').exists())

    def test_freeze_refuses_overwrite_and_missing_external_ids(self):
        args=['freeze','--producer',str(self.producer),'--request',str(self.request_path)]
        with redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(args),1)
            self.assertEqual(runner.main([*args,'--producer-request-identity',self.parent.rid,
                '--producer-completion-identity',self.parent.cid,'--model-identity',self.parent.mid]),1)
        self.assertEqual(self.request_path.read_bytes(),self.raw)


if __name__ == '__main__':
    unittest.main()
