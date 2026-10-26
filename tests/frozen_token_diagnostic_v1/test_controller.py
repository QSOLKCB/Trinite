"""Controller fakes test failures only; never certify custody or numerical outputs."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from frozen_token_diagnostic_v1 import controller, request
from addition_development_v1.request import cells, cell_name


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.producer = self.root/'producer'; self.producer.mkdir()
        self.req = self.root/'request.json'; self.raw = b'{}\n'; self.req.write_bytes(self.raw)
        self.rid = identity(self.raw)

    @contextmanager
    def failure_scope(self):
        # This isolates lifecycle failure tests, not producer/custody conformance.
        with patch.object(controller,'validate',return_value={}), \
             patch.object(controller,'snapshot_matrix',return_value={}), \
             patch.object(controller,'unchanged_matrix'):
            yield

    def matrix(self, launch):
        with self.failure_scope(), patch.object(controller,'launch',side_effect=launch):
            result = controller.run_matrix(self.root/'output',self.req,self.rid,self.producer)
            receipt = controller.audit_matrix(self.root/'output',self.raw,self.rid,self.producer,result['matrix_identity'])
        return result,receipt

    def test_all_failed_cells_retained_and_audited_as_incomplete(self):
        result,receipt = self.matrix(lambda *args: 7)
        self.assertEqual(result['outcome'],'incomplete');self.assertEqual(receipt['reports'],{})
        self.assertEqual(set(result['cells']),{cell_name(a,s) for a,s in cells()})
        for cell in result['cells'].values():
            self.assertEqual(cell['outcome'],'failed');self.assertIsNone(cell['completion_identity'])
            self.assertIn('attempt.json',cell['retained_files'])

    def test_audit_rejects_oversized_retained_failure_log(self):
        def failed(*args):
            (args[0]/'worker.log').write_bytes(b'x'*(controller.PROTOCOL['max_worker_log_bytes']+1))
            return 7
        with self.failure_scope(),patch.object(controller,'launch',side_effect=failed):
            result=controller.run_matrix(self.root/'output',self.req,self.rid,self.producer)
            with self.assertRaisesRegex(ContractError,'bounded regular artifact'):
                controller.audit_matrix(self.root/'output',self.raw,self.rid,self.producer,result['matrix_identity'])

    def test_timeout_has_no_retry_and_preserves_all_six(self):
        called=[]
        def launch(*args):
            called.append(args[4:6]);raise subprocess.TimeoutExpired('fixed-worker',900)
        result,_ = self.matrix(launch)
        self.assertEqual(len(called),6)
        self.assertTrue(all(c['outcome']=='timed_out' for c in result['cells'].values()))

    def test_interruption_marks_remaining_unattempted(self):
        calls=[]
        def launch(*args): calls.append(1);raise KeyboardInterrupt()
        result,_ = self.matrix(launch)
        self.assertEqual(calls,[1]);self.assertEqual(result['cells']['control-0']['outcome'],'interrupted')
        self.assertTrue(all(result['cells'][cell_name(a,s)]['outcome']=='not_attempted' for a,s in cells()[1:]))

    def test_relative_producer_is_absolute_at_worker_handoff(self):
        seen=[]
        previous=Path.cwd()
        try:
            os.chdir(self.root)
            with self.failure_scope(), patch.object(controller,'launch',side_effect=lambda *a: seen.append(a[3]) or 2):
                controller.run_matrix(self.root/'output',self.req,self.rid,Path('producer'))
        finally: os.chdir(previous)
        self.assertEqual(seen,[self.producer]*6)

    def test_output_and_request_overlap_rejected_without_writes(self):
        with self.failure_scope():
            for output in (self.producer/'nested',self.root,request.ROOT/'nested'):
                with self.assertRaises(ContractError): controller.run_matrix(output,self.req,self.rid,self.producer)
        self.assertEqual(list(self.producer.iterdir()),[])

    def test_existing_output_not_reused(self):
        out=self.root/'output';out.mkdir();(out/'keep').write_text('unchanged')
        with self.failure_scope(),self.assertRaises(FileExistsError):
            controller.run_matrix(out,self.req,self.rid,self.producer)
        self.assertEqual((out/'keep').read_text(),'unchanged')

    def test_symlink_ancestor_rejected(self):
        link=self.root/'alias';link.symlink_to(self.producer,target_is_directory=True)
        with self.assertRaises(ContractError):controller.protect(link/'out',self.req)

    def test_run_needs_separate_authorization(self):
        with patch.object(controller,'configure'),patch.object(controller,'run_matrix') as run:
            self.assertEqual(controller.main(['run','--request',str(self.req),'--request-identity',self.rid,
                '--producer',str(self.producer),'--output',str(self.root/'out')]),1)
            run.assert_not_called()

    def test_freeze_needs_external_commitments(self):
        with patch.object(controller,'configure'),patch.object(controller,'core') as core:
            self.assertEqual(controller.main(['freeze','--request',str(self.root/'new.json'),
                '--producer',str(self.producer)]),1)
            core.assert_not_called()

    def test_new_source_binding_checks_actual_import_origins(self):
        fake=types.ModuleType('frozen_token_diagnostic_v1.injected');fake.__file__=str(self.root/'outside.py')
        with patch.dict(sys.modules,{'frozen_token_diagnostic_v1.injected':fake}):
            with self.assertRaisesRegex(ContractError,'import outside'):request.sources()
        bound=request.sources()
        for name in ('scripts/run_frozen_token_diagnostic.py','docs/FROZEN-TOKEN-DIAGNOSTIC.md',
                     'scripts/frozen_token_diagnostic_v1/producer.py','docs/ADDITION-DEVELOPMENT.md'):
            self.assertIn(name,bound)

    def test_bound_log_overflow_terminates_fixed_worker(self):
        root=self.root/'worker';root.mkdir();st=root.stat()
        class Fake:
            stdout=io.BytesIO(b'x'*(controller.PROTOCOL['max_worker_log_bytes']+1))
            killed=False
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def kill(self):self.killed=True
            def wait(self,timeout=None):return 1
        fake=Fake()
        with patch.object(controller.subprocess,'Popen',return_value=fake) as popen:
            with self.assertRaisesRegex(ContractError,'log exceeds'):
                controller.launch(root,self.req,self.rid,self.producer,'control',0,(st.st_dev,st.st_ino))
        self.assertTrue(fake.killed)
        self.assertEqual((root/'worker.log').stat().st_size,controller.PROTOCOL['max_worker_log_bytes'])
        argv=popen.call_args.args[0]
        self.assertEqual(argv[0],sys.executable);self.assertIn(str(request.ROOT/'scripts/run_frozen_token_diagnostic.py'),argv)
        self.assertFalse(popen.call_args.kwargs['shell'])


if __name__=='__main__':unittest.main()
