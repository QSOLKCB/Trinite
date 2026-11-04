# SPDX-License-Identifier: MPL-2.0
"""Software process doubles test failure custody, never certify native execution."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes
from teacher_fit_v1 import custody as io, spec, supervisor


class FakeProcess:
    pid=987654
    def __init__(self, failure):self.failure=failure;self.returncode=None;self.waits=0
    def poll(self):return self.returncode
    def wait(self,timeout):
        self.waits+=1
        if self.waits==1:raise self.failure
        self.returncode=-9;return self.returncode


class SupervisorContracts(unittest.TestCase):
    def exercise(self,failure):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        request={'seed':17,'experiment_directory':str(root),'output_directory':str(root/'seed-17'),
                 'source':{'commit':'synthetic'}}
        req=root/'request.json';approval=root/'approval.json';io.record(req,request);io.record(approval,{'synthetic':True})
        process=FakeProcess(failure)
        grants={'17':{'identity':identity(io.read(approval))}}
        with patch.dict(os.environ,{},clear=True),patch.object(spec,'verify_request',return_value=request), \
                patch.object(spec,'approval'),patch.object(spec,'source',return_value=request['source']), \
                patch.object(spec,'approval_set',return_value=(grants,{})), \
                patch.object(subprocess,'Popen',return_value=process),patch.object(os,'killpg') as kill:
            receipt=supervisor.dispatch(req,identity(io.read(req)),root,approval,identity(io.read(approval)))
            kill.assert_called_once()
            with self.assertRaises(ContractError):
                supervisor.dispatch(req,identity(io.read(req)),root,approval,identity(io.read(approval)))
        self.assertEqual(receipt['child_returncode'],-9)
        self.assertEqual(receipt['status'],'FAILED');self.assertTrue(receipt['process_resolved'])
        io.verify_seal(root/'seed-17',receipt['manifest_identity'])
        self.assertTrue((root/'seed-17'/'stderr.log').exists())
        return receipt,root

    def test_timeout_kills_and_reaps_original_process_preserving_exit(self):
        receipt,_=self.exercise(subprocess.TimeoutExpired('synthetic',900))
        self.assertTrue(receipt['timed_out']);self.assertEqual(receipt['supervisor_exit_status'],124)

    def test_interruption_preserves_attempt_and_prevents_retry(self):
        receipt,_=self.exercise(KeyboardInterrupt())
        self.assertTrue(receipt['interrupted']);self.assertEqual(receipt['supervisor_exit_status'],130)

    def test_unresolved_preceding_process_blocks_next_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=io.fresh(root/'seed-17')
            io.record(out/'supervisor.json',{'process_resolved':False,'child_returncode':None})
            with self.assertRaises(ContractError):supervisor.previous_resolved({'seed':29,'experiment_directory':str(root)})

    def test_reaped_leader_does_not_certify_surviving_group(self):
        p=FakeProcess(KeyboardInterrupt());p.returncode=0
        clock=iter([0.,6.])
        with patch.object(supervisor,'group_stopped',return_value=False),patch.object(os,'killpg') as kill, \
                patch.object(supervisor.time,'monotonic',side_effect=lambda:next(clock)):
            self.assertFalse(supervisor.resolve_group(p));kill.assert_called_once()

    def test_missing_approval_never_creates_output_or_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);request={'seed':17,'output_directory':str(root/'seed-17')}
            with patch.dict(os.environ,{},clear=True),patch.object(spec,'verify_request',return_value=request), \
                    patch.object(subprocess,'Popen') as popen:
                with self.assertRaises(FileNotFoundError):
                    supervisor.dispatch(root/'request','sha256:'+'a'*64,root,root/'missing','sha256:'+'b'*64)
                popen.assert_not_called();self.assertFalse((root/'seed-17').exists())


if __name__=='__main__':unittest.main()
