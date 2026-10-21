"""Offline CLI end-to-end with fake artifacts, no teacher or GPU execution."""
from pathlib import Path
import subprocess
import sys
import unittest
from trinite.contracts import identity, json_bytes, parse_json
import test_intake

ROOT=Path(__file__).resolve().parents[2]
CLI=ROOT/'scripts/run_teacher_intake.py'


class Transport(unittest.TestCase):
    def setUp(self):
        self.fixture=test_intake.IntakeContracts();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root

    def call(self,*args):
        return subprocess.run([sys.executable,'-I','-S',str(CLI),*map(str,args)],
                              capture_output=True,check=False,timeout=30)

    def test_freeze_seal_verify_immutable(self):
        manifest=self.root/'manifest.json';manifest.write_bytes(json_bytes(self.fixture.manifest))
        request=self.root/'request.json'
        result=self.call('freeze','--manifest',manifest,'--asset-root',self.root,'--output',request)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(request.read_bytes(),json_bytes(self.fixture.frozen))
        capture=self.root/'capture.json';capture.write_bytes(json_bytes(self.fixture.capture()))
        output=self.root/'packet.json'
        args=('seal','--request',request,'--request-identity',identity(request.read_bytes()),
              '--capture',capture,'--output',output)
        result=self.call(*args);self.assertEqual(result.returncode,0,result.stderr)
        before=output.read_bytes()
        result=self.call('verify','--packet',output,'--packet-identity',identity(before))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(output.read_bytes(),before)
        result=self.call(*args);self.assertNotEqual(result.returncode,0)
        self.assertEqual(output.read_bytes(),before)
        result=self.call('verify','--packet',output,'--packet-identity','sha256:'+'0'*64)
        self.assertNotEqual(result.returncode,0)

    def test_invalid_capture_retained_with_failure(self):
        request=self.root/'request.json';request.write_bytes(json_bytes(self.fixture.frozen))
        capture=self.root/'capture.json';capture.write_bytes(b'not JSON')
        output=self.root/'report.txt'
        result=self.call('seal','--request',request,'--request-identity',identity(request.read_bytes()),
                         '--capture',capture,'--output',output)
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(output.exists())
        folders=list(self.root.glob('report.txt.attempt-*'))
        self.assertEqual(len(folders),1)
        self.assertEqual((folders[0]/'capture.json').read_bytes(),b'not JSON')
        self.assertEqual(parse_json((folders[0]/'failure.json').read_bytes())['phase'],'failed')

    def test_unisolated_entrypoint_rejected(self):
        result=subprocess.run([sys.executable,str(CLI),'--help'],capture_output=True,check=False)
        self.assertNotEqual(result.returncode,0)
        self.assertIn(b'python -I -S',result.stderr)
