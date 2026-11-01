"""Publication boundary tests; temporary copies only, no training/model loading."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

HERE=Path(__file__).resolve().parent


class DepositBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.deposit=self.root/'deposit'
        shutil.copytree(HERE,self.deposit)

    def replay(self):
        return subprocess.run([sys.executable,'-B',str(self.deposit/'verify.py')],
            capture_output=True,text=True,timeout=30)

    def reseal_untrusted_outer(self,changed_file=None,raw=None):
        p=self.deposit/'manifest.json'
        value=json.loads(p.read_bytes())
        value['files']['payload.zip']=hashlib.sha256((self.deposit/'payload.zip').read_bytes()).hexdigest()
        if changed_file is not None:
            value['payload_files'][changed_file]=hashlib.sha256(raw).hexdigest()
        p.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')

    def test_changed_zip_fails_without_resealing(self):
        p=self.deposit/'payload.zip'
        value=bytearray(p.read_bytes());value[0]^=1;p.write_bytes(value)
        result=self.replay()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('deposit file hash: payload.zip',result.stderr)

    def test_resealed_progress_tamper_still_fails_external_run_anchor(self):
        p=self.deposit/'payload.zip';out=self.deposit/'changed.zip'
        name='run/fit-17/training.json'
        with zipfile.ZipFile(p) as original,zipfile.ZipFile(out,'w') as changed:
            for entry in original.infolist():
                raw=original.read(entry)
                if entry.filename==name:
                    value=json.loads(raw);value['step']=127
                    raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode('ascii')
                    altered=raw
                changed.writestr(entry,raw)
        out.replace(p)
        self.reseal_untrusted_outer(name,altered)
        result=self.replay()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('retained run file changed: fit-17/training.json',result.stderr)

    def test_resealed_path_traversal_is_rejected_before_extraction(self):
        with zipfile.ZipFile(self.deposit/'payload.zip','a') as z:
            z.writestr('../escape','untrusted')
        self.reseal_untrusted_outer()
        result=self.replay()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('unsafe ZIP path',result.stderr)
        self.assertFalse((self.root/'escape').exists())


if __name__=='__main__':
    unittest.main()
