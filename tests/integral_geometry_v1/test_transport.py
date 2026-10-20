import subprocess
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
LAUNCHER=ROOT/'scripts/run_integral_geometry.py'

class Transport(unittest.TestCase):
    def test_requires_isolation(self):
        p=subprocess.run([sys.executable,str(LAUNCHER),'--help'],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('python -I',p.stderr)

    def test_atomic_no_clobber_and_failure_cleanup(self):
        # Ordinary filesystem error simulation, not untrusted import execution.
        code='''
import runpy,tempfile
from pathlib import Path
from unittest.mock import patch
ns=runpy.run_path(%r)
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'report.json'
    ns['publish'](p,b'complete\\n')
    try:
        ns['publish'](p,b'replacement\\n')
    except FileExistsError:
        pass
    else:
        raise RuntimeError('clobbered an existing artifact')
    if p.read_bytes()!=b'complete\\n':
        raise RuntimeError('existing bytes changed')
    q=Path(d)/'retry.json'
    with patch('os.link',side_effect=OSError('simulated publication failure')):
        try:
            ns['publish'](q,b'complete\\n')
        except OSError:
            pass
        else:
            raise RuntimeError('expected write failure')
    if q.exists() or list(Path(d).iterdir())!=[p]:
        raise RuntimeError('failed publication left final or temporary bytes')
    ns['publish'](q,b'complete\\n')
''' % str(LAUNCHER)
        p=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_worker_failure_and_timeout_retain_attempts(self):
        code = r'''
import json,runpy,tempfile,subprocess,sys
from pathlib import Path
from unittest.mock import patch
ns=runpy.run_path(%r)
g=ns['main'].__globals__
g['numerical']=lambda: {'prospective':'controller test only'}
with tempfile.TemporaryDirectory() as d:
    folder=Path(d)
    request=folder/'request.json'
    raw=g['json_bytes'](g['numerical']())
    request.write_bytes(raw)
    for name,error in [('failed.json',None),('timeout.json',subprocess.TimeoutExpired('worker',900))]:
        output=folder/name
        argv=['launcher','run','--request',str(request),'--request-identity',g['identity'](raw),'--output',str(output)]
        with patch.object(sys,'argv',argv),patch('subprocess.run',return_value=subprocess.CompletedProcess([],9),side_effect=error):
            if ns['main']()!=1:
                raise RuntimeError('failure falsely accepted')
        attempts=list(folder.glob(name+'.attempt-*'))
        if len(attempts)!=1 or output.exists():
            raise RuntimeError('failed output published or attempt missing')
        status=json.loads((attempts[0]/'status.json').read_text())
        if status['timed_out']!=(error is not None):
            raise RuntimeError('incorrect timeout receipt')
        if (attempts[0]/'request.json').read_bytes()!=raw:
            raise RuntimeError('request not retained')
        if not all((attempts[0]/f).exists() for f in ('stdout.log','stderr.log')):
            raise RuntimeError('failure logs missing')
''' % str(LAUNCHER)
        p=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

if __name__=='__main__':unittest.main()
