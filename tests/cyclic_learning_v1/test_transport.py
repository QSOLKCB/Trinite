import subprocess
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
LAUNCHER=ROOT/'scripts/run_cyclic_learning.py'

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
        if status['controller_exit_code']!=1 or status['publication']!='not_attempted':
            raise RuntimeError('worker failure marked as publication success')
        if status['worker_resource_usage']['after']['units']!='KiB':
            raise RuntimeError('failure peak memory missing')
        if status['timed_out']!=(error is not None):
            raise RuntimeError('incorrect timeout receipt')
        if (attempts[0]/'request.json').read_bytes()!=raw:
            raise RuntimeError('request not retained')
        if not all((attempts[0]/f).exists() for f in ('stdout.log','stderr.log')):
            raise RuntimeError('failure logs missing')
''' % str(LAUNCHER)
        p=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)


class FinalPublication(unittest.TestCase):
    def test_success_collision_io_failure_and_wrong_request_receipts(self):
        code=r'''
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
    request_id=g['identity'](raw)
    for mode in ('success','collision','io_error','wrong_binding','status_error','journal_error','storage_failure','intent_error','status_and_journal_error'):
        output=folder/(mode+'.json')
        payload=g['json_bytes']({'schema':'trinite.cyclic-learning-report.v2',
                   'request_identity':request_id if mode!='wrong_binding' else 'sha256:'+'0'*64})
        def worker(command,**kwargs):
            Path(command[command.index('--output')+1]).write_bytes(payload)
            return subprocess.CompletedProcess(command,0)
        publish=g['publish']
        def publication(path,raw):
            if mode in ('status_error','storage_failure','status_and_journal_error') and path.name=='status.json':
                raise OSError('simulated status publication failure')
            if path==output:
                if mode=='collision':
                    output.write_bytes(b'other process output')
                if mode=='io_error':
                    raise OSError('simulated final-link failure')
            return publish(path,raw)
        record=g['journal_record']
        def journal_write(path,value,**kwargs):
            if mode=='intent_error' and value['phase']=='publication_intent':
                raise OSError('simulated intent write failure')
            if mode=='status_and_journal_error' and value['phase']=='status_publication_failed':
                raise OSError('simulated subsequent journal failure')
            if mode in ('journal_error','storage_failure') and value['phase'] in ('publication_outcome','status_publication_failed'):
                raise OSError('simulated journal storage failure')
            return record(path,value,**kwargs)
        argv=['launcher','run','--request',str(request),'--request-identity',request_id,'--output',str(output)]
        with patch.object(sys,'argv',argv),patch('subprocess.run',side_effect=worker),patch.dict(g,{'publish':publication,'journal_record':journal_write}):
            rc=ns['main']()
        attempt=next(folder.glob(mode+'.json.attempt-*'))
        journal=[json.loads(line) for line in (attempt/'controller.jsonl').read_text().splitlines()]
        if mode=='storage_failure':
            if rc!=1 or (attempt/'status.json').exists() or output.read_bytes()!=payload:
                raise RuntimeError('full storage failure handled dishonestly')
            if [r['phase'] for r in journal]!=['attempt_started','publication_intent']:
                raise RuntimeError('incomplete journal invented a durable success')
            continue
        if mode in ('status_error','status_and_journal_error'):
            if rc!=1 or (attempt/'status.json').exists() or output.read_bytes()!=payload:
                raise RuntimeError('status failure did not preserve the completed output')
            observations=[r for r in journal if r['phase']=='publication_outcome']
            if len(observations)!=1 or observations[0]['publication']!='published' or 'controller_exit_code' in observations[0]:
                raise RuntimeError('durable journal does not explain actual publication')
            if mode=='status_error':
                if journal[-1]['phase']!='status_publication_failed' or journal[-1]['controller_exit_code']!=1:
                    raise RuntimeError('status failure controller receipt missing')
            elif journal[-1]['phase']!='publication_outcome':
                raise RuntimeError('failed journal append invented a receipt')
            continue
        status=json.loads((attempt/'status.json').read_text())
        if status['exit_code']!=0 or status['controller_exit_code']!=rc:
            raise RuntimeError('worker/controller results conflated')
        if (attempt/'report.json').read_bytes()!=payload:
            raise RuntimeError('completed worker report not retained')
        usage=status['worker_resource_usage']
        if any(v['units']!='KiB' or v['platform']!='linux' or v['peak_rss_kib']<0 for v in usage.values()):
            raise RuntimeError('missing scoped Linux peak memory measurement')
        if mode=='journal_error':
            if rc!=1 or status['publication']!='published' or 'journal_error' not in status or output.read_bytes()!=payload:
                raise RuntimeError('journal failure missing from retained status')
            continue
        if mode=='success':
            if rc!=0 or status['publication']!='published' or output.read_bytes()!=payload:
                raise RuntimeError('success not correctly recorded')
        else:
            if rc!=1 or 'controller_error' not in status:
                raise RuntimeError('controller failure was lost')
            if status['publication']!=('not_attempted' if mode in ('wrong_binding','intent_error') else 'failed'):
                raise RuntimeError('incorrect publication outcome')
            if mode=='collision' and output.read_bytes()!=b'other process output':
                raise RuntimeError('colliding destination was modified')
            if mode!='collision' and output.exists():
                raise RuntimeError('failed result was published')
''' % str(LAUNCHER)
        p=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_worker_checks_completed_request_binding(self):
        code=r'''
import runpy,tempfile,sys
from pathlib import Path
from unittest.mock import patch
ns=runpy.run_path(%r)
g=ns['main'].__globals__
g['numerical']=lambda: {'prospective':'worker controller test only'}
with tempfile.TemporaryDirectory() as d:
    request=Path(d)/'request.json'
    output=Path(d)/'report.json'
    raw=g['json_bytes'](g['numerical']())
    request.write_bytes(raw)
    def report(*,expected_request,on_cell):
        if expected_request!=g['numerical']():
            raise RuntimeError('validated request was not passed into report')
        return {'schema':'trinite.cyclic-learning-report.v2','request_identity':'sha256:'+'0'*64}
    argv=['launcher','_worker','--request',str(request),'--request-identity',g['identity'](raw),'--output',str(output)]
    with patch.object(sys,'argv',argv),patch('cyclic_learning_v1.experiment.report',side_effect=report):
        if ns['main']()!=1 or output.exists():
            raise RuntimeError('worker accepted a different request binding')
        if not output.with_name(output.name+'.rejected.json').exists():
            raise RuntimeError('rejected worker report not retained')
''' % str(LAUNCHER)
        p=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_actual_child_peak_memory_has_explicit_scope(self):
        code=r'''
import runpy,subprocess,sys
ns=runpy.run_path(%r)
before=ns['child_usage']()
subprocess.run([sys.executable,'-I','-c','data=bytearray(1048576)'],check=True)
after=ns['child_usage']()
if after['peak_rss_kib']<=0 or after['peak_rss_kib']<before['peak_rss_kib']:
    raise RuntimeError('missing child resource measurement')
if after['units']!='KiB' or 'cumulative reaped-child' not in after['scope']:
    raise RuntimeError('memory measurement lacks units or scope')
''' % str(LAUNCHER)
        p=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)


class ReplayBaselineCustody(unittest.TestCase):
    def test_verified_baseline_retained_before_worker_and_on_all_outcomes(self):
        code=r'''
import json,runpy,tempfile,subprocess,sys
from pathlib import Path
from unittest.mock import patch
ns=runpy.run_path(%r)
g=ns['main'].__globals__
g['numerical']=lambda: {'prospective':'synthetic replay custody test only'}
with tempfile.TemporaryDirectory() as d:
    folder=Path(d)
    request=folder/'request.json'
    raw=g['json_bytes'](g['numerical']());request.write_bytes(raw)
    request_id=g['identity'](raw)
    expected_raw=g['json_bytes']({'schema':'trinite.cyclic-learning-report.v2',
                                'request_identity':request_id,'synthetic':'expected'})
    expected_id=g['identity'](expected_raw)
    for mode in ('exact','mismatch','failure','timeout'):
        expected=folder/(mode+'-external.json');expected.write_bytes(expected_raw)
        output=folder/(mode+'.json')
        payload=expected_raw if mode=='exact' else g['json_bytes']({
            'schema':'trinite.cyclic-learning-report.v2','request_identity':request_id,'synthetic':'different'})
        def worker(command,**kwargs):
            report=Path(command[command.index('--output')+1]);attempt=report.parent
            if (attempt/'expected-report.json').read_bytes()!=expected_raw:
                raise RuntimeError('baseline was not retained before worker')
            first=json.loads((attempt/'controller.jsonl').read_text().splitlines()[0])
            if first['expected_report_identity']!=expected_id:
                raise RuntimeError('initial journal does not bind baseline')
            expected.unlink()  # Receipt must survive external baseline removal.
            if mode=='timeout':raise subprocess.TimeoutExpired('worker',900)
            if mode=='failure':return subprocess.CompletedProcess(command,9)
            report.write_bytes(payload)
            return subprocess.CompletedProcess(command,0)
        argv=['launcher','replay','--request',str(request),'--request-identity',request_id,
              '--output',str(output),'--expected-report',str(expected),'--expected-identity',expected_id]
        with patch.object(sys,'argv',argv),patch('subprocess.run',side_effect=worker):
            rc=ns['main']()
        attempt=next(folder.glob(mode+'.json.attempt-*'))
        status=json.loads((attempt/'status.json').read_text())
        if status['expected_report_identity']!=expected_id or g['identity']((attempt/'expected-report.json').read_bytes())!=expected_id:
            raise RuntimeError('retained baseline cannot be audited')
        if rc!=(0 if mode=='exact' else 1):raise RuntimeError('incorrect replay outcome')
        if mode in ('exact','mismatch'):
            if status['report_identity']!=g['identity'](payload) or status['exact_replay']!=(mode=='exact'):
                raise RuntimeError('comparison does not bind both identities')
        if output.exists()!=(mode=='exact'):raise RuntimeError('failed replay published output')
''' % str(LAUNCHER)
        p=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)


class EvidenceSchemaVersions(unittest.TestCase):
    def test_v1_report_rejected_even_with_matching_request_identity(self):
        code=r"""
import runpy,tempfile,sys
from pathlib import Path
from unittest.mock import patch
ns=runpy.run_path(%r)
g=ns['main'].__globals__
g['numerical']=lambda: {'schema':'trinite.cyclic-learning-request.v2',
                      'protocol':{'schema':'trinite.cyclic-learning-protocol.v2'},'synthetic':'schema test'}
with tempfile.TemporaryDirectory() as d:
    folder=Path(d)
    request=folder/'request.json';raw=g['json_bytes'](g['numerical']());request.write_bytes(raw)
    request_id=g['identity'](raw)
    old=g['json_bytes']({'schema':'trinite.cyclic-learning-report.v1','request_identity':request_id})
    new=g['json_bytes']({'schema':'trinite.cyclic-learning-report.v2','request_identity':request_id})
    try:g['validate_report_binding'](old,request_id)
    except g['ContractError']:pass
    else:raise RuntimeError('v1 report accepted as v2 evidence')
    g['validate_report_binding'](new,request_id)
    expected=folder/'expected.json';expected.write_bytes(old)
    output=folder/'output.json'
    argv=['launcher','replay','--request',str(request),'--request-identity',request_id,
          '--output',str(output),'--expected-report',str(expected),'--expected-identity',g['identity'](old)]
    with patch.object(sys,'argv',argv),patch('subprocess.run') as worker:
        if ns['main']()!=1:raise RuntimeError('old baseline schema accepted')
        worker.assert_not_called()
    if output.exists() or expected.read_bytes()!=old or list(folder.glob('output.json.attempt-*')):
        raise RuntimeError('rejected baseline started a run or changed existing evidence')
    # Correct identity cannot turn a v1 worker envelope into a v2 report.
    argv=['launcher','_worker','--request',str(request),'--request-identity',request_id,'--output',str(output)]
    with patch.object(sys,'argv',argv),patch('cyclic_learning_v1.experiment.report',return_value=g['parse_json'](old,canonical=True)):
        if ns['main']()!=1 or output.exists():raise RuntimeError('old worker report accepted')
    if output.with_name(output.name+'.rejected.json').read_bytes()!=old:
        raise RuntimeError('old worker report not retained as rejected')
    # Independently reject obsolete request and nested protocol identifiers.
    for field in ('request','protocol'):
        value=g['numerical']()
        if field=='request':value['schema']='trinite.cyclic-learning-request.v1'
        else:value['protocol']['schema']='trinite.cyclic-learning-protocol.v1'
        obsolete=g['json_bytes'](value);request.write_bytes(obsolete)
        argv=['launcher','run','--request',str(request),'--request-identity',g['identity'](obsolete),'--output',str(folder/(field+'.json'))]
        with patch.object(sys,'argv',argv),patch('subprocess.run') as worker:
            if ns['main']()!=1:raise RuntimeError('obsolete request/protocol accepted')
            worker.assert_not_called()
        if request.read_bytes()!=obsolete:raise RuntimeError('rejected request changed')

""" % str(LAUNCHER)
        p=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

if __name__=='__main__':unittest.main()
