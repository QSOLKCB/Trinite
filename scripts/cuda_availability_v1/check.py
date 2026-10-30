"""Sanitized owner-local availability observations, not GPU health/admission."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import time

ROOT = Path(os.path.abspath(__file__)).parents[2]
SOURCE_FILES = ('scripts/check_cuda_availability.py', 'scripts/cuda_availability_v1/__init__.py',
                'scripts/cuda_availability_v1/check.py')
TIMEOUT = 10
OUTPUT_LIMIT = 16 * 1024
SOURCE_LIMIT = 1024 * 1024
MAX_DEVICES = 16
SMI = '/usr/bin/nvidia-smi'
SMI_ARGUMENTS = ('--query-gpu=driver_version,name,memory.total', '--format=csv,noheader,nounits')

# Fixed child programs, never caller-selected code, flags, module or executable.
# They deliberately emit no exception messages, filesystem paths or environment.
METADATA_CODE = r'''
import importlib.metadata
import json
result = {}
for name in ('torch', 'numpy', 'safetensors'):
    try:
        result[name] = {'status':'present','version':importlib.metadata.version(name)}
    except importlib.metadata.PackageNotFoundError:
        result[name] = {'status':'absent'}
    except Exception:
        result[name] = {'status':'metadata_error'}
print(json.dumps(result))
'''
TORCH_CODE = r'''
import json
try:
    import torch
except Exception:
    print(json.dumps({'status':'torch_import_failed'}))
else:
    result = {'status':'query_failed','torch_version':str(torch.__version__),
              'compiled_cuda':torch.version.cuda}
    try:
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
        result.update(available=available, device_count=count)
        if type(available) is not bool or type(count) is not int or not 0 <= count <= 16:
            result['status'] = 'invalid_response'
        elif not available or not count:
            result['status'] = 'no_visible_cuda'
        else:
            devices = []
            for index in range(count):
                p = torch.cuda.get_device_properties(index)
                devices.append({'index':index,'name':str(p.name),'total_memory_bytes':p.total_memory,
                                'capability':[p.major,p.minor]})
            result.update(status='visible_cuda',devices=devices)
    except Exception:
        result['status'] = 'query_failed'
    print(json.dumps(result))
'''


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii') + b'\n'


def _identity(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def _read_source(path):
    path = Path(os.path.abspath(path))
    if any(not stat.S_ISDIR(p.lstat().st_mode) for p in path.parents):
        raise ValueError('unsafe source ancestor')
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > SOURCE_LIMIT:
        raise ValueError('unsafe source file')
    sig = lambda s: (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if sig(os.fstat(fd)) != sig(before):raise ValueError('source replaced')
        with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(SOURCE_LIMIT+1)
        if (len(raw)>SOURCE_LIMIT or sig(os.fstat(fd))!=sig(before)
                or sig(path.lstat())!=sig(before)):raise ValueError('source changed')
        return raw
    finally:os.close(fd)


def source_snapshot():
    return {name:_identity(_read_source(ROOT/name)) for name in SOURCE_FILES}


def visibility_class():
    value=os.environ.get('CUDA_VISIBLE_DEVICES')
    if value is None:return 'unset'
    if value=='':return 'empty'
    if value=='-1':return 'minus-one'
    if len(value)<=256 and re.fullmatch(r'[0-9]+(?:,[0-9]+)*',value):return 'index-list'
    return 'other-redacted'


def _terminate(process):
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    try:process.wait(timeout=1)
    except subprocess.TimeoutExpired:pass


def bounded_process(argv, *, timeout=TIMEOUT, limit=OUTPUT_LIMIT):
    """Drain both pipes concurrently with a combined byte cap; no raw errors out.

    Linux process groups prevent a child retaining inherited pipes indefinitely.
    This is not a sandbox for a hostile executable that escapes its process group.
    """
    if not 0 < timeout <= TIMEOUT or type(limit) is not int or not 0 < limit <= OUTPUT_LIMIT:
        raise ValueError('invalid fixed process bounds')
    if sys.platform!='linux':
        return {'status':'unsupported_platform','returncode':None,'stdout':b'', 'stderr_seen':False}
    start=time.monotonic(); process=None; selector=selectors.DefaultSelector(); finished=False
    output=bytearray(); count=0; stderr_seen=False; status='completed'; code=None
    try:
        try:
            process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,cwd='/',shell=False,close_fds=True,start_new_session=True)
        except (OSError,ValueError,subprocess.SubprocessError):
            return {'status':'spawn_failed','returncode':None,'stdout':b'', 'stderr_seen':False}
        for stream in (process.stdout,process.stderr):
            os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ)
        while selector.get_map():
            remaining=timeout-(time.monotonic()-start)
            if remaining<=0:status='timed_out';break
            for key,_ in selector.select(min(remaining,.1)):
                chunk=os.read(key.fileobj.fileno(),4096)
                if not chunk:selector.unregister(key.fileobj);continue
                count+=len(chunk)
                if key.fileobj is process.stderr:stderr_seen=True
                if count>limit:status='output_limit';break
                if key.fileobj is process.stdout:output.extend(chunk)
            if status!='completed':break
        if status=='completed':
            remaining=timeout-(time.monotonic()-start)
            if remaining<=0:status='timed_out'
            else:
                try:code=process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:status='timed_out'
        if status!='completed':_terminate(process)
        elif code!=0:status='command_failed'
        finished=True
        return {'status':status,'returncode':code,'stdout':bytes(output) if status=='completed' else b'',
                'stderr_seen':stderr_seen}
    finally:
        selector.close()
        if process is not None:
            if not finished or process.poll() is None:_terminate(process)
            for stream in (process.stdout,process.stderr):
                if stream is not None:stream.close()


def _version(value):
    if type(value) is not str or not re.fullmatch(r'[0-9][A-Za-z0-9.+_-]{0,79}',value):
        raise ValueError('invalid version')
    return value


def _name(value):
    if (type(value) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,95}',value)
            or re.search(r'(?:GPU-|[a-f0-9]{8}-[a-f0-9]{4}-|[a-f0-9]{24})',value,re.I)):
        raise ValueError('invalid device name')
    return value


def _integer(value,low,high):
    if type(value) is not int or not low<=value<=high:raise ValueError('invalid numeric field')
    return value


def _parse(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate child field')
            result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError()))


def _public_process(result):
    return {'status':result['status'],'returncode':result['returncode'],
            'stderr_observed':result['stderr_seen']}


def _python():
    # Do not resolve an ordinary venv bin/python symlink to a different interpreter.
    if not os.path.isabs(sys.executable):raise ValueError('absolute interpreter required')
    return sys.executable


def metadata_query():
    result=bounded_process([_python(),'-I','-B','-c',METADATA_CODE])
    receipt=_public_process(result)
    if result['status']!='completed':return receipt
    try:
        data=_parse(result['stdout'])
        if type(data) is not dict or set(data)!={'torch','numpy','safetensors'}:raise ValueError()
        sanitized={}
        for name,value in data.items():
            if value=={'status':'absent'} or value=={'status':'metadata_error'}:sanitized[name]=value
            elif type(value) is dict and set(value)=={'status','version'} and value['status']=='present':
                sanitized[name]={'status':'present','version':_version(value['version'])}
            else:raise ValueError()
        receipt['distributions']=sanitized
    except (ValueError,TypeError,KeyError,UnicodeError):receipt['status']='invalid_response'
    return receipt


def _trusted_smi():
    """Fixed system executable only, no PATH/cwd search or user-selected binary."""
    path=Path(SMI)
    try:
        # Root-owned symlink targets are allowed only through root-controlled paths.
        for current in (path,*path.parents):
            s=current.lstat()
            if s.st_uid!=0 or (not stat.S_ISLNK(s.st_mode) and s.st_mode & 0o022):
                return ('unsafe_executable',None)
        resolved=path.resolve(strict=True)
        for current in (resolved,*resolved.parents):
            s=current.lstat()
            if s.st_uid!=0 or s.st_mode & 0o022:return ('unsafe_executable',None)
        s=resolved.stat()
        if not stat.S_ISREG(s.st_mode) or not s.st_mode & 0o111:return ('unsafe_executable',None)
    except FileNotFoundError:return ('tool_unavailable',None)
    except (OSError,RuntimeError):return ('unsafe_executable',None)
    return ('available',str(resolved))


def smi_query():
    if sys.platform!='linux':return {'status':'unsupported_platform'}
    availability, executable=_trusted_smi()
    if availability!='available':return {'status':availability}
    # Execute the checked root-controlled target, not a mutable symlink chain.
    result=bounded_process([executable,*SMI_ARGUMENTS]);receipt=_public_process(result)
    if result['status']!='completed':return receipt
    try:
        rows=list(csv.reader(io.StringIO(result['stdout'].decode('ascii')),strict=True))
        if not 1<=len(rows)<=MAX_DEVICES:raise ValueError()
        devices=[]
        for row in rows:
            if len(row)!=3:raise ValueError()
            driver,name,memory=(x.strip() for x in row)
            if not re.fullmatch(r'[0-9]{1,9}',memory):raise ValueError()
            devices.append({'driver_version':_version(driver),'name':_name(name),
                            'total_memory_mib':_integer(int(memory),1,1024*1024)})
        receipt['devices']=devices
    except (ValueError,TypeError,UnicodeError,csv.Error):receipt['status']='invalid_response'
    return receipt


def cuda_query():
    result=bounded_process([_python(),'-I','-B','-c',TORCH_CODE]);receipt=_public_process(result)
    if result['status']!='completed':return receipt
    try:
        data=_parse(result['stdout'])
        if data=={'status':'torch_import_failed'}:
            receipt['status']='torch_unavailable';return receipt
        if type(data) is not dict or data.get('status') not in ('visible_cuda','no_visible_cuda','query_failed','invalid_response'):
            raise ValueError()
        allowed={'status','torch_version','compiled_cuda','available','device_count','devices'}
        if set(data)-allowed:raise ValueError()
        values={'status':data['status'],'torch_version':_version(data['torch_version']),
                'compiled_cuda':None if data['compiled_cuda'] is None else _version(data['compiled_cuda'])}
        if data['status'] in ('visible_cuda','no_visible_cuda'):
            if type(data['available']) is not bool:raise ValueError()
            count=_integer(data['device_count'],0,MAX_DEVICES)
            values.update(available=data['available'],device_count=count)
            if data['status']=='visible_cuda':
                if not data['available'] or not count or type(data['devices']) is not list or len(data['devices'])!=count:raise ValueError()
                devices=[]
                for index,p in enumerate(data['devices']):
                    if type(p) is not dict or set(p)!={'index','name','total_memory_bytes','capability'}:raise ValueError()
                    if type(p['index']) is not int or p['index']!=index:raise ValueError()
                    cap=p['capability']
                    if type(cap) is not list or len(cap)!=2:raise ValueError()
                    devices.append({'index':index,'name':_name(p['name']),
                        'total_memory_bytes':_integer(p['total_memory_bytes'],1,1024**4),
                        'capability':[_integer(x,0,99) for x in cap]})
                values['devices']=devices
            elif data['available'] and count:raise ValueError()
        receipt.update(values)
    except (ValueError,TypeError,KeyError,UnicodeError):receipt['status']='invalid_response'
    return receipt


def diagnose(query_cuda=False):
    if type(query_cuda) is not bool:raise ValueError('explicit query selector required')
    before=source_snapshot()
    result={'schema':'trinite.cuda-availability.v1','source_files':before,
        'platform':'linux' if sys.platform=='linux' else 'unsupported',
        'python':{'version':'.'.join(str(v) for v in sys.version_info[:3]),
                  'implementation':sys.implementation.name if sys.implementation.name in ('cpython','pypy') else 'other',
                  'virtual_environment':sys.prefix!=sys.base_prefix,
                  'executable_symlink':Path(_python()).is_symlink(),
                  'interpreter_selection':'same absolute lexical executable; path withheld'},
        'cuda_visible_devices':visibility_class(),'metadata':metadata_query(),
        'nvidia_smi':smi_query(),'cuda_query':cuda_query() if query_cuda else {'status':'not_requested'},
        'limits':{'seconds_per_query':TIMEOUT,'combined_output_bytes_per_query':OUTPUT_LIMIT,
                  'max_queries':3 if query_cuda else 2,'max_devices':MAX_DEVICES},
        'root_cause':'unknown','hardware_attested':False,'gpu_health_established':False,'trial_admission_established':False,
        'model_executed':False,'tensor_allocation_requested':False,'training_performed':False,
        'repair_attempted':False,'driver_initialization_possible':True,
        'scope':'local observations only; no explanation of an earlier failure or proof of hardware health'}
    if source_snapshot()!=before:raise ValueError('diagnostic source changed')
    return {'report':result,'report_identity':_identity(_json(result))}


class _Parser(argparse.ArgumentParser):
    def error(self,message):raise ValueError('invalid diagnostic arguments')


def main(argv=None):
    p=_Parser(description=__doc__)
    p.add_argument('--query-cuda',action='store_true',help='explicitly allow isolated Torch CUDA metadata queries; may initialize driver')
    try:
        args=p.parse_args(argv)
        print(_json(diagnose(args.query_cuda)).decode('ascii'),end='')
        return 0
    except (ValueError,OSError,RuntimeError,TypeError):
        print('{"status":"diagnostic_failed","root_cause":"unknown"}',file=sys.stderr)
        return 1
