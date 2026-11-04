# SPDX-License-Identifier: MPL-2.0
"""One-shot owner-local dispatch with an independent whole-process supervisor."""
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from trinite.contracts import ContractError, identity, json_bytes
from . import custody as io, spec


def previous_resolved(request, grants=None):
    seeds=(17,29,43)
    root=Path(request['experiment_directory'])
    for seed in seeds[:seeds.index(request['seed'])]:
        out=root/f'seed-{seed}'
        status=io.load(out/'supervisor.json')
        if (status.get('process_resolved') is not True or
                (type(status.get('child_returncode')) is not int and status.get('process_started') is not False)):
            raise ContractError('preceding attempt process state unresolved')
        anchor=io.load(root/f'seed-{seed}-anchor.json')['manifest_identity']
        manifest=io.verify_seal(out,anchor)
        if manifest.get('seed')!=seed or manifest.get('status')!=status['status']:
            raise ContractError('preceding attempt custody mismatch')
        if grants is not None and manifest.get('request_identity')!=grants[str(seed)]['request_identity']:
            raise ContractError('preceding attempt belongs to a different frozen request')


def group_stopped(pgid):
    """Linux process-group observation: a reaped leader alone is insufficient."""
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
            belongs=int(fields[2])==pgid
        except FileNotFoundError:
            continue  # A process that exited during enumeration.
        except (OSError,IndexError,ValueError):
            return False
        if belongs and fields[0] not in ('Z','X'):
            return False
    return True


def resolve_group(proc):
    if group_stopped(proc.pid):return True
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        if group_stopped(proc.pid):return True
        time.sleep(.05)
    return False


def dispatch(request_path, request_anchor, dataset, approval_path, approval_anchor,
             approvals_path=None, approvals_anchor=None):
    start=time.monotonic()
    if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
        raise ContractError('target FIT execution is owner-local only; CI forbidden')
    request=spec.verify_request(request_path,request_anchor,dataset)
    spec.approval(approval_path,approval_anchor,request,request_anchor)
    grants,records=spec.approval_set(approvals_path,approvals_anchor,request,dataset)
    if grants[str(request['seed'])]['identity']!=approval_anchor:
        raise ContractError('current approval differs from the three-approval inventory')
    previous_resolved(request,grants)
    copies=[]
    for name,path,anchor in (('request.json',request_path,request_anchor),
                            ('approval.json',approval_path,approval_anchor)):
        raw=io.read(path)
        if identity(raw)!=anchor:
            raise ContractError('request/approval changed at dispatch boundary')
        copies.append((name,raw))
    # The frozen absolute output path is the durable single-use claim. It is
    # never removed, including when process creation or approval recheck fails.
    out=io.fresh(request['output_directory'])
    proc=None; interrupted=False; timed_out=False; resolved=True; parent_error=None
    try:
        for name,raw in copies:
            io.write(out/name,raw)
        io.record(out/'started.json',{'schema':spec.SCHEMA+'.attempt','seed':request['seed'],
            'request_identity':request_anchor,'approval_identity':approval_anchor,
            'supervisor_pid':os.getpid(),'started_unix_ns':str(time.time_ns()),
            'retry_allowed':False,'outer_seconds':900})
        io.record(out/'approval-set.json',{'inventory':grants,'records':records,
            'inventory_identity':approvals_anchor})
        remaining=900-(time.monotonic()-start)
        if remaining<=0:
            raise ContractError('whole-process budget exhausted during preflight')
        read_fd,write_fd=os.pipe()
        try:
            os.write(write_fd,json_bytes({'supervisor_pid':os.getpid(),'request_identity':request_anchor}))
        finally:
            os.close(write_fd)
        def limits():
            resource.setrlimit(resource.RLIMIT_AS,(spec.LIMITS['address_space_bytes'],)*2)
            resource.setrlimit(resource.RLIMIT_FSIZE,(io.MAX_FILE,)*2)
        command=[sys.executable,'-I','-B',str(spec.ROOT/'scripts/run_teacher_fit.py'),
            '_worker','--request',str(out/'request.json'),'--identity',request_anchor,
            '--dataset',str(Path(dataset).resolve()),'--approval',str(out/'approval.json'),
            '--approval-identity',approval_anchor,'--capability-fd',str(read_fd)]
        try:
            with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
                proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,
                    start_new_session=True,pass_fds=(read_fd,),preexec_fn=limits,cwd=spec.ROOT)
                io.record(out/'process.json',{'pid':proc.pid,'process_group':proc.pid,'command':command})
                try:
                    proc.wait(timeout=max(.001,900-(time.monotonic()-start)))
                except subprocess.TimeoutExpired:
                    timed_out=True
                except KeyboardInterrupt:
                    interrupted=True
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGKILL)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            resolved=False
        finally:
            os.close(read_fd)
        if spec.source()!=request['source']:
            raise ContractError('source drift during execution; outcome retained')
    except BaseException as exc:
        parent_error={'type':type(exc).__name__,'message':str(exc)[:2000]}
        if isinstance(exc,KeyboardInterrupt):
            interrupted=True
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
            except (ProcessLookupError,subprocess.TimeoutExpired):
                resolved=proc.poll() is not None
    child_code=None if proc is None else proc.returncode
    if proc is not None:
        try:
            resolved=resolved and resolve_group(proc)
        except BaseException as exc:
            resolved=False
            parent_error={'type':type(exc).__name__,'message':str(exc)[:2000]}
    completed=(out/'completed.json').is_file()
    status=('COMPLETED' if resolved and child_code==0 and completed and time.monotonic()-start<=900 and not
            (timed_out or interrupted or parent_error) else 'FAILED')
    receipt={'schema':spec.SCHEMA+'.supervisor','status':status,'seed':request['seed'],
        'child_returncode':child_code,'process_started':proc is not None,'process_resolved':resolved,
        'timed_out':timed_out,'interrupted':interrupted,'parent_error':parent_error,
        'elapsed_seconds_hex':(time.monotonic()-start).hex(),
        'supervisor_exit_status':0 if status=='COMPLETED' else 124 if timed_out else 130 if interrupted else 1}
    io.record(out/'supervisor.json',receipt)
    anchor=io.seal(out,{'schema':spec.SCHEMA+'.attempt-manifest','seed':request['seed'],
        'status':status,'request_identity':request_anchor,'approval_identity':approval_anchor})
    io.record(Path(request['experiment_directory'])/f"seed-{request['seed']}-anchor.json",{'manifest_identity':anchor})
    return dict(receipt,manifest_identity=anchor)


def worker(args):
    """No direct public execution command; require the live supervisor descriptor."""
    from trinite.contracts import parse_json
    raw=os.read(args.capability_fd,4097);os.close(args.capability_fd)
    capability=parse_json(raw,canonical=True)
    if capability!={'supervisor_pid':os.getppid(),'request_identity':args.identity}:
        raise ContractError('live matching supervisor required')
    request=spec.verify_request(args.request,args.identity,args.dataset)
    spec.approval(args.approval,args.approval_identity,request,args.identity)
    out=Path(request['output_directory'])
    if (Path(args.request)!=out/'request.json' or Path(args.approval)!=out/'approval.json'
            or io.load(out/'started.json')['supervisor_pid']!=os.getppid()):
        raise ContractError('worker attempt custody mismatch')
    saved_set=io.load(out/'approval-set.json')
    spec.verify_approval_records(saved_set,request,args.dataset)
    # Revalidate the whole set at the actual execution boundary, before model import.
    for seed,grant in saved_set['inventory'].items():
        req=spec.verify_request(Path(request['experiment_directory'])/'requests'/f'seed-{seed}.json',
            grant['request_identity'],args.dataset)
        if spec.approval(grant['path'],grant['identity'],req,grant['request_identity'])!=saved_set['records'][seed]:
            raise ContractError('owner approval changed before execution')
    # Only after source, request and separate owner approval validation may the
    # worker import the native model implementation or allocate a model.
    from .native import execute
    execute(request,args.identity,out)
