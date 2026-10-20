"""Trusted-checkout CPU launcher. Requires isolated startup, not a hostile-code auditor."""
import sys
if not sys.flags.isolated:
    raise SystemExit('Invoke with python -I scripts/run_integral_geometry.py ...')
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
sys.dont_write_bytecode = True
from trinite.contracts import ContractError, identity, json_bytes, parse_json

LIMIT = 8_388_608


def publish(path, raw):
    """Complete same-directory temporary, then atomic no-clobber hardlink."""
    path = Path(path)
    fd,name = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name,path)
    finally:
        os.unlink(name)


def read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise ContractError('regular input file required')
    with path.open('rb') as stream:
        raw=stream.read(LIMIT+1)
    if len(raw)>LIMIT:
        raise ContractError('artifact size limit exceeded')
    return raw


def numerical():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from trinite.training import require_environment
    from integral_geometry_v1.spec import request
    return request(require_environment())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run','replay','_worker'))
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--request-identity')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--expected-report',type=Path)
    parser.add_argument('--expected-identity')
    args=parser.parse_args()
    try:
        if args.action=='freeze':
            raw=json_bytes(numerical())
            publish(args.request,raw)
            print(identity(raw))
            return 0
        raw=read(args.request)
        if identity(raw)!=args.request_identity or raw!=json_bytes(numerical()):
            raise ContractError('request source/input/environment/identity mismatch')
        if args.output is None:
            raise ContractError('fresh output path required')
        if args.output.exists() or args.output.is_symlink():
            raise ContractError('output already exists; preserve it and choose a fresh path')
        if args.action=='_worker':
            from integral_geometry_v1.experiment import report
            with args.output.with_name(args.output.name+'.cells.jsonl').open('xb') as partial:
                def retain(cell):
                    encoded=json_bytes(cell)  # json_bytes includes its LF delimiter.
                    if partial.tell()+len(encoded)>LIMIT:
                        raise ContractError('partial report size bound exceeded')
                    partial.write(encoded)
                    partial.flush()
                    os.fsync(partial.fileno())
                completed=json_bytes(report(on_cell=retain))
            if len(completed)>LIMIT:
                raise ContractError('report size bound exceeded')
            publish(args.output,completed)
            return 0
        expected=None
        if args.action=='replay':
            if args.expected_report is None or args.expected_identity is None:
                raise ContractError('replay requires independently retained report identity and path')
            expected=read(args.expected_report)
            if identity(expected)!=args.expected_identity:
                raise ContractError('expected report digest mismatch')
            parsed=parse_json(expected,canonical=True)
            if type(parsed) is not dict or parsed.get('request_identity')!=args.request_identity:
                raise ContractError('expected report request differs')
        folder=args.output.with_name(args.output.name+'.attempt-'+uuid.uuid4().hex)
        folder.mkdir()
        publish(folder/'request.json',raw)
        command=[sys.executable,'-I',str(Path(__file__).resolve()),'_worker',
                 '--request',str((folder/'request.json').resolve()),'--request-identity',args.request_identity,
                 '--output',str((folder/'report.json').resolve())]
        started=time.monotonic()
        with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
            try:
                result=subprocess.run(command,stdout=out,stderr=err,timeout=900,check=False)
                outcome={'exit_code':result.returncode,'timed_out':False}
            except subprocess.TimeoutExpired:
                outcome={'exit_code':None,'timed_out':True}
        outcome['elapsed_seconds_hex']=(time.monotonic()-started).hex()
        outcome['action']=args.action
        if outcome['timed_out'] or outcome['exit_code']!=0:
            publish(folder/'status.json',json_bytes(outcome))
            raise ContractError('worker failed; retained attempt: '+str(folder))
        completed=read(folder/'report.json')
        outcome['report_identity']=identity(completed)
        outcome['exact_replay']=None if expected is None else expected==completed
        publish(folder/'status.json',json_bytes(outcome))
        if expected is not None and expected!=completed:
            raise ContractError('exact replay mismatch; all new bytes retained at '+str(folder))
        publish(args.output,completed)
        print(identity(completed))
        print(folder)
        return 0
    except (ContractError,OSError,ValueError) as error:
        print(str(error),file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
