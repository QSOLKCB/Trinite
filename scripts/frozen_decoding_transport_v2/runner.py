"""New transport version around the immutable v1 scientific producer."""
import argparse
from pathlib import Path
import resource
import time
import uuid
import sys
import torch
from trinite.contracts import ContractError, identity, json_bytes
from frozen_decoding_v1.experiment import report, request as original_request, ROOT
from run_frozen_decoding import read
from .artifacts import publish, sidecar, write_all


def sources():
    paths=list((ROOT/'scripts/frozen_decoding_transport_v2').glob('*.py'))
    paths+=list((ROOT/'tests/frozen_decoding_transport_v2').glob('*.py'))
    paths+=[ROOT/'scripts/run_frozen_decoding_v2.py',ROOT/'.github/workflows/frozen-decoding-transport-v2.yml']
    return {p.relative_to(ROOT).as_posix():identity(p.read_bytes()) for p in sorted(paths)}


def request():
    return {'schema':'trinite.frozen-decoding-transport-request.v2','transport_source':sources(),
            'inference_request':original_request(),
            'publication':'same-directory complete staging, fsync, atomic no-clobber hard link; full-name unique-attempt sidecars',
            'partial_records':'canonical json_bytes already includes one LF; completed cells retained on failure'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run','verify'))
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--request-identity')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--expected-identity')
    args=parser.parse_args(argv)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    try:
        binding=request();expected=json_bytes(binding)
        if args.action=='freeze':
            publish(args.request,expected);print(identity(expected));return 0
        if not args.output or not args.request_identity:
            raise ContractError('run/replay requires bound request and output')
        raw=read(args.request,524288)
        if identity(raw)!=args.request_identity or raw!=expected:
            raise ContractError('v2 transport/inference request mismatch')
        started=time.monotonic()
        if args.action=='run':
            # Early rejection is only a convenience; publish enforces the race-safe guard.
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError('output already exists')
            attempt='.attempt-'+uuid.uuid4().hex
            cells=sidecar(args.output,attempt+'.cells.jsonl')
            timing_path=sidecar(args.output,attempt+'.timings.json')
            if timing_path.exists() or timing_path.is_symlink():
                raise FileExistsError('timing sidecar already exists')
            timings=[]
            with cells.open('xb') as partial:
                print(json_bytes({'attempt_cells_path':str(cells),'attempt_timings_path':str(timing_path)}).decode(),end='',file=sys.stderr)
                def retain(cell):
                    record=json_bytes(cell)  # Already exactly one trailing LF.
                    if partial.tell()+len(record)>16_777_216:
                        raise ContractError('partial records exceed bound')
                    write_all(partial,record);partial.flush()
                inference=report(on_cell=retain,timings=timings)
            timing=json_bytes({'scope':'observed serial latency; excluded from exact replay; no speed claim','items':timings})
        else:
            inference=report()
        if request()!=binding:
            raise ContractError('transport/inference bindings changed during execution')
        actual=json_bytes({'schema':'trinite.frozen-decoding-transport-report.v2','request':binding,'inference_report':inference})
        if len(actual)>16_777_216:raise ContractError('report exceeds bound')
        if args.action=='verify':
            old=read(args.output,16_777_216)
            if identity(old)!=args.expected_identity or actual!=old:
                raise ContractError('v2 report identity or inference replay mismatch')
        else:
            publish(timing_path,timing)
            publish(args.output,actual)
        print(identity(actual))
        if args.action=='run':
            print(json_bytes({'cells_path':str(cells),'timings_path':str(timing_path)}).decode(),end='')
        print(json_bytes({'wall_seconds_hex':(time.monotonic()-started).hex(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          'scope':'whole process; inference only, not matched compute'}).decode(),end='')
        return 0
    except (ContractError,OSError,ValueError) as error:
        print(str(error));return 1

if __name__=='__main__':raise SystemExit(main())
