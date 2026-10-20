"""Freeze, execute or read-only replay the inference-only frozen decoding comparison."""
import argparse
from pathlib import Path
import resource
import time
import torch
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.training import require_environment
from frozen_decoding_v1.experiment import report, request



def read(path, limit):
    if path.is_symlink() or not path.is_file():
        raise ContractError('bounded regular file required')
    with path.open('rb') as stream:
        raw=stream.read(limit+1)
    if not 1<=len(raw)<=limit:
        raise ContractError('artifact size bound exceeded')
    return raw


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
        expected=json_bytes(request())
        if args.action=='freeze':
            with args.request.open('xb') as f:f.write(expected)
            print(identity(expected));return 0
        if not args.output or not args.request_identity:
            raise ContractError('run/replay requires bound request and output')
        raw=read(args.request,262144)
        if identity(raw)!=args.request_identity or raw!=expected:
            raise ContractError('request source/environment/protocol mismatch')
        started=time.monotonic()
        if args.action == 'run':
            # Keep each completed cell if a later cell fails; never replace an old run.
            with args.output.with_suffix('.cells.jsonl').open('xb') as partial:
                def retain(cell):
                    raw_cell = json_bytes(cell)
                    if partial.tell() + len(raw_cell) > 16_777_216:
                        raise ContractError('partial decoding records exceed size bound')
                    partial.write(raw_cell);partial.flush()
                timings=[]
                actual=json_bytes(report(on_cell=retain,timings=timings))
                with args.output.with_suffix('.timings.json').open('xb') as timing_file:
                    timing_file.write(json_bytes({'scope':'observed serial per-item wall latency including syntax mask; fixed greedy-then-syntax order, no speed claim; excluded from exact replay','items':timings}))
        else:
            actual=json_bytes(report())
        if len(actual)>16_777_216:raise ContractError('decoding report exceeds size bound')
        if args.action=='verify':
            old=read(args.output,16_777_216)
            if identity(old)!=args.expected_identity or actual!=old:
                raise ContractError('decoding report identity or complete inference replay mismatch')
        else:
            with args.output.open('xb') as f:f.write(actual)
        print(identity(actual))
        print(json_bytes({'wall_seconds_hex':(time.monotonic()-started).hex(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          'scope':'whole process including inference replay; not matched compute or speed comparison'}).decode(),end='')
        return 0
    except (ContractError,OSError,ValueError) as error:
        print(str(error));return 1

if __name__=='__main__':raise SystemExit(main())
