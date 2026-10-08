"""Compare full standalone CPU suites with grouping; preserve test IDs/order."""
import argparse
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

from trinite.contracts import json_bytes

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('benchmark output already exists')
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    samples={'separate':[],'combined':[]}
    with tempfile.TemporaryDirectory() as directory:
        for trial in range(3):
            inventories={}
            for mode in (('separate','combined') if trial%2==0 else ('combined','separate')):
                commands=([[sys.executable,'-m','unittest','discover','-s','tests/'+d,'-v']
                           for d in ('model','training','geometry')] if mode=='separate'
                          else [[sys.executable,'scripts/check_cpu.py']])
                start=time.perf_counter();tests=[]
                for i,command in enumerate(commands):
                    log=Path(directory)/f'{trial}-{mode}-{i}.log'
                    with log.open('w') as out:
                        p=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=120)
                    if p.returncode:raise RuntimeError('CPU benchmark failed: '+log.read_text())
                    for line in log.read_text().splitlines():
                        if line.startswith('test_') and ' ... ' in line:tests.append(line.split(' ... ')[0])
                elapsed=time.perf_counter()-start;samples[mode].append(elapsed)
                inventories[mode]=tests
                print(f'trial {trial+1}/3 {mode}: {elapsed:.3f}s; {len(tests)} tests',flush=True)
            if not inventories['separate'] or inventories['separate']!=inventories['combined']:
                raise AssertionError('test ID/order parity failed')
    from trinite.training import environment
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    result={'schema':'trinite.cpu-benchmark.v1','repetitions':3,
            'separate_seconds_hex':[x.hex() for x in samples['separate']],
            'combined_seconds_hex':[x.hex() for x in samples['combined']],
            'separate_median_seconds_hex':statistics.median(samples['separate']).hex(),
            'combined_median_seconds_hex':statistics.median(samples['combined']).hex(),
            'median_speedup_hex':(statistics.median(samples['separate'])/statistics.median(samples['combined'])).hex(),
            'test_count':len(tests),'test_ids':tests,
            'scope':'full suites/current optimized sources; fresh invocation per mode; explicit replay subprocesses retained; acquisition excluded',
            'environment':environment()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as out:out.write(json_bytes(result))


if __name__=='__main__':main()
