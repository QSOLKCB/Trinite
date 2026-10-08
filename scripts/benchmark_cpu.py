"""Compare full standalone CPU suites with grouping; preserve test IDs/order."""
import argparse
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

from trinite.contracts import identity, json_bytes

ROOT=Path(__file__).resolve().parents[1]
THREAD_ENV={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
SCOPE='full required suites on current optimized sources; fresh process per invocation; explicit replay subprocesses retained; excludes dependency acquisition'


def run_suite(suite, *, env, out):
    # Only these fixed repo tasks may launch, using the current interpreter.
    # CLI output paths, environment values and selectors never become argv text.
    commands={
        'model':[sys.executable,'-m','unittest','discover','-s','tests/model','-v'],
        'training':[sys.executable,'-m','unittest','discover','-s','tests/training','-v'],
        'geometry':[sys.executable,'-m','unittest','discover','-s','tests/geometry','-v'],
        'combined':[sys.executable,'scripts/check_cpu.py'],
    }
    if type(suite) is not str or suite not in commands:
        raise ValueError('unknown fixed CPU benchmark suite')
    # Audited dynamic interpreter/closed argv lookup; no shell or external argv.
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit, dangerous-subprocess-use-audit
    return subprocess.run(commands[suite],shell=False,cwd=ROOT,env=env,stdout=out,
                          stderr=subprocess.STDOUT,timeout=120)


def result_record(samples, tests, environment, runner_identity):
    return {'schema':'trinite.cpu-benchmark.v1','repetitions':3,
            'separate_seconds_hex':[x.hex() for x in samples['separate']],
            'combined_seconds_hex':[x.hex() for x in samples['combined']],
            'separate_median_seconds_hex':statistics.median(samples['separate']).hex(),
            'combined_median_seconds_hex':statistics.median(samples['combined']).hex(),
            'median_speedup_hex':(statistics.median(samples['separate'])/statistics.median(samples['combined'])).hex(),
            'test_count':len(tests),'test_ids':tests,'scope':SCOPE,
            'environment':environment,'runner_source_identity':runner_identity,
            'process_thread_environment':dict(THREAD_ENV)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('benchmark output already exists')
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),**THREAD_ENV)
    runner_identity=identity((ROOT/'scripts/check_cpu.py').read_bytes())
    samples={'separate':[],'combined':[]}
    with tempfile.TemporaryDirectory() as directory:
        for trial in range(3):
            inventories={}
            for mode in (('separate','combined') if trial%2==0 else ('combined','separate')):
                suites=('model','training','geometry') if mode=='separate' else ('combined',)
                start=time.perf_counter();tests=[]
                for i,suite in enumerate(suites):
                    log=Path(directory)/f'{trial}-{mode}-{i}.log'
                    with log.open('w') as out:
                        p=run_suite(suite,env=env,out=out)
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
    if identity((ROOT/'scripts/check_cpu.py').read_bytes())!=runner_identity:
        raise AssertionError('CPU runner source changed during characterization')
    result=result_record(samples,tests,environment(),runner_identity)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as out:out.write(json_bytes(result))


if __name__=='__main__':main()
