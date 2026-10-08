"""Run every frozen comparison cell in fresh CPU processes; retain failures."""
import argparse
import itertools
import os
from pathlib import Path
import subprocess
import sys

import torch

from trinite.comparison import (LANES, WORKLOADS, SEEDS, read_request, summarize, sources)
from trinite.contracts import identity, json_bytes

ROOT = Path(__file__).resolve().parents[1]


def launch(stage, directory, workload, lane, seed, request, request_identity, out, env):
    if stage not in ('train', 'evaluate') or workload not in WORKLOADS or lane not in LANES or seed not in SEEDS:
        raise ValueError('unknown fixed comparison worker')
    # Current locked interpreter, fixed module/subcommand and validated selectors.
    # Paths and identity remain individual arguments; no shell interpretation.
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit, dangerous-subprocess-use-audit
    return subprocess.run([sys.executable, '-m', 'trinite', 'comparison-cell', str(directory),
        '--stage', stage, '--workload', workload, '--lane', lane, '--seed', str(seed),
        '--request', str(request), '--request-identity', request_identity], shell=False,
        cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-identity', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    request = args.request.resolve()
    read_request(request, args.request_identity)
    if identity(Path(__file__).read_bytes()) != sources()['scripts/run_comparison.py']:
        raise RuntimeError('comparison runner differs from frozen source receipt')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/'request.json').write_bytes(request.read_bytes())
    (output/'runner-source.json').write_bytes(json_bytes({'source_identity': identity(Path(__file__).read_bytes())}))
    runner_identity = identity(Path(__file__).read_bytes())
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    failures = {}
    # Rotate execution order by seed; every lane still consumes the same examples.
    cells = [(w, s, lane) for w, s in itertools.product(WORKLOADS, SEEDS)
             for lane in LANES[s:]+LANES[:s]]
    for stage in ('train', 'evaluate'):
        # No locked test evaluation occurs before all training cells terminate.
        for workload, seed, lane in cells:
            name = f'{workload}-{seed}-{lane}'
            if name in failures:
                continue
            with (output/(name+'-'+stage+'.log')).open('x') as out:
                try:
                    result = launch(stage, output/name, workload, lane, seed, request,
                                    args.request_identity, out, env)
                    if result.returncode:
                        failures[name] = f'{stage}: worker exit {result.returncode}; see retained log'
                except subprocess.TimeoutExpired:
                    failures[name] = stage+': worker exceeded 180 seconds'
            print(stage, name, 'failed' if name in failures else 'completed', flush=True)
    if identity(Path(__file__).read_bytes()) != runner_identity:
        failures['runner'] = 'comparison runner source changed'
    result = summarize(output, request, args.request_identity, failures)
    if 'runner' in failures:
        raise RuntimeError(failures['runner'])
    print(result['outcome'], flush=True)
    return int(result['outcome'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
