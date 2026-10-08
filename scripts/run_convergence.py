"""Explicit bounded fresh-process training diagnostics; no held-out stage."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def launch(workload, seed, candidate, request, request_identity, output, log):
    # Executable and module/script are fixed; only validated cell names and data
    # paths reach argv. No shell, arbitrary executable or user-supplied command.
    if workload not in ('arithmetic', 'fold', 'lattice') or seed not in (0, 1, 2) or candidate not in ('low', 'reference', 'high'):
        raise ValueError('invalid worker cell')
    return subprocess.run([sys.executable, str(ROOT/'scripts/run_convergence.py'), '--worker',
                           '--workload', workload, '--seed', str(seed), '--candidate', candidate,
                           '--request', str(request), '--request-identity', request_identity,
                           '--output', str(output)], cwd=ROOT,
                          env=dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='1',
                                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
                          shell=False, stdout=log, stderr=subprocess.STDOUT, timeout=240)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-request', type=Path)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--request-identity')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--workload', choices=('arithmetic', 'fold', 'lattice'))
    parser.add_argument('--seed', type=int, choices=(0, 1, 2))
    parser.add_argument('--candidate', choices=('low', 'reference', 'high'))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    from trinite.contracts import ContractError, identity, json_bytes, parse_json
    from trinite.convergence import (RUNNER_IDENTITY, cells, cell_name, freeze_request,
                                     read_request, train_cell, summarize, read_bytes)
    try:
        if identity(Path(__file__).read_bytes()) != RUNNER_IDENTITY:
            raise ContractError('convergence runner source mismatch')
        if args.freeze_request:
            if args.request or args.output or args.worker or args.verify_only:
                raise ContractError('freeze cannot execute/verify workers')
            print(json_bytes(freeze_request(args.freeze_request)).decode(), end=''); return 0
        if not args.request or not args.request_identity or not args.output:
            raise ContractError('request, identity and output are required')
        args.request = args.request.resolve(); args.output = args.output.resolve()
        read_request(args.request, args.request_identity)
        if args.worker:
            if args.verify_only or args.workload is None or args.seed is None or args.candidate is None:
                raise ContractError('worker requires one complete cell')
            train_cell(args.output, args.workload, args.seed, args.candidate, args.request, args.request_identity)
            return 0
        if args.verify_only:
            old = parse_json(read_bytes(args.output/'summary.json'), canonical=True)
            current = summarize(args.output, args.request, args.request_identity, old['worker_errors'], publish=False)
            if json_bytes(old) != json_bytes(current) or current['outcome'] != 'completed':
                raise ContractError('convergence summary does not match verified completed run')
            print(json_bytes({'integrity_verified': True, 'summary_identity': identity(json_bytes(old)),
                              'held_out_scored': False}).decode(), end=''); return 0
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'request.json').write_bytes(args.request.read_bytes())
        errors = {}
        for workload, seed, candidate in cells():
            name = cell_name(workload, seed, candidate)
            with (args.output/(name+'.log')).open('x') as log:
                try:
                    result = launch(workload, seed, candidate, args.request, args.request_identity,
                                    args.output/name, log)
                    if result.returncode:
                        errors[name] = f'worker exited {result.returncode}'
                except subprocess.TimeoutExpired:
                    errors[name] = 'worker exceeded 240 seconds'
                except OSError as error:
                    errors[name] = type(error).__name__+': '+str(error)
            print(name, errors.get(name, 'completed'), flush=True)
        result = summarize(args.output, args.request, args.request_identity, errors)
        print(json_bytes({'outcome': result['outcome'], 'summary_identity': identity(json_bytes(result)),
                          'held_out_scored': False}).decode(), end='')
        return 0 if result['outcome'] == 'completed' else 1
    except (ContractError, OSError) as error:
        print(type(error).__name__+': '+str(error), file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
