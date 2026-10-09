"""Bounded fresh-process matched training, gated evaluation and verification."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def launch(stage, request, request_identity, output, log, cell=None):
    if stage not in ('train', 'evaluate', 'summarize', 'verify'):
        raise ValueError('unknown scalar_learning worker stage')
    arguments = []
    if stage == 'train':
        if (type(cell) is not tuple or len(cell) != 3 or cell[0] not in ('arithmetic', 'fold', 'lattice')
                or type(cell[1]) is not int or cell[1] not in (0, 1, 2)
                or cell[2] not in ('dense', 'ternary', 'four-state')):
            raise ValueError('invalid scalar_learning worker cell')
        arguments = ['--workload', cell[0], '--seed', str(cell[1]), '--lane', cell[2]]
    elif cell is not None:
        raise ValueError('matrix worker cannot select a cell')
    # Fixed interpreter/script, closed selectors and separate argv data. No
    # shell or externally chosen executable, module, option or command string.
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit, dangerous-subprocess-use-audit
    return subprocess.run([sys.executable, str(ROOT/'scripts/run_scalar_learning.py'), '--worker', stage,
                           '--request', str(request), '--request-identity', request_identity,
                           '--output', str(output), *arguments], cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='1',
                 MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), shell=False,
        stdout=log, stderr=subprocess.STDOUT, timeout=600 if stage == 'train' else 1200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-request', type=Path)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--request-identity')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--worker', choices=('train', 'evaluate', 'summarize', 'verify'))
    parser.add_argument('--workload', choices=('arithmetic', 'fold', 'lattice'))
    parser.add_argument('--seed', type=int, choices=(0, 1, 2))
    parser.add_argument('--lane', choices=('dense', 'ternary', 'four-state'))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    from trinite.contracts import ContractError, identity, json_bytes, parse_json
    from trinite.scalar_learning import (cells, cell_name, freeze_request,
        read_request, train_cell, evaluate_matrix, summarize, failed_summary, read_bytes, write)
    try:
        selectors = (args.workload, args.seed, args.lane)
        if args.freeze_request:
            if args.request or args.request_identity or args.output or args.worker or args.verify_only or any(s is not None for s in selectors):
                raise ContractError('freeze cannot execute/verify workers')
            print(json_bytes(freeze_request(args.freeze_request)).decode(), end=''); return 0
        if not args.request or not args.request_identity or not args.output:
            raise ContractError('request, identity and output are required')
        for path in (args.request, args.output):
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ContractError('scalar paths cannot traverse symlinks')
        args.request = args.request.resolve(); args.output = args.output.resolve()
        if args.output == args.request or args.output in args.request.parents:
            raise ContractError('scalar output overlaps its immutable request')
        read_request(args.request, args.request_identity)
        if args.worker:
            if args.verify_only:
                raise ContractError('worker cannot verify retained summaries')
            if args.worker == 'train':
                if any(s is None for s in selectors):
                    raise ContractError('training requires one complete cell')
                train_cell(args.output, args.workload, args.seed, args.lane, args.request, args.request_identity)
            else:
                if any(s is not None for s in selectors):
                    raise ContractError('matrix worker cannot select a cell')
                if args.worker == 'evaluate':
                    errors = parse_json(read_bytes(args.output/'training-worker-errors.json'), canonical=True)
                    evaluate_matrix(args.output, args.request, args.request_identity, errors)
                elif args.worker == 'verify':
                    if read_bytes(args.output/'request.json') != read_bytes(args.request):
                        raise ContractError('retained scalar request mismatch')
                    old = parse_json(read_bytes(args.output/'summary.json'), canonical=True)
                    current = summarize(args.output, args.request, args.request_identity, old['worker_errors'], publish=False)
                    if json_bytes(old) != json_bytes(current):
                        raise ContractError('scalar summary differs from fresh verification')
                    print(json_bytes({'integrity_verified': True, 'outcome': current['outcome'],
                                      'summary_identity': identity(json_bytes(old))}).decode(), end='')
                else:
                    errors = parse_json(read_bytes(args.output/'worker-errors.json'), canonical=True)
                    summarize(args.output, args.request, args.request_identity, errors)
            # A retained failed/blocked scientific result is a successful worker
            # invocation. The parent reads the result to set its overall status.
            return 0
        if any(s is not None for s in selectors):
            raise ContractError('matrix invocation cannot select one cell')
        if args.verify_only:
            try:
                return launch('verify', args.request, args.request_identity, args.output, None).returncode
            except subprocess.TimeoutExpired as error:
                raise ContractError('read-only scalar verification exceeded 1200 seconds') from error
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'request.json').write_bytes(read_bytes(args.request))
        errors = {}
        def execute(stage, name, output, cell=None):
            with (args.output/(name+'.log')).open('x') as log:
                try:
                    result = launch(stage, args.request, args.request_identity, output, log, cell)
                    if result.returncode:
                        errors[name] = f'{stage} worker exited {result.returncode}'
                except subprocess.TimeoutExpired:
                    errors[name] = f'{stage} worker exceeded {600 if stage == "train" else 1200} seconds'
                except OSError as error:
                    errors[name] = type(error).__name__+': '+str(error)
            print(name, errors.get(name, 'completed'), flush=True)
        for cell in cells():
            name = cell_name(*cell); execute('train', name, args.output/name, cell)
        write(args.output, 'training-worker-errors.json', errors)
        execute('evaluate', 'evaluation-matrix', args.output)
        if (args.output/'evaluation-errors.json').exists():
            try:
                evaluation_errors = parse_json(read_bytes(args.output/'evaluation-errors.json'), canonical=True)
                from trinite.scalar_learning import validate_errors
                validate_errors(evaluation_errors, training=True)
                errors.update(evaluation_errors)
            except (ContractError, OSError, TypeError, ValueError) as error:
                errors['evaluation-matrix'] = type(error).__name__+': '+str(error)
        write(args.output, 'worker-errors.json', errors)
        execute('summarize', 'summary', args.output)
        if 'summary' in errors or not (args.output/'summary.json').is_file():
            errors.setdefault('summary', 'summary worker did not publish a result')
            result = failed_summary(args.output, args.request_identity, errors)
        else:
            result = parse_json(read_bytes(args.output/'summary.json'), canonical=True)
        print(json_bytes({'outcome': result['outcome'], 'summary_identity': identity(json_bytes(result))}).decode(), end='')
        return 0 if result['outcome'] in ('completed', 'blocked') else 1
    except (ContractError, OSError, KeyError, TypeError, ValueError) as error:
        print(type(error).__name__+': '+str(error), file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
