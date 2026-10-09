"""Fixed fresh workers for arithmetic rehearsal pilot."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def launch(stage, request, request_identity, output, log, cell=None):
    if stage not in ('train', 'summarize', 'verify'):
        raise ValueError('unknown retention worker')
    arguments = []
    if stage == 'train':
        if (type(cell) is not tuple or len(cell) != 2 or type(cell[0]) is not int
                or cell[0] not in (0, 1) or type(cell[1]) is not str or cell[1] not in ('pooled', 'rehearsal')):
            raise ValueError('invalid retention cell')
        arguments = ['--seed', str(cell[0]), '--profile', str(cell[1])]
    elif cell is not None: raise ValueError('summary cannot select a cell')
    # Closed argv, fixed interpreter/script and no shell or external executable.
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit, dangerous-subprocess-use-audit
    return subprocess.run([sys.executable, str(ROOT/'scripts/run_maths_retention.py'), '--worker', stage,
        '--request', str(request), '--request-identity', request_identity, '--output', str(output), *arguments],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), shell=False,
        stdout=log, stderr=subprocess.STDOUT, timeout=1200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-request', type=Path)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--request-identity')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--worker', choices=('train', 'summarize', 'verify'))
    parser.add_argument('--seed', type=int, choices=(0, 1))
    parser.add_argument('--profile', choices=('pooled', 'rehearsal'))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    from trinite.maths_retention import (ContractError, cells, cell_name, freeze_request, read_request,
        train_cell, summarize, read_bytes, write, parse_json, json_bytes, identity)
    try:
        selectors = (args.seed, args.profile)
        if args.freeze_request:
            if args.request or args.request_identity or args.output or args.worker or args.verify_only or any(s is not None for s in selectors):
                raise ContractError('freeze cannot run or verify workers')
            if any(p.is_symlink() for p in (args.freeze_request, *args.freeze_request.parents)):
                raise ContractError('request path cannot traverse symlinks')
            print(json_bytes(freeze_request(args.freeze_request)).decode(), end=''); return 0
        if not args.request or not args.request_identity or not args.output:
            raise ContractError('request, identity and output are required')
        for path in (args.request, args.output):
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ContractError('retention paths cannot traverse symlinks')
        args.request = args.request.resolve(); args.output = args.output.resolve()
        if args.output == args.request or args.output in args.request.parents:
            raise ContractError('output overlaps immutable request')
        read_request(args.request, args.request_identity)
        if args.worker:
            if args.verify_only: raise ContractError('worker cannot select verify-only')
            if args.worker == 'train':
                if any(s is None for s in selectors): raise ContractError('training requires complete cell')
                train_cell(args.output, *selectors, args.request, args.request_identity)
            else:
                if any(s is not None for s in selectors): raise ContractError('summary cannot select a cell')
                errors = parse_json(read_bytes(args.output/'worker-errors.json'), canonical=True)
                current = summarize(args.output, args.request, args.request_identity, errors,
                                    publish=args.worker == 'summarize')
                if args.worker == 'verify':
                    if read_bytes(args.output/'summary.json') != json_bytes(current):
                        raise ContractError('retained summary differs from fresh verification')
                    if current['outcome'] != 'completed': raise ContractError('matrix contains failed/unverified cells')
                    print(json_bytes({'integrity_verified': True, 'summary_identity': identity(json_bytes(current))}).decode(), end='')
            return 0
        if any(s is not None for s in selectors): raise ContractError('matrix cannot select a cell')
        if args.verify_only:
            try: return launch('verify', args.request, args.request_identity, args.output, None).returncode
            except subprocess.TimeoutExpired as error: raise ContractError('verification exceeded 1200 seconds') from error
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'request.json').write_bytes(read_bytes(args.request))
        errors = {}
        for cell in cells():
            name = cell_name(*cell)
            with (args.output/(name+'.log')).open('x') as log:
                try:
                    r = launch('train', args.request, args.request_identity, args.output/name, log, cell)
                    if r.returncode: errors[name] = f'training worker exited {r.returncode}'
                except subprocess.TimeoutExpired: errors[name] = 'training worker exceeded 1200 seconds'
                except OSError as error: errors[name] = type(error).__name__+': '+str(error)
            print(name+': '+errors.get(name, 'completed'), flush=True)
        write(args.output, 'worker-errors.json', errors)
        with (args.output/'summary.log').open('x') as log:
            try:
                r = launch('summarize', args.request, args.request_identity, args.output, log)
                if r.returncode: errors['summary'] = f'summary worker exited {r.returncode}'
            except subprocess.TimeoutExpired: errors['summary'] = 'summary worker exceeded 1200 seconds'
            except OSError as error: errors['summary'] = type(error).__name__+': '+str(error)
        if 'summary' in errors:
            write(args.output, 'summary-worker-failure.json', {'outcome': 'failed', 'worker_errors': errors,
                'integrity_verified': False, 'held_out_scored': None, 'phase5_ready': False})
            return 1
        summary = parse_json(read_bytes(args.output/'summary.json'), canonical=True)
        print(json_bytes({'outcome': summary['outcome'], 'contrasts': summary['contrasts'],
            'summary_identity': identity(json_bytes(summary)), 'selected_candidate': None,
            'held_out_scored': summary['held_out_scored'], 'phase5_ready': False}).decode(), end='')
        return 0 if summary['outcome'] == 'completed' else 1
    except (ContractError, OSError, ValueError) as error:
        print('retention: '+str(error), file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
