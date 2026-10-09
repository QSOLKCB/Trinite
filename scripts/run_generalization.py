"""Bounded read-only parent verification, diagnostics and scalar preparation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-request', type=Path)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--request-identity')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    from trinite.contracts import ContractError, identity, json_bytes
    from trinite.generalization import freeze_request, read_request, read, analyze_run, protocol
    try:
        import torch
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        if args.freeze_request:
            if any((args.request, args.request_identity, args.run, args.output, args.verify_only, args.worker)):
                raise ContractError('freeze cannot execute workers')
            print(json_bytes(freeze_request(args.freeze_request)).decode(), end=''); return 0
        if not all((args.request, args.request_identity, args.run, args.output)):
            raise ContractError('request, identity, parent run and output are required')
        for path in (args.request, args.run, args.output):
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ContractError('diagnostic paths cannot traverse symlinks')
        args.request = args.request.resolve(); args.run = args.run.resolve(); args.output = args.output.resolve()
        # Keep outputs entirely outside the immutable parent and request.
        if (args.output == args.run or args.output in args.run.parents or args.run in args.output.parents
                or args.output == args.request or args.output in args.request.parents):
            raise ContractError('diagnostic output overlaps parent/request')
        read_request(args.request, args.request_identity)
        if args.worker:
            result = analyze_run(args.run, args.request, args.request_identity)
            if args.verify_only:
                if json_bytes(result) != read(args.output/'report.json'):
                    raise ContractError('retained diagnostic report differs from fresh verification')
            else:
                from trinite.curriculum_data import dataset
                # Directory was allocated by the parent; every publication is exclusive.
                for workload in protocol()['matrix']['workloads']:
                    directory = args.output/'curriculum'/workload; directory.mkdir(parents=True, exist_ok=False)
                    for name, raw in zip(('dataset.json', 'manifest.json'), dataset(workload)):
                        with (directory/name).open('xb') as stream: stream.write(raw)
            # Verify all candidate copies, including on read-only re-verification.
            from trinite.curriculum_data import audit_bytes
            for workload in protocol()['matrix']['workloads']:
                directory = args.output/'curriculum'/workload
                audit_bytes(read(directory/'dataset.json'), read(directory/'manifest.json'))
            if not args.verify_only:
                with (args.output/'report.json').open('xb') as stream: stream.write(json_bytes(result))
                from trinite.generalization import retain_evidence
                retain_evidence(args.output, args.run)
            else:
                from trinite.generalization import verify_evidence
                verify_evidence(args.output, args.run)
            print(json_bytes({'outcome': result['outcome'], 'report_identity': identity(json_bytes(result))}).decode(), end='')
            return 0
        if not args.verify_only:
            args.output.mkdir(parents=True, exist_ok=False)
            with (args.output/'request.json').open('xb') as stream: stream.write(read(args.request))
        else:
            if read(args.output/'request.json') != read(args.request): raise ContractError('retained diagnostic request mismatch')
        command = [sys.executable, str(ROOT/'scripts/run_generalization.py'), '--worker',
                   '--request', str(args.request), '--request-identity', args.request_identity,
                   '--run', str(args.run), '--output', str(args.output)]
        if args.verify_only: command.append('--verify-only')
        # Fixed interpreter/runner, separate argv data, no shell/command injection.
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit, dangerous-subprocess-use-audit
        try:
            result = subprocess.run(command, cwd=ROOT, shell=False,
                env=dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='1',
                         MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
                capture_output=True, timeout=protocol()['resources']['worker_seconds'])
        except subprocess.TimeoutExpired as error:
            if not args.verify_only:
                (args.output/'worker.stdout').write_bytes(error.stdout or b'')
                (args.output/'worker.stderr').write_bytes(error.stderr or b'')
                (args.output/'failure.json').write_bytes(json_bytes({'outcome': 'failed', 'stage': 'diagnostic',
                    'error': 'worker timed out', 'request_identity': args.request_identity, **protocol()['gates']}))
            raise ContractError('diagnostic worker timed out; partial evidence retained') from error
        if not args.verify_only:
            (args.output/'worker.stdout').write_bytes(result.stdout)
            (args.output/'worker.stderr').write_bytes(result.stderr)
            if result.returncode:
                (args.output/'failure.json').write_bytes(json_bytes({'outcome': 'failed', 'stage': 'diagnostic',
                    'error': 'worker exited '+str(result.returncode), 'request_identity': args.request_identity,
                    **protocol()['gates']}))
        sys.stdout.buffer.write(result.stdout); sys.stderr.buffer.write(result.stderr)
        return result.returncode
    except (ContractError, OSError, KeyError, TypeError, ValueError) as error:
        print(type(error).__name__+': '+str(error), file=sys.stderr); return 1


if __name__ == '__main__': raise SystemExit(main())
