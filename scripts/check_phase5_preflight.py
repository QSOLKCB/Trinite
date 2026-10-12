"""Offline signed Phase 5 preflight; only creates an immutable rejection receipt."""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.phase5_preflight import assess, read_json, safe_read


def verify_foundations(root, request_identity, summary_identity):
    """Run the existing read-only verifier in a separate, bounded process."""
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)) or not root.is_dir():
        raise ContractError('unsafe or missing foundations run root')
    runner = ROOT/'scripts/verify_foundations_run.py'
    argv = [sys.executable, str(runner), str(root), '--request-identity', request_identity,
            '--summary-identity', summary_identity]
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    try:
        result = subprocess.run(argv, cwd=ROOT, shell=False, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=2400)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError('foundations verification could not complete') from exc
    if result.returncode:
        raise ContractError('authoritative foundations verification rejected the run')
    record = parse_json(result.stdout, canonical=True)
    if (type(record) is not dict or record.get('request_identity') != request_identity
            or record.get('summary_identity') != summary_identity):
        raise ContractError('foundations verifier returned an unbound receipt')
    return {'schema': 'trinite.phase5-foundations-verification.v1', **record}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--signatures', type=Path, required=True)
    p.add_argument('--trust-policy', type=Path, required=True)
    p.add_argument('--trust-policy-identity', required=True,
                   help='Out-of-band SHA256 of canonical JSON trust policy; never read this from the packet')
    p.add_argument('--foundations-run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        for path in (args.packet, args.signatures, args.trust_policy, args.output):
            if path.is_symlink() or any(parent.is_symlink() for parent in path.absolute().parents):
                raise ContractError('phase5 path traversal through symlink denied')
        output = args.output.absolute()
        sources = [a.absolute() for a in (args.packet, args.signatures, args.trust_policy)]
        if output in sources or output in (s for source in sources for s in source.parents):
            raise ContractError('phase5 output overlaps inputs')
        packet = read_json(args.packet)
        signatures = parse_json(safe_read(args.signatures), canonical=True)
        policy = read_json(args.trust_policy)
        prior = None
        prior_error = None
        clock = datetime.now(timezone.utc)
        result = assess(packet, signatures, policy, args.trust_policy_identity, now=clock)
        if args.foundations_run and all(result['checks'][key] for key in (
                'protocol_and_matrix', 'external_policy_identity', 'signature_quorum', 'request_window')):
            try:
                refs = packet['prerequisites']
                prior = verify_foundations(args.foundations_run, refs['foundations_request_identity'],
                                           refs['foundations_summary_identity'])
            except (ContractError, KeyError, TypeError, ValueError) as exc:
                prior_error = str(exc)
            result = assess(packet, signatures, policy, args.trust_policy_identity,
                            now=clock, foundations_result=prior)
        if prior_error:
            result['errors'].append('foundations verification: '+prior_error)
        # Exclusive, fail-closed retention; never overwrite old evidence.
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            stream.write(json_bytes(result))
        print(json_bytes({'receipt_identity':identity(json_bytes(result)),
                          'outcome': result['outcome'], 'phase5_authorized':False}).decode(), end='')
        return 1
    except (ContractError, OSError, KeyError, TypeError, ValueError, UnicodeError) as exc:
        print('phase5-preflight: '+str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
