"""Produce or freshly replay a bounded PCGrad engineering receipt."""
import argparse
from pathlib import Path
import sys
import torch
from pcgrad_v1.evidence import report
from pcgrad_v1.spec import specification
from trinite.contracts import ContractError, identity, json_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', default='engineering')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--expected-identity')
    args = parser.parse_args(argv)
    try:
        specification(args.profile)
        if args.verify_only != bool(args.expected_identity):
            raise ContractError('verification requires independently retained expected identity')
        # Reject preexisting generation targets before doing numerical work.
        if not args.verify_only and (args.output.exists() or args.output.is_symlink()):
            raise ContractError('output must be new; historical evidence cannot be overwritten')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        actual = json_bytes(report())
        if len(actual) > 2*1024*1024:
            raise ContractError('report exceeds 2 MiB')
        if args.verify_only:
            if args.output.is_symlink() or not args.output.is_file() or args.output.stat().st_size > 2*1024*1024:
                raise ContractError('verification requires bounded regular report')
            retained = args.output.read_bytes()
            if identity(retained) != args.expected_identity or actual != retained:
                raise ContractError('retained identity/source/environment/numerical replay mismatch')
        else:
            with args.output.open('xb') as stream:
                stream.write(actual)
        print(identity(actual))
        return 0
    except (ContractError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
