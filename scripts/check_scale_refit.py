"""Produce or independently recompute a source-bound scale-refit engineering receipt."""
import argparse
from pathlib import Path
import sys

import torch

from scale_refit.diagnostic import canonical, identity, report
from trinite.contracts import ContractError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--expected-identity')
    args = parser.parse_args(argv)
    try:
        if args.verify_only != bool(args.expected_identity):
            raise ContractError('verification requires an independently retained expected identity')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        actual = canonical(report())
        if args.verify_only:
            if args.output.is_symlink() or not args.output.is_file() or args.output.stat().st_size > 1024*1024:
                raise ContractError('verification requires a bounded regular report')
            retained = args.output.read_bytes()
            if identity(retained) != args.expected_identity or actual != retained:
                raise ContractError('retained identity/source/environment/numerical report mismatch')
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
