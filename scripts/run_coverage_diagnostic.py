#!/usr/bin/env python3
"""Inspect historical coverage and retain/verify a derived report; never run a model."""
import argparse
from pathlib import Path
import sys

from trinite.contracts import ContractError, identity, json_bytes, require_identity
from coverage_diagnostic_v1.analysis import FIXTURE, analyze, read_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    report = commands.add_parser('report')
    report.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--report', type=Path, required=True)
    verify.add_argument('--expected-identity', required=True)
    args = parser.parse_args(argv)
    try:
        raw = json_bytes(analyze())
        if args.command == 'report':
            if args.output.resolve().is_relative_to(FIXTURE.resolve()):
                raise ContractError('cannot write into the immutable public subset')
            # Exclusive creation: never overwrite retained evidence. Parent directory
            # is caller-controlled; this is not hostile-filesystem containment.
            with args.output.open('xb') as stream:
                stream.write(raw)
            print(identity(raw))
        else:
            expected = require_identity(args.expected_identity)
            supplied = read_bytes(args.report)
            if identity(supplied) != expected or supplied != raw:
                raise ContractError('derived report identity, sources or contents differ')
            print('verified ' + expected)
        return 0
    except (ContractError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
