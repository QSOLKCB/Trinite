#!/usr/bin/env python3
"""Explicit engineering build/prepare/verify only. There is no train command."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from procedural_dataset_v1 import admission, transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('build')
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--out', required=True)
    prepare.add_argument('--seed', type=int, default=17)
    verify = commands.add_parser('verify')
    verify.add_argument('--bundle', required=True)
    verify.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'build':
        result = transport.build()
    elif args.command == 'prepare':
        result = {'manifest_sha256': admission.prepare(args.out, args.seed),
                  'training_executed': False, 'exposure': admission.PROTOCOL['exposure']}
    else:
        result = {'verified': True, 'audit': admission.verify(args.bundle, args.manifest_sha256)['manifest.json']['audit'],
                  'training_executed': False}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
