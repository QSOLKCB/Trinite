"""Offline engineering build/token inspection only; no training command."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import omi_module_v1 as omi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('build')
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--word', type=int, required=True)
    inspect.add_argument('--task', choices=omi.TASKS, default='next')
    inspect.add_argument('--horizon', type=int, default=1)
    args = parser.parse_args()
    if args.operation == 'build':
        omi.build()
    else:
        observation = omi.public(args.word, args.task, args.horizon)
        print(json.dumps({'public': observation, 'prompt': omi.prompt(observation), 'ledger': omi.ledger(observation)}, sort_keys=True))


if __name__ == '__main__':
    main()
