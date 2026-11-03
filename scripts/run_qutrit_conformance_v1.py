"""Offline engineering build and token inspection, never training."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import qutrit_conformance_v1 as q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('build')
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--word', required=True)
    inspect.add_argument('--circuit', choices=q.CIRCUITS, default='S1')
    inspect.add_argument('--task', choices=q.TASKS, default='apply')
    inspect.add_argument('--candidate', default='')
    args = parser.parse_args()
    if args.operation == 'build':
        q.build()
    else:
        value = q.public(args.word, args.circuit, args.task, args.candidate)
        print(json.dumps(dict(public=value, prompt=q.prompt(value), ledger=q.ledger(value)), sort_keys=True))


if __name__ == '__main__':
    main()
