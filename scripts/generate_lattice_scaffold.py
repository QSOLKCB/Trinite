"""Create or read-only verify a frozen integer lattice reference artifact."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trinite.contracts import ContractError, identity, json_bytes
from trinite.lattice_scaffold import construct, verify


def safe_path(value):
    path = Path(value)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ContractError('lattice paths cannot traverse symlinks')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--depth', type=int)
    action.add_argument('--verify', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            if args.output is not None:
                raise ContractError('read-only verification cannot write output')
            print(json_bytes(verify(safe_path(args.verify).read_bytes())).decode(), end='')
        else:
            if args.output is None:
                raise ContractError('generation requires --output')
            path = safe_path(args.output)
            raw = json_bytes(construct(args.depth))
            with path.open('xb') as stream:
                stream.write(raw)
            print(json_bytes({'artifact_identity': identity(raw), 'bytes': len(raw),
                              'output': str(path), 'outcome': 'created'}).decode(), end='')
        return 0
    except (ContractError, OSError, ValueError, KeyError, TypeError) as error:
        print('lattice scaffold: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
