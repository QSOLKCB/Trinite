"""Offline CPU position-interface conformance; never launches target training."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

from trinite.contracts import ContractError, identity, json_bytes


def sources():
    paths = list((ROOT/'src/trinite').rglob('*.py'))
    paths += list((ROOT/'scripts/position_coupling_v1').glob('*.py'))
    paths += list((ROOT/'tests/position_coupling_v1').glob('*.py'))
    paths += [Path(__file__).resolve(), ROOT/'requirements.lock',
              ROOT/'src/trinite/cpu-dependencies.lock',
              ROOT/'docs/POSITION-COUPLING.md',
              ROOT/'.github/workflows/position-coupling-v1.yml']
    result = {}
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2**20:
            raise ContractError('invalid conformance source file')
        result[path.relative_to(ROOT).as_posix()] = identity(path.read_bytes())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='new report file; refuses to overwrite')
    args = parser.parse_args()
    if args.output and args.output.resolve().is_relative_to(ROOT):
        parser.error('report output must be outside the repository tree')
    # Open exclusively before computation, preserving partial/failure output.
    stream = args.output.open('xb') if args.output else sys.stdout.buffer
    result = {'schema': 'trinite.position-conformance-report.v1',
              'scope': 'synthetic CPU forward/gradient conformance only',
              'target_training_run': False, 'learning_claim': False,
              'closed_custody': False}
    code = 1
    try:
        import torch
        import trinite.model
        if Path(trinite.model.__file__).resolve() != ROOT/'src/trinite/model.py':
            raise ContractError('native model import is not checkout-bound')
        from position_coupling_v1.conformance import run_checks
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        before = sources()
        result['source'] = before
        result['checks'] = run_checks()
        if sources() != before:
            raise ContractError('sources changed during conformance')
        result['outcome'] = 'passed'
        code = 0
    except Exception as exc:
        result['outcome'] = 'failed'
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        stream.write(json_bytes(result))
        stream.flush()
        if args.output:
            stream.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
