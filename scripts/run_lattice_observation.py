"""Freeze, execute or independently verify retrospective native lattice observation.

Uses the exact historical reference packet already retained in fixtures, never
downloads a model or trains one. Request/result files are exclusive, verification
never writes or changes the retained inputs.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from trinite.contracts import ContractError, json_bytes
from trinite.lattice_observation import experiment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze-request', type=Path)
    action.add_argument('--run', action='store_true')
    action.add_argument('--verify-only', action='store_true')
    p.add_argument('--request', type=Path)
    p.add_argument('--request-identity')
    p.add_argument('--output', type=Path)
    p.add_argument('--artifact-identity')
    a = p.parse_args(argv)
    try:
        if a.freeze_request is not None:
            if any(x is not None for x in (a.request, a.request_identity, a.output, a.artifact_identity)):
                raise ContractError('freeze cannot accept run or verification identities')
            receipt = experiment.freeze(a.freeze_request)
        else:
            if a.request is None or a.request_identity is None or a.output is None:
                raise ContractError('request, external request identity and output are required')
            if a.run:
                if a.artifact_identity is not None:
                    raise ContractError('run cannot borrow external analysis identity')
                receipt = experiment.run(a.request, a.request_identity, a.output)
            else:
                if a.artifact_identity is None:
                    raise ContractError('read-only verification requires external analysis identity')
                receipt = experiment.verify(a.request, a.request_identity, a.output, a.artifact_identity)
        print(json_bytes(receipt).decode(), end='')
        return 0
    except (ContractError, OSError, ValueError, TypeError, KeyError, EOFError) as error:
        print('activation lattice: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
