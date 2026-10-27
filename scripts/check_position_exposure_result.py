#!/usr/bin/env python3
"""Read-only check of the partial public position-exposure evidence; no model run."""
import sys
from trinite.contracts import ContractError, json_bytes
from position_exposure_result_v1.check import verify

if __name__ == '__main__':
    try:
        sys.stdout.buffer.write(json_bytes(verify()))
    except (ContractError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
