#!/usr/bin/env python3
"""Read-only partial music pilot evidence check; no training or inference."""
import sys
from trinite.contracts import ContractError, json_bytes
from music_learning_result_v1.check import verify

if __name__ == '__main__':
    try:
        sys.stdout.buffer.write(json_bytes(verify()))
    except (ContractError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
