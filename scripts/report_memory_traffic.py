#!/usr/bin/env python3
"""Print deterministic analytic accounting; never construct/run a model."""
import sys
if not sys.flags.isolated or not sys.dont_write_bytecode:
    raise SystemExit('Use python -I -B scripts/report_memory_traffic.py --plan PLAN.json')
import argparse
import dataclasses
import hashlib
import json
import os
import stat
from pathlib import Path
import re
import typing
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trinite.contracts import ContractError, json_bytes, parse_json
from memory_traffic_v1.accounting import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    try:
        # Nonblocking open avoids FIFO hangs; descriptor check avoids path races.
        fd = os.open(args.plan, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
                raise ContractError('plan must be a regular file of at most 64 KiB')
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ContractError('plan exceeds 64 KiB')
        value = report(parse_json(raw))
    except (ContractError, OSError) as exc:
        parser.exit(2, str(exc)+'\n')
    sys.stdout.buffer.write(json_bytes(value))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
