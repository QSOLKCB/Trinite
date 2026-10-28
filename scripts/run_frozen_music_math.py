#!/usr/bin/env python3
"""Checkout-only entrypoint; target inference requires its own authorization."""
from pathlib import Path

# -B only prevents cache writes. Reject alternate cached/importable package
# bytes before importing this probe, rather than hashing .py while executing .pyc.
package = Path(__file__).absolute().parent / 'frozen_music_math_v1'
for entry in package.rglob('*'):
    if (entry.is_symlink() or entry.name == '__pycache__'
            or entry.suffix in ('.pyc', '.pyo', '.so', '.pyd')):
        raise SystemExit('probe import caches/shadows forbidden; use a fresh checkout with -B')

from frozen_music_math_v1.workflow import main

if __name__ == '__main__':
    raise SystemExit(main())
