#!/usr/bin/env python3
"""Isolated checkout-only frozen evaluation; separately authorized inference only."""
import sys

# sys is built in: reject unsafe invocation before any filesystem import.
# This cannot undo Python startup/site code run before the script without -I.
if not sys.flags.isolated or not sys.dont_write_bytecode:
    raise SystemExit('Use python -I -B scripts/run_frozen_music_eval.py; ambient import paths are not admitted')

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for base in (ROOT / 'scripts', ROOT / 'src'):
    for entry in base.rglob('*'):
        if (entry.is_symlink() or entry.name == '__pycache__'
                or entry.suffix in ('.pyc', '.pyo', '.so', '.pyd')):
            raise SystemExit('runtime caches/shadows forbidden; use a fresh checkout with -B')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from frozen_music_eval_v1.workflow import main

if __name__ == '__main__':
    raise SystemExit(main())
