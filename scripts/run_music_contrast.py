#!/usr/bin/env python3
"""Checkout-only bounded musical contrast training/evaluation controller."""
import sys
from pathlib import Path

if not sys.flags.isolated or not sys.dont_write_bytecode:
    raise SystemExit('Use python -I -B scripts/run_music_contrast.py; ambient import paths are not admitted')
ROOT = Path(__file__).resolve().parents[1]
# -B disables writing but still reads caches. Reject them BEFORE project imports.
for base in (ROOT/'src', ROOT/'scripts'):
    for path in base.rglob('*'):
        if path.is_symlink() or path.name == '__pycache__' or path.suffix in ('.pyc', '.pyo', '.so', '.pyd'):
            raise SystemExit('Checkout import caches, links and extension shadows are forbidden')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from music_contrast_v1.controller import main
if __name__ == '__main__':
    raise SystemExit(main())
