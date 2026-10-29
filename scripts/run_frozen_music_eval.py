#!/usr/bin/env python3
"""Isolated checkout-only frozen evaluation; separately authorized inference only."""
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
for base in (ROOT / 'scripts', ROOT / 'src'):
    for entry in base.rglob('*'):
        if (entry.is_symlink() or entry.name == '__pycache__'
                or entry.suffix in ('.pyc', '.pyo', '.so', '.pyd')):
            raise SystemExit('runtime caches/shadows forbidden; use a fresh checkout with -B')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from frozen_music_eval_v1.workflow import main

if __name__ == '__main__':
    raise SystemExit(main())
