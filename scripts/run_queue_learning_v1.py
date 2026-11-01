#!/usr/bin/env python3
"""Fixed checkout-only queue-learning launcher; works with Python -I -B."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

from queue_learning_v1.controller import main

if __name__ == '__main__':
    raise SystemExit(main())
