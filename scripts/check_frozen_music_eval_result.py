#!/usr/bin/env python3
"""Read-only reports-only frozen musical evaluation check; no model execution."""
import sys
if not sys.flags.isolated or not sys.dont_write_bytecode:
    raise SystemExit('Use python -I -B scripts/check_frozen_music_eval_result.py')
# Startup/site code before this guard is outside the script's control.
# Preload every direct stdlib dependency of the pinned scorer closure while
# isolated startup paths are still in force. Repository files must not shadow
# hashlib/json/subprocess (or other stdlib imports) after adding project paths.
import __future__
import argparse
import collections
import dataclasses
import fractions
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import threading
import typing
ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'scripts/frozen_music_eval_result_v1').is_symlink():
    raise SystemExit('linked verifier package forbidden')
for entry in (ROOT / 'scripts/frozen_music_eval_result_v1').rglob('*'):
    if (entry.is_symlink() or entry.name == '__pycache__'
            or entry.suffix in ('.pyc', '.pyo', '.so', '.pyd')):
        raise SystemExit('verifier import caches/shadows forbidden')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
# Authenticate resolution before find_spec is allowed to import the parent.
package = importlib.util.find_spec('frozen_music_eval_result_v1')
if package is None or package.origin != str(ROOT / 'scripts/frozen_music_eval_result_v1/__init__.py'):
    raise SystemExit('foreign verifier package resolution')
module = importlib.util.find_spec('frozen_music_eval_result_v1.check')
if module is None or module.origin != str(ROOT / 'scripts/frozen_music_eval_result_v1/check.py'):
    raise SystemExit('foreign verifier check resolution')
from frozen_music_eval_result_v1.check import main

if __name__ == '__main__':
    raise SystemExit(main())
