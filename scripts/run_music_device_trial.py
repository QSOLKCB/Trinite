#!/usr/bin/env python3
"""Isolated matched CPU/CUDA compatibility trial; separate run authorization required."""
import sys
if not sys.flags.isolated or not sys.dont_write_bytecode:
    raise SystemExit('Use python -I -B scripts/run_music_device_trial.py')
# A script cannot undo site/startup execution before it was invoked without -I.
# Load the bootstrap's standard library before adding any repository path.
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'scripts/music_device_trial_v1'
if PACKAGE.is_symlink():
    raise SystemExit('linked study package forbidden')
for member in PACKAGE.rglob('*'):
    if member.is_symlink() or member.name == '__pycache__' or member.suffix in ('.pyc','.pyo','.so','.pyd'):
        raise SystemExit('study package caches/shadows forbidden')
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
# Resolving the top-level package does not execute its initializer. Do not use
# find_spec on a child module before validating that initializer's tracked bytes.
spec = importlib.util.find_spec('music_device_trial_v1')
if spec is None or spec.origin != str(PACKAGE/'__init__.py'):
    raise SystemExit('foreign study bootstrap resolution')
# The launcher and this small stdlib-only bootstrap are the trusted entrypoint.
# Load it directly, without executing any other project module first.
spec = importlib.util.spec_from_file_location('_trinite_device_bootstrap', PACKAGE/'bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
bootstrap.check(ROOT)
from music_device_trial_v1.controller import main
if __name__ == '__main__':
    raise SystemExit(main())
