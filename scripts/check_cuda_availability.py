#!/usr/bin/env python3
"""Bounded read-only owner-local availability observations; never an experiment."""
import sys
if not sys.flags.isolated or not sys.dont_write_bytecode:
    raise SystemExit('Use python -I -B scripts/check_cuda_availability.py')
# The guard cannot undo startup/site code that already ran in the interpreter.
import importlib.util
import os
from pathlib import Path
import stat
ROOT=Path(os.path.abspath(__file__)).parents[1]
for target in (Path(os.path.abspath(__file__)), ROOT/'scripts/cuda_availability_v1/check.py'):
    if not stat.S_ISREG(target.lstat().st_mode) or any(not stat.S_ISDIR(p.lstat().st_mode) for p in target.parents):
        raise SystemExit('non-link diagnostic source paths required')
for member in (ROOT/'scripts/cuda_availability_v1').iterdir():
    if member.is_symlink() or member.name=='__pycache__' or member.suffix in ('.pyc','.pyo','.so','.pyd'):
        raise SystemExit('diagnostic import caches and links forbidden')
# Load only this trusted file, without adding repo directories to import paths.
spec=importlib.util.spec_from_file_location('_trinite_cuda_availability',ROOT/'scripts/cuda_availability_v1/check.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if __name__=='__main__':
    raise SystemExit(module.main())
