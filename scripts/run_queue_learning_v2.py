#!/usr/bin/env python3
"""Fixed v2 checkout launcher: admission and explicit approval remain mandatory."""
from pathlib import Path
import os
import resource
import sys

# Set before Torch or native libraries load, inherited by all descendants.
if sys.platform != 'linux':
    raise SystemExit('queue-learning-v2 requires Linux')
if len(sys.argv) > 1 and sys.argv[1].startswith('_'):
    if os.environ.get('TRINITE_QUEUE_V2_SUPERVISOR') != str(os.getppid()):
        raise SystemExit('internal worker requires the fixed resource supervisor')
resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from queue_learning_v2.controller import main

if __name__ == '__main__':
    raise SystemExit(main())
