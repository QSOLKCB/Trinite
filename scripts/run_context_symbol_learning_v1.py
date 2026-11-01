#!/usr/bin/env python3
"""Fixed Linux checkout launcher; applies limits before native imports."""
import os
from pathlib import Path
import resource
import sys

if sys.platform!='linux':
    raise SystemExit('context-symbol-learning-v1 requires Linux /proc, subreaper and rlimits')
if len(sys.argv)>1 and sys.argv[1].startswith('_'):
    if os.environ.get('TRINITE_QUEUE_V2_SUPERVISOR')!=str(os.getppid()):
        raise SystemExit('internal worker requires fixed resource supervisor')
resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from context_symbol_learning_v1.controller import main

if __name__=='__main__':
    raise SystemExit(main())
