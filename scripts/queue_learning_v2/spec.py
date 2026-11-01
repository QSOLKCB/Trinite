"""Execution contract for v2; the preserved proposal is not an authorization."""
from copy import deepcopy

import queue_evaluation_v1 as gate
from queue_learning_v1.spec import ROOT, git_identity
from .protocol import PROTOCOL as PROPOSAL, plan

SEEDS = (17, 29)
PROTOCOL = deepcopy(PROPOSAL)
PROTOCOL.update(schema='trinite.queue-learning-protocol.v2', execution_enabled=True,
    restrictions='exact-freeze owner approval required; no retries, resume, extra seeds, search, GPU or paid compute; geometry loss zero')
PROTOCOL['authorization'] = 'fresh reviewed merged green implementation, feasibility, freeze and exact v2 owner approval'
PROTOCOL['attempts'] = ['fit-17', 'main-17', 'main-29']
PROTOCOL['commit_barrier'] = 'all held-out outputs for both seeds and all three baselines before any held-out score'


def sources():
    """Overbind the full runtime, including unchanged numerical/safe-reader imports."""
    names = {'AGENTS.md', 'requirements.lock', 'scripts/run_queue_learning_v2.py',
             'scripts/run_queue_learning_v1.py', 'scripts/queue_evaluation_v1.py',
             'docs/QUEUE-LEARNING-V1.md', 'docs/QUEUE-LEARNING-V2.md'}
    for base in ('src', 'scripts/queue_learning_v1', 'scripts/queue_learning_v2'):
        names.update(str(p.relative_to(ROOT)) for p in (ROOT / base).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.json', '.lock', '.txt'))
    names.update(gate.SOURCE_FILES)
    # The observation gate's own closure is intentionally historical. V2 also
    # binds both Rust lock/toolchain files without rewriting that v1 closure.
    for crate in ('queue-sim-v1', 'queue-evaluator-v1'):
        names.update(f'external/{crate}/{name}' for name in
                     ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'))
        names.update(str(p.relative_to(ROOT)) for p in
                     (ROOT / 'external' / crate / 'src').rglob('*.rs'))
    return {name: gate.digest(gate.read(ROOT / name, 4 * 1024 * 1024)) for name in sorted(names)}
