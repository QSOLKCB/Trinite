"""Source-bound prospective protocol; no scientific run is authorized here."""
from pathlib import Path
import subprocess

import queue_evaluation_v1 as gate
from trinite.contracts import ModelConfig
from trinite.foundations_plan import FoundationsPlan
from .codec import CONTEXT, GENERATION

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (17, 29)
PROTOCOL = {
    'schema': 'trinite.queue-learning-protocol.v1', 'target': 'forecast.dropped',
    'episodes': 384, 'families': 48, 'split_episodes': {'train': 256, 'validation': 64, 'test': 64},
    'grouping': 'kind/parameter; all initial-state, mask and hidden-suffix variants together',
    'selection': 'final update 64 for each seed; no early stopping or held-out selection',
    'seeds': list(SEEDS), 'generation_tokens': GENERATION, 'context': CONTEXT,
    'missing': 'lossless masks; no imputation; causal baseline abstains if last queue/arrival/service missing',
    'baseline': 'train-episode-weighted lower median; mode ties smallest; last public queue/arrival/effective-service continuation',
    'primary': 'numeric MAE only with full planned coverage; otherwise undefined all-case MAE; report conditional MAE and denominators',
    'secondary': 'exact matches/all planned; valid syntax including explicit abstention/all planned; numeric coverage/all planned',
    'comparison': 'per seed and split; improvement requires full coverage and lower MAE than each fixed baseline; descriptive only',
    'commit_barrier': 'all model seeds and baselines, validation and test, committed before any held-out target',
    'limits': {'target_tokens_per_seed': 32768, 'update_seconds': 120, 'whole_worker_seconds': 900},
    'restrictions': 'CPU float32 dense; native AdamW; no GPU, paid compute, retries, search, RL or cross-domain inputs',
    'authorization': 'reviewed merged green implementation then explicit owner approval of source/environment-bound freeze',
}


def plan(seed):
    gate.require(type(seed) is int and seed in SEEDS, 'seed outside frozen queue pilot')
    return FoundationsPlan(model=ModelConfig(context_length=256, blocks=1, width=32,
        heads=4, feed_forward_width=64), seed=seed, steps=64, batch_size=8,
        learning_rate='0.001', validation_every=64, max_seconds=120, max_target_tokens=32768)


PROTOCOL['plans'] = {str(seed): plan(seed).to_dict() for seed in SEEDS}
PROTOCOL['order'] = 'SHA-256 of train episode ID; two cycles; identical across seeds'
PROTOCOL['native_validation_field'] = 'train-only diagnostic at final step; never held-out selection'


def sources():
    # Deliberately overbind the native source tree: includes the transitive safe
    # reader imports, source pins, licenses and CPU lock, not only direct imports.
    names = {'AGENTS.md', 'requirements.lock', 'scripts/run_queue_learning_v1.py',
             'scripts/queue_evaluation_v1.py', 'docs/QUEUE-LEARNING-V1.md'}
    for base in ('src', 'scripts/queue_learning_v1'):
        names.update(str(p.relative_to(ROOT)) for p in (ROOT / base).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.json', '.lock', '.txt'))
    names.update(gate.SOURCE_FILES)
    return {name: gate.digest(gate.read(ROOT / name, 4 * 1024 * 1024)) for name in sorted(names)}


def git_identity():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, timeout=5).decode('ascii').strip()
    return {'head': git('rev-parse', 'HEAD'),
            'clean': not bool(git('status', '--porcelain', '--untracked-files=normal')),
            'main': git('rev-parse', 'origin/main')}
