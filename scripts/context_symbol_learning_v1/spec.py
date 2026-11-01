"""Exact bounded schedule and complete executable source closure."""
from pathlib import Path
import subprocess

import queue_evaluation_v1 as gate
from trinite.contracts import ModelConfig
from trinite.foundations_plan import FoundationsPlan

ROOT = Path(__file__).resolve().parents[2]
ATTEMPTS = (('fit-17',17,True,128,2432), ('main-17',17,False,256,4900), ('main-29',29,False,256,4900))
TOKEN_CEILING = 12232  # Exact fixed train order, answer bytes AND EOS.
AUTHORIZATION = 'execute context-symbol-learning-v1 fit seed 17 once at step 128; only after final 8/8, main seeds 17 and 29 once at step 256'
PROTOCOL = dict(schema='trinite.context-symbol-learning-protocol.v1', family_version=1,
    data_version='context-symbol-scientific-family-v1',
    development_exposed_inventory_sha256='a73e56c3f3fb7a2dd89cd14b0fa0bb909e1c885808208247756951a8b82ec43e',
    attempts=[dict(name=n, seed=s, fit=f, updates=u, target_ceiling=t) for n,s,f,u,t in ATTEMPTS],
    optimizer_updates_ceiling=640, supervised_token_ceiling=TOKEN_CEILING,
    context=256, parameters=24768, batch=8, learning_rate='0.001', geometry_loss=0,
    precision='CPU float32; pinned eager deterministic native decoder and single-tensor AdamW',
    generation='unrestricted greedy; three emitted tokens maximum; EOS mandatory; no repair/retry',
    selection='final checkpoint only: fit 128, main 256; validation reporting-only after all commitments',
    split='nine table parent families; SHA-256 table rank: five train, two validation, two test',
    overlap='exact public and full-table orbit under context/symbol/meaning bijections; masks/query ignored; declared mutations grouped',
    limitations='public construction; fixed-size table-family transfer only; no unseen algorithm, size, domain, secret benchmark or psychological claim',
    commitment='all heldout predictions for BOTH seeds and fixed baselines before any heldout oracle evaluation',
    limits=dict(update_seconds=120, attempt_seconds=900, total_seconds=2700,
        process_rss=1<<30, aggregate_rss=2<<30, address_space=4<<30),
    restrictions='no retries, resume, extra seeds, search, GPU, paid compute or budget expansion')


def plan(seed, fit=False):
    gate.require(type(seed) is int and seed in (17,29) and (not fit or seed == 17), 'unapproved seed/attempt')
    return FoundationsPlan(model=ModelConfig(context_length=256, blocks=1,width=32,heads=4,feed_forward_width=64),
        seed=seed, steps=128 if fit else 256, batch_size=8, learning_rate='0.001',
        validation_every=128 if fit else 256, max_seconds=120, max_target_tokens=2432 if fit else 4900)


def sources():
    names = {'AGENTS.md','requirements.lock','scripts/run_context_symbol_learning_v1.py',
        'scripts/queue_evaluation_v1.py','scripts/tombstone_adapter_v1.py','docs/CONTEXT-SYMBOL-LEARNING-V1.md',
        'docs/CONTEXT-SYMBOL-RELATION-V1.md','docs/PROCEDURAL-DATASET-V1.md'}
    # Overbind native transitive imports, authoritative safe reader, shared
    # process/artifact utilities and tokenizer admission without modifying them.
    for base in ('src','scripts/context_symbol_learning_v1','scripts/queue_learning_v1',
                 'scripts/queue_learning_v2','scripts/procedural_dataset_v1','external/context-symbol-relation-v1'):
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/base).rglob('*')
            if p.is_file() and 'target' not in p.relative_to(ROOT/base).parts
            and p.suffix in ('.py','.rs','.json','.lock','.toml','.txt','.md'))
    return {n:gate.digest(gate.read(ROOT/n,4<<20)) for n in sorted(names)}


def git_identity():
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,timeout=10).decode('ascii').strip()
    return dict(head=git('rev-parse','HEAD'), main=git('rev-parse','origin/main'),
                clean=not bool(git('status','--porcelain','--untracked-files=normal')))


def clean_main():
    value = git_identity()
    gate.require(value['clean'] and value['head'] == value['main'], 'clean actual merged main required')
    return value
