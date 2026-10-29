"""Isolated original-producer admission and prompt-only inference worker.

Never import the current checkout's native modules in this process. The original
checkout supplies its own unchanged request validator, safe reader and greedy.
"""
import argparse
import hashlib
import os
import stat
from pathlib import Path
import subprocess
import sys

for entry in Path(__file__).absolute().parent.rglob('*'):
    if (entry.is_symlink() or entry.name == '__pycache__'
            or entry.suffix in ('.pyc', '.pyo', '.so', '.pyd')):
        raise SystemExit('probe import caches/shadows forbidden; use a fresh checkout with -B')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frozen_music_eval_v1.artifacts import (ROOT, digest, encoded, identity,
    inventory, parse, publish, read, safe_path, sources)

ORIGINAL_COMMIT = 'c239acb6e7b0b5c49798debbae92ca1e1264a64c'
ORIGINAL_TREE = '2bf799bf24edca25b784be36c60fe921011042d1'


def reject_import_caches(root):
    """-B prevents writes, not reads: refuse ignored executable shadow bytes."""
    for base in (root / 'src', root / 'scripts'):
        for current, directories, files in os.walk(base, followlinks=False):
            for name in directories:
                if name == '__pycache__' or (Path(current) / name).is_symlink():
                    raise ValueError('original runtime caches/linked directories forbidden; use a fresh checkout')
            if any(name.endswith(('.pyc', '.pyo', '.so', '.pyd')) for name in files):
                raise ValueError('original runtime bytecode/extension shadows forbidden')


def original_source(path):
    root = safe_path(path)
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError('original producer checkout required')
    def git(*args):
        # Replacement refs can substitute malicious trees behind a pinned SHA.
        # Also discard ambient repository/config selectors before every query.
        env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
        env.update(GIT_NO_REPLACE_OBJECTS='1', GIT_CONFIG_NOSYSTEM='1',
                   GIT_CONFIG_GLOBAL=os.devnull)
        return subprocess.check_output(
            ['git', '--no-replace-objects', '--work-tree=' + str(root),
             '-c', 'core.fsmonitor=false', '-C', str(root), *args],
            env=env, timeout=30, stderr=subprocess.PIPE).decode().strip()
    # Repository-local core.worktree must never redirect cleanliness checks
    # away from the exact directory whose source bytes will be imported.
    if git('rev-parse', '--show-toplevel') != str(root):
        raise ValueError('original producer effective worktree differs')
    if (git('rev-parse', 'HEAD') != ORIGINAL_COMMIT
            or git('rev-parse', 'HEAD^{tree}') != ORIGINAL_TREE
            or git('status', '--porcelain', '--untracked-files=all')):
        raise ValueError('clean exact original checkpoint producer required')
    reject_import_caches(root)
    if git('ls-files', '--others', '--ignored', '--exclude-standard', '--', 'src', 'scripts'):
        raise ValueError('ignored original runtime files forbidden; use a fresh checkout')
    # Git's clean status can omit assume-unchanged/skip-worktree entries. Compare
    # actual source bytes to pinned tree blobs before importing native modules.
    for line in git('ls-tree', '-r', '--full-tree', 'HEAD', '--', 'src', 'scripts', 'docs', 'AGENTS.md', 'requirements.lock', 'configs').splitlines():
        metadata, name = line.split('\t', 1)
        mode, kind, object_id = metadata.split()
        if kind != 'blob' or mode not in ('100644', '100755'):
            raise ValueError('original runtime requires ordinary tracked files')
        raw = read(root / name)
        observed = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if observed != object_id:
            raise ValueError('original runtime bytes differ from pinned Git tree')
    return root


def inspect_original(source, run, request_id, matrix_id):
    root = original_source(source)
    identity(request_id); identity(matrix_id)
    # Refuse a mixed runtime before importing any original native code.
    for name in tuple(sys.modules):
        if name == 'trinite' or name.startswith(('trinite.', 'music_exposure_v1',
                'music_learning_v1', 'music_contrast_v1', 'symbolic_music_v1')):
            raise ValueError('original inspection requires a fresh isolated interpreter')
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
    from music_exposure_v1.controller import configure
    configure()
    from frozen_music_eval_v1.producer import inspect_matrix
    binding = inspect_matrix(safe_path(run), request_id, matrix_id)
    return dict(binding, commit=ORIGINAL_COMMIT, tree=ORIGINAL_TREE)


def replay_gate():
    return {'schema': 'trinite.frozen-music-eval-replay-gate.v1',
        'seeds': [17, 29], 'step': 640, 'replay_decodes': 80,
        'exact_retained_predictions': True, 'both_training_fit': True}


def evaluate_models(models, inputs, retained, emit, gate):
    """One fixed inference-only pass; both replay stages precede every test call.

    ``inputs`` contains prompt strings and row identities only. Retained raw
    outputs are compared after each generation; they never enter greedy inputs.
    This same-process guard detects accidental state mutation, not hostile code.
    """
    import random
    import numpy as np
    import torch
    from trinite.comparison import greedy
    from trinite.training import model_identity
    if set(models) != {17, 29} or set(retained) != {17, 29}:
        raise ValueError('both fixed seed models and retained predictions required')
    train = [r for r in inputs if r['split'] == 'train']
    heldout = [r for r in inputs if r['split'] == 'test']
    if len(inputs) != 160 or len(train) != 40 or len(heldout) != 120:
        raise ValueError('fixed replay/heldout inventory required')
    if any(set(r) != {'example_identity', 'prompt', 'split'} for r in inputs):
        raise ValueError('inference inputs must not contain gold or metadata')
    calls = 0
    decodes = 0
    states = {}
    hooks = []
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    grad_mode = torch.is_grad_enabled()

    def guard(model, state):
        params = [(n, id(p), p.requires_grad) for n, p in model.named_parameters()]
        buffers = [(n, id(b)) for n, b in model.named_buffers()]
        modes = [(n, id(m), m.training) for n, m in model.named_modules()]
        if (model_identity(model) != state['identity'] or params != state['params']
                or buffers != state['buffers'] or modes != state['eval_modes']
                or any(p.grad is not None for p in model.parameters())
                or any(not torch.equal(b, state['buffer_values'][n]) for n,b in model.named_buffers())):
            raise ValueError('frozen inference mutated model, gradients or modes')
        now = np.random.get_state()
        if (random.getstate() != rng[0] or now[0] != rng[1][0]
                or not np.array_equal(now[1], rng[1][1]) or now[2:] != rng[1][2:]
                or not torch.equal(torch.get_rng_state(), rng[2])):
            raise ValueError('frozen inference mutated RNG state')

    def count_call(module, args):
        nonlocal calls
        calls += 1
        if calls > 20480:
            raise ValueError('aggregate forward budget exceeded')

    try:
        for seed in (17, 29):
            model = models[seed]
            if model.config.context_length != 256 or any(p.device.type != 'cpu' or p.dtype != torch.float32
                    or p.grad is not None for p in model.parameters()):
                raise ValueError('frozen CPU float32 model with absent gradients required')
            states[seed] = state = {'identity': model_identity(model),
                'params': [(n, id(p), p.requires_grad) for n,p in model.named_parameters()],
                'buffers': [(n, id(b)) for n,b in model.named_buffers()],
                'buffer_values': {n:b.detach().clone() for n,b in model.named_buffers()},
                'modes': [(m,m.training) for m in model.modules()]}
            model.eval()
            state['eval_modes'] = [(n,id(m),m.training) for n,m in model.named_modules()]
            hooks.append(model.register_forward_pre_hook(count_call))
        for stage, selected in (('train_replay', train), ('heldout', heldout)):
            for seed in (17, 29):
                model, state = models[seed], states[seed]
                if stage == 'train_replay' and set(retained[seed]) != {r['example_identity'] for r in train}:
                    raise ValueError('retained replay inventory differs')
                for row in selected:
                    prompt = row['prompt']
                    if type(prompt) is not str or not prompt.isascii() or 1+len(prompt)+64 > 256:
                        raise ValueError('prompt exceeds frozen byte/context budget')
                    guard(model, state)
                    start = calls
                    with torch.no_grad():
                        generated = greedy(model, prompt, 64)
                    if (type(generated) is not list or not 1 <= len(generated) <= 64
                            or any(type(t) is not int or not 0 <= t <= 258 for t in generated)
                            or 257 in generated[:-1] or (generated[-1] != 257 and len(generated) != 64)
                            or calls-start != len(generated)):
                        raise ValueError('native raw EOS/cap or forward accounting differs')
                    guard(model, state)
                    record = {'seed': seed, 'step': 640, 'stage': stage,
                        'example_identity': row['example_identity'], 'generated_tokens': generated}
                    emit(record)
                    decodes += 1
                    if stage == 'train_replay' and generated != retained[seed][row['example_identity']]:
                        raise ValueError('train replay differs; heldout evaluation forbidden')
            if stage == 'train_replay':
                # Check BOTH models after the last journal callback, not only
                # the model whose heldout stage happens to execute first.
                for seed in (17, 29):
                    guard(models[seed], states[seed])
                gate(replay_gate())  # Durably retained before the first heldout call.
                for seed in (17, 29):
                    guard(models[seed], states[seed])
        for seed in (17, 29):
            guard(models[seed], states[seed])
    finally:
        for hook in hooks:
            hook.remove()
        for state in states.values():
            for module, mode in state['modes']:
                module.training = mode
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        torch.set_grad_enabled(grad_mode)
    if decodes != 320:
        raise ValueError('fixed inference coverage incomplete')
    return {'decodes': decodes, 'forward_calls': calls,
        'model_identities_unchanged': True, 'training_performed': False,
        'train_replay_decodes': 80, 'heldout_decodes': 240}


def infer(args):
    # Pin old runtime bytes BEFORE importing anything with native dependencies.
    observed = inspect_original(args.original_source, args.original_run,
                                args.original_request_identity, args.original_matrix_identity)
    from frozen_music_eval_v1.workflow import validate_request
    request_raw = read(args.request, 4 * 1024 * 1024)
    request = validate_request(request_raw, args.request_identity)
    if observed != request['original']:
        raise ValueError('original input differs from frozen evaluation request')
    from frozen_music_eval_v1.producer import load_models
    output = safe_path(args.output)
    if (output.stat().st_dev, output.stat().st_ino) != tuple(args.output_identity):
        raise ValueError('attempt directory identity differs')
    run = safe_path(args.original_run)
    binding = {k:v for k,v in observed.items() if k not in ('commit', 'tree')}
    models = load_models(run, binding)
    retained = {}
    for seed in (17, 29):
        prediction_raw = read(run / f'train-{seed}/step-640-predictions.json')
        checkpoint = next(c for c in observed['checkpoints'] if c['seed'] == seed)
        if digest(prediction_raw) != checkpoint['training_predictions_identity']:
            raise ValueError('retained final predictions changed after admission')
        value = parse(prediction_raw)
        if len(value['predictions']) != 40 or any(r['correct'] is not True for r in value['predictions']):
            raise ValueError('both retained final training fits required')
        retained[seed] = {r['example_identity']:r['generated_tokens'] for r in value['predictions']}
    fd = os.open(output / 'predictions.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    written = 0
    with os.fdopen(fd, 'wb') as journal:
        def emit(record):
            nonlocal written
            raw = encoded(record); written += len(raw)
            if written > 1048576:
                raise ValueError('prediction journal bound exceeded')
            journal.write(raw); journal.flush(); os.fsync(journal.fileno())
        def gate(value):
            if inventory(run) != observed['inventory']:
                raise ValueError('original input changed before heldout barrier')
            publish(output / 'replay-gate.json', encoded(value))
        completion = evaluate_models(models, request['input_rows'], retained, emit, gate)
    if inventory(run) != observed['inventory']:
        raise ValueError('original input mutated during inference')
    from music_exposure_v1.request import validate
    validate(read(run / 'request.json'), observed['request_identity'])
    original_source(args.original_source)
    if sources() != request['probe_source']:
        raise ValueError('evaluation producer changed during inference')
    completion.update(original_inventory_unchanged=True,
                      replay_gate_identity=digest(read(output / 'replay-gate.json')))
    publish(output / 'worker-completion.json', encoded(completion))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('inspect', 'infer'))
    for name in ('original-source', 'original-run', 'original-request-identity', 'original-matrix-identity'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--request'); parser.add_argument('--request-identity')
    parser.add_argument('--output'); parser.add_argument('--output-identity', type=int, nargs=2)
    parser.add_argument('--authorize-inference', action='store_true')
    args = parser.parse_args()
    if args.action == 'inspect':
        print(encoded(inspect_original(args.original_source, args.original_run,
              args.original_request_identity, args.original_matrix_identity)).decode(), end='')
    else:
        if not args.authorize_inference or args.output_identity is None:
            raise ValueError('separate inference authorization and output custody required')
        infer(args)


if __name__ == '__main__':
    main()
