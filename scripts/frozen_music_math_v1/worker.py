"""Isolated original-producer admission and prompt-only inference worker.

Never import the current checkout's native modules in this process. The original
checkout supplies its own unchanged request validator, safe reader and greedy.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

for entry in Path(__file__).absolute().parent.rglob('*'):
    if (entry.is_symlink() or entry.name == '__pycache__'
            or entry.suffix in ('.pyc', '.pyo', '.so', '.pyd')):
        raise SystemExit('probe import caches/shadows forbidden; use a fresh checkout with -B')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frozen_music_math_v1.artifacts import (ROOT, digest, encoded, identity,
    inventory, parse, publish, read, safe_path, sources)
from frozen_music_math_v1.protocol import CAP, CHECKPOINTS, CONTEXT, input_questions

ORIGINAL_COMMIT = 'de28d4ec0a779bb91d6e464f12e6220aceb31fd9'
ORIGINAL_TREE = '8282d8223bb22c155c15beff673ffecca77b22be'


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
    if not root.is_dir():
        raise ValueError('original producer checkout required')
    def git(*args):
        # Fixed read-only Git operations, never shell interpolation.
        return subprocess.check_output(['git', '-c', 'core.fsmonitor=false', '-C', str(root), *args],
                                       timeout=30, stderr=subprocess.PIPE).decode().strip()
    if (git('rev-parse', 'HEAD') != ORIGINAL_COMMIT
            or git('rev-parse', 'HEAD^{tree}') != ORIGINAL_TREE
            or git('status', '--porcelain', '--untracked-files=all')):
        raise ValueError('clean exact original checkpoint producer required')
    reject_import_caches(root)
    if git('ls-files', '--others', '--ignored', '--exclude-standard', '--', 'src', 'scripts'):
        raise ValueError('ignored original runtime files forbidden; use a fresh checkout')
    # Git's clean status can omit assume-unchanged/skip-worktree entries. Compare
    # actual source bytes to pinned tree blobs before importing native modules.
    for line in git('ls-tree', '-r', '--full-tree', 'HEAD', '--', 'src', 'scripts').splitlines():
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
    # The -I interpreter ignores ambient PYTHONPATH; original package roots win.
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
    from music_learning_v1.controller import configure
    from music_overfit_v1.request import validate, cell_name
    from music_overfit_v1.controller import audit_matrix
    from trinite.contracts import parse_json
    configure()
    run = safe_path(run)
    before = inventory(run)
    raw = read(run / 'request.json', 1024 * 1024)
    request = validate(raw, request_id)
    checked = audit_matrix(run, raw, request_id, matrix_id)
    if checked['outcome'] != 'completed' or not checked['tensor_structure_verified']:
        raise ValueError('complete safe original matrix required; reports-only is insufficient')
    checkpoints = []
    for subset, seed, step in CHECKPOINTS:
        prefix = cell_name(seed, subset) + f'/step-{step:03d}'
        metadata = parse_json(read(run / (prefix + '-checkpoint.json')), canonical=True)
        checkpoints.append({'subset': subset, 'seed': seed, 'step': step,
            'payload': prefix + '.safetensors', 'metadata': prefix + '-checkpoint.json',
            'payload_identity': before[prefix + '.safetensors'],
            'metadata_identity': before[prefix + '-checkpoint.json'],
            'model_identity': metadata['model_identity']})
    if inventory(run) != before:
        raise ValueError('original run changed during admission')
    return {'commit': ORIGINAL_COMMIT, 'tree': ORIGINAL_TREE,
        'request_identity': request_id, 'matrix_identity': matrix_id,
        'source': request['source'], 'environment': request['environment'],
        'admission_identity': request['admission_identity'],
        'inventory': before, 'checkpoints': checkpoints,
        'safe_structure_verified': True, 'optimizer_replayed': False,
        'predictions_replayed': False}


def infer(args):
    request_raw = read(args.request, 4 * 1024 * 1024)
    if digest(request_raw) != identity(args.request_identity):
        raise ValueError('probe request identity differs')
    from frozen_music_math_v1.workflow import validate_request
    request = validate_request(request_raw, args.request_identity)
    observed = inspect_original(args.original_source, args.original_run,
                                args.original_request_identity, args.original_matrix_identity)
    if observed != request['original']:
        raise ValueError('original input differs from frozen probe request')
    from music_overfit_v1.numerical import verify_checkpoint
    from trinite.comparison import greedy
    from trinite.contracts import parse_json
    from trinite.training import model_identity
    import torch
    output = safe_path(args.output)
    if (output.stat().st_dev, output.stat().st_ino) != tuple(args.output_identity):
        raise ValueError('attempt directory identity differs')
    run = safe_path(args.original_run)
    calls = 0
    count = 0
    fd = os.open(output / 'predictions.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as journal:
        for checkpoint in observed['checkpoints']:
            payload = read(run / checkpoint['payload'])
            metadata_raw = read(run / checkpoint['metadata'])
            if (digest(payload) != checkpoint['payload_identity']
                    or digest(metadata_raw) != checkpoint['metadata_identity']):
                raise ValueError('checkpoint changed after admission')
            state = verify_checkpoint(payload, parse_json(metadata_raw, canonical=True),
                checkpoint['seed'], checkpoint['subset'], observed['admission_identity'],
                observed['request_identity'])
            model = state.model
            model.eval()
            before = model_identity(model)
            if before != checkpoint['model_identity'] or model.config.context_length != CONTEXT:
                raise ValueError('restored model/context differs')
            def count_call(module, inputs):
                nonlocal calls
                calls += 1
                if calls > 4096:
                    raise ValueError('aggregate forward budget exceeded')
            hook = model.register_forward_pre_hook(count_call)
            try:
                for question in request['questions']:
                    prompt = question['prompt']
                    if not prompt.isascii() or 1 + len(prompt.encode('ascii')) + CAP > CONTEXT:
                        raise ValueError('prompt exceeds fixed context')
                    if any(p.grad is not None for p in model.parameters()):
                        raise ValueError('inference requires absent gradients')
                    start_calls = calls
                    # No gold, task oracle, grammar, repair or example history is passed.
                    with torch.no_grad():
                        generated = greedy(model, prompt, CAP)
                    if calls - start_calls != len(generated):
                        raise ValueError('native forward/token accounting differs')
                    row = {key: checkpoint[key] for key in ('subset', 'seed', 'step')}
                    row.update(question_id=question['question_id'], generated_tokens=generated)
                    journal.write(encoded(row)); journal.flush(); os.fsync(journal.fileno())
                    count += 1
                    if model_identity(model) != before or any(p.grad is not None for p in model.parameters()):
                        raise ValueError('inference changed model/gradients')
            finally:
                hook.remove()
            del state, model
    if count != 256 or inventory(run) != observed['inventory']:
        raise ValueError('coverage incomplete or original input mutated')
    # Revalidate imports and exact original source after inference without forwards.
    from music_overfit_v1.request import validate
    validate(read(run / 'request.json'), observed['request_identity'])
    if sources() != request['probe_source']:
        raise ValueError('probe producer changed during inference')
    publish(output / 'worker-completion.json', encoded({'decodes': count,
        'forward_calls': calls, 'model_identities_unchanged': True,
        'original_inventory_unchanged': True, 'training_performed': False}))


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
