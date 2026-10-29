"""Strict standard-library receipt structure, not native admission or replay.

The original isolated verifier independently checks all claimed byte identities
and safe tensor structure; these checks prevent malformed receipts from being
accepted by a later read-only probe report audit.
"""
from pathlib import PurePosixPath

from .artifacts import identity
from .protocol import SEEDS

ORIGINAL_COMMIT = 'c239acb6e7b0b5c49798debbae92ca1e1264a64c'
ORIGINAL_TREE = '2bf799bf24edca25b784be36c60fe921011042d1'
ATTEMPT_SCHEMA = 'trinite.frozen-music-eval-attempt.v1'
SELECTION = 'fixed final640 seeds17/29; both-seed replay barrier; one attempt; no training'


def _keys(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(label + ' fields differ')


def _text(value, label, limit=16384):
    if type(value) is not str or not value or len(value) > limit:
        raise ValueError(label + ' requires bounded nonempty text')


def _path(value):
    _text(value, 'receipt path', 512)
    path = PurePosixPath(value)
    if (not path.parts or path.is_absolute() or str(path) != value or '\\' in value
            or any(part in ('.', '..') for part in path.parts)
            or any(ord(c) < 32 or ord(c) > 126 for c in value)):
        raise ValueError('canonical relative receipt path required')


def inventory_names():
    """The complete successful original controller layout, including all steps."""
    names = {'request.json', 'matrix.json'}
    for seed in SEEDS:
        cell = f'train-{seed}/'
        names.update(cell + name for name in (
            'request.json', 'report.json', 'completion.json', 'worker.log', 'attempt.json'))
        names.update(cell + f'step-{step:03d}' + suffix
                     for step in (0, 64, 640)
                     for suffix in ('.safetensors', '-checkpoint.json', '-predictions.json', '-journal.jsonl'))
    return names


def _environment(value):
    _keys(value, ('python', 'platform', 'machine', 'torch', 'cpu', 'cpu_kernel_capability',
        'default_dtype', 'default_device', 'threads', 'interop_threads', 'deterministic',
        'dtype', 'device', 'optimizer', 'packages'), 'original environment')
    for key in ('python', 'platform', 'machine', 'torch', 'cpu_kernel_capability',
                'default_dtype', 'default_device', 'dtype', 'device', 'optimizer'):
        _text(value[key], 'environment ' + key)
    for key in ('threads', 'interop_threads'):
        if type(value[key]) is not int or not 1 <= value[key] <= 65536:
            raise ValueError('original environment thread count differs')
    if (value['threads'] != 1 or value['deterministic'] is not True
            or value['machine'] != 'x86_64' or not value['platform'].startswith('Linux')
            or value['torch'] != '2.8.0+cpu' or value['default_dtype'] != 'torch.float32'
            or value['default_device'] != 'cpu' or value['dtype'] != 'float32'
            or value['device'] != 'cpu' or value['optimizer'] != 'AdamW-single-tensor'):
        raise ValueError('original native CPU environment differs')
    _keys(value['cpu'], ('vendor_id', 'model name', 'flags'), 'original CPU')
    for text in value['cpu'].values():
        _text(text, 'original CPU field')
    # Exact c239acb native CPU contract, not a newly chosen inference environment.
    packages = {'torch': '2.8.0+cpu', 'numpy': '2.3.5', 'safetensors': '0.6.2',
        'packaging': '25.0', 'filelock': '3.32.3', 'typing-extensions': '4.16.0',
        'setuptools': '78.1.0', 'sympy': '1.14.0', 'networkx': '3.6.1', 'jinja2': '3.1.6',
        'fsspec': '2026.7.0', 'mpmath': '1.3.0', 'markupsafe': '3.0.3'}
    if value['packages'] != packages:
        raise ValueError('original pinned package versions differ')


def validate_original(value):
    _keys(value, ('commit', 'tree', 'request_identity', 'matrix_identity', 'source',
        'environment', 'admission_identity', 'inventory', 'checkpoints',
        'safe_structure_verified', 'optimizer_replayed', 'predictions_replayed'),
        'original receipt')
    if value['commit'] != ORIGINAL_COMMIT or value['tree'] != ORIGINAL_TREE:
        raise ValueError('original producer pin differs')
    for key in ('request_identity', 'matrix_identity', 'admission_identity'):
        identity(value[key])
    source = value['source']
    if type(source) is not dict or not 1 <= len(source) <= 512:
        raise ValueError('bounded original source map required')
    for path, digest in source.items():
        _path(path); identity(digest)
    required = {'scripts/music_exposure_v1/numerical.py',
                'src/trinite/comparison_checkpoint.py', 'src/trinite/comparison.py'}
    if not required <= set(source):
        raise ValueError('original source map missing loader/decoder')
    _environment(value['environment'])
    _keys(value['inventory'], inventory_names(), 'complete original inventory')
    for digest in value['inventory'].values():
        identity(digest)
    if (value['inventory']['request.json'] != value['request_identity']
            or value['inventory']['matrix.json'] != value['matrix_identity']):
        raise ValueError('original root commitment differs from inventory')
    for seed in SEEDS:
        if value['inventory'][f'train-{seed}/request.json'] != value['request_identity']:
            raise ValueError('cell original request identity differs')
    checkpoints = value['checkpoints']
    if type(checkpoints) is not list or len(checkpoints) != 2:
        raise ValueError('both fixed final checkpoint endpoints required')
    for record, seed in zip(checkpoints, SEEDS):
        _keys(record, ('seed', 'step', 'payload', 'metadata', 'payload_identity',
                       'metadata_identity', 'model_identity', 'training_predictions_identity'), 'original checkpoint')
        prefix = f'train-{seed}/step-640'
        if (type(record['seed']) is not int or type(record['step']) is not int
                or (record['seed'], record['step']) != (seed, 640)
                or record['payload'] != prefix + '.safetensors'
                or record['metadata'] != prefix + '-checkpoint.json'):
            raise ValueError('original checkpoint ordered identity/path differs')
        for key in ('payload_identity', 'metadata_identity', 'model_identity', 'training_predictions_identity'):
            identity(record[key])
        if any(record[key + '_identity'] != value['inventory'][record[key]]
               for key in ('payload', 'metadata')):
            raise ValueError('checkpoint identity differs from complete inventory')
        if record['training_predictions_identity'] != value['inventory'][prefix + '-predictions.json']:
            raise ValueError('training predictions differ from complete inventory')
    if (value['safe_structure_verified'] is not True or value['optimizer_replayed'] is not False
            or value['predictions_replayed'] is not False):
        raise ValueError('unsupported original assurance claim')
    return value


def validate_attempt(value, request_id):
    identity(request_id)
    _keys(value, ('schema', 'request_identity', 'authorization', 'selection'), 'probe attempt')
    if (value['schema'] != ATTEMPT_SCHEMA or value['request_identity'] != request_id
            or value['authorization'] != 'explicit-inference-flag' or value['selection'] != SELECTION):
        raise ValueError('probe attempt authorization/context differs')
    return value
