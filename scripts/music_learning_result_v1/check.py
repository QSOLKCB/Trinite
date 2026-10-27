"""Verify original public pilot bytes and metrics without constructing a model.

Use quiescent local artifacts; pre/post checks are not hostile-writer attestation.
The intentionally partial subset cannot verify omitted checkpoints or custody.
"""
import math
import os
from pathlib import Path
import stat
import sys

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from music_learning_v1.admission import dataset, audit_bytes
from music_learning_v1.numerical import check_score, schedule_receipt
from music_learning_v1.reporting import describe
from unseen_evaluation_v1.packet import inventory, unchanged

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/music-learning-result-20261027-v1'
REQUEST = 'sha256:e534fa4beae47aade39acd1b06a71919e02120b08cbe22e36037b8dcb09b1d65'
MATRIX = 'sha256:a36434c73ee4643caed241a202d5e5b4cf6c986c453588ea73009d1467b773ad'
SOURCE_RECEIPT = 'sha256:fcecf23369db5cef63132b2c09b61c3677a2bfd91887cd9aa91cc2b85aa5e8a2'
CELLS = ('dense-17', 'dense-29')
STAGES = ('untrained', 'trained')
SPLITS = ('train', 'validation', 'test')
FILES = ('attempt.json', 'completion.json', 'report.json', 'predictions.jsonl',
         *(f'{stage}-{kind}.json' for stage in STAGES for kind in ('description', *SPLITS)))
CHECKER_FILES = ('scripts/check_music_learning_result.py',
                 'scripts/music_learning_result_v1/__init__.py',
                 'scripts/music_learning_result_v1/check.py')
GATES = ('fresh_confirmation', 'learning_adequate', 'phase5_ready', 'release_ready')


def read(path):
    """Bounded no-follow read with observed replacement/mutation rejection."""
    path = Path(path).absolute()
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ContractError('unsafe artifact ancestor')
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 4 * 1024 * 1024:
        raise ContractError('expected bounded regular public artifact')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ContractError('public artifact replaced during open')
        raw = stream.read(4 * 1024 * 1024 + 1)
        after = os.fstat(stream.fileno())
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if len(raw) > 4 * 1024 * 1024 or any(getattr(before, k) != getattr(after, k) for k in fields):
        raise ContractError('public artifact changed or exceeds bound')
    return raw


def source_check(sources):
    for name, expected in sources.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ContractError('invalid source path')
        if identity(read(ROOT / name)) != expected:
            raise ContractError('historical producer source differs: ' + name)
        parts = relative.parts
        if parts[0] not in ('src', 'scripts') or not name.endswith('.py'):
            continue
        names = list(parts[1:])
        names[-1] = names[-1][:-3]
        if names[-1] == '__init__':
            names.pop()
        module = sys.modules.get('.'.join(names))
        if module is not None:
            origin = getattr(module, '__file__', None)
            if origin is None or Path(origin).absolute() != ROOT / name:
                raise ContractError('repository dependency imported from a foreign root')


def verify(fixture=FIXTURE):
    fixture = Path(fixture)
    before = inventory(fixture)
    expected = {'.', 'README.md', 'matrix.json', 'source-bindings.json', *CELLS,
                *(f'{cell}/{name}' for cell in CELLS for name in FILES)}
    if set(before['members']) != expected:
        raise ContractError('public subset membership differs')
    checker_sources = {name: identity(read(ROOT / name)) for name in CHECKER_FILES}
    raw = read(fixture / 'source-bindings.json')
    if identity(raw) != SOURCE_RECEIPT:
        raise ContractError('derived source binding receipt changed')
    receipt = parse_json(raw, canonical=True)
    if receipt['request_identity'] != REQUEST or receipt['matrix_identity'] != MATRIX:
        raise ContractError('source binding identity differs')
    source_check(receipt['sources'])
    raw = read(fixture / 'matrix.json')
    if identity(raw) != MATRIX:
        raise ContractError('original matrix identity differs')
    matrix = parse_json(raw, canonical=True)
    if (matrix['request_identity'] != REQUEST or set(matrix['cells']) != set(CELLS)
            or matrix['outcome'] != 'completed' or any(matrix[k] is not False for k in GATES)):
        raise ContractError('matrix request/cells/outcome/gates differ')
    data, admission = dataset()
    audit_bytes(data, admission)
    if (identity(data), identity(admission)) != (receipt['dataset_identity'], receipt['admission_identity']):
        raise ContractError('original admission differs')
    # Reject any tampered retained member before expensive admission/scoring work.
    for cell in CELLS:
        for name in FILES:
            if identity(read(fixture / cell / name)) != matrix['cells'][cell]['retained_files'][name]:
                raise ContractError('original retained artifact differs: ' + cell + '/' + name)
    result = {}
    prediction_count = 0
    for cell in CELLS:
        bound = matrix['cells'][cell]
        originals, records = {}, {}
        for name in FILES:
            raw = read(fixture / cell / name)
            if identity(raw) != bound['retained_files'][name]:
                raise ContractError('original retained artifact differs: ' + cell + '/' + name)
            originals[name] = raw
            if name.endswith('.json'):
                records[name] = parse_json(raw, canonical=True)
        completion, attempt = records['completion.json'], records['attempt.json']
        for record in (completion, attempt):
            if record['request_identity'] != REQUEST or record['cell'] != cell:
                raise ContractError('cell request linkage differs')
        if (bound['completion_identity'] != identity(originals['completion.json'])
                or bound['attempt_identity'] != identity(originals['attempt.json'])
                or attempt['completion_identity'] != bound['completion_identity']
                or attempt['outcome'] != 'completed' or attempt['worker_exit'] != 0
                or bound['outcome'] != 'completed'):
            raise ContractError('completion/attempt linkage differs')
        for name in FILES:
            if name not in ('completion.json', 'attempt.json') and completion['artifacts'][name] != bound['retained_files'][name]:
                raise ContractError('completion artifact linkage differs')
        journal, descriptions = [], {}
        for stage in STAGES:
            scores = {split: records[f'{stage}-{split}.json'] for split in SPLITS}
            model_id = scores['train']['model_identity']
            for split, score in scores.items():
                check_score(score, split, model_id)
                journal.extend({'stage': stage, 'selector': split, 'prediction': p}
                               for p in score['predictions'])
                prediction_count += len(score['predictions'])
            description = describe(scores)
            if json_bytes(description) != originals[f'{stage}-description.json']:
                raise ContractError('retained descriptive map differs')
            descriptions[stage] = {key: description[key]['total'] for key in
                                  ('train_fit', 'held_out_validation', 'held_out_test')}
        if b''.join(json_bytes(row) for row in journal) != originals['predictions.jsonl']:
            raise ContractError('raw journal differs from score predictions')
        seed = int(cell.rsplit('-', 1)[1])
        report = records['report.json']
        wanted = {'seed': seed, 'outcome': 'completed', 'step': 64,
                  'target_tokens': 3955, 'request_identity': REQUEST,
                  'model_identity': records['trained-train.json']['model_identity'],
                  'schedule': schedule_receipt(seed)}
        if any(json_bytes(report[k]) != json_bytes(v) for k, v in wanted.items()) or any(report[k] is not False for k in GATES):
            raise ContractError('reported schedule/model/gates differ')
        elapsed = float.fromhex(report['update_seconds_hex'])
        if not math.isfinite(elapsed) or not 0 <= elapsed <= 120:
            raise ContractError('reported update time exceeds budget')
        result[cell] = descriptions
    unchanged(fixture, before)
    source_check(receipt['sources'])
    if any(identity(read(ROOT / name)) != digest for name, digest in checker_sources.items()):
        raise ContractError('checker source changed during verification')
    return {'schema': 'trinite.music-learning-public-subset-check.v1',
            'request_identity': REQUEST, 'matrix_identity': MATRIX,
            'producer_source_binding_identity': SOURCE_RECEIPT,
            'checker_sources': checker_sources, 'cells': result,
            'original_files': 25, 'prediction_records': prediction_count,
            'closed_custody': False, 'checkpoint_tensors_verified': False,
            'inference_replayed': False, 'optimizer_replayed': False,
            **{name: False for name in GATES}}
