"""Explicit frozen inputs, layouts and source closure; no model execution."""
import gzip
import io
import os
import stat
from pathlib import Path
from trinite.contracts import ContractError, identity, json_bytes, parse_json
from scale_refit_training_v1.spec import inputs

ROOT = Path(__file__).resolve().parents[2]
BASE = 'eb7adcf9b545d8b3e78908f8b091981797f2c80b'
REFERENCE_ID = 'sha256:7f5b8c6819ab5162546f881a8105148e409b17c37689bd72e52e9799fc06a688'
MAX_REPORT = 64 * 1024 * 1024
MAX_REQUEST = 1024 * 1024
MAX_SECONDS = 300
LAYOUTS = ('singleton_min', 'singleton_32', 'batch8_32', 'partition4_32',
           'reverse8_32', 'recompose8_32', 'batch8_min', 'payload8_32')
CELLS = ((0, 'absmean'), (0, 'refit'), (1, 'absmean'), (1, 'refit'))


def read(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ContractError('input must be a regular non-symlink file')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ContractError('opened input is not a regular file')
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ContractError('input exceeds byte bound')
    return raw


def protocol():
    return parse_json(read(ROOT/'scripts/batch_composition_v1/protocol.json', MAX_REQUEST), canonical=True)


def workload():
    dataset, _ = inputs()
    rows = parse_json(dataset, canonical=True)['examples']
    result = []
    for split, expected in (('train', 36), ('validation', 6)):
        selected = [r for r in rows if r['split'] == split]
        if len(selected) != expected:
            raise ContractError('frozen workload split inventory mismatch')
        for row in selected:
            # Labels, text-with-answer, encodings and family metadata never cross this boundary.
            result.append({'item_id': row['example_identity'], 'split': split,
                           'prompt': row['prompt'], 'prompt_identity': identity(row['prompt'].encode())})
    validate_rows(result)
    return result


def validate_rows(rows):
    if type(rows) is not list or not rows:
        raise ContractError('nonempty workload list required')
    ids = []
    for row in rows:
        if type(row) is not dict or set(row) != {'item_id', 'split', 'prompt', 'prompt_identity'}:
            raise ContractError('workload requires label-free exact fields')
        if (type(row['item_id']) is not str or type(row['prompt']) is not str
                or row['split'] not in ('train', 'validation')
                or row['prompt_identity'] != identity(row['prompt'].encode())
                or not 1 <= len(row['prompt'].encode()) + 1 <= 32):
            raise ContractError('invalid workload prompt/identity/split/context')
        ids.append(row['item_id'])
    if len(set(ids)) != len(ids):
        raise ContractError('workload item identities must be unique')


def schedule(rows, layout):
    validate_rows(rows)
    if layout not in LAYOUTS:
        raise ContractError('unknown layout')
    order = list(range(len(rows)))
    if layout == 'recompose8_32':
        order = order[::2] + order[1::2]
    size = 1 if layout.startswith('singleton') else 4 if layout == 'partition4_32' else 8
    groups = [order[i:i+size] for i in range(0, len(order), size)]
    if layout == 'reverse8_32':
        groups = [list(reversed(g)) for g in groups]
    validate_schedule(rows, groups)
    return groups


def validate_schedule(rows, groups):
    if type(groups) is not list or any(type(g) is not list or not 1 <= len(g) <= 8 for g in groups):
        raise ContractError('invalid batch groups')
    flat = [i for g in groups for i in g]
    if any(type(i) is not int for i in flat) or sorted(flat) != list(range(len(rows))):
        raise ContractError('schedule must cover every row exactly once')


def reference_report():
    compressed = read(ROOT/'fixtures/frozen-decoding-v1/report.json.gz', 16 * 1024 * 1024)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024 or identity(raw) != REFERENCE_ID:
        raise ContractError('reference trajectory report identity/size mismatch')
    old = parse_json(raw, canonical=True)
    if [(c['seed'], c['arm']) for c in old['cells']] != list(CELLS):
        raise ContractError('reference endpoint inventory mismatch')
    return old


def references(rows):
    old = reference_report()
    result = {}
    for cell in old['cells']:
        predictions = cell['modes']['syntax']['predictions']
        expected_ids = [r['item_id'] for r in rows]
        if [r['example_identity'] for r in predictions] != expected_ids:
            raise ContractError('reference trajectory item order mismatch')
        tokens = []
        for row, prediction in zip(rows, predictions):
            value = prediction['generated_tokens']
            if (prediction['prompt_identity'] != row['prompt_identity']
                    or type(value) is not list or len(value) != 2
                    or type(value[0]) is not int or not 48 <= value[0] <= 57 or value[1] != 257):
                raise ContractError('reference syntax trajectory malformed')
            tokens.append(value)
        result[(cell['seed'], cell['arm'])] = tokens
    return result


def sources():
    from frozen_decoding_v1.experiment import sources as historical_sources
    paths = []
    for folder in ('scripts/batch_composition_v1', 'tests/batch_composition_v1', 'tests/batch_composition_review'):
        paths.extend((ROOT/folder).glob('*.py'))
    paths.extend(ROOT/p for p in (
        'scripts/batch_composition_v1/protocol.json', 'scripts/run_batch_composition.py',
        'scripts/check_batch_composition_cpu.py', '.github/workflows/batch-composition.yml',
        'scripts/frozen_decoding_transport_v2/__init__.py',
        'scripts/frozen_decoding_transport_v2/artifacts.py'))
    return {**historical_sources(), **{p.relative_to(ROOT).as_posix(): identity(read(p, MAX_REPORT)) for p in sorted(paths)}}


def request():
    from frozen_decoding_v1.experiment import original, REPORT_ID
    from trinite.training import require_environment
    old = original()
    rows = workload()
    refs = references(rows)
    return {'schema': 'trinite.batch-composition-request.v2', 'protocol': protocol(),
            'source': sources(), 'environment': require_environment(),
            'original_report_identity': REPORT_ID, 'reference_report_identity': REFERENCE_ID,
            'checkpoint_identities': [identity(json_bytes(c['final_checkpoint'])) for c in old['cells']],
            'workload': rows, 'reference_trajectory_identities': [identity(json_bytes(refs[k])) for k in CELLS],
            'schedules': {layout: schedule(rows, layout) for layout in LAYOUTS}}
