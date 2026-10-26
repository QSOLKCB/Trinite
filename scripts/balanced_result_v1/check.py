"""Authenticate original public bytes and recompute scores; no numerical replay."""
from pathlib import Path
import sys

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from balanced_addition_v1.data import dataset, audit_bytes, exposure
from balanced_addition_v1.numerical import check_score
from balanced_addition_v1.reporting import describe
from unseen_evaluation_v1.packet import inventory, unchanged

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/balanced-addition-result-20261026-v1'
MATRIX = 'sha256:930040ebe307083052bf8f1c004af20f65ca697bce94b4e5a2c4331b1294d412'
REQUEST = 'sha256:897d9dae78eed8c50e6c899610810c49867ffec5067a941cc60ea7e4828f76d7'
SOURCE_RECEIPT = 'sha256:b86ed33bac62577742e50a14892e5bc665d51604ab55a7dd091a26117b86804c'
CELLS = tuple(f'{arm}-{seed}' for arm in ('control', 'candidate') for seed in range(3))
STRATA = ('shared-answer', 'broadened-answer', 'range-length')
FILES = ('completion.json', 'training-4000.json', 'report.json', 'description.json',
         *(f'development-{s}.json' for s in STRATA))


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4*1024*1024:
        raise ContractError('expected bounded regular public artifact')
    with path.open('rb') as stream:
        raw = stream.read(4*1024*1024+1)
    if len(raw) > 4*1024*1024:
        raise ContractError('public artifact too large')
    return raw


def verify(fixture=FIXTURE):
    fixture = Path(fixture)
    before = inventory(fixture)
    checker_sources = {name: identity(read(ROOT/name)) for name in (
        'scripts/check_balanced_result.py', 'scripts/balanced_result_v1/__init__.py',
        'scripts/balanced_result_v1/check.py')}
    expected = {'.', 'README.md', 'matrix.json', 'source-bindings.json', *CELLS,
                *(f'{cell}/{name}' for cell in CELLS for name in FILES)}
    if set(before['members']) != expected:
        raise ContractError('public subset membership differs')
    source_raw = read(fixture/'source-bindings.json')
    if identity(source_raw) != SOURCE_RECEIPT:
        raise ContractError('derived source binding receipt changed')
    receipt = parse_json(source_raw, canonical=True)
    if receipt['request_identity'] != REQUEST:
        raise ContractError('source binding request differs')
    for name, expected_hash in receipt['sources'].items():
        if identity(read(ROOT/name)) != expected_hash:
            raise ContractError('historical producer source differs: '+name)
    # Bind canonical module names, not suffixes of potentially foreign paths.
    for name in receipt['sources']:
        parts = Path(name).parts
        if parts[0] not in ('src', 'scripts') or not name.endswith('.py'):
            continue
        names = list(parts[1:])
        names[-1] = names[-1][:-3]
        if names[-1] == '__init__':
            names.pop()
        module = sys.modules.get('.'.join(names))
        if module is None:
            continue
        file = getattr(module, '__file__', None)
        if file is None or Path(file).resolve() != (ROOT/name).resolve():
            raise ContractError('repository dependency imported from a foreign root')
    raw = read(fixture/'matrix.json')
    if identity(raw) != MATRIX:
        raise ContractError('original matrix identity differs')
    matrix = parse_json(raw, canonical=True)
    if matrix['request_identity'] != REQUEST or set(matrix['cells']) != set(CELLS):
        raise ContractError('matrix request/cell membership differs')
    data, admission = dataset()
    audit_bytes(data, admission)
    if (identity(data), identity(admission)) != (receipt['dataset_identity'], receipt['admission_identity']):
        raise ContractError('original admission differs')
    result = {}
    for cell in CELLS:
        bound = matrix['cells'][cell]
        records = {}
        for name in FILES:
            raw = read(fixture/cell/name)
            if identity(raw) != bound['retained_files'][name]:
                raise ContractError('original retained artifact differs: '+cell+'/'+name)
            records[name] = parse_json(raw, canonical=True)
        completion = records['completion.json']
        if completion['request_identity'] != REQUEST or completion['cell'] != cell:
            raise ContractError('completion linkage differs')
        for name in FILES:
            if name != 'completion.json' and completion['artifacts'][name] != bound['retained_files'][name]:
                raise ContractError('completion artifact linkage differs')
        arm, seed = cell.split('-')
        report = records['report.json']
        model = report['final_model_identity']
        training = records['training-4000.json']
        development = {s: records[f'development-{s}.json'] for s in STRATA}
        training_summary = check_score(training, arm, 'train', model)
        summaries = {s: check_score(record, arm, s, model) for s, record in development.items()}
        description = describe(arm, training, development)
        if json_bytes(description) != json_bytes(records['description.json']):
            raise ContractError('retained descriptive map differs')
        if report['training']['4000'] != training_summary or report['development'] != summaries:
            raise ContractError('original report scoring differs')
        if report['exposure'] != exposure(arm) or report['steps'] != 4000 or report['target_tokens'] != 76800:
            raise ContractError('reported schedule differs')
        if report['cell'] != cell or report['seed'] != int(seed) or report['arm'] != arm:
            raise ContractError('report cell differs')
        if any(report[k] is not False for k in ('learning_adequate', 'fresh_confirmation', 'phase5_ready', 'release_ready')):
            raise ContractError('unsupported scientific gate')
        result[cell] = {'model_identity': model,
            'unique_training': description['unique_training']['total'],
            'seen_anchors': description['original_control_seen_anchors']['total'],
            'primary_in_range': description['primary_in_range_development']['total'],
            'development': {s: description['shared_development'][s]['total'] for s in STRATA}}
    unchanged(fixture, before)
    if any(identity(read(ROOT/name)) != digest for name, digest in checker_sources.items()):
        raise ContractError('checker source changed during verification')
    return {'schema': 'trinite.balanced-addition-public-subset-check.v1',
            'matrix_identity': MATRIX, 'request_identity': REQUEST, 'cells': result,
            'checker_sources': checker_sources, 'producer_source_binding_identity': SOURCE_RECEIPT,
            'original_files': 43, 'closed_custody': False, 'checkpoint_tensors_verified': False,
            'inference_replayed': False, 'optimizer_replayed': False,
            'fresh_confirmation': False, 'learning_adequate': False,
            'phase5_ready': False, 'release_ready': False}
