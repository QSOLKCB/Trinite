"""Bind original public bytes and recompute retained metrics; no model execution."""
from pathlib import Path
import sys

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from position_study_v1.data import dataset, audit_bytes, exposure
from position_study_v1.numerical import check_score
from position_study_v1.reporting import describe, criteria, controls
from position_study_v1.protocol import MILESTONES, STRATA
from unseen_evaluation_v1.packet import inventory, unchanged

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'fixtures/position-study-result-20261026-v1'
MATRIX = 'sha256:bcfee16fbbdddff24442027d22e32fe321dd0294479e40e78f44f0a929b5a762'
REQUEST = 'sha256:bdf6d3bfc7e05467f324e2794cfec70791b082ac08fad384b00fbcfcd6c09623'
SOURCE_RECEIPT = 'sha256:69bc803e9ce63d9a3e0a0ce8d3aa5c5bed39f309f9f54f681d27432681c29962'
CELLS = tuple(f'{arm}-{seed}' for arm in ('control', 'candidate') for seed in range(3))
LABELS = tuple((f'training-{step}', 'train', step) for step in MILESTONES) + tuple(
    (f'development-{s}', s, 4000) for s in STRATA)
FILES = ('completion.json', 'report.json', 'description.json',
         *(f'{label}.{suffix}' for label, _, _ in LABELS for suffix in ('json',)),
         *(f'raw-{label}.jsonl' for label, _, _ in LABELS))
CHECKER_FILES = ('scripts/check_position_result.py', 'scripts/position_result_v1/__init__.py',
                 'scripts/position_result_v1/check.py')


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4*1024*1024:
        raise ContractError('expected bounded regular public artifact')
    with path.open('rb') as stream:
        raw = stream.read(4*1024*1024+1)
    if len(raw) > 4*1024*1024:
        raise ContractError('public artifact too large')
    return raw


def source_check(sources):
    for name, expected in sources.items():
        if identity(read(ROOT/name)) != expected:
            raise ContractError('historical producer source differs: '+name)
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
        origin = getattr(module, '__file__', None)
        if origin is None or Path(origin).resolve() != (ROOT/name).resolve():
            raise ContractError('repository dependency imported from a foreign root')


def verify(fixture=FIXTURE):
    fixture = Path(fixture)
    before = inventory(fixture)
    checker_sources = {name: identity(read(ROOT/name)) for name in CHECKER_FILES}
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
    source_check(receipt['sources'])
    raw = read(fixture/'matrix.json')
    if identity(raw) != MATRIX:
        raise ContractError('original matrix identity differs')
    matrix = parse_json(raw, canonical=True)
    if (matrix['request_identity'] != REQUEST or set(matrix['cells']) != set(CELLS)
            or matrix['outcome'] != 'completed'):
        raise ContractError('matrix request/cell/outcome differs')
    gates = ('learning_adequate', 'fresh_confirmation', 'phase5_ready', 'release_ready')
    if any(matrix[k] is not False for k in gates):
        raise ContractError('unsupported matrix scientific gate')
    data, admission = dataset()
    audit_bytes(data, admission)
    if (identity(data), identity(admission)) != (receipt['dataset_identity'], receipt['admission_identity']):
        raise ContractError('original admission differs')
    result, descriptions = {}, {}
    prediction_count = 0
    for cell in CELLS:
        bound = matrix['cells'][cell]
        records, originals = {}, {}
        for name in FILES:
            raw = read(fixture/cell/name)
            if identity(raw) != bound['retained_files'][name]:
                raise ContractError('original retained artifact differs: '+cell+'/'+name)
            originals[name] = raw
            if name.endswith('.json'):
                records[name] = parse_json(raw, canonical=True)
        completion = records['completion.json']
        if (completion['request_identity'] != REQUEST or completion['cell'] != cell
                or bound['completion_identity'] != identity(originals['completion.json'])
                or bound['outcome'] != 'completed'):
            raise ContractError('completion linkage differs')
        for name in FILES:
            if name != 'completion.json' and completion['artifacts'][name] != bound['retained_files'][name]:
                raise ContractError('completion artifact linkage differs')
        arm, seed = cell.split('-')
        report = records['report.json']
        summaries = {}
        for label, selector, step in LABELS:
            score = records[label+'.json']
            summaries[label] = check_score(score, arm, selector, score['model_identity'])
            predictions = score['predictions']
            expected_journal = b''.join(json_bytes({
                'schema': 'trinite.position-study-prediction-journal.v1',
                'request_identity': REQUEST, 'cell': cell, 'step': step,
                'selector': selector, 'addressing': score['addressing'],
                'model_identity': score['model_identity'], 'prediction': p}) for p in predictions)
            if originals['raw-'+label+'.jsonl'] != expected_journal:
                raise ContractError('raw journal differs from score predictions')
            prediction_count += len(predictions)
        training = records['training-4000.json']
        development = {s: records[f'development-{s}.json'] for s in STRATA}
        if any(r['model_identity'] != report['final_model_identity'] for r in (training, *development.values())):
            raise ContractError('final score model linkage differs')
        description = describe(arm, training, development)
        if json_bytes(description) != json_bytes(records['description.json']):
            raise ContractError('retained descriptive map differs')
        if (report['training'] != {str(s): summaries[f'training-{s}'] for s in MILESTONES}
                or report['development'] != {s: summaries[f'development-{s}'] for s in STRATA}
                or report['controls'] != {s: controls(arm, s) for s in STRATA}):
            raise ContractError('original report scoring differs')
        if report['exposure'] != exposure(arm) or report['steps'] != 4000 or report['target_tokens'] != 121600:
            raise ContractError('reported schedule differs')
        if (report['cell'] != cell or report['seed'] != int(seed) or report['arm'] != arm
                or report['request_identity'] != REQUEST or report['outcome'] != 'completed'):
            raise ContractError('report cell differs')
        if any(report[k] is not False for k in gates):
            raise ContractError('unsupported scientific gate')
        descriptions[cell] = description
        result[cell] = {'model_identity': report['final_model_identity'],
            'training_milestones': report['training'],
            'seen': description['seen_training_anchors']['total'],
            'development': {s: description['development'][s]['total'] for s in STRATA}}
    unchanged(fixture, before)
    source_check(receipt['sources'])
    if any(identity(read(ROOT/name)) != digest for name, digest in checker_sources.items()):
        raise ContractError('checker source changed during verification')
    return {'schema': 'trinite.position-study-public-subset-check.v1',
            'matrix_identity': MATRIX, 'request_identity': REQUEST, 'cells': result,
            'criteria': criteria(descriptions), 'prediction_records': prediction_count,
            'checker_sources': checker_sources, 'producer_source_binding_identity': SOURCE_RECEIPT,
            'original_files': 115, 'closed_custody': False, 'checkpoint_tensors_verified': False,
            'inference_replayed': False, 'optimizer_replayed': False,
            'fresh_confirmation': False, 'learning_adequate': False,
            'phase5_ready': False, 'release_ready': False}
