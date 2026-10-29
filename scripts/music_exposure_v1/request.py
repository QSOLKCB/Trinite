"""Exact source/environment/admission-bound train-only request."""
from pathlib import Path
import sys
from trinite.contracts import ContractError, identity, json_bytes, parse_json, require_identity
from music_learning_v1.request import ROOT, read, sources as upstream_sources
from .spec import SEEDS, CHECKPOINTS, PROTOCOL, dataset, schedule_receipt, plan_for


def cell_name(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ContractError('unknown frozen exposure cell')
    return 'train-' + str(seed)


def sources():
    result = upstream_sources()
    paths = list((ROOT/'scripts/music_exposure_v1').glob('*.py'))
    paths += [ROOT/p for p in ('scripts/run_music_exposure.py', 'docs/MUSIC-EXPOSURE.md',
        'scripts/music_contrast_v1/__init__.py', 'scripts/music_contrast_v1/spec.py',
        'scripts/music_contrast_v1/reporting.py')]
    allowed = {p.absolute() for p in paths}
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in ('music_exposure_v1', 'music_contrast_v1'):
            source = getattr(module, '__file__', None)
            if source is None or Path(source).absolute() not in allowed or Path(source).is_symlink():
                raise ContractError('foreign or unbound exposure import: ' + name)
    for base in (ROOT/'src', ROOT/'scripts'):
        for path in base.rglob('*'):
            if path.is_symlink() or path.name == '__pycache__' or path.suffix in ('.pyc', '.pyo', '.so', '.pyd'):
                raise ContractError('import caches, links or extension shadows are forbidden')
    result.update({p.relative_to(ROOT).as_posix(): identity(read(p)) for p in paths})
    return result


def core():
    from trinite.training import require_environment
    raw, receipt = dataset()
    return {'schema': 'trinite.music-exposure-request.v1', 'protocol': PROTOCOL,
        'source': sources(), 'environment': require_environment(),
        'dataset_identity': identity(raw), 'admission_identity': identity(receipt),
        'cells': {cell_name(seed): {'seed': seed, 'plan': plan_for(seed).to_dict(),
            'schedules': {str(step): schedule_receipt(step) for step in CHECKPOINTS},
            'requested_work': {'training_updates': 640, 'decode_rows': 120, 'heldout_decodes': 0}}
            for seed in SEEDS}}


def validate(raw, expected):
    require_identity(expected)
    if type(raw) is not bytes or len(raw) > 1024*1024 or identity(raw) != expected:
        raise ContractError('exposure request size/identity differs')
    value = parse_json(raw, canonical=True)
    if raw != json_bytes(core()):
        raise ContractError('exposure source/environment/admission/protocol differs')
    return value
