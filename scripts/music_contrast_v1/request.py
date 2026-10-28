"""Exact source/environment/admission commitment, independent of older requests."""
from pathlib import Path
import sys
from trinite.contracts import ContractError, identity, json_bytes, parse_json, require_identity
from music_learning_v1.request import ROOT, read, sources as upstream_sources
from music_learning_v1.numerical import plan_for
from .spec import SEEDS, PROTOCOL, dataset, schedule_receipt


def cell_name(seed, phase):
    if type(seed) is not int or seed not in SEEDS or phase not in ('train', 'heldout'):
        raise ContractError('unknown frozen contrast cell')
    return phase + '-' + str(seed)


def sources():
    result = upstream_sources()
    paths = list((ROOT/'scripts/music_contrast_v1').glob('*.py'))
    paths += [ROOT/'scripts/run_music_contrast.py', ROOT/'docs/MUSIC-CONTRAST.md']
    allowed = {p.absolute() for p in paths}
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] == 'music_contrast_v1':
            source = getattr(module, '__file__', None)
            if source is None or Path(source).absolute() not in allowed or Path(source).is_symlink():
                raise ContractError('foreign contrast import: ' + name)
    for base in (ROOT/'src', ROOT/'scripts/music_contrast_v1',
                 ROOT/'scripts/music_learning_v1', ROOT/'scripts/symbolic_music_v1'):
        for path in base.rglob('*'):
            if path.is_symlink() or path.name == '__pycache__' or path.suffix in ('.pyc', '.pyo', '.so', '.pyd'):
                raise ContractError('import caches, links or extension shadows are forbidden')
    result.update({p.relative_to(ROOT).as_posix(): identity(read(p)) for p in paths})
    return result


def core():
    from trinite.training import require_environment
    raw, receipt = dataset()
    return {'schema': 'trinite.music-contrast-request.v1', 'protocol': PROTOCOL,
            'source': sources(), 'environment': require_environment(),
            'dataset_identity': identity(raw), 'admission_identity': identity(receipt),
            'cells': {cell_name(seed, phase): {'seed': seed, 'phase': phase,
                **({'plan': plan_for(seed).to_dict(), 'schedule': schedule_receipt(),
                    'requested_work': {'training_updates': 64, 'decode_rows': 80}}
                   if phase == 'train' else {'source_checkpoint_plan': plan_for(seed).to_dict(),
                    'requested_work': {'training_updates': 0, 'decode_rows': 120,
                                       'checkpoint_step': 64, 'conditional_on_both_training_seeds': True}})}
                for phase in ('train', 'heldout') for seed in SEEDS}}


def validate(raw, expected):
    require_identity(expected)
    if type(raw) is not bytes or len(raw) > 1024*1024 or identity(raw) != expected:
        raise ContractError('contrast request size/identity differs')
    value = parse_json(raw, canonical=True)
    if raw != json_bytes(core()):
        raise ContractError('contrast source/environment/admission/protocol differs')
    return value
