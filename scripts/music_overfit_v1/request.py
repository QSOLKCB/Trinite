"""Source-bound new request, preserving the original music request unchanged."""
from pathlib import Path
import sys
from trinite.contracts import ContractError, identity, json_bytes, parse_json, require_identity
from music_learning_v1.request import ROOT, read, sources as original_sources
from music_learning_v1.numerical import plan_for
from .spec import CELLS, PROTOCOL, admitted, schedule_receipt


def cell_name(seed, subset):
    if type(seed) is not int or (seed, subset) not in CELLS:
        raise ContractError('unknown frozen overfit cell')
    return subset + '-dense-' + str(seed)


def sources():
    result = original_sources()
    paths = sorted((ROOT / 'scripts/music_overfit_v1').glob('*.py'))
    paths += [ROOT / 'scripts/run_music_overfit.py', ROOT / 'docs/MUSIC-OVERFIT.md']
    allowed = {str(p) for p in paths}
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] == 'music_overfit_v1':
            if getattr(module, '__file__', None) not in allowed:
                raise ContractError('foreign overfit import: ' + name)
    result.update({p.relative_to(ROOT).as_posix(): identity(read(p)) for p in paths})
    return result


def core():
    from trinite.training import require_environment
    raw, receipt, _ = admitted()
    return {'schema': 'trinite.music-overfit-request.v1', 'protocol': PROTOCOL,
            'source': sources(), 'environment': require_environment(),
            'dataset_identity': identity(raw), 'admission_identity': identity(receipt),
            'cells': {cell_name(seed, subset): {'seed': seed, 'subset': subset,
                'plan': plan_for(seed).to_dict(), 'schedule': schedule_receipt(subset)}
                for seed, subset in CELLS}}


def validate(raw, expected):
    require_identity(expected)
    if type(raw) is not bytes or len(raw) > 1048576 or identity(raw) != expected:
        raise ContractError('overfit request identity/size differs')
    value = parse_json(raw, canonical=True)
    if raw != json_bytes(core()):
        raise ContractError('overfit source/environment/admission/protocol differs')
    return value
