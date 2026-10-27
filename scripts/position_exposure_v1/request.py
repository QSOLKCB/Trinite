"""Frozen source/environment/admission request; no model outcomes select inputs."""
from pathlib import Path
import sys

from trinite.contracts import ContractError, identity, json_bytes, parse_json, require_identity
from balanced_addition_v1.request import read
from .data import dataset, audit_bytes, validate_matching
from .protocol import PROTOCOL, ARMS, SEEDS

ROOT = Path(__file__).resolve().parents[2]


def sources():
    from position_study_v1.request import sources as historical_sources
    original = historical_sources()
    paths = list((ROOT/'scripts/position_exposure_v1').glob('*.py'))
    paths += [ROOT/'scripts/run_position_exposure.py', ROOT/'docs/POSITION-EXPOSURE-V1.md']
    paths += list((ROOT/'scripts/position_coupling_v1').glob('*.py'))
    paths += [ROOT/'scripts/check_position_coupling.py',ROOT/'docs/POSITION-COUPLING.md']
    paths += list((ROOT/'tests/position_exposure_v1').glob('*.py'))
    paths += [ROOT/'.github/workflows/position-exposure-v1.yml']
    result = {**original, **{p.relative_to(ROOT).as_posix():identity(read(p)) for p in sorted(paths)}}
    # Canonical names, not suffix matching: reject foreign or flattened imports.
    expected_modules = {}
    for name in result:
        p = Path(name)
        if p.suffix != '.py' or p.parts[0] not in ('scripts','src'): continue
        parts = list(p.parts[1:]); parts[-1] = parts[-1][:-3]
        if parts[-1] == '__init__': parts.pop()
        expected_modules['.'.join(parts)] = (ROOT/p).resolve()
    for name, module in tuple(sys.modules.items()):
        if name in expected_modules:
            file = getattr(module, '__file__', None)
            if file is None or Path(file).resolve() != expected_modules[name]:
                raise ContractError('position exposure import outside source-bound checkout')
        elif name.split('.')[0] in ('position_exposure_v1','position_coupling_v1'):
            raise ContractError('unbound position module')
    return result


def cell_name(arm, seed):
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ContractError('unknown fixed position exposure cell')
    return f'{arm}-{seed}'


def cells():
    return [(arm, seed) for seed in SEEDS for arm in ARMS]


def core():
    from trinite.training import require_environment
    from .numerical import plan_for
    from .adapter import addressing
    from .protocol import MODES
    from .reporting import design_summary
    raw, manifest = dataset(); audit_bytes(raw, manifest)
    return {'schema': 'trinite.position-exposure-request.v1', 'protocol': PROTOCOL,
        'protocol_identity': identity(json_bytes(PROTOCOL)), 'source': sources(),
        'environment': require_environment(), 'dataset_identity': identity(raw),
        'admission_identity': identity(manifest), 'matching': validate_matching(),
        'design_inventory':design_summary(),
        'cells': {cell_name(arm, seed): {'arm': arm, 'seed': seed,
                  'addressing':addressing(arm), 'plan': plan_for(seed).to_dict()}
                  for arm, seed in cells()}}


def validate(raw, expected):
    require_identity(expected)
    if type(raw) is not bytes or len(raw) > 1024*1024 or identity(raw) != expected:
        raise ContractError('position exposure request identity/size differs')
    parse_json(raw, canonical=True)
    if raw != json_bytes(core()):
        raise ContractError('position exposure request source/environment/data/protocol differs')
    return parse_json(raw, canonical=True)
