"""Frozen source/environment/admission request; no model outcomes select inputs."""
from pathlib import Path
import sys

from trinite.contracts import ContractError, identity, integer, json_bytes, parse_json, require_identity
from .data import dataset, audit_bytes, validate_matching
from .protocol import PROTOCOL, ARMS, SEEDS

ROOT = Path(__file__).resolve().parents[2]


def read(path, limit=16*1024*1024):
    import stat
    path=Path(path);before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size>limit:
        raise ContractError('bounded regular artifact required')
    with path.open('rb') as stream:data=stream.read(limit+1)
    after=path.lstat()
    fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
    if len(data)>limit or fields(before)!=fields(after):
        raise ContractError('artifact changed or exceeded bound')
    return data


def sources():
    from failure_map_v1.request import sources as historical_sources
    original = historical_sources()
    paths = list((ROOT/'scripts/balanced_addition_v1').glob('*.py'))
    paths += [ROOT/'scripts/run_balanced_addition.py', ROOT/'docs/BALANCED-ADDITION.md']
    allowed = {p.resolve() for p in paths}
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] == 'balanced_addition_v1':
            file = getattr(module, '__file__', None)
            if file is None or Path(file).resolve() not in allowed:
                raise ContractError('balanced addition import outside source-bound checkout')
    return {**original, **{p.relative_to(ROOT).as_posix(): identity(read(p)) for p in sorted(paths)}}


def cell_name(arm, seed):
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ContractError('unknown fixed addition cell')
    return f'{arm}-{seed}'


def cells():
    return [(arm, seed) for seed in SEEDS for arm in ARMS]


def core():
    from trinite.training import require_environment
    from .numerical import plan_for
    raw, manifest = dataset(); audit_bytes(raw, manifest)
    return {'schema': 'trinite.balanced-addition-request.v1', 'protocol': PROTOCOL,
        'protocol_identity': identity(json_bytes(PROTOCOL)), 'source': sources(),
        'environment': require_environment(), 'dataset_identity': identity(raw),
        'admission_identity': identity(manifest), 'matching': validate_matching(),
        'cells': {cell_name(arm, seed): {'arm': arm, 'seed': seed, 'plan': plan_for(seed).to_dict()}
                  for arm, seed in cells()}}


def validate(raw, expected):
    require_identity(expected)
    if type(raw) is not bytes or len(raw) > 1024*1024 or identity(raw) != expected:
        raise ContractError('addition request identity/size differs')
    parse_json(raw, canonical=True)
    if raw != json_bytes(core()):
        raise ContractError('addition request source/environment/data/protocol differs')
    return parse_json(raw, canonical=True)
