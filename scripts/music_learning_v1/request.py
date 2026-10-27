"""Exact source, environment, admission and plan commitment before outcomes."""
import os
from pathlib import Path
import stat
import sys
from trinite.contracts import ContractError, identity, json_bytes, parse_json, require_identity
from .protocol import PROTOCOL, SEEDS
from .admission import dataset, audit_bytes
ROOT = Path(__file__).resolve().parents[2]


def read(path, limit=16*1024*1024):
    path = Path(path).absolute()
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ContractError('artifact ancestors must be non-link directories')
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ContractError('bounded regular artifact required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(fd)
        signature = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
        if signature(opened) != signature(before):
            raise ContractError('artifact changed at open')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(limit+1)
        if (len(data)>limit or signature(os.fstat(fd))!=signature(before)
                or signature(path.lstat())!=signature(before)):
            raise ContractError('artifact changed while reading')
        return data
    finally:
        os.close(fd)


def sources():
    paths = list((ROOT/'src').rglob('*.py'))
    paths += list((ROOT/'src/trinite').glob('*.json')) + list((ROOT/'src/trinite').glob('*.lock'))
    paths += list((ROOT/'scripts/music_learning_v1').glob('*.py'))
    paths += list((ROOT/'scripts/symbolic_music_v1').glob('*.py'))
    paths += [ROOT/p for p in ('AGENTS.md','requirements.lock','scripts/run_music_learning.py',
        'scripts/run_symbolic_music.py','docs/MUSIC-LEARNING.md','docs/SYMBOLIC-MUSIC.md',
        'scripts/native_diagnostic_v1/__init__.py','scripts/native_diagnostic_v1/artifacts.py',
        'scripts/unseen_evaluation_v1/__init__.py','scripts/unseen_evaluation_v1/packet.py')]
    pin = parse_json(read(ROOT/'src/trinite/provenance-pin.json'), canonical=True)
    paths += [ROOT/p for p in pin['files']]
    allowed = {p.absolute() for p in paths}
    for name,module in tuple(sys.modules.items()):
        if name.split('.')[0] in ('trinite','music_learning_v1','symbolic_music_v1',
                'native_diagnostic_v1','unseen_evaluation_v1') or name.startswith('provenance_'):
            file = getattr(module,'__file__',None)
            if file is None or Path(file).absolute() not in allowed or Path(file).is_symlink():
                raise ContractError('music import outside bound checkout: '+name)
    return {p.relative_to(ROOT).as_posix(): identity(read(p)) for p in sorted(set(paths))}


def cell_name(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ContractError('unknown frozen music cell')
    return 'dense-'+str(seed)


def core():
    from trinite.training import require_environment
    from .numerical import plan_for, schedule_receipt
    raw, admitted = dataset(); audit_bytes(raw, admitted)
    return {'schema':'trinite.music-learning-request.v1','protocol':PROTOCOL,
        'source':sources(),'environment':require_environment(),
        'dataset_identity':identity(raw),'admission_identity':identity(admitted),
        'cells':{cell_name(seed):{'seed':seed,'plan':plan_for(seed).to_dict(),
                               'schedule':schedule_receipt(seed)} for seed in SEEDS}}


def validate(raw, expected):
    require_identity(expected)
    if type(raw) is not bytes or len(raw)>1024*1024 or identity(raw)!=expected:
        raise ContractError('music request identity/size differs')
    value = parse_json(raw,canonical=True)
    if raw != json_bytes(core()):
        raise ContractError('music request source/environment/admission/protocol differs')
    return value
