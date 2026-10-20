"""Frozen pilot protocol and complete relevant source/input bindings."""
from pathlib import Path
from trinite.contracts import ContractError, identity, parse_json
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'scripts/scale_refit_training_v1/protocol.json'
PROTOCOL_ID = 'sha256:bf9f382a88ad66e6926c4f48755f8cc364813bbd3d1996213464da3f1c792c2a'

def specification():
    raw = PROTOCOL.read_bytes()
    if identity(raw) != PROTOCOL_ID:
        raise ContractError('pilot protocol differs from frozen bytes')
    return parse_json(raw, canonical=True)

def inputs():
    spec = specification()
    result = [(ROOT/'fixtures/formal-v1'/n).read_bytes() for n in ('dataset.json', 'manifest.json')]
    if any(identity(raw) != spec['data'][n] for n, raw in zip(('dataset.json','manifest.json'), result)):
        raise ContractError('pilot formal inputs differ from frozen identities')
    return result

def sources():
    paths = list((ROOT/'src/trinite').rglob('*.py'))
    for folder in ('scripts/scale_refit', 'scripts/scale_refit_training_v1', 'tests/scale_refit_training_v1'):
        paths += list((ROOT/folder).glob('*.py'))
    paths += [PROTOCOL, ROOT/'scripts/run_scale_refit_training.py', ROOT/'scripts/check_scale_refit_training_cpu.py', ROOT/'requirements.lock',
              ROOT/'src/trinite/cpu-dependencies.lock', ROOT/'.github/workflows/scale-refit-training.yml']
    return {p.relative_to(ROOT).as_posix(): identity(p.read_bytes()) for p in sorted(paths)}
