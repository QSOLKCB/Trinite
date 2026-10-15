"""Published pre-outcome specification and explicit source closure."""
from pathlib import Path
from trinite.contracts import ContractError, identity, parse_json

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'scripts/learned_scales_v1/protocol.json'
SPEC_COMMIT = '430b988a013ec1fa9523c75a293b5c698ebc7b2c'
PROTOCOL_ID = 'sha256:08554101a99dde509336f213d2a0277a4700e4d77e4d7086a920f317e821b4be'


def specification(profile='engineering'):
    if type(profile) is not str or profile != 'engineering':
        raise ContractError('scientific execution requires a new reviewed protocol and resource authorization')
    raw = PROTOCOL.read_bytes()
    if identity(raw) != PROTOCOL_ID:
        raise ContractError('engineering specification differs from published bytes')
    return parse_json(raw)


def sources():
    paths = list((ROOT/'src/trinite').rglob('*.py'))
    paths += list((ROOT/'scripts/learned_scales_v1').glob('*.py'))
    paths += list((ROOT/'tests/learned_scales_v1').glob('*.py'))
    paths += [PROTOCOL, ROOT/'scripts/check_learned_scales.py', ROOT/'scripts/check_learned_scales_cpu.py',
              ROOT/'requirements.lock', ROOT/'src/trinite/cpu-dependencies.lock',
              ROOT/'.github/workflows/learned-scales.yml']
    return {p.relative_to(ROOT).as_posix(): identity(p.read_bytes()) for p in sorted(paths)}
