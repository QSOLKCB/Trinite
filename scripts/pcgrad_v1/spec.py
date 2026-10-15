"""Published engineering spec; no scientific profile or override exists."""
from pathlib import Path
from trinite.contracts import ContractError, identity, parse_json

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'scripts/pcgrad_v1/protocol.json'
SPEC_COMMIT = '56b253af6f6630ade09b001cff7023d42c652d01'
PROTOCOL_ID = 'sha256:514220181e9f9e85782318c252911336e081bf6ef67587a8922ee806da1f3ad1'


def specification(profile='engineering'):
    if type(profile) is not str or profile != 'engineering':
        raise ContractError('scientific execution requires a new reviewed protocol and resource authorization')
    raw = PROTOCOL.read_bytes()
    if identity(raw) != PROTOCOL_ID:
        raise ContractError('engineering specification differs from published bytes')
    return parse_json(raw)


def sources():
    paths = list((ROOT/'src/trinite').rglob('*.py'))
    paths += list((ROOT/'scripts/pcgrad_v1').glob('*.py'))
    paths += list((ROOT/'tests/pcgrad_v1').glob('*.py'))
    paths += [PROTOCOL, ROOT/'scripts/check_pcgrad.py', ROOT/'scripts/check_pcgrad_cpu.py',
              ROOT/'requirements.lock', ROOT/'src/trinite/cpu-dependencies.lock',
              ROOT/'.github/workflows/pcgrad.yml']
    return {p.relative_to(ROOT).as_posix(): identity(p.read_bytes()) for p in sorted(paths)}
