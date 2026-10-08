"""Pinned offline QEC dependency; no source prose enters training examples."""
import hashlib
import importlib
from pathlib import Path

from .contracts import ContractError, identity, parse_json

REVISION = '32fdf9c88f4ac1b873c8acc4d3d54c50f87d6e0a'
PIN_IDENTITY = 'sha256:168301a4c4568f289a641ffe51c64972ff5c194ca0d0a089cc57dc4b2e37cf52'


def verify_pin(package=None):
    root = Path(package) if package is not None else Path(__file__).resolve().parent
    raw = (root/'qec-pin.json').read_bytes()
    if identity(raw) != PIN_IDENTITY:
        raise ContractError('QEC pin changed')
    pin = parse_json(raw, canonical=True)
    if pin['revision'] != REVISION:
        raise ContractError('unsupported QEC revision')
    for name, record in pin['files'].items():
        path = root/name
        if any(p.is_symlink() for p in [path, *path.parents]) or not path.is_file():
            raise ContractError('unsafe or missing QEC source: '+name)
        with path.open('rb') as stream:
            content = stream.read(65537)
        blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if identity(content) != record['content_identity'] or blob != record['git_blob']:
            raise ContractError('QEC source differs from pin: '+name)
    return {'revision': REVISION, 'pin_identity': PIN_IDENTITY,
            'files': {n: e['content_identity'] for n, e in pin['files'].items()}}


def oracle(kind):
    verify_pin()
    if kind == 'qutrit':
        codes = importlib.import_module('trinite._qec.qutrit.codes')
        algebra = importlib.import_module('trinite._qec.qutrit.stabilizer')
        exact = importlib.import_module('trinite._qec.qutrit.exact')
        code = codes.cyclic_five_qutrit_code()
    elif kind == 'ququart':
        codes = importlib.import_module('trinite._qec.ququart.codes')
        algebra = importlib.import_module('trinite._qec.ququart.packed')
        exact = importlib.import_module('trinite._qec.ququart.exact')
        code = codes.packed_five_ququart_code()
    else:
        raise ContractError('unknown fixed QEC oracle')
    return code, exact.ExactDecoder(code, max_weight=1), algebra
