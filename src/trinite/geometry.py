"""Bounded analysis of detached coordinates using the frozen upstream kernel."""
import hashlib
from decimal import Context, DivisionByZero, InvalidOperation, Overflow, localcontext
import math
from pathlib import Path

from .contracts import ContractError, identity, integer, json_bytes, parse_json

GEO_COMMIT = 'e770f585bf3136b47a657772156116d5f02b1e77'
PIN_IDENTITY = 'sha256:4ac17fa8f210d61d6c582aedc72c460e8d70aca29ba5cfbb08f3f0de810addee'


def verify_geometry_pin(root=None):
    package = Path(__file__).resolve().parent
    raw = (package/'geometry-pin.json').read_bytes()
    if identity(raw) != PIN_IDENTITY:
        raise ContractError('unsupported geometry source pin')
    pin = parse_json(raw, canonical=True)
    if pin['commit'] != GEO_COMMIT: raise ContractError('unsupported geometry revision')
    checked = []
    for local, entry in pin['files'].items():
        if root is None and not local.startswith('src/'): continue
        path = Path(root)/local if root is not None else package/local.split('/')[-1]
        if path.is_symlink() or not path.is_file(): raise ContractError('unsafe/missing geometry source')
        with path.open('rb') as stream: content = stream.read(65537)
        blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if identity(content) != entry['content_identity'] or blob != entry['git_blob']:
            raise ContractError('geometry source differs from pinned bytes: '+local)
        checked.append(local)
    return {'commit': GEO_COMMIT, 'pin_identity': PIN_IDENTITY, 'checked_files': checked}


def reference():
    verify_geometry_pin()
    from . import _geo_reference
    return _geo_reference


def points(value):
    if type(value) is not list or not 1 <= len(value) <= 256:
        raise ContractError('trajectory requires 1..256 points')
    if type(value[0]) is not list or not 1 <= len(value[0]) <= 128:
        raise ContractError('trajectory requires 1..128 dimensions')
    width = len(value[0])
    if any(type(row) is not list or len(row) != width for row in value):
        raise ContractError('trajectory dimension mismatch')
    try:
        if any(type(x) not in (float, int) or not math.isfinite(float(x)) for row in value for x in row):
            raise ValueError('not finite numeric coordinates')
        return [[float(x) for x in row] for row in value]
    except (ValueError, OverflowError) as error:
        raise ContractError('trajectory requires finite binary64 coordinates') from error


def encoded(value):
    if type(value) is list: return [encoded(x) for x in value]
    if not math.isfinite(value): raise ContractError('nonfinite geometry output')
    return float(value).hex()


def metrics(trajectory):
    p, kernel = points(trajectory), reference()
    try:
        with localcontext(Context(prec=28, rounding='ROUND_HALF_EVEN', Emin=-999999, Emax=999999,
                                  capitals=1, clamp=0, traps=[InvalidOperation, DivisionByZero, Overflow])):
            curvature = kernel.menger_curvature_sequence(p)
        return {'path_length_hex': encoded(kernel.path_length(p)),
                'order_1_hex': encoded(kernel.finite_difference(p, 1)),
                'order_2_hex': encoded(kernel.finite_difference(p, 2)),
                'menger_hex': encoded(curvature),
                'mean_menger_hex': encoded(math.fsum(curvature)/len(curvature)) if curvature else None,
                'curvature_status': 'available' if curvature else 'unavailable: fewer than three points'}
    except (ValueError, OverflowError) as error:
        raise ContractError('geometry outside pinned numerical domain: '+str(error)) from error


def alignment(left, right, *, order=1, mode='arclength'):
    a, b = points(left), points(right)
    integer(order, 'difference order', 0, 8)
    if mode not in ('error', 'truncate', 'arclength'): raise ContractError('unsupported alignment')
    kernel = reference()
    try:
        return kernel.cosine_alignment(a, b, order=order, align=mode)
    except (ValueError, OverflowError) as error:
        raise ContractError('undefined/invalid geometry comparison: '+str(error)) from error


def rademacher(trajectory, key, seed=17):
    p = points(trajectory)
    integer(seed, 'control seed', 0, 2**32-1)
    if type(key) is not str: raise ContractError('control key must be a string')
    # Each sign has a stable coordinate address; no global/model RNG is used.
    return [[1. if hashlib.sha256(json_bytes({'seed': seed, 'key': key, 'point': i,
                                             'coordinate': j})).digest()[0] & 1 else -1.
             for j in range(len(row))] for i, row in enumerate(p)]
