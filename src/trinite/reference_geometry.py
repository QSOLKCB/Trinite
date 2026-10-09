"""Bounded detached float32 geometry. No model, observer or optional backend."""
import math
import struct

from .contracts import ContractError, exact_keys, identity


def encode_vector(values):
    if not 1 <= len(values) <= 4096: raise ContractError('vector width budget')
    raw = struct.pack('<'+'f'*len(values), *values)
    rounded = struct.unpack('<'+'f'*len(values), raw)
    if not all(math.isfinite(x) for x in rounded): raise ContractError('nonfinite capture')
    return {'float_hex': [x.hex() for x in rounded], 'bytes_identity': identity(raw), 'width': len(values)}


def decode_vector(record):
    exact_keys(record, {'float_hex', 'bytes_identity', 'width'}, 'float32 vector')
    if type(record['width']) is not int or not 1 <= record['width'] <= 4096:
        raise ContractError('vector width budget')
    if type(record['float_hex']) is not list or len(record['float_hex']) != record['width']:
        raise ContractError('vector shape mismatch')
    try:
        values = [float.fromhex(x) for x in record['float_hex']]
        if encode_vector(values) != record: raise ContractError('noncanonical or tampered vector')
    except (ValueError, TypeError, OverflowError, struct.error) as error:
        raise ContractError('invalid float32 capture') from error
    return values


def matrix(rows):
    if type(rows) is not list or not 2 <= len(rows) <= 96: raise ContractError('sample budget')
    width = len(rows[0])
    if not 1 <= width <= 4096 or any(len(r) != width for r in rows): raise ContractError('matrix shape')
    if not all(type(x) is float and math.isfinite(x) for r in rows for x in r):
        raise ContractError('finite float matrix required')
    return rows


def dot(x, y):
    return math.fsum(a*b for a, b in zip(x, y))


def centered_gram(rows):
    rows = matrix(rows); n = len(rows)
    gram = [[dot(x,y) for y in rows] for x in rows]
    means = [math.fsum(r)/n for r in gram]; mean = math.fsum(means)/n
    return [[gram[i][j]-means[i]-means[j]+mean for j in range(n)] for i in range(n)]


def cka(x, y):
    if len(x) != len(y): raise ContractError('sample alignment mismatch')
    a, b = centered_gram(x), centered_gram(y)
    energy_a, energy_b = math.fsum(v*v for r in a for v in r), math.fsum(v*v for r in b for v in r)
    if energy_a == 0 or energy_b == 0: return None
    return math.fsum(a[i][j]*b[i][j] for i in range(len(a)) for j in range(len(a)))/math.sqrt(energy_a*energy_b)


def distances(rows):
    rows = matrix(rows)
    return [[math.fsum((a-b)**2 for a,b in zip(x,y)) for y in rows] for x in rows]


def cosine_distances(rows):
    rows = matrix(rows); norms = [math.sqrt(dot(r,r)) for r in rows]
    return [[None if norms[i]*norms[j] == 0 else 1-dot(x,y)/(norms[i]*norms[j])
             for j,y in enumerate(rows)] for i,x in enumerate(rows)]


def hex_matrix(rows):
    return [[None if x is None else x.hex() for x in r] for r in rows]
