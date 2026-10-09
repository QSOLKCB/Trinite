"""Independent exact-coordinate oracle; does not import the floating kernel."""
from fractions import Fraction
import math


def centered(rows):
    values = [[Fraction.from_float(x) for x in row] for row in rows]
    means = [sum(row[k] for row in values)/len(values) for k in range(len(values[0]))]
    return [[x-means[k] for k,x in enumerate(row)] for row in values]


def cka_squared(x,y):
    a,b = centered(x), centered(y)
    # Coordinate centering precedes Gram construction, independently of H K H.
    gx = [[sum(u*v for u,v in zip(p,q)) for q in a] for p in a]
    gy = [[sum(u*v for u,v in zip(p,q)) for q in b] for p in b]
    ex = sum(v*v for r in gx for v in r); ey = sum(v*v for r in gy for v in r)
    if not ex or not ey: return None
    cross = sum(gx[i][j]*gy[i][j] for i in range(len(a)) for j in range(len(a)))
    return cross*cross/(ex*ey)


def squared_distances(rows):
    values = [[Fraction.from_float(x) for x in row] for row in rows]
    return [[sum((a-b)**2 for a,b in zip(x,y)) for y in values] for x in values]


def close(actual, expected):
    if actual is None or expected is None: return actual is None and expected is None
    return math.isfinite(actual) and abs(actual-float(expected)) <= 1e-12*(1+abs(float(expected)))


def verify(x,y,value,rdm):
    exact = cka_squared(x,y)
    if not close(None if value is None else value*value, exact): raise ValueError('CKA oracle mismatch')
    expected = squared_distances(x)
    if any(not close(rdm[i][j], expected[i][j]) for i in range(len(x)) for j in range(len(x))):
        raise ValueError('distance oracle mismatch')
    return {'cka_squared_exact': None if exact is None else {'numerator':str(exact.numerator),'denominator':str(exact.denominator)},
            'distance_entries_verified':len(x)**2,'tolerance':'abs+rel 1e-12'}
