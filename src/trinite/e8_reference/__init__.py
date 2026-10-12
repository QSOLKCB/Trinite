# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Trent Slade / QSOL-IMC.
"""Exact E8 reference algebra; no model, training or activation mapping.

New Python implementation of the root/triality formulas in QSOLKCB/E8 at
292cc9a41dd08bb8bc48b060c9e36157dbe3d1f0. See docs/E8-REFERENCE.md.
Roots use doubled integer coordinates; general linear actions use Fractions.
"""
from fractions import Fraction
from itertools import combinations, product

from ..contracts import ContractError

DONOR_COMMIT = '292cc9a41dd08bb8bc48b060c9e36157dbe3d1f0'
REFERENCE_ID = 'TRI-E8-REF-001'
D4_NUMERATOR = (
    (1, 1, 1, 1),
    (1, 1, -1, -1),
    (1, -1, 1, -1),
    (-1, 1, 1, -1),
)
# P selects x_0, x_1, x_2. This receiver is deliberately explicit and lossy.
DISPLAY_PROJECTION = tuple(tuple(int(i == j) for j in range(8)) for i in range(3))


def _vector(value, width):
    if type(value) not in (list, tuple) or len(value) != width:
        raise ContractError(f'exact vector requires {width} coordinates')
    result = []
    for coordinate in value:
        if type(coordinate) not in (int, Fraction):
            raise ContractError('exact coordinates require integers or Fractions')
        rational = Fraction(coordinate)
        if max(abs(rational.numerator).bit_length(), rational.denominator.bit_length()) > 256:
            raise ContractError('exact coordinate exceeds 256-bit input bound')
        result.append(rational)
    return tuple(result)


def roots_doubled():
    """Return 240 lexicographically ordered immutable roots, storing 2r."""
    roots = []
    for first, second in combinations(range(8), 2):
        for a, b in product((-2, 2), repeat=2):
            root = [0] * 8
            root[first], root[second] = a, b
            roots.append(tuple(root))
    roots.extend(signs for signs in product((-1, 1), repeat=8)
                 if signs.count(-1) % 2 == 0)
    return tuple(sorted(roots))


def triality(vector):
    """Apply T=A direct-sum A exactly, A=D4_NUMERATOR/2.

    Accepts general exact 8D coordinates, not only roots. Scaling conventions
    are preserved: conventional roots remain conventional, doubled stay doubled.
    """
    x = _vector(vector, 8)
    return tuple(sum((Fraction(c) * x[offset + j] for j, c in enumerate(row)),
                     Fraction(0)) / 2
                 for offset in (0, 4) for row in D4_NUMERATOR)


def root_orbits():
    """Return all fixed singlets/three-cycles in deterministic action order."""
    roots = roots_doubled()
    root_set, seen, orbits = set(roots), set(), []
    for root in roots:
        if root in seen:
            continue
        second = triality(root)
        third = triality(second)
        if triality(third) != root or second not in root_set or third not in root_set:
            raise ContractError('triality failed exact root preservation/closure')
        orbit = (root,) if second == root else (root, second, third)
        seen.update(orbit)
        orbits.append(orbit)
    return tuple(orbits)


def project_first_three(vector):
    """Apply the declared display-only P=(e_0, e_1, e_2)^T."""
    return _vector(vector, 8)[:3]


def rotate_z_quarter(vector):
    """Active right-handed +pi/2 rotation about z: (x,y,z) -> (-y,x,z)."""
    x, y, z = _vector(vector, 3)
    return (-y, x, z)


def reference_report():
    """JSON-compatible analytic demonstration, not a neural observation."""
    roots, orbits = roots_doubled(), root_orbits()
    v = (1, -2, 1)
    rotated = rotate_z_quarter(v)
    # Integer-only examples are converted without silently rounding Fractions.
    def integers(vector):
        if any(Fraction(x).denominator != 1 for x in vector):
            raise ContractError('demonstration expected integral coordinates')
        return [int(x) for x in vector]

    root = (2, 0, 0, 0, 2, 0, 0, 0)
    return {
        'schema': 'trinite.e8-reference.v1',
        'reference_id': REFERENCE_ID,
        'evidence_class': 'ANALYTIC_REFERENCE',
        'donor_commit': DONOR_COMMIT,
        'coordinate_convention': 'doubled roots 2r; squared norm 8',
        'root_count': len(roots),
        'integer_family_count': sum(0 in r for r in roots),
        'half_integer_family_count': sum(0 not in r for r in roots),
        'd4_triality_numerator': [list(row) for row in D4_NUMERATOR],
        'triality_denominator': 2,
        'triality_fixed_roots': sum(len(o) == 1 for o in orbits),
        'triality_three_cycles': sum(len(o) == 3 for o in orbits),
        'display_projection': [list(row) for row in DISPLAY_PROJECTION],
        'root_example': integers(root),
        'triality_root_example': integers(triality(root)),
        'rotated_projection_example': integers(rotate_z_quarter(project_first_three(triality(root)))),
        'rotation_example': {
            'axis': 'z', 'angle': 'pi/2', 'original': list(v),
            'rotated': integers(rotated), 'norm_squared': 6,
            'dot_product': int(sum(a * b for a, b in zip(v, rotated))),
            'angle_between_vectors_cosine': '1/6',
        },
        'neural_mapping': 'not defined',
        'geometry_loss': 0,
        'learning_benefit': 'not measured',
    }
