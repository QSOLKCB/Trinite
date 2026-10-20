# SPDX-License-Identifier: MPL-2.0
"""Readable integer reference for the pinned NEXUS Z4 transition.

Adapted from QSOLKCB/NEXUS; see fixtures/nexus-donor-v1/NOTICE.md.
The independently compiled Rust and Lean implementations are the parity oracles.
"""
from itertools import combinations, product

MASK = (1 << 32) - 1
VERTICES = tuple(sorted(p for p in product((-1, 0, 1), repeat=3) if p.count(0) == 1))
EDGES = tuple((a, b) for a, b in combinations(range(12), 2)
              if sum((x-y)**2 for x, y in zip(VERTICES[a], VERTICES[b])) == 2)
FACES = tuple(t for t in combinations(range(12), 3)
              if all(e in EDGES for e in combinations(t, 2)))
OWNERS = tuple(next(i for i, t in enumerate(FACES) if a in t and b in t) for a, b in EDGES)


def _colours():
    colours, used = [-1]*24, [set() for _ in range(12)]
    def visit(k):
        if k == 24:
            return True
        a, b = EDGES[k]
        for colour in range(4):
            if colour not in used[a] and colour not in used[b]:
                colours[k] = colour
                used[a].add(colour); used[b].add(colour)
                if visit(k+1):
                    return True
                used[a].remove(colour); used[b].remove(colour)
        return False
    if not visit(0):
        raise ValueError('no matching partition')
    return tuple(colours)


COLOURS = _colours()


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('integer outside exact domain')
    return value


def stir(x):
    x = integer(x, 0, MASK)
    x ^= x >> 16
    x = x * 0x21f0aaad & MASK
    x ^= x >> 15
    x = x * 0x735a2d97 & MASK
    return x ^ (x >> 15)


def plan(seed, r):
    integer(seed, 0, MASK); integer(r, 0, MASK)
    h = stir(((seed << 7 | seed >> 25) & MASK) ^ (r * 0x2545f491 & MASK) ^ 0x734c4154)
    return h & 3, bool(h >> 2 & 1), h >> 3 & 7


def validate(seed, rounds, state):
    integer(seed, 0, MASK); integer(rounds, 0, 128)
    if type(state) not in (list, tuple) or len(state) not in (24, 72):
        raise ValueError('only 2/6 cells are admitted')
    for value in state:
        integer(value, 0, 3)


def step(seed, r, state, inverse=False):
    if type(inverse) is not bool:
        raise ValueError('inverse must be boolean')
    integer(r, 0, MASK)
    validate(seed, 0, state)
    colour, flip, face = plan(seed, r)
    values = list(state)
    cells = len(values)//12
    def couple():
        for pair in range(cells//2):
            left = (2*pair + (r & 1)) % cells
            right = (left+1) % cells
            for v in FACES[face]:
                a, b = left*12+v, right*12+v
                values[a], values[b] = values[b], values[a]
    if inverse:
        couple()
    for cell in range(cells):
        for e, ((a, b), c) in enumerate(zip(EDGES, COLOURS)):
            if c != colour:
                continue
            ctrl, target = (b, a) if flip else (a, b)
            word = stir((seed + r*0x27d4eb2f + cell*0x165667b1 + OWNERS[e]*0x9e3779b1) & MASK) % 3
            delta = 1 + ((values[cell*12+ctrl] + word + e + cell) & MASK) % 3
            values[cell*12+target] = (values[cell*12+target] + (-delta if inverse else delta)) % 4
    if not inverse:
        couple()
    return values


def evolve(seed, rounds, state, inverse=False):
    if type(inverse) is not bool:
        raise ValueError('inverse must be boolean')
    validate(seed, rounds, state)
    result = list(state)
    for r in (reversed(range(rounds)) if inverse else range(rounds)):
        result = step(seed, r, result, inverse)
    return result
