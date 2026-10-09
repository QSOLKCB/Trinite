"""Independent exact oracle and strict minimal-symbol prompt parser."""
from fractions import Fraction
from math import comb
import re

from .contracts import ContractError

TASKS = ('union-count', 'intersection-count', 'choose', 'mod-add', 'dot',
         'vector-add-0', 'vector-add-1', 'determinant', 'matvec-0', 'matvec-1',
         'solve-0', 'solve-1')
TAGS = dict(zip(TASKS, ('UC', 'IC', 'BC', 'MA', 'VD', 'VA0', 'VA1', 'DT', 'MV0', 'MV1', 'LS0', 'LS1')))


def validate(spec):
    if type(spec) is not dict or set(spec) != {'task', 'values'}:
        raise ContractError('maths spec requires exactly task and values')
    task, v = spec['task'], spec['values']
    if type(task) is not str or task not in TASKS or type(v) is not list or any(type(x) is not int for x in v):
        raise ContractError('unknown maths task or noninteger operands')
    if task in ('union-count', 'intersection-count'):
        valid = len(v) == 2 and all(0 <= x <= 7 for x in v)
    elif task == 'choose':
        valid = len(v) == 2 and 0 <= v[0] <= 8 and -1 <= v[1] <= v[0]+1
    elif task == 'mod-add':
        valid = len(v) == 3 and 2 <= v[2] <= 6 and all(0 <= x < v[2] for x in v[:2])
    else:
        valid = len(v) == (6 if task.startswith(('matvec', 'solve')) else 4) and all(-1 <= x <= 1 for x in v)
    if not valid:
        raise ContractError('maths operands outside frozen domain')
    return task, v


def solve(spec):
    task, v = validate(spec)
    if task in ('union-count', 'intersection-count'):
        a, b = ({i for i in range(3) if x & (1 << i)} for x in v)
        return str(len(a | b if task == 'union-count' else a & b))
    if task == 'choose':
        n, k = v
        return str(comb(n, k) if 0 <= k <= n else 0)
    if task == 'mod-add':
        return str((v[0]+v[1]) % v[2])
    a, b, c, d = v[:4]
    if task == 'dot':
        return str(sum(x*y for x, y in zip(v[:2], v[2:])))
    if task.startswith('vector-add'):
        i = int(task[-1]); return str(v[i]+v[i+2])
    if task == 'determinant':
        # Leibniz permutation expansion, separate from generator procedure.
        return str(sum(sign*rows[0][p[0]]*rows[1][p[1]] for p, sign in
                       (((0, 1), 1), ((1, 0), -1)) for rows in ([[a, b], [c, d]],)))
    if task.startswith('matvec'):
        row = [[a, b], [c, d]][int(task[-1])]
        return str(sum(x*y for x, y in zip(row, v[4:])))
    # Rational Gaussian elimination; no adjugate/determinant label formula.
    rows = [[Fraction(a), Fraction(b), Fraction(v[4])],
            [Fraction(c), Fraction(d), Fraction(v[5])]]
    for column in range(2):
        pivot = next((r for r in range(column, 2) if rows[r][column]), None)
        if pivot is None: return 'ERR'
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [x/scale for x in rows[column]]
        for r in range(2):
            if r != column:
                scale = rows[r][column]
                rows[r] = [x-scale*y for x, y in zip(rows[r], rows[column])]
    return str(rows[int(task[-1])][2])


def parse_prompt(prompt):
    if type(prompt) is not str or len(prompt) > 64:
        raise ContractError('bounded maths prompt required')
    match = re.fullmatch(r'([A-Z]+[01]?)(?::(-?\d+(?:,-?\d+)*)=|\((-?\d+(?:,-?\d+)*)\):)', prompt)
    if match is None or match[1] not in TAGS.values():
        raise ContractError('invalid maths prompt')
    tokens = (match[2] or match[3]).split(',')
    if any(str(int(t)) != t for t in tokens): raise ContractError('noncanonical maths integer')
    spec = {'task': next(k for k, v in TAGS.items() if v == match[1]), 'values': list(map(int, tokens))}
    validate(spec)
    return spec
