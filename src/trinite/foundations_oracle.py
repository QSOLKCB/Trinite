"""Independent bounded formal and prompt checks; no generator/model imports."""
from collections import Counter
from fractions import Fraction
import re

from .contracts import ContractError, exact_keys, integer

WORKLOADS = ('arithmetic', 'fold', 'lattice')
TAGS = {'arithmetic': {'A': 'add', 'S': 'subtract', 'M': 'multiply', 'D': 'divide'},
        'fold': {'F': 'fold'}, 'lattice': {'R': 'role-to-address', 'A': 'address-to-role',
        'V': 'valid-address', 'T': 'traversal'}}
ROLES = ('QRE', 'ODU', 'CHX')


def address(text):
    if type(text) is not str or len(text) > 32:
        raise ContractError('bounded address required')
    match = re.fullmatch(r'L\[([0-2]),([0-2]),([0-2])\]', text, flags=re.ASCII)
    if not match:
        raise ContractError('invalid lattice address')
    return [int(v) for v in match.groups()]


def solve(kind, spec):
    if kind not in WORKLOADS or type(spec) is not dict:
        raise ContractError('unknown foundations specification')
    task = spec.get('task')
    if kind == 'arithmetic':
        exact_keys(spec, {'task', 'a', 'b'}, 'arithmetic item')
        a = integer(spec['a'], 'operand', -3, 3)
        b = integer(spec['b'], 'operand', -3, 3)
        if task not in TAGS[kind].values():
            raise ContractError('unknown arithmetic task')
        if task == 'add': return str(a+b)
        if task == 'subtract': return str(a-b)
        if task == 'multiply': return str(a*b)
        if b == 0: return 'ERR'
        result = Fraction(a, b)
        return str(result.numerator) if result.denominator == 1 else str(result)
    if kind == 'fold':
        exact_keys(spec, {'task', 'values'}, 'fold item')
        values = spec['values']
        if task != 'fold' or type(values) is not list or not 2 <= len(values) <= 4:
            raise ContractError('invalid fold item')
        for v in values: integer(v, 'fold value', -2, 2)
        counts = Counter(values)
        return f'{sum(counts.values())} {sum(v*c for v,c in counts.items())} {sum(pow(v,2)*c for v,c in counts.items())}'
    if task == 'traversal':
        exact_keys(spec, {'task', 'index'}, 'traversal item')
        value = (17*integer(spec['index'], 'traversal index', 0, 26)) % 27
        return f'L[{value//9},{(value//3)%3},{value%3}]'
    exact_keys(spec, {'task', 'coordinates'}, 'lattice item')
    coordinates = spec['coordinates']
    if type(coordinates) is not list or len(coordinates) != 3:
        raise ContractError('invalid coordinates')
    for v in coordinates: integer(v, 'coordinate', 0, 3 if task == 'valid-address' else 2)
    if task == 'valid-address': return str(int(all(v < 3 for v in coordinates)))
    if task == 'role-to-address': return 'L['+','.join(map(str,coordinates))+']'
    if task == 'address-to-role': return ''.join(ROLES[i][v] for i,v in enumerate(coordinates))
    raise ContractError('unknown lattice task')


def parse_prompt(kind, prompt):
    if kind not in WORKLOADS or type(prompt) is not str or len(prompt) > 48 or not prompt.isascii():
        raise ContractError('invalid foundations prompt')
    match = re.fullmatch(r'([A-Z]):(.+)=', prompt, flags=re.ASCII)
    if match is None:
        match = re.fullmatch(r'([A-Z])\((.+)\):', prompt, flags=re.ASCII)
    if match is None or match[1] not in TAGS[kind]:
        raise ContractError('invalid prompt grammar')
    task, body = TAGS[kind][match[1]], match[2]
    if kind in ('arithmetic', 'fold'):
        parts = body.split(',')
        if any(re.fullmatch(r'-?(?:0|[1-9][0-9]*)', v, flags=re.ASCII) is None for v in parts):
            raise ContractError('invalid decimal operands')
        values = list(map(int,parts))
        if kind == 'arithmetic':
            if len(values) != 2: raise ContractError('arithmetic arity')
            spec = {'task':task,'a':values[0],'b':values[1]}
        else: spec = {'task':task,'values':values}
    elif task == 'traversal':
        if re.fullmatch(r'(?:0|[1-9][0-9]*)', body, flags=re.ASCII) is None:
            raise ContractError('invalid traversal index')
        spec = {'task':task,'index':int(body)}
    elif task == 'role-to-address':
        if len(body)!=3 or any(v not in ROLES[i] for i,v in enumerate(body)):
            raise ContractError('invalid symbolic roles')
        spec = {'task':task,'coordinates':[ROLES[i].index(v) for i,v in enumerate(body)]}
    elif task == 'valid-address':
        match = re.fullmatch(r'L\[([0-3]),([0-3]),([0-3])\]', body, flags=re.ASCII)
        if match is None: raise ContractError('invalid validity-task grammar')
        spec = {'task':task,'coordinates':list(map(int,match.groups()))}
    else: spec = {'task':task,'coordinates':address(body)}
    solve(kind,spec)  # The independently parsed specification must be bounded.
    return spec
