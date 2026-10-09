"""Independent bounded scalar-label and prompt oracle; no generator or model."""
from collections import Counter
import re

from .contracts import ContractError, exact_keys, integer
from .foundations_oracle import solve as parent_solve, parse_prompt as parent_parse


def solve(workload, spec):
    if type(spec) is not dict:
        raise ContractError('scalar specification required')
    if workload == 'fold':
        exact_keys(spec, {'task', 'values'}, 'scalar fold')
        task, values = spec['task'], spec['values']
        if task not in ('count', 'sum', 'squares') or type(values) is not list or not 2 <= len(values) <= 4:
            raise ContractError('invalid scalar fold')
        for v in values:
            integer(v, 'scalar fold value', -2, 2)
        histogram = Counter(values)
        if task == 'count': return str(sum(histogram.values()))
        return str(sum((v if task == 'sum' else v**2)*n for v, n in histogram.items()))
    if workload == 'lattice' and spec.get('task') in ('traversal-value', 'traversal-x', 'traversal-y', 'traversal-z'):
        exact_keys(spec, {'task', 'index'}, 'scalar traversal')
        value = (17*integer(spec['index'], 'scalar index', 0, 26)) % 27
        return str({'traversal-value': value, 'traversal-x': value//9,
                    'traversal-y': value//3 % 3, 'traversal-z': value % 3}[spec['task']])
    return parent_solve(workload, spec)


def parse_prompt(workload, prompt):
    if type(prompt) is not str or len(prompt) > 48 or not prompt.isascii():
        raise ContractError('invalid scalar prompt')
    match = re.fullmatch(r'([CQSXYZ]):(.+)=', prompt, flags=re.ASCII)
    if match is None:
        match = re.fullmatch(r'([CQSXYZ])\((.+)\):', prompt, flags=re.ASCII)
    if match is None or workload == 'arithmetic':
        return parent_parse(workload, prompt)
    tag, body = match.groups()
    if workload == 'fold':
        if tag not in 'CSQ': raise ContractError('invalid scalar fold tag')
        parts = body.split(',')
        if any(re.fullmatch(r'(?:0|-?[1-9][0-9]*)', p, flags=re.ASCII) is None for p in parts):
            raise ContractError('noncanonical scalar operands')
        spec = {'task': {'C': 'count', 'S': 'sum', 'Q': 'squares'}[tag], 'values': list(map(int, parts))}
    elif workload == 'lattice' and tag in 'QXYZ':
        if re.fullmatch(r'(?:0|[1-9][0-9]*)', body, flags=re.ASCII) is None:
            raise ContractError('invalid scalar traversal index')
        spec = {'task': {'Q': 'traversal-value', 'X': 'traversal-x',
                          'Y': 'traversal-y', 'Z': 'traversal-z'}[tag], 'index': int(body)}
    else:
        return parent_parse(workload, prompt)
    solve(workload, spec)
    return spec
