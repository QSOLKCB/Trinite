"""Exact finite checks against integer-counter and rational arithmetic oracles."""
from fractions import Fraction
from .reference import CyclicElement, CyclicGroupConformanceError, to_sexagesimal, from_sexagesimal
from . import spec


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counter_order(k, n):
    # Independent of CyclicElement.add and of math.gcd; bounded by the fixed suite.
    require(type(n) is int and 1 <= n <= 360 and type(k) is int and 0 <= k < n,
            'counter oracle input outside finite scope')
    position = 0
    for count in range(1, n+1):
        position += k
        while position >= n:
            position -= n
        if position == 0:
            return count
    raise ValueError('counter did not return to identity')


def evaluate(case):
    n, k = case['n'], case['k']
    value = CyclicElement(k, n)
    if case['operation'] == 'element':
        expected_order = counter_order(k, n)
        require(value.order() == expected_order, 'gcd/counter order mismatch')
        require(value.inverse().k == (n-k) % n, 'inverse mismatch')
        require(value.add(value.inverse()).k == 0, 'inverse composition mismatch')
        require(CyclicElement(k-n, n) == value == CyclicElement(k+n, n), 'normalization mismatch')
        require(CyclicElement.from_turn(Fraction(k, n), n) == value, 'turn notation mismatch')
        require(CyclicElement.from_degrees(Fraction(k*360, n), n) == value, 'degree notation mismatch')
        require(CyclicElement.from_dms(*value.dms(), n) == value, 'DMS notation mismatch')
        digits = to_sexagesimal(k)
        require(from_sexagesimal(digits) == k, 'base60 roundtrip mismatch')
        # Independently reconstruct positional value (not decoder reuse).
        require(sum(d*60**i for i, d in enumerate(reversed(digits))) == k, 'base60 positional mismatch')
        for target_k in (0, k, n-1):
            target = CyclicElement(target_k, n)
            d = value.signed_displacement_to(target, 'positive')
            require((k+d) % n == target_k and 2*abs(d) <= n, 'shortest displacement mismatch')
        return {'order': expected_order, 'inverse': (n-k) % n,
                'turn': str(Fraction(k, n)), 'degrees': str(Fraction(k*360, n)),
                'sexagesimal': list(digits)}
    m = case['m']
    exact_ticks = Fraction(k, n)*m
    if exact_ticks.denominator != 1:
        try:
            value.project_exact(m)
        except CyclicGroupConformanceError:
            return {'projection': 'rejected-nonexact', 'turn': str(Fraction(k, n))}
        raise ValueError('inexact projection accepted')
    projected = value.project_exact(m)
    require(projected.k == exact_ticks.numerator, 'exact projection mismatch')
    require(projected.project_exact(n) == value, 'projection roundtrip mismatch')
    require(projected.order() == counter_order(k, n), 'projection order mismatch')
    return {'projection': projected.k, 'turn': str(Fraction(k, n))}


def rows(req):
    for index, case in enumerate(req['cases']):
        answer = evaluate(case)
        prompt = spec.canonical(case).decode().rstrip('\n')
        require(__import__('json').loads(prompt) == case, 'rendered formal input mismatch')
        yield {'source_id': spec.SOURCE_ID, 'generator_revision': req['generator_revision'],
               'seed': None, 'split_procedure': req['split'], 'generation_procedure': req['procedure'],
               'formal_specification': case, 'family_id': 'circle-modulus-'+str(case['n']),
               'rendered_text': prompt, 'expected_answer': answer,
               'verifier_outcome': 'passed: exact bounded counter/rational/positional checks',
               'transformation_lineage': {'index': index, 'parent': {'n': case['n'], 'k': case['k']}},
               'request_identity': spec.identity(spec.canonical(req))}
