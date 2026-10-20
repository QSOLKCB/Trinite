"""Bounded rational polynomial notation and exact semantic scoring, no model IO."""
from fractions import Fraction
import re
from trinite.contracts import ContractError

MAX_TEXT = 96
RATIONAL = re.compile(r'-?(?:0|[1-9][0-9]{0,5})(?:/[1-9][0-9]{0,5})?\Z')
INTEGRAL = re.compile(r'I\(([^;]+);([^;]+);([^;]+)\)\Z')


def rational(text):
    if type(text) is not str or not RATIONAL.fullmatch(text):
        raise ContractError('bounded rational token required')
    return Fraction(text)


def trim(values):
    result = tuple(values)
    while len(result) > 1 and result[-1] == 0:
        result = result[:-1]
    return result


def integral(poly, a, b):
    return sum((c * (b**(i+1)-a**(i+1)) / (i+1) for i,c in enumerate(poly)), Fraction())


def render(a, b, numerator, denominator=None):
    def term(p):
        return 'I(' + str(a) + ';' + str(b) + ';' + ','.join(map(str, trim(p))) + ')'
    result = term(numerator)
    if denominator is not None:
        result += '/' + term(denominator)
    if len(result) > MAX_TEXT:
        raise ContractError('setup length exceeds grammar bound')
    return result


def parse_setup(text):
    if type(text) is not str or not 1 <= len(text) <= MAX_TEXT or not text.isascii():
        raise ContractError('bounded ASCII integral setup required')
    terms = text.split(')/I(')
    if len(terms) == 2:
        terms = [terms[0]+')', 'I('+terms[1]]
    elif len(terms) != 1:
        raise ContractError('at most one ratio allowed')
    result = []
    for term in terms:
        match = INTEGRAL.fullmatch(term)
        if match is None:
            raise ContractError('integral grammar mismatch')
        a,b = map(rational, match.group(1,2))
        coefficients = match.group(3).split(',')
        if not 1 <= len(coefficients) <= 4 or not a < b:
            raise ContractError('invalid bounds or polynomial degree')
        result.append((a,b,trim(tuple(map(rational, coefficients)))))
    if len(result) == 2 and result[0][:2] != result[1][:2]:
        raise ContractError('ratio requires shared bounds')
    return tuple(result)


def setup_correct(prediction, expected):
    try:
        return parse_setup(prediction) == parse_setup(expected)
    except (ContractError, ValueError, ZeroDivisionError):
        return False


def answer_correct(prediction, expected):
    try:
        return rational(prediction) == rational(expected)
    except (ContractError, ValueError, ZeroDivisionError):
        return False


def evaluate_setup(text):
    """Calculator oracle ONLY for data audit; never called by model generation."""
    parts = parse_setup(text)
    values = [integral(p,a,b) for a,b,p in parts]
    if len(values) == 1:
        return values[0]
    if values[1] <= 0:
        raise ContractError('centroid requires positive mass')
    return values[0]/values[1]
