"""Strict exact-count criteria, adapted from the supplied PR #20 review proposal."""
from ..contracts import ContractError, exact_keys, integer


def criterion_failures(tasks, expected_counts, numerator=9, denominator=10):
    integer(denominator, 'criterion denominator', 1, 2**31-1)
    integer(numerator, 'criterion numerator', 0, denominator)
    if (type(expected_counts) is not dict or not expected_counts
            or any(type(k) is not str or not k for k in expected_counts)
            or type(tasks) is not dict or set(tasks) != set(expected_counts)):
        raise ContractError('criterion needs the complete admitted task set')
    failures = []
    for task, value in sorted(tasks.items()):
        exact_keys(value, {'examples', 'correct', 'baseline_correct'}, 'criterion task')
        expected = integer(expected_counts[task], 'admitted task examples', 1, 2**31-1)
        n = integer(value['examples'], 'task examples', 1, 2**31-1)
        if n != expected: raise ContractError('task count differs from admitted scoring view')
        correct = integer(value['correct'], 'task correct', 0, n)
        integer(value['baseline_correct'], 'task baseline', 0, n)
        if correct*denominator < n*numerator: failures.append(task)
    return failures


def training_failures(scores, expected_counts):
    if type(scores) is not dict or not {'tasks', 'examples', 'correct'} <= set(scores):
        raise ContractError('criterion aggregate score fields missing')
    failures = criterion_failures(scores['tasks'], expected_counts)
    n = integer(scores['examples'], 'aggregate examples', 1, 2**31-1)
    correct = integer(scores['correct'], 'aggregate correct', 0, n)
    if n != sum(v['examples'] for v in scores['tasks'].values()) or correct != sum(v['correct'] for v in scores['tasks'].values()):
        raise ContractError('aggregate/task score totals disagree')
    return failures
