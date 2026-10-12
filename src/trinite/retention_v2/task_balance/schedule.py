"""Bounded integer scheduling over already admitted, seed-ranked training rows."""
from ...contracts import ContractError, integer

MATHS_TASKS = ('choose', 'determinant', 'dot', 'intersection-count', 'matvec-0',
               'matvec-1', 'mod-add', 'solve-0', 'solve-1', 'union-count',
               'vector-add-0', 'vector-add-1')


def task_visits(steps):
    integer(steps, 'task-balanced updates', 1, 8192)
    rounds, remainder = divmod(4 * steps, len(MATHS_TASKS))
    return {task: rounds + int(index < remainder) for index, task in enumerate(MATHS_TASKS)}


def schedule_rows(legacy, maths, task_for, steps, profile):
    """Preserve arithmetic slots and rank within each maths task; no RNG or fitting."""
    task_visits(steps)
    if type(profile) is not str or profile not in ('control', 'candidate'):
        raise ContractError('unknown task-balance schedule')
    if not legacy or not maths:
        raise ContractError('task balance requires both training streams')
    ids = [row['identity'] for row in (*legacy, *maths)]
    if len(set(ids)) != len(ids):
        raise ContractError('task balance requires disjoint unique training rows')
    if set(task_for) != {row['identity'] for row in maths} or set(task_for.values()) != set(MATHS_TASKS):
        raise ContractError('task balance needs the exact twelve-task training map')
    groups = {task: [row for row in maths if task_for[row['identity']] == task] for task in MATHS_TASKS}
    rows = []
    for step in range(steps):
        rows.extend(legacy[(4 * step + j) % len(legacy)] for j in range(4))
        for j in range(4):
            slot = 4 * step + j
            if profile == 'control':
                rows.append(maths[slot % len(maths)])
            else:
                visit, task_index = divmod(slot, len(MATHS_TASKS))
                group = groups[MATHS_TASKS[task_index]]
                rows.append(group[visit % len(group)])
    return rows
