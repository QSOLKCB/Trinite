"""Fixed structured schedules, partitioned before any target is requested.

This module produces requests, never observations, labels or queue transitions.
All mask/initial-state/hidden-future variants of a structural group stay together.
"""
import hashlib
from copy import deepcopy

KINDS = ('steady', 'periodic', 'ramp', 'outage')
MASKS = ((True,) * 6, (True, True, True, False, False, False))


def families():
    result = {}
    for kind in KINDS:
        ranked = sorted(range(12), key=lambda j: hashlib.sha256(
            f'queue-learning-v1/split/{kind}/{j}'.encode('ascii')).digest())
        for rank, j in enumerate(ranked):
            result[f'{kind}:{j:02d}'] = 'train' if rank < 8 else 'validation' if rank < 10 else 'test'
    return dict(sorted(result.items()))


def episodes():
    rows = []
    for family, split in families().items():
        kind, raw_j = family.split(':')
        j = int(raw_j)
        capacity = 4 + j
        arrivals, service, outages = [], [], []
        for tick in range(16):
            if kind == 'steady':
                a, s, o = 2 + j % 6, 1 + j // 4, False
            elif kind == 'periodic':
                a, s, o = 1 + j % 4 + 2 * ((tick + j) % 3), 2 + j % 3, False
            elif kind == 'ramp':
                a, s, o = 1 + j // 3 + tick // 2, 2 + j % 4, False
            else:
                a, s, o = 2 + j // 4, 4 + j % 3, tick % 4 == 3
            arrivals.append(a)
            service.append(s)
            outages.append(o)
        for initial in range(2):
            for future in range(2):
                for mask in range(2):
                    request = {'schema': 'trinite.queue-request.v1', 'capacity': capacity,
                               'initial_queue': initial * (capacity // 2),
                               'arrivals': [a + (3 * future if t >= 8 else 0)
                                            for t, a in enumerate(arrivals)],
                               'service': service[:], 'outages': outages[:], 'observe_ticks': 8,
                               'observed': list(MASKS[mask])}
                    rows.append({'id': f'{kind}-{j:02d}-{initial}{future}{mask}',
                                 'family': family, 'split': split, 'request': deepcopy(request)})
    return rows
