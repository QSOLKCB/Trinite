"""Preregistered preparation only; intentionally has no training entry point."""
import hashlib
from collections import defaultdict
from copy import deepcopy

import queue_evaluation_v1 as gate
from queue_learning_v1.producer import episodes as historical, MASKS, KINDS
from queue_learning_v1.spec import plan as v1_plan
from trinite.foundations_plan import FoundationsPlan


def plan(seed, *, fit=False):
    if seed not in ((17,) if fit else (17, 29)):
        raise ValueError('seed outside queue v2 proposal')
    value = v1_plan(seed).to_dict()
    value.update(steps=128 if fit else 256, validation_every=128 if fit else 256,
                 max_target_tokens=4096 if fit else 8192)
    value['model'] = v1_plan(seed).model
    return FoundationsPlan(**value)


def fit_rows(rows):
    """Eight fixed train prefixes, one label each; selection never uses label values."""
    from .diagnostics import check_rows
    check_rows(rows)
    seen, selected = set(), []
    for row in sorted(rows, key=lambda r: hashlib.sha256(r['identity'].encode('ascii')).digest()):
        key = gate.encode(row['observation'])
        if key not in seen:
            seen.add(key)
            selected.append(row)
        if len(selected) == 8:
            return selected
    raise ValueError('eight distinct training prefixes required')


def fit_report(model, rows):
    """Prepared forward-only fit measurement; only final step 128 may gate a run."""
    from .diagnostics import supervision, generation
    selected = fit_rows(rows)
    raw = generation(model, selected)
    return {'scope': 'eight train prefixes only; no generalization evidence',
            'ids': [r['identity'] for r in selected], 'supervision': supervision(model, selected),
            'generation': raw, 'meets_fit_criterion': raw['exact'] == raw['planned'] == 8}


def episodes():
    rows = [deepcopy(r) for r in historical() if r['split'] == 'train']
    for kind in KINDS:
        ranked = sorted(range(12, 16), key=lambda j: hashlib.sha256(
            f'queue-learning-v2/fresh/{kind}/{j}'.encode('ascii')).digest())
        for j in range(12, 16):
            capacity = 4 + j
            arrivals, service, outages = [], [], []
            for tick in range(16):
                if kind == 'steady':
                    a, s, o = 2 + j % 6, 1 + j // 4, False
                elif kind == 'periodic':
                    a, s, o = 1 + j % 4 + 3 * ((tick + j) % 3), 2 + j % 3, False
                elif kind == 'ramp':
                    a, s, o = 1 + j // 3 + tick // 2, 2 + j % 4, False
                else:
                    a, s, o = 2 + j // 3, 4 + j % 3, tick % 4 == 3
                arrivals.append(a); service.append(s); outages.append(o)
            for initial in range(2):
                for future in range(2):
                    for mask in range(2):
                        request = {'schema': 'trinite.queue-request.v1', 'capacity': capacity,
                                   'initial_queue': initial * (capacity // 2),
                                   'arrivals': [a + (3 * future if t >= 8 else 0) for t, a in enumerate(arrivals)],
                                   'service': service[:], 'outages': outages[:], 'observe_ticks': 8,
                                   'observed': list(MASKS[mask])}
                        rows.append({'id': f'v2-{kind}-{j:02d}-{initial}{future}{mask}',
                                     'family': f'v2-{kind}:{j:02d}',
                                     'split': 'validation' if ranked.index(j) < 2 else 'test',
                                     'request': request})
    return rows


def related_key(row):
    """Ignore capacity, initial state, measurement mask and all hidden futures.

    Equal prefix driving schedules are conservatively related even across kinds.
    Broader algorithmic similarity remains a declared limitation, not secrecy.
    """
    request = row['request']
    n = request['observe_ticks']
    return gate.encode({k: request[k][:n] for k in ('arrivals', 'service', 'outages')})


def audit(rows, observations, old_observations):
    """Observations only; rejects leakage without requesting a single target."""
    if rows != episodes() or set(observations) != {r['id'] for r in rows}:
        raise ValueError('v2 exact inventory required')
    old = historical()
    if set(old_observations) != {r['id'] for r in old}:
        raise ValueError('complete exposed v1 observation inventory required')
    seen = defaultdict(set)
    families = defaultdict(set)
    old_prefixes = set(old_observations.values())
    old_related = {related_key(r) for r in old}
    for row in rows:
        raw = observations[row['id']]
        # Use the same exact prompt and context bounds, without labels.
        from queue_learning_v1.codec import prompt
        prompt(raw)
        if row['split'] != 'train' and (raw in old_prefixes or related_key(row) in old_related):
            raise ValueError('fresh evaluation overlaps exposed v1 prefix/related family')
        seen[raw].add(row['split'])
        families[related_key(row)].add(row['split'])
        families[row['family']].add(row['split'])
    if any(len(v) != 1 for v in [*seen.values(), *families.values()]):
        raise ValueError('prefix or related variants cross v2 splits')
    return {'episodes': len(rows), 'unique_prefixes': len(seen), 'cross_split_prefixes': 0,
            'cross_split_related_families': 0, 'fresh_overlap_with_exposed_v1': 0,
            'requests_sha256': gate.digest(gate.encode(rows)),
            'observations_sha256': gate.digest(gate.encode({k: gate.digest(v) for k, v in sorted(observations.items())}))}


PROTOCOL = {
    'schema': 'trinite.queue-learning-proposal.v2', 'execution_enabled': False,
    'model_parameters': 24768, 'precision': 'CPU float32 dense; one Torch thread; strict determinism',
    'fit': {'seed': 17, 'plan': plan(17, fit=True).to_dict(), 'attempts': 1,
            'selection': 'eight first distinct train prefixes in SHA256 episode-ID order; no label-value selection',
            'gate': 'final update 128 only: 8/8 exact raw numeric answers and EOS; otherwise stop entire study'},
    'pilot': {'seeds': [17, 29], 'plans': {str(s): plan(s).to_dict() for s in (17, 29)},
              'attempts_per_seed': 1, 'selection': 'final update 256 only; no validation selection',
              'scheduled_visits_per_seed': 2048, 'expected_supervised_tokens_per_seed': 5600},
    'limits': {'update_seconds_per_attempt': 120, 'whole_attempt_seconds': 900,
               'whole_study_seconds': 2700, 'rss_bytes_per_process': 1073741824,
               'aggregate_rss_bytes': 2147483648, 'address_space_bytes_per_process': 4294967296,
               'total_optimizer_updates': 640, 'total_supervised_token_cap': 20480},
    'restrictions': 'no run command; no extra seeds, retries, search, larger model, GPU, paid compute; geometry loss zero',
}
