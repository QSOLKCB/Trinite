"""Commit-all barrier over the unchanged two-anchor gate; no model imports."""
from collections import Counter

import queue_evaluation_v1 as gate
from .artifacts import anchored_json, append_json, read_json, retain_failure, write_json
from .codec import output

STATUSES = ('prediction', 'abstain', 'malformed', 'failure', 'timeout', 'not_attempted')
METHODS = ('seed17', 'seed29', 'median', 'mode', 'causal')


def constants(train_rows):
    gate.require(train_rows and all(r['split'] == 'train' for r in train_rows), 'baseline pool must be train-only')
    labels = sorted(r['answer'] for r in train_rows)
    counts = Counter(labels)
    return {'median': labels[(len(labels) - 1) // 2],
            'mode': min(counts, key=lambda k: (-counts[k], k))}


def causal(observation_bytes):
    obs = gate.observation(observation_bytes)
    row = obs['rows'][-1]
    if not all(row['observed'][:3]):
        return {'status': 'abstain'}
    q, a, s = row['values'][:3]
    h = obs['horizon']
    # A dummy visible tick leaves q unchanged. The authoritative queue owns all
    # admission-before-service transitions; this baseline duplicates no dynamics.
    request = {'schema': 'trinite.queue-request.v1', 'capacity': obs['capacity'], 'initial_queue': q,
               'arrivals': [0] + [a] * h, 'service': [0] + [s] * h,
               'outages': [False] * (h + 1), 'observe_ticks': 1, 'observed': [True] * 6}
    raw, _ = gate.run('target', gate.encode(request))
    value = gate.decode(raw)
    gate.keys(value, ('schema', 'dropped'))
    gate.require(value['schema'] == 'trinite.queue-target.v1', 'causal continuation target schema')
    return {'status': 'prediction', 'value': gate.integer(value['dropped'], 0, 32 * h)}


def check_raw(raw, observed):
    gate.require(type(raw) is dict and raw.get('status') in STATUSES, 'unknown raw prediction status')
    if 'tokens' in raw:
        gate.keys(raw, ('status', 'tokens'))
        gate.require(raw['status'] == output(raw['tokens'], observed)['status'], 'raw generation classification changed')
    elif raw['status'] == 'prediction':
        gate.keys(raw, ('status', 'value'))
        gate.integer(raw['value'], 0, 32 * gate.observation(observed)['horizon'])
    else:
        gate.keys(raw, ('status',))
    return raw


def commit_all(frozen, episodes, predictions, out, barrier_file):
    frozen, out, barrier_file = gate.disjoint(frozen, out, barrier_file)
    planned = [(method, r) for method in METHODS for r in episodes if r['split'] != 'train']
    gate.require(set(predictions) == {(m, r['id']) for m, r in planned}, 'incomplete planned prediction inventory')
    destination = gate.fresh(out)
    entries = []
    try:
        journal = barrier_file.with_suffix('.jsonl')
        gate.write(journal, b'')
        for method, row in planned:
            observed = gate.read(frozen / row['id'] / 'public/observation.json')
            raw = check_raw(predictions[method, row['id']], observed)
            name = method + '-' + row['id']
            write_json(destination / (name + '-raw.json'), raw)
            pred = {'schema': 'trinite.queue-prediction.v1', 'observation_sha256': gate.digest(observed),
                    'status': 'prediction' if raw['status'] == 'prediction' else 'abstain'}
            if raw['status'] == 'prediction':
                pred['value'] = raw['value'] if 'value' in raw else output(raw['tokens'], observed)['value']
            write_json(destination / (name + '-prediction.json'), pred)
            anchor = gate.commit(frozen / row['id'] / 'public', destination / (name + '-prediction.json'),
                                 destination / name)
            entries.append({'method': method, 'id': row['id'], 'split': row['split'],
                            'family': row['family'], 'commit_sha256': anchor,
                            'raw_sha256': gate.digest(gate.read(destination / (name + '-raw.json')))})
            append_json(journal, entries[-1])
        # This separately stored barrier is complete before the first private score.
        value = {'schema': 'trinite.queue-learning-commit-barrier.v1', 'entries': entries,
                 'journal_sha256': gate.digest(gate.read(journal, 4 * 1024 * 1024))}
        return write_json(barrier_file, value)
    except Exception as error:
        retain_failure(destination, error)
        raise


def public_barrier(commits, barrier_file, barrier_sha, episodes):
    value = anchored_json(barrier_file, barrier_sha)
    gate.keys(value, ('schema', 'entries', 'journal_sha256'))
    gate.require(value['schema'] == 'trinite.queue-learning-commit-barrier.v1', 'wrong queue barrier schema')
    journal = gate.read(barrier_file.with_suffix('.jsonl'), 4 * 1024 * 1024)
    gate.require(gate.digest(journal) == value['journal_sha256']
                 and journal == b''.join(gate.encode(e) for e in value['entries']), 'queue independent commitment journal changed')
    expected = [(method, r) for method in METHODS for r in episodes if r['split'] != 'train']
    gate.require(len(value['entries']) == len(expected), 'queue commit barrier missing predictions')
    for entry, (method, row) in zip(value['entries'], expected):
        gate.keys(entry, ('method', 'id', 'split', 'family', 'commit_sha256', 'raw_sha256'))
        gate.require(all(entry[k] == v for k, v in {'method': method, 'id': row['id'],
                     'split': row['split'], 'family': row['family']}.items()), 'queue commit barrier inventory changed')
        name = method + '-' + row['id']
        files, pred = gate.committed(commits / name, entry['commit_sha256'])
        raw_bytes = gate.read(commits / (name + '-raw.json'))
        gate.require(gate.digest(raw_bytes) == entry['raw_sha256'], 'queue raw output changed')
        raw = check_raw(gate.decode(raw_bytes), files['observation.json'])
        gate.require(pred['status'] == ('prediction' if raw['status'] == 'prediction' else 'abstain'), 'raw/commit status mismatch')
        if pred['status'] == 'prediction':
            expected_value = raw.get('value') if 'value' in raw else output(raw['tokens'], files['observation.json'])['value']
            gate.require(pred['value'] == expected_value, 'raw/committed value mismatch')
    return value['entries']


def score_all(frozen, prepares, commits, barrier_file, barrier_sha, episodes, out):
    # Validate EVERY public commitment before opening ANY private preparation.
    entries = public_barrier(commits, barrier_file, barrier_sha, episodes)
    destination = gate.fresh(out)
    receipts = []
    try:
        for e in entries:
            name = e['method'] + '-' + e['id']
            args = (frozen / e['id'] / 'private', commits / name, prepares[e['id']], e['commit_sha256'])
            gate.score(*args, destination / name)
            gate.verify(*args, destination / name)
            receipt = {'id': e['id'], 'method': e['method'], 'prepare_sha256': prepares[e['id']],
                       'commit_sha256': e['commit_sha256'], 'verified': True,
                       'result_sha256': gate.digest(gate.read(destination / name / 'evaluation.json'))}
            write_json(destination / (name + '-verification.json'), receipt)
            receipts.append(receipt)
        write_json(destination / 'verification.json', {'barrier_sha256': barrier_sha, 'receipts': receipts})
        report = summarize(entries, commits, destination)
        write_json(destination / 'summary.json', report)
        return report
    except Exception as error:
        retain_failure(destination, error)
        raise


def summarize(entries, commits, results):
    report = {}
    for method in METHODS:
        report[method] = {}
        for split in ('validation', 'test'):
            group = [e for e in entries if e['method'] == method and e['split'] == split]
            statuses, errors, exact = Counter(), [], 0
            for entry in group:
                name = method + '-' + entry['id']
                raw = read_json(commits / (name + '-raw.json'))
                score = read_json(results / name / 'evaluation.json')
                statuses[raw['status']] += 1
                if raw['status'] == 'prediction':
                    gate.require(score['status'] == 'prediction', 'numeric score missing')
                    errors.append(score['absolute_error'])
                    exact += int(score['exact_match'])
                else:
                    gate.require(not {'predicted', 'absolute_error', 'exact_match'} & set(score), 'invalid output received numeric score')
            n, numeric = len(group), len(errors)
            ratio = lambda a, b: {'numerator': a, 'denominator': b}
            report[method][split] = {'planned': n, 'statuses': {s: statuses[s] for s in STATUSES},
                'attempted': n - statuses['not_attempted'], 'numeric': numeric,
                'mae': ratio(sum(errors), numeric) if numeric == n and n else None,
                'conditional_numeric_mae': ratio(sum(errors), numeric) if numeric else None,
                'exact_match': ratio(exact, n), 'numeric_exact_match': ratio(exact, numeric),
                'valid_output_rate': ratio(statuses['prediction'] + statuses['abstain'], n),
                'coverage': ratio(numeric, n)}
    comparisons = {}
    for seed in ('seed17', 'seed29'):
        comparisons[seed] = {}
        for split in ('validation', 'test'):
            comparisons[seed][split] = {}
            model = report[seed][split]['mae']
            for baseline in ('median', 'mode', 'causal'):
                other = report[baseline][split]['mae']
                comparisons[seed][split][baseline] = {
                    'mae_difference': {'numerator': model['numerator'] - other['numerator'],
                                       'denominator': model['denominator']}
                    if model and other and model['denominator'] == other['denominator'] else None,
                    'lower_mae': model['numerator'] < other['numerator'] if model and other else None}
    return {'schema': 'trinite.queue-learning-summary.v1', 'methods': report,
            'comparisons': comparisons, 'learning_claim': False,
            'limitations': 'descriptive synthetic family holdout; hashes do not prove operator blindness or chronology'}
