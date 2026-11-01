"""Retained exposed gate exercise; no model, training or pilot target outcomes."""
import queue_evaluation_v1 as gate
from .artifacts import append_json, inventory, retain_failure, write_json
from .evaluation import METHODS, commit_all, score_all
from .spec import sources


def execute(out, anchors_out):
    out, anchors_out = gate.disjoint(out, anchors_out)
    destination = gate.fresh(out)
    anchors = gate.fresh(anchors_out)
    try:
        frozen = gate.fresh(destination / 'episodes')
        gate.write(anchors / 'preparations.jsonl', b'')
        rows, prepares = [], {}
        for i, split in enumerate(('validation', 'test')):
            name = 'exposed-engineering-' + str(i)
            slot = gate.fresh(frozen / name)
            request = {'schema': 'trinite.queue-request.v1', 'capacity': 2, 'initial_queue': 0,
                       'arrivals': [1, 4 + i, 3 + i], 'service': [1, 0, 1], 'outages': [False] * 3,
                       'observe_ticks': 1, 'observed': [True] * 6}
            gate.write(slot / 'request.json', gate.encode(request))
            prepares[name] = gate.prepare(slot / 'request.json', slot / 'private', slot / 'public')
            append_json(anchors / 'preparations.jsonl', {'id': name, 'prepare_sha256': prepares[name]})
            rows.append({'id': name, 'family': 'exposed-engineering-only', 'split': split})
        prepare_sha = write_json(anchors / 'preparations.json', prepares)
        predictions = {(m, r['id']): {'status': 'prediction', 'value': 0} for m in METHODS for r in rows}
        predictions['seed17', rows[0]['id']] = {'status': 'malformed', 'tokens': [48]}
        predictions['seed17', rows[1]['id']] = {'status': 'abstain', 'tokens': [63, 257]}
        predictions['seed29', rows[0]['id']] = {'status': 'timeout'}
        predictions['seed29', rows[1]['id']] = {'status': 'failure'}
        barrier_sha = commit_all(frozen, rows, predictions, destination / 'commits', anchors / 'barrier.json')
        summary = score_all(frozen, prepares, destination / 'commits', anchors / 'barrier.json',
                            barrier_sha, rows, destination / 'scores')
        write_json(destination / 'engineering.json', {'sources': sources(), 'summary': summary,
            'scope': 'exposed synthetic engineering gate, fixed outputs; method names are placeholders, not models or fitted baselines',
            'optimizer_updates': 0, 'pilot_outcomes_scored': 0})
        run_sha = write_json(anchors / 'run-inventory.json', inventory(destination))
        return {'preparation_anchors_sha256': prepare_sha, 'barrier_sha256': barrier_sha, 'run_sha256': run_sha,
                'scope': 'actual prepare/commit/score/verify; exposed engineering only; no model training or pilot evidence'}
    except Exception as error:
        retain_failure(destination, error)
        raise
