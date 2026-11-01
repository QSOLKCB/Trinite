"""Read-only closed-inventory v2 verification; never optimizer replay."""
from pathlib import Path
import math

import queue_evaluation_v1 as gate
from queue_learning_v1 import evaluation
from queue_learning_v1.artifacts import anchored_json, inventory, read_json
from trinite.contracts import parse_json
from . import admission
from .protocol import episodes
from .spec import PROTOCOL, sources
from .controller import (ATTEMPTS, empty_predictions, stopped_summary,
                         validate_checkpoint, validate_fit_certificate)


def _public_context(frozen, freeze_sha):
    """Only independently anchored manifest bytes; never private preparations."""
    frozen = gate.safe_path(frozen)
    manifest = anchored_json(frozen / 'manifest.json', freeze_sha)
    gate.require(manifest['schema'] == 'trinite.queue-learning-freeze.v2'
                 and manifest['sources'] == sources() and manifest['protocol'] == PROTOCOL,
                 'v2 public freeze source/protocol context changed')
    rows = episodes()
    gate.require(type(manifest['episodes']) is list and len(manifest['episodes']) == len(rows),
                 'v2 public freeze episode count')
    for entry, row in zip(manifest['episodes'], rows):
        gate.keys(entry, ('id', 'family', 'split', 'request_sha256', 'observation_sha256'))
        gate.hash_arg(entry['observation_sha256'])
        gate.require(all(entry[k] == row[k] for k in ('id', 'family', 'split'))
                     and entry['request_sha256'] == gate.digest(gate.encode(row['request'])),
                     'v2 public freeze inventory differs from exact protocol')
    return frozen, manifest


def _attempts(destination, stopped):
    planned = [{'name': n, 'seed': s, 'fit': f, 'planned_updates': u, 'planned_targets': t,
                'status': 'not_attempted'} for n, s, f, u, t in ATTEMPTS]
    gate.require(read_json(destination / 'planned-attempts.json') == planned, 'v2 planned attempts changed')
    actual = read_json(destination / 'attempts.json')
    gate.require(type(actual) is list and len(actual) == 3, 'v2 complete attempt denominator required')
    for row, expected in zip(actual, planned):
        gate.require(set(row) == set(expected) and
                     all(row[k] == expected[k] for k in expected if k != 'status') and
                     row['status'] in ('not_attempted', 'completed', 'failure', 'timeout'),
                     'v2 attempt inventory/status changed')
    if not stopped:
        gate.require(all(r['status'] == 'completed' for r in actual), 'successful v2 run missing completed attempt')
    # A fit failure never permits either main attempt.
    if actual[0]['status'] != 'completed':
        gate.require(all(r['status'] == 'not_attempted' for r in actual[1:]), 'v2 main attempted without completed fit')
    return actual


def _stopped_predictions(destination, frozen, manifest, record):
    predictions = empty_predictions(manifest)
    heldout = [r for r in manifest['episodes'] if r['split'] != 'train']
    allowed = [r['id'] for r in heldout]
    for seed in (17, 29):
        name = 'seed' + str(seed)
        folder = destination / ('infer-' + str(seed))
        attempted, actual = [], []
        if (folder / 'attempts.jsonl').exists():
            attempted = [parse_json(line + b'\n', canonical=True)['id'] for line in
                         gate.read(folder / 'attempts.jsonl', 4 << 20).splitlines()]
        gate.require(attempted == allowed[:len(attempted)], 'stopped v2 inference attempt order')
        receipt = record['workers'].get(name, {}).get('infer', {})
        failure = 'timeout' if receipt.get('failure') in ('timeout', 'update_timeout') else 'failure'
        for identity in attempted:
            predictions[name, identity] = {'status': failure}
        if (folder / 'raw.jsonl').exists():
            actual = [parse_json(line + b'\n', canonical=True) for line in
                      gate.read(folder / 'raw.jsonl', 4 << 20).splitlines()]
        gate.require([r['id'] for r in actual] == attempted[:len(actual)], 'stopped v2 raw journal order')
        for row in actual:
            predictions[name, row['id']] = evaluation.check_raw(row['raw'],
                gate.read(destination / 'public' / (row['id'] + '.json')))
    retained = read_json(destination / 'predictions.json')
    expected_keys = sorted(predictions)
    gate.require(type(retained) is list and len(retained) == len(expected_keys),
                 'stopped v2 full prediction inventory required')
    for row, (method, identity) in zip(retained, expected_keys):
        gate.keys(row, ('method', 'id', 'raw'))
        gate.require((row['method'], row['id']) == (method, identity),
                     'stopped v2 prediction inventory order changed')
        raw = row['raw']
        observed = gate.read(frozen / identity / 'public/observation.json') if (
            'tokens' in raw or 'value' in raw) else b''
        evaluation.check_raw(raw, observed)
        if method in ('seed17', 'seed29'):
            gate.require(raw == predictions[method, identity],
                         'stopped v2 predictions differ from raw worker journal')
        predictions[method, identity] = raw
    return predictions


def _seconds(value):
    number = float.fromhex(value)
    gate.require(math.isfinite(number) and number >= 0 and number.hex() == value,
                 'finite canonical nonnegative resource timing required')
    return number


def _healthy(receipt, cap, *, allowed_exit=(0,)):
    expected = {'rss_bytes_per_process': 1 << 30, 'aggregate_rss_bytes': 2 << 30,
                'address_space_bytes_per_process': 4 << 30}
    if (type(receipt['exit_code']) is not int or receipt['exit_code'] not in allowed_exit
            or receipt['failure'] is not None):
        return False
    wall, deadline = _seconds(receipt['wall_seconds_hex']), _seconds(receipt['deadline_seconds_hex'])
    return (receipt['limits'] == expected and 0 < deadline <= cap and wall < deadline
            and type(receipt['peak_process_rss_bytes']) is int
            and 0 < receipt['peak_process_rss_bytes'] < (1 << 30)
            and type(receipt['peak_aggregate_rss_bytes']) is int
            and 0 < receipt['peak_aggregate_rss_bytes'] < (2 << 30)
            and all(r['peak_rss_kib'] * 1024 < (1 << 30) for r in receipt['reaped']))


def _successful_attempt_receipts(destination, record):
    gate.require(_seconds(record['wall_seconds_hex']) < 2700, 'v2 inner study wall limit exceeded')
    for name, _, fit, updates, targets in ATTEMPTS:
        worker = record['workers'][name]
        phases = ('train', 'validation') if fit else ('train', 'validation', 'infer')
        for phase in phases:
            receipt = worker[phase]
            gate.require(receipt == read_json(destination / (name + '-' + phase + '-worker.json'))
                         and _healthy(receipt, 900), 'v2 unsuccessful or overbudget attempt receipt')
        gate.require(_seconds(worker['whole_attempt_seconds_hex']) < 900,
                     'v2 whole attempt wall limit exceeded')
        gate.require(worker['train']['observed_updates'] <= updates
                     and worker['train']['observed_target_tokens'] <= targets,
                     'v2 observed update/target budget exceeded')


def verify(args):
    destination, anchor_dir = gate.safe_path(args.run), gate.safe_path(args.run_anchors)
    retained = anchored_json(anchor_dir / 'final-inventory.json', args.run_sha)
    names = {str(p.relative_to(destination)) + ('/' if p.is_dir() else '')
             for p in destination.rglob('*')}
    gate.require(names == set(retained), 'v2 final run inventory paths changed')
    if not (destination / 'run.json').exists():
        gate.require(retained == inventory(destination), 'v2 final run inventory changed')
        supervisor = read_json(destination / 'supervisor.json')
        gate.require(supervisor['exit_code'] != 0 or supervisor['failure'] is not None,
                     'missing v2 run record without failed supervisor')
        return {'verified': True, 'scope': 'retained incomplete inventory only; freeze/checkpoint custody not verified',
                'complete_study': False, 'optimizer_replay': False, 'heldout_verified': False}
    frozen, manifest = _public_context(Path(args.frozen), args.freeze_sha)
    record = anchored_json(destination / 'run.json', retained['run.json'])
    gate.require(record['schema'] == 'trinite.queue-learning-run.v2'
                 and record['freeze_sha256'] == args.freeze_sha
                 and record['optimizer_replay'] is False and record['learning_claim'] is False,
                 'v2 run context changed')
    gate.require(type(record['scoring_started']) is bool and type(record['scoring_completed']) is bool
                 and (not record['scoring_completed'] or record['scoring_started']), 'v2 scoring lifecycle changed')
    entries = None
    if record['scoring_started']:
        # Even byte-inventory reads of scored artifacts follow the public barrier.
        entries = evaluation.public_barrier(destination / 'commits', anchor_dir / 'barrier.json',
                                            record['barrier_sha256'], manifest['episodes'])
    else:
        gate.require(not any(n.startswith('scores/') for n in retained), 'scores exist without scoring start')
    gate.require(retained == inventory(destination), 'v2 final run inventory changed')
    supervisor = read_json(destination / 'supervisor.json')
    if not _healthy(supervisor, 2700, allowed_exit=(0, 2) if record['stopped'] is not None else (0,)):
        return {'verified': True, 'scope': 'retained incomplete study; outer resource supervision failed',
                'complete_study': False, 'optimizer_replay': False, 'heldout_verified': False,
                'heldout_scoring_started': record['scoring_started']}
    _attempts(destination, record['stopped'])
    if record['stopped'] is not None:
        expected = stopped_summary(_stopped_predictions(destination, frozen, manifest, record))
        expected['heldout_scored'] = record['scoring_started']
        expected['scoring_completed'] = record['scoring_completed']
        gate.require(record['summary'] == expected, 'stopped v2 summary differs from retained public attempts')
        gate.require(retained == inventory(destination), 'v2 verifier mutated stopped run')
        return {'verified': True, 'scope': ('closed stopped attempt; partial scores retained unverified; no scoring rerun'
                                      if record['scoring_started'] else 'closed stopped attempt; no heldout scoring'),
                'complete_study': False, 'optimizer_replay': False, 'heldout_verified': False}
    # EVERY public output is validated before safe train checkpoint restoration,
    # training-target regeneration or any heldout private verification.
    gate.require(record['scoring_started'] and record['scoring_completed'] and entries is not None,
                 'v2 success requires completed scoring')
    _successful_attempt_receipts(destination, record)
    # Only after the complete public barrier may preparation/private files open.
    frozen, checked_manifest, prepares = admission.frozen_context(
        frozen, args.freeze_sha, args.anchors, args.anchors_sha)
    gate.require(checked_manifest == manifest, 'v2 manifest changed across public barrier')
    checkpoint = anchored_json(anchor_dir / 'checkpoints.json', record['checkpoint_sha256'])
    gate.require(checkpoint['schema'] == 'trinite.queue-learning-checkpoints.v2'
                 and checkpoint['freeze_sha256'] == args.freeze_sha
                 and set(checkpoint['checkpoints']) == {'17', '29'}, 'v2 checkpoint anchors context')
    validate_fit_certificate(destination / 'fit', anchor_dir / 'fit-gate.json',
                             record['fit_gate_sha256'], args.freeze_sha)
    for seed in (17, 29):
        entry = validate_checkpoint(destination / ('seed' + str(seed)), seed, False, args.freeze_sha)
        gate.require(entry == checkpoint['checkpoints'][str(seed)], 'v2 checkpoint differs from external anchor')
    receipts = []
    for entry in entries:
        name = entry['method'] + '-' + entry['id']
        gate.verify(frozen / entry['id'] / 'private', destination / 'commits' / name,
                    prepares[entry['id']], entry['commit_sha256'], destination / 'scores' / name)
        expected = {'id': entry['id'], 'method': entry['method'],
            'prepare_sha256': prepares[entry['id']], 'commit_sha256': entry['commit_sha256'],
            'verified': True, 'result_sha256': gate.digest(gate.read(destination / 'scores' / name / 'evaluation.json'))}
        gate.require(read_json(destination / 'scores' / (name + '-verification.json')) == expected,
                     'v2 individual score verification receipt changed')
        receipts.append(expected)
    gate.require(read_json(destination / 'scores/verification.json') == {
        'barrier_sha256': record['barrier_sha256'], 'receipts': receipts}, 'v2 aggregate score receipt changed')
    summary = evaluation.summarize(entries, destination / 'commits', destination / 'scores')
    gate.require(summary == record['summary'] == read_json(destination / 'scores/summary.json'),
                 'v2 score summary differs from committed predictions')
    gate.require(retained == inventory(destination), 'v2 verifier mutated run artifacts')
    return {'verified': True, 'complete_study': True, 'heldout_verified': True,
            'optimizer_replay': False, 'operator_blindness_proved': False}
