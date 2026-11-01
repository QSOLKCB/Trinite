"""Fresh v2 admission, zero-update feasibility and exact-freeze authorization.

The approval is an operator declaration, not cryptographic evidence of the
owner's identity or remote CI. The caller must obtain that owner's approval.
"""
import math
from pathlib import Path
import resource
import time

import queue_evaluation_v1 as gate
from trinite.contracts import json_bytes
from queue_learning_v1 import admission as previous
from queue_learning_v1.artifacts import anchored_json, append_json, read_json, retain_failure, write_json
from queue_learning_v1.codec import prompt
from queue_learning_v1.producer import episodes as historical
from .protocol import episodes, audit
from .spec import PROTOCOL, ROOT, git_identity, sources


def families():
    return dict(sorted({r['family']: r['split'] for r in episodes()}.items()))


def record(rows, source):
    gate.require(rows == episodes() and len(rows) == 384 and
                 len({gate.encode(r['request']) for r in rows}) == 384,
                 'exact independently regenerated v2 requests required')
    return {'schema': 'trinite.queue-learning-admission.v2', 'outcome': 'admitted',
        'source_id': 'commissioned-queue-formal-v2',
        'origin': 'original finite integer schedules; no external data, teacher or corpus',
        'author': 'Codex generator commissioned by Trent Slade / QSOLKCB',
        'generation': 'unchanged 256 v1 train requests; 16 fresh evaluation families j=12..15; protocol.episodes',
        'rights_basis': 'new formal queue counts and minimal symbolic notation; no third-party expressive content',
        'scope': 'only this exact 384-request v2 inventory; v1 approval does not authorize v2',
        'verification': 'exact regeneration; unchanged authoritative Rust observation/target; observation-only overlap audit',
        'sources': source, 'requests_sha256': gate.digest(json_bytes(rows)),
        'split_sha256': gate.digest(json_bytes(families())),
        'counts': {'train': 256, 'validation': 64, 'test': 64, 'families': 48},
        'execution_authorization': 'separate exact-freeze v2 owner approval required',
        'limitations': 'public construction; not legal certification, operator blindness or a secret benchmark'}


def validate_admission(value, rows, source):
    gate.require(json_bytes(value) == json_bytes(record(rows, source)), 'v2 admission mismatch')
    return value


def observed_inventory(rows):
    result = {}
    for row in rows:
        raw, _ = gate.run('observe', gate.encode(row['request']))
        prompt(raw)
        result[row['id']] = raw
    return result


def structural(rows, observations):
    return audit(rows, observations, observed_inventory(historical()))


def _positive_hex(value):
    number = float.fromhex(value)
    gate.require(math.isfinite(number) and number > 0, 'invalid feasibility timing')
    return number


def _validate_feasibility(proof, source, git, environment):
    gate.require(git['clean'] and git['main'] is not None and git['head'] == git['main'],
                 'feasibility must bind clean actual merged main')
    gate.require(proof['schema'] == 'trinite.queue-learning-feasibility.v2'
                 and proof['sources'] == source and proof['git'] == git
                 and proof['binary_sha256'] == gate.binary_identity()
                 and proof['protocol'] == PROTOCOL, 'fresh v2 feasibility source/git/protocol required')
    kernel = proof['kernel']
    gate.require(kernel['environment'] == environment and kernel['parameters'] == 24768
                 and type(kernel['optimizer_updates']) is int and kernel['optimizer_updates'] == 0
                 and kernel['gradient_calls'] == 3 and len(kernel['kernel_seconds_hex']) == 3,
                 'v2 feasibility must use three zero-update worst-context gradient calls')
    estimate = max(_positive_hex(t) for t in kernel['kernel_seconds_hex']) * 256 * 4
    gate.require(_positive_hex(kernel['conservative_256_update_estimate_seconds_hex']) == estimate
                 and estimate < 120, '256-update fourfold feasibility estimate exceeds budget')
    overhead = _positive_hex(kernel['diagnostic_io_seconds_hex'])
    whole = _positive_hex(kernel['conservative_whole_attempt_seconds_hex'])
    gate.require(whole >= estimate + overhead * 4 and whole < 900,
                 'whole-attempt diagnostic/IO feasibility budget')
    setup = _positive_hex(proof['admission_import_setup_seconds_hex'])
    attempt = _positive_hex(proof['conservative_whole_attempt_seconds_hex'])
    gate.require(attempt >= whole + setup * 4 and attempt < 900,
                 'admission/import/setup whole-attempt feasibility budget')
    gate.require(type(proof['peak_rss_kib']) is int and 0 < proof['peak_rss_kib'] * 1024 < 1073741824,
                 'feasibility process RSS exceeds bound')


def validate_feasibility(proof, source, git, environment):
    """Only the outer, reaped successful supervisor can seal a usable receipt."""
    supervisor = proof.get('supervisor')
    gate.require(type(supervisor) is dict and supervisor.get('exit_code') == 0
                 and supervisor.get('failure') is None, 'successful outer feasibility supervision required')
    gate.require(supervisor.get('limits') == {
        'rss_bytes_per_process': 1073741824, 'aggregate_rss_bytes': 2147483648,
        'address_space_bytes_per_process': 4294967296}, 'feasibility supervisor limit identity')
    for name, limit in [('peak_process_rss_bytes', 1073741824),
                        ('peak_aggregate_rss_bytes', 2147483648)]:
        value = supervisor.get(name)
        gate.require(type(value) is int and 0 < value < limit, 'feasibility supervised RSS budget')
    elapsed = _positive_hex(supervisor['wall_seconds_hex'])
    deadline = _positive_hex(supervisor['deadline_seconds_hex'])
    gate.require(elapsed < deadline == 900, 'feasibility whole-attempt deadline')
    _validate_feasibility(proof, source, git, environment)


def feasibility(out):
    destination = gate.fresh(out)
    try:
        started = time.monotonic()
        from .numerical import feasibility_kernel
        source, git = sources(), git_identity()
        gate.require(git['clean'] and git['head'] == git['main'], 'feasibility requires clean actual merged main')
        rows = episodes()
        validate_admission(record(rows, source), rows, source)
        observed = observed_inventory(rows)
        audit_result = structural(rows, observed)
        kernel = feasibility_kernel()
        elapsed = time.monotonic() - started
        setup = elapsed - sum(float.fromhex(t) for t in kernel['kernel_seconds_hex']) - float.fromhex(kernel['diagnostic_io_seconds_hex'])
        gate.require(setup > 0, 'invalid feasibility accounting')
        whole = float.fromhex(kernel['conservative_whole_attempt_seconds_hex']) + setup * 4
        proof = {'schema': 'trinite.queue-learning-feasibility.v2', 'sources': source,
                 'git': git, 'binary_sha256': gate.binary_identity(), 'protocol': PROTOCOL,
                 'structural': audit_result, 'kernel': kernel,
                 'wall_seconds_hex': elapsed.hex(),
                 'admission_import_setup_seconds_hex': setup.hex(),
                 'conservative_whole_attempt_seconds_hex': whole.hex(),
                 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        _validate_feasibility(proof, sources(), git_identity(), kernel['environment'])
        return write_json(destination / 'feasibility.json', proof)
    except Exception as error:
        retain_failure(destination, error)
        raise


def freeze(feasibility_file, feasibility_sha, out, anchors_file):
    feasibility_file, out, anchors_file = gate.disjoint(feasibility_file, out, anchors_file)
    gate.require(ROOT not in anchors_file.parents, 'preparation anchors must be outside checkout')
    proof = anchored_json(feasibility_file, feasibility_sha)
    from .numerical import configure
    source, git, environment = sources(), git_identity(), configure()
    gate.require(git['clean'] and git['head'] == git['main'], 'freeze requires clean actual merged main')
    validate_feasibility(proof, source, git, environment)
    destination = gate.fresh(out)
    try:
        rows = episodes()
        journal = anchors_file.with_suffix('.jsonl')
        gate.disjoint(destination, journal, anchors_file)
        gate.write(journal, b'')
        for name, value in [('splits.json', families()), ('protocol.json', PROTOCOL),
                            ('admission.json', record(rows, source)), ('feasibility.json', proof)]:
            write_json(destination / name, value)
        entries, prepares, observations = [], {}, {}
        for row in rows:
            slot = gate.fresh(destination / row['id'])
            request = gate.encode(row['request'])
            gate.write(slot / 'request.json', request)
            anchor = gate.prepare(slot / 'request.json', slot / 'private', slot / 'public')
            observed = gate.read(slot / 'public/observation.json')
            observations[row['id']], prepares[row['id']] = observed, anchor
            append_json(journal, {'id': row['id'], 'prepare_sha256': anchor})
            entries.append({k: row[k] for k in ('id', 'family', 'split')} |
                           {'request_sha256': gate.digest(request), 'observation_sha256': gate.digest(observed)})
        audit_result = structural(rows, observations)
        gate.require(audit_result == proof['structural'] and sources() == source
                     and git_identity() == git and configure() == environment, 'freeze context changed')
        manifest = {'schema': 'trinite.queue-learning-freeze.v2', 'sources': source, 'git': git,
            'environment': environment, 'binary_sha256': gate.binary_identity(), 'protocol': PROTOCOL,
            'episodes': entries, 'structural': audit_result,
            'files': {n: gate.digest(gate.read(destination / n, 4 * 1024 * 1024)) for n in
                      ('splits.json', 'protocol.json', 'admission.json', 'feasibility.json')}}
        frozen_sha = write_json(destination / 'manifest.json', manifest)
        anchor_sha = write_json(anchors_file, {'schema': 'trinite.queue-learning-preparation-anchors.v2',
            'freeze_sha256': frozen_sha, 'prepares': prepares,
            'journal_sha256': gate.digest(gate.read(journal, 4 * 1024 * 1024))})
        return {'freeze_sha256': frozen_sha, 'preparation_anchors_sha256': anchor_sha,
                'scope': 'observations only; zero optimizer steps; exact v2 owner approval still required'}
    except Exception as error:
        retain_failure(destination, error)
        raise


def frozen_context(frozen, freeze_sha, anchors_file, anchors_sha):
    frozen, anchors_file = gate.disjoint(frozen, anchors_file)
    gate.require(ROOT not in anchors_file.parents, 'anchors must be outside checkout')
    manifest = anchored_json(frozen / 'manifest.json', freeze_sha)
    anchors = anchored_json(anchors_file, anchors_sha)
    gate.keys(manifest, ('schema', 'sources', 'git', 'environment', 'binary_sha256', 'protocol', 'episodes', 'structural', 'files'))
    gate.keys(anchors, ('schema', 'freeze_sha256', 'prepares', 'journal_sha256'))
    gate.keys(manifest['files'], ('splits.json', 'protocol.json', 'admission.json', 'feasibility.json'))
    rows, source = episodes(), sources()
    gate.require(manifest['schema'] == 'trinite.queue-learning-freeze.v2'
                 and manifest['sources'] == source and manifest['protocol'] == PROTOCOL
                 and manifest['binary_sha256'] == gate.binary_identity(), 'v2 frozen context changed')
    gate.require(anchors['schema'] == 'trinite.queue-learning-preparation-anchors.v2'
                 and anchors['freeze_sha256'] == freeze_sha
                 and set(anchors['prepares']) == {r['id'] for r in rows}, 'v2 independent anchors context')
    journal = gate.read(anchors_file.with_suffix('.jsonl'), 4 * 1024 * 1024)
    gate.require(gate.digest(journal) == anchors['journal_sha256'] and journal == b''.join(
        json_bytes({'id': r['id'], 'prepare_sha256': anchors['prepares'][r['id']]}) for r in rows), 'preparation journal changed')
    gate.require(set(p.name for p in frozen.iterdir()) == {r['id'] for r in rows} |
                 {'manifest.json', *manifest['files']}, 'extra/missing freeze artifacts')
    for name, sha in manifest['files'].items():
        gate.require(gate.digest(gate.read(frozen / name, 4 * 1024 * 1024)) == sha, 'freeze file changed')
    gate.require(read_json(frozen / 'splits.json') == families() and
                 read_json(frozen / 'protocol.json') == PROTOCOL, 'frozen split/protocol mismatch')
    validate_admission(read_json(frozen / 'admission.json'), rows, source)
    observations, entries = {}, []
    for row in rows:
        slot = frozen / row['id']
        gate.require(set(p.name for p in slot.iterdir()) == {'request.json', 'private', 'public'}, 'episode artifacts changed')
        request = gate.encode(row['request'])
        gate.require(gate.read(slot / 'request.json') == request, 'frozen request changed')
        private = previous.prepared(slot / 'private', anchors['prepares'][row['id']])
        public = gate.bundle(slot / 'public', gate.PUBLIC)
        gate.require(private['request.json'] == request and public['observation.json'] == private['observation.json']
                     and public['protocol.json'] == gate.encode(gate.PROTOCOL), 'frozen public/private request mismatch')
        observations[row['id']] = public['observation.json']
        entries.append({k: row[k] for k in ('id', 'family', 'split')} |
                       {'request_sha256': gate.digest(request), 'observation_sha256': gate.digest(public['observation.json'])})
    gate.require(manifest['episodes'] == entries and manifest['structural'] == structural(rows, observations),
                 'frozen inventory/overlap mismatch')
    from .numerical import configure
    environment = configure()
    gate.require(environment == manifest['environment'], 'frozen runtime changed')
    validate_feasibility(read_json(frozen / 'feasibility.json'), source, manifest['git'], environment)
    gate.require(sources() == source, 'sources changed during freeze verification')
    return frozen, manifest, anchors['prepares']


def authorize(manifest, freeze_sha, approval_file):
    gate.hash_arg(freeze_sha)
    expected = {'schema': 'trinite.queue-learning-owner-approval.v2', 'freeze_sha256': freeze_sha,
        'implementation_commit': manifest['git']['head'], 'approved_by': 'Trent Slade',
        'reviewed_merged_green': True,
        'authorization': 'execute queue-learning-v2 fit seed 17 once; only after final 8/8 at step 128, main seeds 17 and 29 once at step 256'}
    gate.require(json_bytes(read_json(approval_file, 16384)) == json_bytes(expected), 'explicit exact-freeze v2 owner approval required')
    git = git_identity()
    gate.require(git['clean'] and git['head'] == git['main'] == manifest['git']['head']
                 and manifest['git']['clean'], 'execution requires clean reviewed merged main matching freeze')


def training_rows(frozen, manifest, anchors):
    # Reuse the sole authoritative train admission; the unchanged 256 train
    # requests retain their exact identities. Never materialize held-out labels.
    validate_admission(read_json(frozen / 'admission.json'), episodes(), sources())
    gate.require([{k: r[k] for k in ('id', 'family', 'split')} for r in manifest['episodes']] ==
                 [{k: r[k] for k in ('id', 'family', 'split')} for r in episodes()],
                 'train admission requires exact v2 split inventory')
    result = previous.training_rows(frozen, manifest, anchors)
    previous.check_training_rows(result)
    return result


check_training_rows = previous.check_training_rows
