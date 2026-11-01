"""Fixed checkout-only workers. Run is separate from engineering feasibility/freeze."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import queue_evaluation_v1 as gate
from trinite.contracts import json_bytes, parse_json
from . import admission, evaluation
from .artifacts import anchored_json, append_json, inventory, read_json, retain_failure, write_json
from .codec import prompt
from .producer import episodes, families
from .spec import PROTOCOL, ROOT, SEEDS, git_identity, plan, sources


def observed_inventory(rows):
    result = {}
    for row in rows:
        raw, _ = gate.run('observe', gate.encode(row['request']))
        prompt(raw)
        result[row['id']] = raw
    return result


def feasibility(out):
    destination = gate.fresh(out)
    try:
        started = time.monotonic()
        from .numerical import feasibility_kernel
        source = sources()
        rows = episodes()
        admission.validate_admission(admission.record(rows, source), rows, source)
        structural = admission.audit_observations(rows, observed_inventory(rows))
        kernel = feasibility_kernel()
        gate.require(sources() == source, 'sources changed during queue feasibility')
        record = {'schema': 'trinite.queue-learning-feasibility.v1', 'sources': source,
                  'binary_sha256': gate.binary_identity(), 'protocol': PROTOCOL,
                  'structural': structural, 'kernel': kernel, 'git': git_identity(),
                  'wall_seconds_hex': (time.monotonic() - started).hex(),
                  'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        return write_json(destination / 'feasibility.json', record)
    except Exception as error:
        retain_failure(destination, error)
        raise


def freeze(feasibility_file, feasibility_sha, out, anchors_file):
    feasibility_file, out, anchors_file = gate.disjoint(feasibility_file, out, anchors_file)
    proof = anchored_json(feasibility_file, feasibility_sha)
    source = sources()
    gate.require(proof['schema'] == 'trinite.queue-learning-feasibility.v1'
                 and proof['sources'] == source and proof['binary_sha256'] == gate.binary_identity()
                 and proof['protocol'] == PROTOCOL and proof['kernel']['optimizer_updates'] == 0
                 and proof['kernel']['parameters'] == plan(17).model.parameter_count
                 and 0 < float.fromhex(proof['kernel']['conservative_64_update_estimate_seconds_hex']) < 120,
                 'fresh queue feasibility proof required before freeze')
    from .numerical import configure
    gate.require(configure() == proof['kernel']['environment'], 'queue feasibility environment changed')
    git = git_identity()
    gate.require(git['clean'], 'freeze requires a clean source commit')
    destination = gate.fresh(out)
    try:
        rows = episodes()
        prepare_journal = anchors_file.with_suffix('.jsonl')
        gate.write(prepare_journal, b'')
        # Written before any target invocation. Freeze invokes observe only.
        write_json(destination / 'splits.json', families())
        write_json(destination / 'protocol.json', PROTOCOL)
        write_json(destination / 'admission.json', admission.record(rows, source))
        write_json(destination / 'feasibility.json', proof)
        inventory, prepares, observations = [], {}, {}
        for row in rows:
            slot = gate.fresh(destination / row['id'])
            request = gate.encode(row['request'])
            gate.write(slot / 'request.json', request)
            anchor = gate.prepare(slot / 'request.json', slot / 'private', slot / 'public')
            observed = gate.read(slot / 'public/observation.json')
            observations[row['id']] = observed
            prepares[row['id']] = anchor
            append_json(prepare_journal, {'id': row['id'], 'prepare_sha256': anchor})
            inventory.append({k: row[k] for k in ('id', 'family', 'split')} |
                             {'request_sha256': gate.digest(request), 'observation_sha256': gate.digest(observed)})
        structural = admission.audit_observations(rows, observations)
        gate.require(structural == proof['structural'] and sources() == source, 'queue freeze feasibility/source changed')
        manifest = {'schema': 'trinite.queue-learning-freeze.v1', 'sources': source, 'git': git,
                    'environment': configure(), 'binary_sha256': gate.binary_identity(),
                    'protocol': PROTOCOL, 'episodes': inventory, 'structural': structural,
                    'files': {n: gate.digest(gate.read(destination / n, 4 * 1024 * 1024)) for n in
                              ('splits.json', 'protocol.json', 'admission.json', 'feasibility.json')}}
        frozen_sha = write_json(destination / 'manifest.json', manifest)
        anchor_sha = write_json(anchors_file, {'schema': 'trinite.queue-learning-preparation-anchors.v1',
                                'freeze_sha256': frozen_sha, 'prepares': prepares,
                                'journal_sha256': gate.digest(gate.read(prepare_journal, 4 * 1024 * 1024))})
        return {'freeze_sha256': frozen_sha, 'preparation_anchors_sha256': anchor_sha,
                'scope': 'no targets computed, no training; explicit approval still required'}
    except Exception as error:
        retain_failure(destination, error)
        raise


def frozen_context(frozen, freeze_sha, anchors_file, anchors_sha):
    frozen = gate.safe_path(frozen)
    manifest = anchored_json(frozen / 'manifest.json', freeze_sha)
    anchors = anchored_json(anchors_file, anchors_sha)
    gate.keys(manifest, ('schema', 'sources', 'git', 'environment', 'binary_sha256', 'protocol', 'episodes', 'structural', 'files'))
    gate.keys(anchors, ('schema', 'freeze_sha256', 'prepares', 'journal_sha256'))
    gate.keys(manifest['files'], ('splits.json', 'protocol.json', 'admission.json', 'feasibility.json'))
    gate.require(anchors['schema'] == 'trinite.queue-learning-preparation-anchors.v1'
                 and anchors['freeze_sha256'] == freeze_sha, 'queue preparation anchors context')
    journal = gate.read(Path(anchors_file).with_suffix('.jsonl'), 4 * 1024 * 1024)
    gate.require(gate.digest(journal) == anchors['journal_sha256']
                 and journal == b''.join(gate.encode({'id': r['id'], 'prepare_sha256': anchors['prepares'][r['id']]})
                                         for r in episodes()), 'queue independent preparation journal changed')
    gate.require(manifest['schema'] == 'trinite.queue-learning-freeze.v1'
                 and manifest['sources'] == sources() and manifest['protocol'] == PROTOCOL
                 and manifest['binary_sha256'] == gate.binary_identity(), 'queue frozen source/protocol changed')
    rows = episodes()
    expected = [{k: r[k] for k in ('id', 'family', 'split')} | {
                'request_sha256': gate.digest(gate.encode(r['request'])),
                'observation_sha256': e['observation_sha256']} for r, e in zip(rows, manifest['episodes'])]
    gate.require(manifest['episodes'] == expected and len(expected) == 384
                 and set(anchors['prepares']) == {r['id'] for r in rows}, 'queue freeze inventory changed')
    gate.require(set(p.name for p in frozen.iterdir()) == {r['id'] for r in rows} |
                 {'manifest.json', 'splits.json', 'protocol.json', 'admission.json', 'feasibility.json'},
                 'extra/missing queue freeze artifacts')
    for name, sha in manifest['files'].items():
        gate.require(name in ('splits.json', 'protocol.json', 'admission.json', 'feasibility.json')
                     and gate.digest(gate.read(frozen / name, 4 * 1024 * 1024)) == sha, 'queue freeze file changed')
    admission.validate_admission(read_json(frozen / 'admission.json'), rows, sources())
    for row, entry in zip(rows, manifest['episodes']):
        public = gate.bundle(frozen / row['id'] / 'public', gate.PUBLIC)
        gate.require(gate.digest(public['observation.json']) == entry['observation_sha256']
                     and public['protocol.json'] == gate.encode(gate.PROTOCOL), 'queue frozen public bytes changed')
        prompt(public['observation.json'])
    return frozen, manifest, anchors['prepares']


def authorize(manifest, freeze_sha, approval_file):
    value = read_json(approval_file, 16384)
    expected = {'schema': 'trinite.queue-learning-owner-approval.v1', 'freeze_sha256': freeze_sha,
                'implementation_commit': manifest['git']['head'], 'approved_by': 'Trent Slade',
                'reviewed_merged_green': True,
                'authorization': 'execute queue-learning-v1 seeds 17 and 29 once'}
    gate.require(json_bytes(value) == json_bytes(expected), 'explicit owner approval of this frozen run required')
    git = git_identity()
    gate.require(git['clean'] and git['head'] == git['main'] == manifest['git']['head'],
                 'queue run requires clean reviewed main commit matching freeze')
    # This is an operator declaration. Local JSON/git cannot certify authorship,
    # remote CI, human review, blindness or chronology; the owner controls release.


def usage(started, before):
    after = resource.getrusage(resource.RUSAGE_SELF)
    return {'wall_seconds_hex': (time.monotonic() - started).hex(), 'peak_rss_kib': after.ru_maxrss,
            'user_seconds_hex': (after.ru_utime - before.ru_utime).hex(),
            'system_seconds_hex': (after.ru_stime - before.ru_stime).hex()}


def train_worker(args):
    from . import numerical
    numerical.configure()
    frozen, manifest, prepares = frozen_context(Path(args.frozen), args.freeze_sha, args.anchors, args.anchors_sha)
    authorize(manifest, args.freeze_sha, args.approval)
    gate.disjoint(frozen, args.anchors, args.approval, args.out)
    destination = gate.fresh(args.out)
    started, before, state = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF), None
    try:
        rows = admission.training_rows(frozen, manifest, prepares)
        write_json(destination / 'train-data.json', rows)
        state = numerical.state_for(rows, args.seed, args.freeze_sha)
        update_start = time.monotonic()
        from trinite.training import run_steps
        run_steps(state)  # ONE authoritative native updater, unchanged.
        update_seconds = time.monotonic() - update_start
        gate.require(update_seconds <= 120 and state.step == 64, 'queue update deadline/completion failed')
        payload, metadata = numerical.snapshot(state, args.freeze_sha)
        gate.write(destination / 'tensors.safetensors', payload)
        gate.write(destination / 'metadata.json', metadata)
        write_json(destination / 'training.json', {'outcome': 'completed', 'seed': args.seed,
                   'schedule': numerical.schedule(rows, args.seed), 'update_seconds_hex': update_seconds.hex(),
                   'resources': usage(started, before), 'checkpoint_sha256': [gate.digest(payload), gate.digest(metadata)],
                   'train_data_sha256': gate.digest(json_bytes(rows)), 'step': state.step,
                   'train_diagnostics_scope': 'native validation history is train-only; no held-out loss/selection'})
    except Exception as error:
        if state is not None:
            try:
                payload, metadata = numerical.snapshot(state, args.freeze_sha)
                gate.write(destination / 'partial.safetensors', payload)
                gate.write(destination / 'partial-metadata.json', metadata)
            except Exception as partial_error:
                write_json(destination / 'partial-failure.json', {'type': type(partial_error).__name__})
        retain_failure(destination, error)
        raise
    finally:
        write_json(destination / 'resources.json', usage(started, before))


def infer_worker(args):
    from . import numerical
    numerical.configure()
    # No frozen/private request path or held-out labels are arguments to this worker.
    public = gate.safe_path(args.public)
    checkpoint = gate.safe_path(args.checkpoint)
    anchor = anchored_json(args.checkpoint_anchors, args.checkpoint_sha)
    entry = anchor['checkpoints'][str(args.seed)]
    gate.disjoint(public, checkpoint, args.checkpoint_anchors, args.out)
    destination = gate.fresh(args.out)
    gate.write(destination / 'raw.jsonl', b'')
    gate.write(destination / 'attempts.jsonl', b'')
    started, before = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    try:
        rows = read_json(checkpoint / 'train-data.json')
        gate.require(gate.digest(json_bytes(rows)) == entry['train_data_sha256'], 'queue train data checkpoint changed')
        state = numerical.restore(gate.read(checkpoint / 'tensors.safetensors', 32 * 1024 * 1024),
            gate.read(checkpoint / 'metadata.json', 1024 * 1024), rows, args.seed,
            anchor['freeze_sha256'], entry['checkpoint_sha256'])
        gate.require(state.step == 64, 'queue inference requires frozen final checkpoint')
        from trinite.training import model_identity
        initial = model_identity(state.model)
        inventory = read_json(public / 'inventory.json')
        for row in inventory:
            observed = gate.read(public / (row['id'] + '.json'))
            gate.require(gate.digest(observed) == row['observation_sha256'], 'queue inference public packet changed')
            append_json(destination / 'attempts.jsonl', {'id': row['id']})
            result = numerical.infer(state.model, observed)
            append_json(destination / 'raw.jsonl', {'id': row['id'], 'raw': result})
        gate.require(model_identity(state.model) == initial, 'queue inference mutated model')
        write_json(destination / 'completed.json', {'model_identity': initial, 'examples': len(inventory)})
    except Exception as error:
        retain_failure(destination, error)
        raise
    finally:
        write_json(destination / 'resources.json', usage(started, before))


def launch(argv, seconds):
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        _, evidence = gate.bounded_process(argv, b'', timeout=seconds)
    except gate.ProcessError as error:
        evidence = error.evidence
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    elapsed = time.monotonic() - started
    return {'process': evidence, 'wall_seconds_hex': elapsed.hex(), 'deadline_seconds_hex': float(seconds).hex(),
            'elapsed_limit_exceeded': elapsed > seconds,
            'child_user_seconds_hex': (after.ru_utime - before.ru_utime).hex(),
            'child_system_seconds_hex': (after.ru_stime - before.ru_stime).hex(),
            'children_peak_rss_kib_high_water': after.ru_maxrss,
            'scope': 'whole child deadline includes imports, admission, IO and diagnostics; bounded stdout/stderr retained'}


def run(args):
    frozen, out, anchor_dir = gate.disjoint(args.frozen, args.out, args.run_anchors)
    frozen, manifest, prepares = frozen_context(frozen, args.freeze_sha, args.anchors, args.anchors_sha)
    from .numerical import configure
    gate.require(configure() == manifest['environment'], 'queue frozen execution environment changed')
    authorize(manifest, args.freeze_sha, args.approval)
    destination, anchors = gate.fresh(out), gate.fresh(anchor_dir)
    started, before = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    workers, checkpoints, predictions = {}, {}, {}
    launcher = [sys.executable, '-I', '-B', str(ROOT / 'scripts/run_queue_learning_v1.py')]
    try:
        common = ['--frozen', str(frozen), '--freeze-sha', args.freeze_sha,
                  '--anchors', str(args.anchors), '--anchors-sha', args.anchors_sha, '--approval', str(args.approval)]
        for seed in SEEDS:
            worker = launch(launcher + ['_train', *common, '--seed', str(seed),
                            '--out', str(destination / f'train-{seed}')], 120)
            workers[str(seed)] = {'train': worker}
            if (worker['process']['exit_code'] == 0 and worker['process']['failure'] is None
                    and not worker['elapsed_limit_exceeded']):
                info = read_json(destination / f'train-{seed}/training.json')
                checkpoints[str(seed)] = {k: info[k] for k in ('checkpoint_sha256', 'train_data_sha256')}
            else:
                checkpoints[str(seed)] = None
            write_json(destination / f'worker-{seed}-train.json', worker)
        # Independently retained checkpoint byte identities BEFORE any held-out inference.
        checkpoint_sha = write_json(anchors / 'checkpoints.json', {'freeze_sha256': args.freeze_sha,
                                    'checkpoints': checkpoints})
        public = gate.fresh(destination / 'public')
        heldout = [e for e in manifest['episodes'] if e['split'] != 'train']
        for row in heldout:
            gate.write(public / (row['id'] + '.json'), gate.read(frozen / row['id'] / 'public/observation.json'))
        write_json(public / 'inventory.json', [{k: row[k] for k in ('id', 'observation_sha256')} for row in heldout])
        for seed in SEEDS:
            attempted = set()
            failure_status = 'not_attempted'
            if checkpoints[str(seed)] is not None:
                remaining = 900 - float.fromhex(workers[str(seed)]['train']['wall_seconds_hex'])
                worker = launch(launcher + ['_infer', '--public', str(public), '--checkpoint',
                    str(destination / f'train-{seed}'), '--checkpoint-anchors', str(anchors / 'checkpoints.json'),
                    '--checkpoint-sha', checkpoint_sha, '--seed', str(seed), '--out', str(destination / f'infer-{seed}')], remaining)
                workers[str(seed)]['infer'] = worker
                write_json(destination / f'worker-{seed}-infer.json', worker)
                failure_status = 'timeout' if worker['process']['failure'] == 'timeout' else 'failure'
                attempts = destination / f'infer-{seed}/attempts.jsonl'
                if attempts.exists():
                    allowed = {e['id'] for e in heldout}
                    for line in gate.read(attempts).splitlines():
                        record = parse_json(line + b'\n', canonical=True)
                        gate.keys(record, ('id',))
                        gate.require(record['id'] in allowed - attempted, 'foreign/duplicate inference attempt')
                        attempted.add(record['id'])
                journal = destination / f'infer-{seed}/raw.jsonl'
                if journal.exists():
                    allowed = {e['id'] for e in heldout}
                    for line in gate.read(journal, 4 * 1024 * 1024).splitlines():
                        row = parse_json(line + b'\n', canonical=True)
                        gate.keys(row, ('id', 'raw'))
                        gate.require(row['id'] in allowed and row['id'] in attempted, 'foreign/duplicate inference journal row')
                        allowed.remove(row['id'])
                        predictions[f'seed{seed}', row['id']] = row['raw']
            for row in heldout:
                predictions.setdefault((f'seed{seed}', row['id']),
                                       {'status': failure_status if row['id'] in attempted else 'not_attempted'})
        # Baselines use TRAIN ONLY; no held-out targets have been computed.
        train_rows = admission.training_rows(frozen, manifest, prepares)
        constants = evaluation.constants(train_rows)
        write_json(destination / 'baselines.json', constants)
        for row in heldout:
            observed = gate.read(public / (row['id'] + '.json'))
            for method, value in constants.items():
                predictions[method, row['id']] = {'status': 'prediction', 'value': value}
            predictions['causal', row['id']] = evaluation.causal(observed)
        barrier_sha = evaluation.commit_all(frozen, manifest['episodes'], predictions,
                                          destination / 'commits', anchors / 'barrier.json')
        report = evaluation.score_all(frozen, prepares, destination / 'commits', anchors / 'barrier.json',
                                      barrier_sha, manifest['episodes'], destination / 'scores')
        write_json(destination / 'run.json', {'freeze_sha256': args.freeze_sha, 'workers': workers,
                   'checkpoint_sha256': checkpoint_sha, 'barrier_sha256': barrier_sha,
                   'resources': usage(started, before), 'summary': report})
        return {'checkpoint_sha256': checkpoint_sha, 'barrier_sha256': barrier_sha,
                'verification_scope': 'gate observation/target replay and safe checkpoint structure; no optimizer replay',
                'failed_workers': any(v is None for v in checkpoints.values()) or
                   any(w.get('infer', {}).get('process', {}).get('exit_code', 0) != 0
                       or w.get('infer', {}).get('elapsed_limit_exceeded', False) for w in workers.values()) or
                   any(predictions[f'seed{seed}', row['id']]['status'] in ('failure', 'timeout', 'not_attempted')
                       for seed in SEEDS for row in heldout)}
    except Exception as error:
        retain_failure(destination, error)
        raise
    finally:
        write_json(destination / 'resources.json', usage(started, before))


def verify(args):
    frozen, manifest, prepares = frozen_context(Path(args.frozen), args.freeze_sha, args.anchors, args.anchors_sha)
    destination = gate.safe_path(args.run)
    anchor_dir = gate.safe_path(args.run_anchors)
    retained = anchored_json(anchor_dir / 'run-inventory.json', args.run_sha)
    gate.require(retained == inventory(destination), 'queue run artifact inventory changed')
    checkpoint = anchored_json(anchor_dir / 'checkpoints.json', args.checkpoint_sha)
    gate.require(checkpoint['freeze_sha256'] == args.freeze_sha, 'queue checkpoint freeze mismatch')
    from .numerical import configure, restore
    gate.require(configure() == manifest['environment'], 'queue verification environment mismatch')
    for seed in SEEDS:
        entry = checkpoint['checkpoints'][str(seed)]
        if entry is not None:
            folder = destination / f'train-{seed}'
            rows = read_json(folder / 'train-data.json')
            gate.require(gate.digest(json_bytes(rows)) == entry['train_data_sha256'], 'queue checkpoint train data changed')
            state = restore(gate.read(folder / 'tensors.safetensors', 32 * 1024 * 1024),
                            gate.read(folder / 'metadata.json', 1024 * 1024), rows, seed,
                            args.freeze_sha, entry['checkpoint_sha256'])
            gate.require(state.step == 64, 'queue final checkpoint step mismatch')
    entries = evaluation.public_barrier(destination / 'commits', anchor_dir / 'barrier.json', args.barrier_sha, manifest['episodes'])
    for entry in entries:
        name = entry['method'] + '-' + entry['id']
        gate.verify(frozen / entry['id'] / 'private', destination / 'commits' / name,
                    prepares[entry['id']], entry['commit_sha256'], destination / 'scores' / name)
        receipt = read_json(destination / 'scores' / (name + '-verification.json'))
        gate.require(receipt == {'id': entry['id'], 'method': entry['method'],
            'prepare_sha256': prepares[entry['id']], 'commit_sha256': entry['commit_sha256'], 'verified': True,
            'result_sha256': gate.digest(gate.read(destination / 'scores' / name / 'evaluation.json'))}, 'queue verification receipt mismatch')
    gate.require(read_json(destination / 'scores/verification.json') == {
        'barrier_sha256': args.barrier_sha,
        'receipts': [read_json(destination / 'scores' / (e['method'] + '-' + e['id'] + '-verification.json')) for e in entries]},
        'queue aggregate verification receipt mismatch')
    summary = evaluation.summarize(entries, destination / 'commits', destination / 'scores')
    gate.require(read_json(destination / 'scores/summary.json') == summary, 'queue summary differs from retained predictions')
    run_record = read_json(destination / 'run.json')
    gate.require(run_record['summary'] == summary and run_record['freeze_sha256'] == args.freeze_sha
                 and run_record['checkpoint_sha256'] == args.checkpoint_sha
                 and run_record['barrier_sha256'] == args.barrier_sha, 'queue run record context mismatch')
    return {'verified': True, 'optimizer_replay': False, 'operator_blindness_proved': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    p = sub.add_parser('feasibility')
    p.add_argument('--out', required=True)
    p = sub.add_parser('freeze')
    p.add_argument('--feasibility', required=True)
    p.add_argument('--feasibility-sha', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--anchors', required=True)
    p = sub.add_parser('engineering')
    p.add_argument('--out', required=True)
    p.add_argument('--anchors', required=True)
    for command in ('run', '_train', 'verify'):
        p = sub.add_parser(command)
        for field in ('frozen', 'freeze-sha', 'anchors', 'anchors-sha'):
            p.add_argument('--' + field, required=True)
        if command in ('run', '_train'):
            p.add_argument('--approval', required=True)
            p.add_argument('--out', required=True)
        if command == '_train':
            p.add_argument('--seed', type=int, choices=SEEDS, required=True)
        elif command == 'run':
            p.add_argument('--run-anchors', required=True)
        else:
            for field in ('run', 'run-anchors', 'checkpoint-sha', 'barrier-sha', 'run-sha'):
                p.add_argument('--' + field, required=True)
    p = sub.add_parser('_infer')
    for field in ('public', 'checkpoint', 'checkpoint-anchors', 'checkpoint-sha', 'out'):
        p.add_argument('--' + field, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    try:
        gate.require(os.name == 'posix', 'queue pilot requires POSIX')
        if args.operation == 'feasibility':
            result = {'feasibility_sha256': feasibility(args.out)}
        elif args.operation == 'freeze':
            result = freeze(args.feasibility, args.feasibility_sha, args.out, args.anchors)
        elif args.operation == 'run':
            result = run(args)
            result['run_sha256'] = write_json(Path(args.run_anchors) / 'run-inventory.json', inventory(Path(args.out)))
        elif args.operation == 'engineering':
            from .engineering import execute
            result = execute(args.out, args.anchors)
        elif args.operation == 'verify':
            result = verify(args)
        else:
            (train_worker if args.operation == '_train' else infer_worker)(args)
            result = {'completed': True}
        print(json_bytes(result).decode('ascii'), end='')
        return 2 if result.get('failed_workers') else 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print('queue-learning-v1 rejected: ' + type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 2
