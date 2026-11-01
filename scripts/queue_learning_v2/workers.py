"""Fixed v2 workers; only training receives private TRAIN request access."""
import math
import os
from pathlib import Path
import resource
import time

import queue_evaluation_v1 as gate
from queue_learning_v1.artifacts import (anchored_json, append_json, read_json,
                                        retain_failure, write_json)
from trinite.contracts import json_bytes


def usage(started, before):
    after = resource.getrusage(resource.RUSAGE_SELF)
    return {'wall_seconds_hex': (time.monotonic() - started).hex(),
            'peak_rss_kib': after.ru_maxrss,
            'user_seconds_hex': (after.ru_utime - before.ru_utime).hex(),
            'system_seconds_hex': (after.ru_stime - before.ru_stime).hex()}


def progress(destination, phase, state=None, update_started=None):
    """One bounded atomic current record, with an append-only transition journal."""
    value = {'phase': phase, 'step': state.step if state is not None else 0,
             'target_tokens': state.target_tokens if state is not None else 0,
             'update_started_seconds_hex': update_started.hex() if update_started is not None else None}
    append_json(destination / 'progress.jsonl', value)
    pending = destination / 'progress.next'
    gate.write(pending, json_bytes(value))
    os.replace(pending, destination / 'progress.json')


def diagnostics(model, rows):
    from .diagnostics import supervision, generation
    result = {'supervision': supervision(model, rows), 'generation': generation(model, rows)}
    s = result['supervision']
    nll = sum(float.fromhex(s[k]['nll_sum_hex']) for k in ('answer', 'eos'))
    native = float.fromhex(s['native']['mean_nll_hex']) * s['native']['targets']
    gate.require(math.isfinite(nll) and math.isfinite(native)
                 and math.isclose(nll, native, rel_tol=1e-6, abs_tol=1e-5),
                 'detached diagnostics disagree with native objective')
    gate.require(s['answer']['targets'] + s['eos']['targets'] == s['native']['targets'],
                 'diagnostic target denominator differs from native objective')
    return result


def train_worker(args):
    from . import admission, numerical
    from .protocol import fit_report, plan
    gate.require(type(args.fit) is bool, 'v2 fit mode must be boolean')
    environment = numerical.configure()
    frozen, manifest, prepares = admission.frozen_context(
        Path(args.frozen), args.freeze_sha, args.anchors, args.anchors_sha)
    gate.require(environment == manifest['environment'], 'v2 train environment changed')
    admission.authorize(manifest, args.freeze_sha, args.approval)
    gate.disjoint(frozen, args.anchors, args.approval, args.out)
    destination = gate.fresh(args.out)
    gate.write(destination / 'progress.jsonl', b'')
    started, before, state, update_start = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF), None, None
    try:
        progress(destination, 'preparing')
        config = plan(args.seed, fit=args.fit)
        if not args.fit:
            from .controller import validate_fit_certificate
            validate_fit_certificate(args.fit_checkpoint, args.fit_gate, args.fit_gate_sha, args.freeze_sha)
        # Independent write-once attempt ledger. Failure consumes the attempt too;
        # deleting it is an operator custody breach, never an automatic retry.
        claim = Path(args.anchors).with_name(Path(args.anchors).name +
                    ('.fit17.attempt.json' if args.fit else f'.main{args.seed}.attempt.json'))
        write_json(claim, {'schema': 'trinite.queue-learning-attempt-claim.v2',
                          'freeze_sha256': args.freeze_sha, 'out': str(destination),
                          'seed': args.seed, 'fit': args.fit})
        rows = admission.training_rows(frozen, manifest, prepares)
        write_json(destination / 'train-data.json', rows)
        state = numerical.state_for(rows, args.seed, args.freeze_sha, fit=args.fit)
        initial = diagnostics(state.model, rows)
        initial_fit = fit_report(state.model, rows) if args.fit else None
        write_json(destination / 'initial-diagnostics.json', {'step': 0, 'train': initial, 'fit': initial_fit})
        update_start = time.monotonic()
        progress(destination, 'updates', state, update_start)
        from trinite.training import run_steps
        for step in range(1, config.steps + 1):
            gate.require(time.monotonic() - update_start < 120, 'v2 cumulative update deadline')
            run_steps(state, stop_after=step)  # Authoritative native updater; no alternative AdamW.
            gate.require(state.step == step and time.monotonic() - update_start < 120,
                         'v2 incomplete update or cumulative update deadline')
            progress(destination, 'updates', state, update_start)
        update_seconds = time.monotonic() - update_start
        gate.require(state.step == config.steps
                     and state.target_tokens == (2816 if args.fit else 5600),
                     'v2 final checkpoint step/target count required')
        progress(destination, 'diagnostics', state, update_start)
        final = diagnostics(state.model, rows)
        final_fit = fit_report(state.model, rows) if args.fit else None
        write_json(destination / 'final-diagnostics.json', {'step': state.step, 'train': final, 'fit': final_fit})
        payload, metadata = numerical.snapshot(state, args.freeze_sha)
        gate.write(destination / 'tensors.safetensors', payload)
        gate.write(destination / 'metadata.json', metadata)
        write_json(destination / 'training.json', {
            'outcome': 'completed', 'seed': args.seed, 'fit': args.fit, 'step': state.step,
            'target_tokens': state.target_tokens, 'targets': state.target_tokens,
            'schedule': numerical.schedule(rows, args.seed, fit=args.fit),
            'update_seconds_hex': update_seconds.hex(), 'resources': usage(started, before),
            'checkpoint_sha256': [gate.digest(payload), gate.digest(metadata)],
            'train_data_sha256': gate.digest(json_bytes(rows)), 'fit_report': final_fit,
            'diagnostics': {'initial_sha256': gate.digest(gate.read(destination / 'initial-diagnostics.json')),
                            'final_sha256': gate.digest(gate.read(destination / 'final-diagnostics.json'))},
            'train_diagnostics_scope': 'all unique train rows once at 0/final; no heldout labels or selection'})
        progress(destination, 'completed', state, update_start)
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
    from trinite.training import model_identity
    environment = numerical.configure()
    # Neither frozen/private requests nor heldout truth is accepted by this API.
    public, checkpoint = gate.safe_path(args.public), gate.safe_path(args.checkpoint)
    anchor = anchored_json(args.checkpoint_anchors, args.checkpoint_sha)
    entry = anchor['checkpoints'][str(args.seed)]
    gate.require(entry is not None, 'missing v2 final checkpoint')
    gate.disjoint(public, checkpoint, args.checkpoint_anchors, args.out)
    destination = gate.fresh(args.out)
    gate.write(destination / 'raw.jsonl', b'')
    gate.write(destination / 'attempts.jsonl', b'')
    started, before = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    try:
        rows = read_json(checkpoint / 'train-data.json')
        gate.require(gate.digest(json_bytes(rows)) == entry['train_data_sha256'], 'v2 checkpoint train data changed')
        state = numerical.restore(gate.read(checkpoint / 'tensors.safetensors', 32 * 1024 * 1024),
            gate.read(checkpoint / 'metadata.json', 1024 * 1024), rows, args.seed,
            anchor['freeze_sha256'], entry['checkpoint_sha256'])
        gate.require(state.step == 256, 'v2 inference requires final step256')
        initial = model_identity(state.model)
        entries = read_json(public / 'inventory.json')
        gate.require(len(entries) == 128 and len({r['id'] for r in entries}) == 128,
                     'complete unique v2 heldout public inventory required')
        for row in entries:
            observed = gate.read(public / (row['id'] + '.json'))
            gate.require(gate.digest(observed) == row['observation_sha256'], 'v2 inference public packet changed')
            append_json(destination / 'attempts.jsonl', {'id': row['id']})
            append_json(destination / 'raw.jsonl', {'id': row['id'], 'raw': numerical.infer(state.model, observed)})
        gate.require(model_identity(state.model) == initial, 'v2 inference mutated model')
        write_json(destination / 'completed.json', {'model_identity': initial, 'examples': len(entries),
                                                  'environment': environment})
    except Exception as error:
        retain_failure(destination, error)
        raise
    finally:
        write_json(destination / 'resources.json', usage(started, before))
