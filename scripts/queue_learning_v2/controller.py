"""Fail-closed fit-first controller; no execution without a fresh exact approval."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import queue_evaluation_v1 as gate
from queue_learning_v1 import evaluation
from queue_learning_v1.artifacts import (anchored_json, inventory, read_json,
                                        retain_failure, write_json)
from trinite.contracts import json_bytes, parse_json
from . import admission
from .protocol import episodes, fit_rows
from .resources import supervise
from .spec import ROOT

ATTEMPTS = (('fit', 17, True, 128, 2816), ('seed17', 17, False, 256, 5600),
            ('seed29', 29, False, 256, 5600))


def launch(arguments, seconds, progress=None):
    return supervise([sys.executable, '-I', '-B', str(ROOT / 'scripts/run_queue_learning_v2.py'),
                      *arguments], seconds, progress=progress,
                      max_updates=128 if '--fit' in arguments else 256,
                      max_targets=4096 if '--fit' in arguments else 8192)


def successful(receipt):
    return receipt['exit_code'] == 0 and receipt['failure'] is None


def validate_checkpoint(folder, seed, fit, binding):
    """Safe restore and independently recomputed fit, never a report boolean."""
    from . import numerical
    from .protocol import fit_report
    folder = gate.safe_path(folder)
    info = read_json(folder / 'training.json')
    expected_step, expected_targets = (128, 2816) if fit else (256, 5600)
    seconds = float.fromhex(info['update_seconds_hex'])
    gate.require(info['outcome'] == 'completed' and type(info['fit']) is bool
        and info['fit'] == fit and type(info['seed']) is int and info['seed'] == seed
        and type(info['step']) is int and info['step'] == expected_step
        and info['target_tokens'] == info['targets'] == expected_targets
        and math.isfinite(seconds) and 0 < seconds < 120, 'incomplete or over-budget v2 checkpoint')
    rows = read_json(folder / 'train-data.json')
    gate.require(gate.digest(json_bytes(rows)) == info['train_data_sha256'], 'checkpoint train data changed')
    state = numerical.restore(gate.read(folder / 'tensors.safetensors', 32 << 20),
        gate.read(folder / 'metadata.json', 1 << 20), rows, seed, binding,
        info['checkpoint_sha256'], fit=fit)
    gate.require(state.step == expected_step and state.target_tokens == expected_targets
                 and info['schedule'] == numerical.schedule(rows, seed, fit=fit),
                 'checkpoint completion/schedule differs from worker')
    for label in ('initial', 'final'):
        raw = gate.read(folder / (label + '-diagnostics.json'))
        gate.require(gate.digest(raw) == info['diagnostics'][label + '_sha256'], 'diagnostic bytes changed')
        gate.require(gate.decode(raw)['step'] == (0 if label == 'initial' else expected_step),
                     'diagnostic checkpoint step changed')
    if fit:
        report = fit_report(state.model, rows)
        selected = fit_rows(rows)
        records = report['generation']['records']
        gate.require(info['fit_report'] == report and len(records) == 8
                     and report['ids'] == [r['identity'] for r in selected]
                     and report['generation']['planned'] == 8
                     and report['generation']['exact'] == 8,
                     'final fit must pass all eight independently regenerated prefixes')
        for record, row in zip(records, selected):
            gate.require(record['id'] == row['identity'] and record['exact'] is True
                         and record['tokens'] == [*str(row['answer']).encode('ascii'), 257]
                         and record['parsed']['status'] == 'prediction',
                         'fit requires canonical decimal plus EOS for every prefix')
    return {k: info[k] for k in ('checkpoint_sha256', 'train_data_sha256')} | {
        'training_sha256': gate.digest(gate.read(folder / 'training.json'))}


def validate_fit_certificate(checkpoint, certificate, certificate_sha, freeze_sha):
    value = anchored_json(Path(certificate), certificate_sha)
    expected = {'schema': 'trinite.queue-learning-fit-gate.v2', 'freeze_sha256': freeze_sha,
                'fit': validate_checkpoint(Path(checkpoint), 17, True, freeze_sha)}
    gate.require(value == expected, 'missing or changed independently verified fit gate')
    return value


def empty_predictions(manifest):
    return {(method, r['id']): {'status': 'not_attempted'} for method in evaluation.METHODS
            for r in manifest['episodes'] if r['split'] != 'train'}


def stopped_summary(predictions):
    # No heldout target reads on a stopped study; full denominators remain visible.
    methods = {}
    for method in evaluation.METHODS:
        methods[method] = {}
        for split in ('validation', 'test'):
            group = [predictions[method, r['id']] for r in episodes() if r['split'] == split]
            methods[method][split] = {'planned': 64,
                'statuses': {s: sum(x['status'] == s for x in group) for s in evaluation.STATUSES},
                'mae': None, 'conditional_numeric_mae': None, 'exact_match': None,
                'numeric_exact_match': None,
                'unscored_exact': {'numerator': None, 'denominator': 64},
                'coverage': {'numerator': sum(x['status'] == 'prediction' for x in group), 'denominator': 64}}
    return {'schema': 'trinite.queue-learning-stopped.v2', 'methods': methods,
            'heldout_scored': False, 'learning_claim': False}


def run(args):
    """Executed only under the outer whole-study supervisor (including admission)."""
    started = time.monotonic()
    frozen, out, anchor_dir = gate.disjoint(args.frozen, args.out, args.run_anchors)
    gate.require(ROOT not in anchor_dir.parents and anchor_dir != ROOT,
                 'independent run anchors must be outside checkout')
    destination, anchors = gate.fresh(out), gate.fresh(anchor_dir)
    attempt_rows = [{'name': n, 'seed': s, 'fit': f, 'planned_updates': u, 'planned_targets': t,
                     'status': 'not_attempted'} for n, s, f, u, t in ATTEMPTS]
    write_json(destination / 'planned-attempts.json', attempt_rows)
    records, checkpoints, begins = {}, {}, {}
    scoring_started = scoring_completed = False
    predictions = empty_predictions({'episodes': episodes()})
    summary = stopped_summary(predictions)
    stopped, checkpoint_sha, barrier_sha, fit_sha = None, None, None, None
    # Static reservations include failed/unattempted work: no budget replenishment.
    gate.require(sum(x['planned_updates'] for x in attempt_rows) == 640 and
                 sum(x['planned_targets'] for x in attempt_rows) == 14016 <= 20480,
                 'v2 planned global budgets changed')
    try:
        frozen, manifest, prepares = admission.frozen_context(frozen, args.freeze_sha, args.anchors, args.anchors_sha)
        admission.authorize(manifest, args.freeze_sha, args.approval)
        # Exclusive durable claim adjacent to independent preparation anchors.
        # A failed attempt consumes the freeze; another directory is not a retry.
        claim = Path(args.anchors).with_name(Path(args.anchors).name + '.execution-claim.json')
        write_json(claim, {'schema': 'trinite.queue-learning-execution-claim.v2',
            'freeze_sha256': args.freeze_sha, 'out': str(destination), 'run_anchors': str(anchors)})
        common = ['--frozen', str(frozen), '--freeze-sha', args.freeze_sha,
                  '--anchors', str(args.anchors), '--anchors-sha', args.anchors_sha,
                  '--approval', str(args.approval)]
        fit_sha = None
        for index, (name, seed, fit, _, _) in enumerate(ATTEMPTS):
            begins[name] = time.monotonic()
            remaining = min(900., 2700. - (begins[name] - started))
            gate.require(remaining > 0, 'whole-study deadline reached')
            folder = destination / name
            command = ['_train', *common, '--seed', str(seed), '--out', str(folder)]
            if fit:
                command += ['--fit']
            else:
                command += ['--fit-checkpoint', str(destination / 'fit'), '--fit-gate',
                            str(anchors / 'fit-gate.json'), '--fit-gate-sha', fit_sha]
            attempt_rows[index]['status'] = 'attempted'
            receipt = launch(command, remaining, folder / 'progress.json')
            records[name] = {'train': receipt}
            write_json(destination / (name + '-train-worker.json'), receipt)
            if not successful(receipt):
                attempt_rows[index]['status'] = 'timeout' if receipt['failure'] in ('timeout', 'update_timeout') else 'failure'
                stopped = name + ': worker failed'
                break
            remaining = min(900 - (time.monotonic() - begins[name]), 2700 - (time.monotonic() - started))
            gate.require(remaining > 0, 'attempt deadline before checkpoint validation')
            validation_dir = destination / (name + '-validation')
            validation = launch(['_validate', '--checkpoint', str(folder), '--seed', str(seed),
                '--freeze-sha', args.freeze_sha, '--out', str(validation_dir), *(['--fit'] if fit else [])], remaining)
            records[name]['validation'] = validation
            write_json(destination / (name + '-validation-worker.json'), validation)
            gate.require(successful(validation), 'final checkpoint/fit validation failed')
            entry = read_json(validation_dir / 'validated.json')
            gate.require(entry['training_sha256'] == gate.digest(gate.read(folder / 'training.json')),
                         'validated checkpoint changed')
            gate.require(time.monotonic() - begins[name] < 900 and time.monotonic() - started < 2700,
                         'checkpoint verification exceeded attempt/study budget')
            attempt_rows[index]['status'] = 'completed'
            if fit:
                records[name]['whole_attempt_seconds_hex'] = (time.monotonic() - begins[name]).hex()
                fit_sha = write_json(anchors / 'fit-gate.json', {
                    'schema': 'trinite.queue-learning-fit-gate.v2', 'freeze_sha256': args.freeze_sha, 'fit': entry})
            else:
                checkpoints[str(seed)] = entry
        if stopped is None:
            gate.require(set(checkpoints) == {'17', '29'}, 'both final checkpoints required before heldout inference')
            checkpoint_sha = write_json(anchors / 'checkpoints.json', {
                'schema': 'trinite.queue-learning-checkpoints.v2', 'freeze_sha256': args.freeze_sha,
                'checkpoints': checkpoints})
            public = gate.fresh(destination / 'public')
            heldout = [r for r in manifest['episodes'] if r['split'] != 'train']
            for row in heldout:
                gate.write(public / (row['id'] + '.json'), gate.read(frozen / row['id'] / 'public/observation.json'))
            write_json(public / 'inventory.json', [{k: r[k] for k in ('id', 'observation_sha256')} for r in heldout])
            # Constants fixed from all train rows, before any heldout inference.
            train_rows = read_json(destination / 'seed17/train-data.json')
            constants = evaluation.constants(train_rows)
            write_json(destination / 'baselines.json', constants)
            for seed in (17, 29):
                name = 'seed' + str(seed)
                remaining = min(900 - (time.monotonic() - begins[name]), 2700 - (time.monotonic() - started))
                if remaining <= 0:
                    stopped = name + ': whole-attempt deadline'
                    break
                folder = destination / ('infer-' + str(seed))
                receipt = launch(['_infer', '--public', str(public), '--checkpoint', str(destination / name),
                    '--checkpoint-anchors', str(anchors / 'checkpoints.json'), '--checkpoint-sha', checkpoint_sha,
                    '--seed', str(seed), '--out', str(folder)], remaining)
                records[name]['infer'] = receipt
                write_json(destination / (name + '-infer-worker.json'), receipt)
                allowed = [r['id'] for r in heldout]
                attempted = []
                if (folder / 'attempts.jsonl').exists():
                    attempted = [parse_json(line + b'\n', canonical=True)['id'] for line in
                                 gate.read(folder / 'attempts.jsonl', 4 << 20).splitlines()]
                gate.require(attempted == allowed[:len(attempted)], 'invalid inference attempts')
                failure_status = 'timeout' if receipt['failure'] in ('timeout', 'update_timeout') else 'failure'
                for identity in attempted:
                    predictions[name, identity] = {'status': failure_status}
                actual = []
                if (folder / 'raw.jsonl').exists():
                    actual = [parse_json(line + b'\n', canonical=True) for line in
                              gate.read(folder / 'raw.jsonl', 4 << 20).splitlines()]
                gate.require([r['id'] for r in actual] == attempted[:len(actual)], 'invalid raw inference journal')
                for row in actual:
                    predictions[name, row['id']] = evaluation.check_raw(row['raw'], gate.read(public / (row['id'] + '.json')))
                if not successful(receipt):
                    records[name]['whole_attempt_seconds_hex'] = (time.monotonic() - begins[name]).hex()
                    stopped = name + ': inference failed'
                    attempt_rows[1 if seed == 17 else 2]['status'] = failure_status
                    break
                gate.require([r['id'] for r in actual] == allowed, 'missing heldout predictions')
                records[name]['whole_attempt_seconds_hex'] = (time.monotonic() - begins[name]).hex()
                gate.require(float.fromhex(records[name]['whole_attempt_seconds_hex']) < 900, 'whole attempt exhausted')
            if stopped is None:
                for row in heldout:
                    observed = gate.read(public / (row['id'] + '.json'))
                    for method, value in constants.items():
                        predictions[method, row['id']] = {'status': 'prediction', 'value': value}
                    predictions['causal', row['id']] = evaluation.causal(observed)
                # All seeds and baselines, both splits, committed before any score.
                barrier_sha = evaluation.commit_all(frozen, manifest['episodes'], predictions,
                    destination / 'commits', anchors / 'barrier.json')
                gate.require(time.monotonic() - started < 2700, 'study deadline before scoring')
                scoring_started = True
                write_json(destination / 'scoring-started.json', {'barrier_sha256': barrier_sha})
                summary = evaluation.score_all(frozen, prepares, destination / 'commits',
                    anchors / 'barrier.json', barrier_sha, manifest['episodes'], destination / 'scores')
                scoring_completed = True
                gate.require(time.monotonic() - started < 2700, 'study deadline after scoring')
    except Exception as error:
        stopped = type(error).__name__ + ': ' + str(error)
        retain_failure(destination, error)
        for row in attempt_rows:
            if row['status'] == 'attempted':
                row['status'] = 'failure'
    finally:
        for name, record in records.items():
            record.setdefault('whole_attempt_seconds_hex', (time.monotonic() - begins[name]).hex())
        if stopped:
            summary = stopped_summary(predictions)
            summary['heldout_scored'] = scoring_started
            summary['scoring_completed'] = scoring_completed
        write_json(destination / 'predictions.json', [{'method': method, 'id': identity, 'raw': raw}
                   for (method, identity), raw in sorted(predictions.items())])
        write_json(destination / 'attempts.json', attempt_rows)
        write_json(destination / 'run.json', {'schema': 'trinite.queue-learning-run.v2',
            'freeze_sha256': args.freeze_sha, 'stopped': stopped, 'workers': records,
            'scoring_started': scoring_started, 'scoring_completed': scoring_completed,
            'checkpoint_sha256': checkpoint_sha, 'barrier_sha256': barrier_sha, 'fit_gate_sha256': fit_sha, 'summary': summary,
            'wall_seconds_hex': (time.monotonic() - started).hex(),
            'optimizer_replay': False, 'learning_claim': False})
        run_sha = write_json(anchors / 'run-inventory.json', inventory(destination))
    return {'failed_workers': stopped is not None, 'stopped': stopped, 'run_sha256': run_sha,
            'checkpoint_sha256': checkpoint_sha, 'barrier_sha256': barrier_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    for command in ('feasibility', '_feasibility'):
        p = sub.add_parser(command)
        p.add_argument('--out', required=True)
    p = sub.add_parser('freeze')
    for field in ('feasibility', 'feasibility-sha', 'out', 'anchors'):
        p.add_argument('--' + field, required=True)
    for command in ('run', '_run', '_train'):
        p = sub.add_parser(command)
        for field in ('frozen', 'freeze-sha', 'anchors', 'anchors-sha', 'approval', 'out'):
            p.add_argument('--' + field, required=True)
        if command in ('run', '_run'):
            p.add_argument('--run-anchors', required=True)
        else:
            p.add_argument('--seed', type=int, choices=(17, 29), required=True)
            p.add_argument('--fit', action='store_true')
            for field in ('fit-checkpoint', 'fit-gate', 'fit-gate-sha'):
                p.add_argument('--' + field)
    p = sub.add_parser('verify')
    for field in ('frozen', 'freeze-sha', 'anchors', 'anchors-sha', 'run', 'run-anchors', 'run-sha'):
        p.add_argument('--' + field, required=True)
    p = sub.add_parser('_validate')
    for field in ('checkpoint', 'freeze-sha', 'out'):
        p.add_argument('--' + field, required=True)
    p.add_argument('--seed', type=int, choices=(17, 29), required=True)
    p.add_argument('--fit', action='store_true')
    p = sub.add_parser('_infer')
    for field in ('public', 'checkpoint', 'checkpoint-anchors', 'checkpoint-sha', 'out'):
        p.add_argument('--' + field, required=True)
    p.add_argument('--seed', type=int, choices=(17, 29), required=True)
    args = parser.parse_args()
    try:
        gate.require(sys.platform == 'linux', 'v2 supervision requires Linux /proc and subreaper')
        if args.operation in ('run', 'feasibility'):
            # Start the whole-study clock before configure/admission/imports in child.
            argv = ['_' + args.operation, *sys.argv[2:]]
            receipt = launch(argv, 2700 if args.operation == 'run' else 900)
            out = Path(args.out)
            if not out.exists():
                gate.fresh(out)
            write_json(out / 'supervisor.json', receipt)
            if args.operation == 'run':
                # Seal only after all descendants reaped and outer receipt retained.
                anchors = Path(args.run_anchors)
                if not anchors.exists():
                    gate.fresh(anchors)
                result = {'process': receipt, 'run_sha256': write_json(anchors / 'final-inventory.json', inventory(out)),
                          'failed_workers': not successful(receipt)}
            else:
                result = {'process': receipt, 'failed_workers': not successful(receipt)}
                if successful(receipt):
                    proof = read_json(out / 'feasibility.json')
                    proof['supervisor'] = receipt
                    result['feasibility_sha256'] = write_json(out / 'supervised-feasibility.json', proof)
        elif args.operation == '_feasibility':
            result = {'feasibility_sha256': admission.feasibility(args.out)}
        elif args.operation == 'freeze':
            result = admission.freeze(args.feasibility, args.feasibility_sha, args.out, args.anchors)
        elif args.operation == 'verify':
            from .verification import verify
            result = verify(args)
        elif args.operation == '_validate':
            destination = gate.fresh(args.out)
            entry = validate_checkpoint(Path(args.checkpoint), args.seed, args.fit, args.freeze_sha)
            result = {'validation_sha256': write_json(destination / 'validated.json', entry)}
        elif args.operation == '_run':
            result = run(args)
        else:
            from .workers import train_worker, infer_worker
            (train_worker if args.operation == '_train' else infer_worker)(args)
            result = {'completed': True}
        print(json_bytes(result).decode('ascii'), end='')
        return 2 if result.get('failed_workers') else 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print('queue-learning-v2 rejected: ' + type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 2
