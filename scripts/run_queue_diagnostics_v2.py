"""Read retained v1 training evidence only. No scientific freeze/run or optimizer path."""
import argparse
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

from trinite.contracts import json_bytes, parse_json
from queue_learning_v1 import numerical
from queue_learning_v1.spec import sources
from queue_learning_v2.diagnostics import supervision, generation


def forensic_restore(payload, metadata, rows, seed, expected):
    """Versioned foreign-environment inspection, NOT v1 restore or optimizer replay.

    Published anchors, exact source/plan/data/init and the authoritative closed
    tensor reader still apply. Recorded initial loss/environment are observations
    from the old host, never substituted into a v1 receipt or used for training.
    """
    from trinite.comparison_checkpoint import restore_tensors
    if [hashlib.sha256(v).hexdigest() for v in (payload, metadata)] != list(expected):
        raise ValueError('checkpoint anchors differ')
    value = parse_json(metadata, canonical=True)
    state = numerical.state_for(rows, seed, value['binding'])
    _, raw = numerical.snapshot(state, value['binding'])
    context = parse_json(raw, canonical=True)
    if value.keys() != context.keys():
        raise ValueError('checkpoint inventory differs')
    observed = {'environment', 'initial_train_loss', 'model_identity', 'step', 'cursor',
                'target_tokens', 'history', 'train_diagnostics'}
    if any(value[k] != context[k] for k in context.keys() - observed):
        raise ValueError('checkpoint source/plan/data/init differs')
    state.step, state.cursor, state.target_tokens = value['step'], value['cursor'], value['target_tokens']
    state.history, state.validation = value['history'], value['train_diagnostics']
    return restore_tensors(state, payload, value['model_identity'])


def diagnose(evidence):
    import torch
    fixture = ROOT / 'fixtures/queue-learning-result-20261101-v1'
    # Independent published file anchors, not identities recovered from mutable metadata.
    for line in (fixture / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if hashlib.sha256((evidence / name).read_bytes()).hexdigest() != expected:
            raise ValueError('retained evidence file changed: ' + name)
    anchors = parse_json((evidence / 'anchors/run-anchors/checkpoints.json').read_bytes(), canonical=True)
    result = {'schema': 'trinite.queue-training-diagnostic.v2', 'optimizer_steps': 0,
              'scope': '256 unique v1 training rows; initial and retained final models; no held-out diagnostics',
              'diagnostic_sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [Path(__file__).resolve(), ROOT / 'scripts/queue_learning_v2/diagnostics.py']},
              'cells': {}}
    numerical.configure()
    from trinite.training import environment
    result['environment'] = environment()
    from collections import defaultdict
    # Fail closed if a future edit accidentally reaches optimization.
    with patch.object(torch.optim.AdamW, 'step', side_effect=RuntimeError('diagnostics forbid optimizer steps')), torch.no_grad():
        for seed in (17, 29):
            slot = evidence / f'run/train-{seed}'
            raw = (slot / 'train-data.json').read_bytes()
            anchor = anchors['checkpoints'][str(seed)]
            if hashlib.sha256(raw).hexdigest() != anchor['train_data_sha256']:
                raise ValueError('train data anchor mismatch')
            rows = parse_json(raw, canonical=True)
            metadata = (slot / 'metadata.json').read_bytes()
            meta = parse_json(metadata, canonical=True)
            if sources() != meta['sources']:
                raise ValueError('v1 source closure differs; use the retained producer, never patch receipts')
            initial = numerical.state_for(rows, seed, meta['binding'])
            restored = forensic_restore((slot / 'tensors.safetensors').read_bytes(), metadata,
                                        rows, seed, anchor['checkpoint_sha256'])
            groups = defaultdict(list)
            for row in rows:
                groups[json_bytes(row['observation'])].append(row['answer'])
            params = dict(initial.model.named_parameters())
            changed = {n: float((p - params[n]).abs().max()).hex()
                       for n, p in restored.model.named_parameters()}
            result['cells'][str(seed)] = {
                'restore_scope': 'v2 foreign-environment forward inspection; v1 environment-bound restore rejected; no replay claim',
                'recorded_environment': meta['environment'],
                'observation_prefixes': len(groups),
                'conflicting_prefixes': sum(len(set(v)) > 1 for v in groups.values()),
                'initial': {'supervision': supervision(initial.model, rows), 'generation': generation(initial.model, rows)},
                'final': {'supervision': supervision(restored.model, rows), 'generation': generation(restored.model, rows)},
                'step': restored.step, 'visits': restored.cursor, 'targets': restored.target_tokens,
                'parameter_max_absolute_changes_hex': changed,
                'optimizer_steps': sorted({int(v['step']) for v in restored.optimizer.state.values()}),
                'initial_train_loss_hex': meta['initial_train_loss'],
                'final_train_loss_hex': meta['train_diagnostics'][-1]['loss_hex']}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = diagnose(args.evidence.resolve())
    with args.out.open('xb') as stream:
        stream.write(json_bytes(value))
