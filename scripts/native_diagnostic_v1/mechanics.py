"""Small train-only mechanics controls; no target experiment or new updater.

Tolerances and workload are constants, fixed before observations. The real CPU
backend is mandatory. Checks raise rather than produce a successful partial gate.
"""
from dataclasses import replace
import math
import random

import numpy as np
import torch

from trinite import training
from trinite.budget_checkpoint import restore_state, snapshot_state
from trinite.budget_training import plan_for, state_for
from trinite.contracts import ContractError, ModelConfig, identity, json_bytes, parse_json
from trinite.foundations_data import audit_bytes, dataset
from trinite.tokenizer import BOS, EOS, PAD, ByteTokenizer

LOSS_ATOL = 2e-6
LOSS_RTOL = 1e-6
REQUEST_ID = identity(b'trinite.native-diagnostic.mechanics.v1')


def _require(value, message):
    if not value:
        raise ContractError('native diagnostic mechanics: ' + message)


def engineering_plan():
    """Three native updates, not the target protocol or a learning-quality test."""
    return replace(plan_for(0, 'extended'), model=ModelConfig(context_length=64,
        blocks=1, width=16, heads=2, feed_forward_width=32), steps=3,
        batch_size=2, validation_every=3, max_target_tokens=128, max_seconds=30)


def _examples():
    raw, manifest = dataset('arithmetic')
    audit_bytes(raw, manifest)
    rows = sorted((r for r in parse_json(raw, canonical=True)['examples']
                   if r['split'] == 'train'), key=lambda r: r['example_identity'])
    short = min(rows, key=lambda r: (len(r['answer']), r['example_identity']))
    long = max(rows, key=lambda r: (len(r['answer']), r['example_identity']))
    _require(len(short['answer']) != len(long['answer']), 'mixed-length control missing')
    return [short, long]


def _manual_batch(rows, context):
    """Independent expected masks are derived from prompt/answer byte lengths."""
    encodings = []
    for row in rows:
        _require(row['split'] == 'train', 'held-out row in mechanics')
        prompt, answer = row['prompt'].encode(), row['answer'].encode()
        _require(row['text'].encode() == prompt + answer, 'prompt/answer concatenation')
        tokens = (BOS, *prompt, *answer, EOS)
        expected_loss = (0, *([0]*len(prompt)), *([1]*len(answer)), 1)
        encoded = ByteTokenizer(context).encode(row['text'], answer_start=len(prompt))
        _require(encoded.input_ids == tokens, 'token sequence/BOS/EOS mismatch')
        _require(encoded.loss_mask == expected_loss, 'prompt/answer/EOS mask mismatch')
        _require(encoded.attention_mask == (1,)*len(tokens), 'live mask mismatch')
        encodings.append({'identity': row['example_identity'], 'ids': encoded.input_ids,
                          'mask': encoded.attention_mask, 'loss': encoded.loss_mask})
    length = max(len(e['ids']) for e in encodings)
    ids = torch.tensor([list(e['ids'])+[PAD]*(length-len(e['ids'])) for e in encodings],
                       dtype=torch.int64, device='cpu')
    mask = torch.tensor([[True]*len(e['ids'])+[False]*(length-len(e['ids'])) for e in encodings],
                        dtype=torch.bool, device='cpu')
    return encodings, ids, mask


def check_loss(model):
    rows = _examples()
    examples, ids, mask = _manual_batch(rows, model.config.context_length)
    with torch.no_grad():
        logits = model(ids[:, :-1], mask[:, :-1]).logits
        losses, counts = [], []
        for row_index, row in enumerate(rows):
            # Each answer byte and EOS is predicted by the preceding position.
            start = len(row['prompt'].encode())
            gold = [*row['answer'].encode(), EOS]
            selected = []
            for offset, token in enumerate(gold):
                values = logits[row_index, start+offset].tolist()
                maximum = max(values)
                selected.append(maximum + math.log(math.fsum(math.exp(v-maximum) for v in values)) - values[token])
            losses.extend(selected)
            counts.append(len(selected))
        expected = math.fsum(losses)/len(losses)
        actual, count = training.target_loss(model, examples)
    _require(count == sum(counts), 'shifted scored-token count mismatch')
    _require(math.isclose(float(actual), expected, rel_tol=LOSS_RTOL, abs_tol=LOSS_ATOL),
             'independent token-mean loss mismatch')
    return {'examples': len(rows), 'scored_tokens': count, 'answer_lengths': [len(r['answer'].encode()) for r in rows],
            'actual_loss_hex': float(actual).hex(), 'oracle_loss_hex': expected.hex()}


def check_attention(model):
    # Same-shape comparisons avoid conflating prefix shape with numerical rounding.
    ids = torch.tensor([[BOS, 49, 43, 50, 61, 51, EOS, PAD]], dtype=torch.int64, device='cpu')
    mask = torch.tensor([[True]*7+[False]], dtype=torch.bool, device='cpu')
    changed_future = ids.clone(); changed_future[0, 4:7] = torch.tensor([45, 52, EOS])
    changed_padding = ids.clone(); changed_padding[0, 7] = 97
    with torch.no_grad():
        expected = model(ids, mask).logits
        future = model(changed_future, mask).logits
        padding = model(changed_padding, mask).logits
    _require(torch.equal(expected[:, :4], future[:, :4]), 'future tokens influence causal prefix')
    _require(torch.equal(expected[:, :7], padding[:, :7]), 'padding IDs influence live logits')
    _require(torch.count_nonzero(expected[:, 7:]).item() == 0 and
             torch.count_nonzero(padding[:, 7:]).item() == 0, 'padded output is nonzero')
    _require(model.output_weight is model.token_embedding, 'output embedding tie broken')
    return {'same_shape_causal_prefix_exact': True, 'masked_suffix_exact': True,
            'padded_logits_zero': True, 'output_embedding_tied': True}


def _snapshot(state):
    return snapshot_state(state, 'arithmetic', 'extended', REQUEST_ID)


def _resume(payload, metadata, plan):
    return restore_state(payload, metadata, 'arithmetic', 0, 'extended', REQUEST_ID,
                         (identity(payload), identity(metadata)), plan)


def check_update_resume():
    plan = engineering_plan()
    plain = state_for('arithmetic', 0, 'extended', plan)
    instrumented = state_for('arithmetic', 0, 'extended', plan)
    _require(plain.data['validation'] is plain.data['train'], 'diagnostics must be train-only')
    initial = training.model_identity(plain.model)
    training.run_steps(plain)
    training.run_steps(instrumented, stop_after=1)
    before = _snapshot(instrumented)
    # Observations do not affect numerical state or optimization history.
    check_loss(instrumented.model)
    check_attention(instrumented.model)
    after = _snapshot(instrumented)
    _require(before == after, 'observation changed model/optimizer/progress')
    resumed = _resume(*after, plan)
    training.run_steps(resumed)
    _require(_snapshot(plain) == _snapshot(resumed), 'split safe-checkpoint replay differs')
    _require(training.model_identity(plain.model) != initial, 'optimizer did not update model')
    _require(all(math.isfinite(float.fromhex(r['gradient_norm_hex'])) and
                 float.fromhex(r['gradient_norm_hex']) > 0 for r in plain.history),
             'nonfinite or zero recorded preclip gradient norm')
    _require(all(p.grad is None for p in plain.model.parameters()), 'completed updates retain gradients')
    return {'updates': plan.steps, 'split_after': 1, 'model_and_optimizer_exact': True,
            'observation_isolation_exact': True, 'model_updated': True,
            'positive_finite_gradient_norms': True, 'completed_gradients_cleared': True}


def _rng_identity():
    array = np.random.get_state()
    return (random.getstate(), torch.get_rng_state().clone(),
            (array[0], array[1].tobytes(), array[2], array[3], array[4]))


def preflight():
    """Run fixed real-backend mechanics; does not authorize a target run."""
    training.require_environment()
    before_rng = _rng_identity()
    state = state_for('arithmetic', 0, 'extended', engineering_plan())
    before = _snapshot(state)
    loss = check_loss(state.model)
    attention = check_attention(state.model)
    _require(before == _snapshot(state), 'mechanics mutated initial state')
    updates = check_update_resume()
    after_rng = _rng_identity()
    _require(before_rng[0] == after_rng[0] and torch.equal(before_rng[1], after_rng[1])
             and before_rng[2] == after_rng[2], 'mechanics consumed global RNG')
    return {'schema': 'trinite.native-diagnostic-mechanics.v1', 'passed': True,
            'scope': 'three-update engineering controls; no target learning claim',
            'loss_atol_hex': LOSS_ATOL.hex(), 'loss_rtol_hex': LOSS_RTOL.hex(),
            'loss': loss, 'attention': attention, 'update_resume': updates,
            'global_rng_unchanged': True}


def validate_preflight(record):
    """Validate a stored mechanics receipt without executing a model.

    This validates the complete fixed control contract and measurement consistency,
    not authenticity: the enclosing source-bound workflow must bind the receipt.
    Counts are pinned to this version's arithmetic control (one/four answer bytes,
    plus two EOS targets); a changed fixture requires an explicit contract update.
    """
    template = {
        'schema': 'trinite.native-diagnostic-mechanics.v1', 'passed': True,
        'scope': 'three-update engineering controls; no target learning claim',
        'loss_atol_hex': LOSS_ATOL.hex(), 'loss_rtol_hex': LOSS_RTOL.hex(),
        'loss': {'examples': 2, 'scored_tokens': 7, 'answer_lengths': [1, 4],
                 'actual_loss_hex': None, 'oracle_loss_hex': None},
        'attention': {'same_shape_causal_prefix_exact': True,
                      'masked_suffix_exact': True, 'padded_logits_zero': True,
                      'output_embedding_tied': True},
        'update_resume': {'updates': 3, 'split_after': 1,
                          'model_and_optimizer_exact': True,
                          'observation_isolation_exact': True, 'model_updated': True,
                          'positive_finite_gradient_norms': True,
                          'completed_gradients_cleared': True},
        'global_rng_unchanged': True,
    }

    def exact(value, expected, path):
        if expected is None:
            _require(type(value) is str, path + ' must be a canonical finite hex measurement')
            try:
                number = float.fromhex(value)
            except (ValueError, OverflowError):
                raise ContractError('native diagnostic mechanics: ' + path + ' invalid hex') from None
            _require(math.isfinite(number) and number > 0 and number.hex() == value,
                     path + ' must be a positive canonical finite hex measurement')
        elif type(expected) is dict:
            _require(type(value) is dict and value.keys() == expected.keys(),
                     path + ' exact keys required')
            for key, wanted in expected.items():
                exact(value[key], wanted, path + '.' + key)
        elif type(expected) is list:
            _require(type(value) is list and len(value) == len(expected),
                     path + ' exact list required')
            for index, wanted in enumerate(expected):
                exact(value[index], wanted, path + '[' + str(index) + ']')
        else:
            _require(type(value) is type(expected) and value == expected,
                     path + ' fixed control mismatch')

    exact(record, template, 'preflight')
    loss = record['loss']
    _require(math.isclose(float.fromhex(loss['actual_loss_hex']),
                         float.fromhex(loss['oracle_loss_hex']),
                         rel_tol=LOSS_RTOL, abs_tol=LOSS_ATOL),
             'preflight independent token-mean loss mismatch')
    return record
