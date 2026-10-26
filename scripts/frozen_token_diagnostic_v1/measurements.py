"""Gold-prefix observations of every admitted answer position, never raw decoding.

The native tokenizer, target loss and model-state identities remain authoritative.
This new diagnostic cannot repair an answer or establish held-out adequacy.
"""
import math

import torch
from torch.nn import functional as F

from trinite.contracts import ContractError, exact_keys, integer, json_bytes, require_identity
from trinite.tokenizer import ByteTokenizer, EOS, VOCABULARY_SIZE
from trinite.training import model_identity, target_loss
from native_diagnostic_v1.measurements import _state, _hex
from addition_development_v1.data import dataset, audit_bytes, examples
from addition_development_v1.protocol import STRATA

CONDITIONING = 'teacher-forced expected gold prefix; not prompt-only raw generation'
CLAIMS = {'training_updates': 0, 'raw_predictions_replayed': False,
          'fresh_confirmation': False, 'learning_adequate': False,
          'phase5_ready': False, 'release_ready': False}


def admitted(stratum):
    if type(stratum) is not str or stratum not in STRATA:
        raise ContractError('unknown frozen-token development stratum')
    raw, manifest = dataset()
    audit_bytes(raw, manifest)
    return examples(stratum)


def _rank(vector, token):
    """One-based descending-logit rank, lowest token ID breaks exact ties."""
    return int((vector > vector[token]).sum()) + int((vector[:token] == vector[token]).sum()) + 1


def _context(item):
    return {**{key: item[key] for key in ('example_identity', 'family_id', 'semantic_identity', 'carrier', 'coverage')},
            'answer_length_bytes': len(item['answer'].encode('ascii')),
            'prompt_length_bytes': len(item['prompt'].encode('ascii')),
            'operand_length_bytes': [len(str(item['formal'][key]).encode('ascii')) for key in ('a', 'b')]}


def _summary(rows):
    targets = [target for row in rows for target in row['targets']]
    groups = {'first_answer': [row['targets'][0] for row in rows],
              'later_answer': [t for t in targets if t['answer_offset'] > 0 and not t['is_eos']],
              'answer_eos': [t for t in targets if t['is_eos']]}
    def counts(items):
        return {'targets': len(items), 'gold_top1': sum(t['correct'] for t in items),
                'eos_top1': sum(t['argmax_token'] == EOS for t in items),
                'gold_rank_above_eos': sum(t['gold_rank'] < t['eos_rank'] for t in items),
                'gold_rank_below_eos': sum(t['gold_rank'] > t['eos_rank'] for t in items)}
    loss = math.fsum(float.fromhex(r['authoritative_loss_hex']) * r['scored_targets'] for r in rows)
    return {'examples': len(rows), 'scored_targets': len(targets),
            'teacher_forced_gold_top1': sum(t['correct'] for t in targets),
            'teacher_forced_all_targets_top1': sum(all(t['correct'] for t in row['targets']) for row in rows),
            'target_mean_loss_hex': (loss / len(targets)).hex(),
            'positions': {key: counts(value) for key, value in groups.items()},
            'by_answer_length_bytes': {str(length): {
                'examples': sum(r['answer_length_bytes'] == length for r in rows),
                'teacher_forced_all_targets_top1': sum(all(t['correct'] for t in r['targets'])
                    for r in rows if r['answer_length_bytes'] == length)}
                for length in sorted({r['answer_length_bytes'] for r in rows})},
            'forward_calls': 2 * len(rows), 'conditioning': CONDITIONING}


def measure(model, stratum, on_row=None):
    """Exactly two forwards per row, no gradient, generation, or checkpoint choice.

    Forward one measures the full causal gold-prefix sequence. Forward two uses
    the unchanged native target_loss to cross-check mask/shift and loss reduction.
    The full causal sequence is not a generated prefix. Earlier wrong decisions
    are not fed forward: later positions remain conditioned on the gold answer.
    """
    items = admitted(stratum)
    if model.config.context_length != 64 or getattr(model, 'lane', None) != 'dense':
        raise ContractError('frozen-token diagnostics require dense context-64 model')
    for parameter in model.parameters():
        if parameter.device.type != 'cpu' or parameter.dtype != torch.float32:
            raise ContractError('frozen-token diagnostic model must be CPU float32')
    encodings = [ByteTokenizer(64).encode(item['text'], answer_start=len(item['prompt'].encode('ascii')))
                 for item in items]
    before = _state(model)
    rows = []
    try:
        with torch.no_grad():
            for item, enc in zip(items, encodings):
                ids = torch.tensor([enc.input_ids], dtype=torch.int64, device='cpu')
                mask = torch.tensor([enc.attention_mask], dtype=torch.bool, device='cpu')
                logits = model(ids[:, :-1], mask[:, :-1]).logits[0]
                if (logits.shape != (len(enc.input_ids)-1, VOCABULARY_SIZE)
                        or logits.device.type != 'cpu' or logits.dtype != torch.float32
                        or not bool(torch.isfinite(logits).all())):
                    raise ContractError('invalid frozen-token diagnostic logits')
                losses = F.cross_entropy(logits, ids[0, 1:], reduction='none')
                probabilities = torch.softmax(logits, dim=-1)
                targets = []
                for position, scored in enumerate(enc.loss_mask):
                    if not scored:
                        continue
                    gold = enc.input_ids[position]
                    vector = logits[position-1]
                    targets.append({'target_position': position, 'answer_offset': len(targets),
                        'gold_token': gold, 'is_eos': gold == EOS,
                        'argmax_token': int(vector.argmax()), 'correct': int(vector.argmax()) == gold,
                        'gold_rank': _rank(vector, gold), 'eos_rank': _rank(vector, EOS),
                        'gold_probability_hex': float(probabilities[position-1, gold]).hex(),
                        'eos_probability_hex': float(probabilities[position-1, EOS]).hex(),
                        'loss_hex': float(losses[position-1]).hex()})
                loss, count = target_loss(model, [{'ids': enc.input_ids, 'mask': enc.attention_mask,
                                                   'loss': enc.loss_mask}])
                if count != len(targets):
                    raise ContractError('native target count differs from observed positions')
                row = {**_context(item), 'targets': targets, 'scored_targets': count,
                       'authoritative_loss_hex': float(loss).hex()}
                rows.append(row)
                if on_row is not None:
                    # Callback gets detached plain data, not mutable retained report.
                    import copy
                    on_row(copy.deepcopy(row))
    finally:
        if json_bytes(_state(model)) != json_bytes(before):
            raise ContractError('frozen-token observation changed model, gradients, mode or RNG')
    report = {'schema': 'trinite.frozen-token-measurements.v1', 'stratum': stratum,
              'model_identity': before['model'], 'conditioning': CONDITIONING,
              'rows': rows, 'summary': _summary(rows), **CLAIMS}
    validate_report(report, stratum, before['model'])
    return report


def validate_report(report, stratum, expected_model):
    """Recompute retained structure/counts, NOT the probabilities or model outputs."""
    items = admitted(stratum)
    require_identity(expected_model)
    exact_keys(report, {'schema','stratum','model_identity','conditioning','rows','summary'} | set(CLAIMS),
               'frozen-token measurements')
    context = {'schema': 'trinite.frozen-token-measurements.v1', 'stratum': stratum,
               'model_identity': expected_model, 'conditioning': CONDITIONING, **CLAIMS}
    if any(json_bytes(report[k]) != json_bytes(v) for k, v in context.items()):
        raise ContractError('frozen-token measurement context mismatch')
    rows = report['rows']
    if type(rows) is not list or len(rows) != len(items):
        raise ContractError('frozen-token rows incomplete or excessive')
    for item, row in zip(items, rows):
        context = _context(item)
        exact_keys(row, set(context) | {'targets','scored_targets','authoritative_loss_hex'}, 'frozen-token row')
        if any(json_bytes(row[k]) != json_bytes(v) for k, v in context.items()):
            raise ContractError('frozen-token admitted row order/context differs')
        enc = ByteTokenizer(64).encode(item['text'], answer_start=len(item['prompt'].encode('ascii')))
        positions = [i for i, flag in enumerate(enc.loss_mask) if flag]
        if type(row['targets']) is not list or len(row['targets']) != len(positions):
            raise ContractError('frozen-token answer/EOS positions incomplete')
        if type(row['scored_targets']) is not int or row['scored_targets'] != len(positions):
            raise ContractError('frozen-token target count differs')
        for offset, (position, target) in enumerate(zip(positions, row['targets'])):
            exact_keys(target, {'target_position','answer_offset','gold_token','is_eos','argmax_token',
                'correct','gold_rank','eos_rank','gold_probability_hex','eos_probability_hex','loss_hex'},
                'frozen-token target')
            token = integer(target['argmax_token'], 'argmax', 0, VOCABULARY_SIZE-1)
            gold_rank = integer(target['gold_rank'], 'gold rank', 1, VOCABULARY_SIZE)
            eos_rank = integer(target['eos_rank'], 'EOS rank', 1, VOCABULARY_SIZE)
            gold = enc.input_ids[position]
            context = {'target_position': position, 'answer_offset': offset, 'gold_token': gold,
                       'is_eos': gold == EOS, 'correct': token == gold}
            if any(json_bytes(target[k]) != json_bytes(v) for k, v in context.items()):
                raise ContractError('frozen-token target shift/label differs')
            if ((gold_rank == 1) != (token == gold) or (eos_rank == 1) != (token == EOS)
                    or (gold_rank == eos_rank) != (gold == EOS)):
                raise ContractError('frozen-token rank/top1 inconsistency')
            probability = _hex(target['gold_probability_hex'], 'gold probability', 0, 1)
            eos_probability = _hex(target['eos_probability_hex'], 'EOS probability', 0, 1)
            loss = _hex(target['loss_hex'], 'loss', 0, 1e30)
            if gold == EOS and (target['gold_probability_hex'] != target['eos_probability_hex']):
                raise ContractError('frozen-token gold EOS probability differs')
            if (probability > eos_probability and gold_rank >= eos_rank
                    or probability < eos_probability and gold_rank <= eos_rank):
                raise ContractError('frozen-token probability ordering contradicts rank')
            # Rounded float32 probabilities may tie (including underflow) when
            # logits do not; do not infer rank ties from softmax equality.
            if not math.isclose(probability, math.exp(-loss), rel_tol=2e-5, abs_tol=1e-7):
                raise ContractError('frozen-token probability/loss inconsistency')
        loss = _hex(row['authoritative_loss_hex'], 'native target loss', 0, 1e30)
        mean = math.fsum(float.fromhex(t['loss_hex']) for t in row['targets']) / len(positions)
        if not math.isclose(loss, mean, rel_tol=1e-5, abs_tol=1e-6):
            raise ContractError('frozen-token native loss/position mean differs')
    summary = _summary(rows)
    if json_bytes(report['summary']) != json_bytes(summary):
        raise ContractError('frozen-token retained summary differs')
    return summary
