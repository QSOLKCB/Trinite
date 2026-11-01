"""Detached training-only measurements; native objective and decoder stay authoritative."""
from collections import Counter

import torch
from torch.nn import functional as F

from queue_learning_v1.codec import output, prompt, supervised
import queue_evaluation_v1 as gate
from trinite.contracts import ContractError
from trinite.tokenizer import BOS, EOS, PAD
from trinite.training import batch, model_identity, target_loss


def check_rows(rows):
    if not rows or len(rows) > 256:
        raise ContractError('diagnostic requires 1..256 training rows')
    for row in rows:
        if row.get('split') != 'train':
            raise ContractError('diagnostics accept training rows only')
        encoded = supervised(gate.encode(row['observation']), row['answer'])
        if any(list(row[k]) != list(v) for k, v in (
                ('ids', encoded.input_ids), ('mask', encoded.attention_mask), ('loss', encoded.loss_mask))):
            raise ContractError('diagnostic tokens/masks differ from queue codec')
        if (row['ids'][0] != BOS or row['ids'][-1] != EOS or PAD in row['ids']
                or sum(row['loss']) != len(str(row['answer'])) + 1):
            raise ContractError('diagnostic answer boundary mismatch')


def supervision(model, rows):
    """Counts and summed NLL, never averages of unequal batch means."""
    check_rows(rows)
    before = model_identity(model)
    totals = {name: {'targets': 0, 'correct': 0, 'nll': 0.0}
              for name in ('answer', 'eos', 'first_answer')}
    native_sum, native_count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), 8):
            group = rows[start:start + 8]
            ids, mask, scored = batch(group)
            logits = model(ids[:, :-1], mask[:, :-1]).logits
            targets, live = ids[:, 1:], scored[:, 1:]
            nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none').reshape_as(targets)
            answer = live & (targets < BOS)
            eos = live & (targets == EOS)
            first = torch.zeros_like(live)
            for i in range(len(group)):
                first[i, torch.where(answer[i])[0][0]] = True
            for name, selected in (('answer', answer), ('eos', eos), ('first_answer', first)):
                totals[name]['targets'] += int(selected.sum())
                totals[name]['correct'] += int(((logits.argmax(-1) == targets) & selected).sum())
                totals[name]['nll'] += float(nll[selected].double().sum())
            loss, count = target_loss(model, group)
            native_sum += float(loss) * count
            native_count += count
    if before != model_identity(model):
        raise ContractError('forward diagnostic changed model')
    for value in totals.values():
        value['nll_sum_hex'] = value.pop('nll').hex()
    totals['native'] = {'targets': native_count, 'mean_nll_hex': (native_sum / native_count).hex()}
    return totals


def generation(model, rows):
    """Raw tokens and strict v1 parsing, without EOS suppression or numeric forcing."""
    from trinite.comparison import greedy
    check_rows(rows)
    before = model_identity(model)
    records = []
    with torch.no_grad():
        for row in rows:
            observed = gate.encode(row['observation'])
            tokens = greedy(model, prompt(observed), 5)
            records.append({'id': row['identity'], 'tokens': tokens, 'parsed': output(tokens, observed),
                            'exact': tokens == [*str(row['answer']).encode('ascii'), EOS]})
    if before != model_identity(model):
        raise ContractError('generation diagnostic changed model')
    return {'planned': len(rows), 'statuses': dict(Counter(r['parsed']['status'] for r in records)),
            'eos_only': sum(r['tokens'] == [EOS] for r in records),
            'exact': sum(r['exact'] for r in records), 'records': records}
