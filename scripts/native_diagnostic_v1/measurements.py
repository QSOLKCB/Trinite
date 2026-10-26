"""Train-only descriptive measurements; no updater, repair or model selection."""
from collections import Counter, defaultdict
import math

import torch
from torch.nn import functional as F

from trinite.checkpoint import rng_snapshot
from trinite.comparison import greedy
from trinite.contracts import ContractError, exact_keys, integer, json_bytes, parse_json, require_identity
from trinite.foundations_data import audit_bytes, dataset
from trinite.foundations_oracle import parse_prompt, solve
from trinite.generalization import answer_body
from trinite.inspection import tensor_identity
from trinite.tokenizer import ByteTokenizer, EOS
from trinite.training import model_identity, target_loss


def _state(model):
    meta, tensors = rng_snapshot()
    return {
        'model': model_identity(model),
        'gradients': {name: None if p.grad is None else tensor_identity(p.grad)
                      for name, p in model.named_parameters()},
        'training': {name: module.training for name, module in model.named_modules()},
        'rng': {'metadata': meta, 'tensors': {name: tensor_identity(t)
                                             for name, t in tensors.items()}},
    }


def admitted_rows(examples, workload):
    """Require exact training rows, not a caller-labelled validation/test item."""
    if workload != 'arithmetic':
        raise ContractError('unknown diagnostic workload')
    if type(examples) is not list or not 1 <= len(examples) <= 2000:
        raise ContractError('diagnostics require a bounded nonempty training list')
    raw, manifest = dataset(workload)
    audit_bytes(raw, manifest)
    train = [r for r in parse_json(raw, canonical=True)['examples'] if r['split'] == 'train']
    admitted = {r['example_identity']: r for r in train}
    seen = set()
    for item in examples:
        if type(item) is not dict or type(item.get('example_identity')) is not str:
            raise ContractError('invalid diagnostic training row')
        key = item['example_identity']
        if key in seen or key not in admitted or json_bytes(item) != json_bytes(admitted[key]):
            raise ContractError('duplicate, foreign or non-training diagnostic row')
        seen.add(key)
    return train


def sensitivity(rows):
    """Same-task pairs with different gold answers; descriptive, not causality."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['task']].append(row)
    pairs = collisions = 0
    witnesses = []
    for task in sorted(groups):
        group = groups[task]
        for i, left in enumerate(group):
            for right in group[i+1:]:
                if left['expected_tokens'] == right['expected_tokens']:
                    continue
                pairs += 1
                same = left['generated_tokens'] == right['generated_tokens']
                collisions += int(same)
                if same and len(witnesses) < 8:
                    witnesses.append([left['example_identity'], right['example_identity']])
    return {'different_answer_pairs': pairs, 'identical_output_pairs': collisions,
            'different_output_pairs': pairs-collisions,
            'all_conflicting_pairs_identical': None if pairs == 0 else pairs == collisions,
            'collision_witnesses': witnesses,
            'claim': 'training-only descriptive pairs; not independent trials or proof of reasoning'}


def carrier_controls(rows):
    """Compare already-admitted surface carriers for the same formal input."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['semantic_identity']].append(row)
    pairs = equal = both_correct = 0
    for group in groups.values():
        for i, left in enumerate(group):
            for right in group[i+1:]:
                if left['carrier'] == right['carrier']:
                    continue
                pairs += 1
                equal += int(left['generated_tokens'] == right['generated_tokens'])
                both_correct += int(left['answer_and_eos_correct'] and right['answer_and_eos_correct'])
    return {'same_semantics_different_carrier_pairs': pairs,
            'identical_output_pairs': equal, 'both_exact_pairs': both_correct,
            'claim': 'admitted prompt-surface controls, not new data or independent trials'}


def _generation_row(item, emitted, cap, context_length):
    expected = [*item['answer'].encode('ascii'), EOS]
    _, malformed = answer_body(emitted)
    answer_tokens = emitted[:-1] if emitted and emitted[-1] == EOS else emitted
    try:
        body = bytes(answer_tokens).decode('ascii')
    except (ValueError, UnicodeDecodeError):
        body = None
    has_eos = bool(emitted and emitted[-1] == EOS)
    if has_eos:
        reason = 'eos'
    elif len(emitted) >= cap:
        reason = 'token-cap'
    else:
        reason = 'context-limit'
    return {'example_identity': item['example_identity'], 'family_id': item['family_id'],
            'task': item['task'], 'semantic_identity': item['semantic_identity'],
            'carrier': item['carrier'], 'split': 'train', 'expected_tokens': expected,
            'generated_tokens': emitted, 'answer_correct': body == item['answer'],
            'answer_and_eos_correct': emitted == expected, 'malformed_reason': malformed,
            'stop_reason': reason, 'premature_eos': has_eos and len(emitted) < len(expected),
            'output_tokens': len(emitted), 'context_length': context_length}


def measure(model, examples, cap, *, workload='arithmetic'):
    """Measure a frozen model on exact admitted TRAIN rows without changing it.

    Teacher forcing reads the gold prefix; raw generation reads only the prompt.
    Both are descriptive training diagnostics, not fresh held-out evaluation.
    Gold rank is one-based with lowest-token-ID tie breaking. Each example has
    two teacher-forced forwards (measurement plus authoritative target_loss),
    then one full-prefix forward per generated token. No caching is implied.
    """
    integer(cap, 'diagnostic generation cap', 1, 32)
    train = admitted_rows(examples, workload)
    tokenizer = ByteTokenizer(model.config.context_length)
    counts = defaultdict(Counter)
    lookup = {}
    for row in train:
        counts[row['task']][row['answer']] += 1
        if row['prompt'] in lookup and lookup[row['prompt']] != row['answer']:
            raise ContractError('training lookup has conflicting labels')
        lookup[row['prompt']] = row['answer']
    majority = {task: min(values, key=lambda answer: (-values[answer], answer))
                for task, values in counts.items()}
    # Validate all encodings before any model invocation.
    encoded = [tokenizer.encode(r['text'], answer_start=len(r['prompt'].encode('ascii')))
               for r in examples]
    before = _state(model)
    rows, targets = [], []
    try:
        with torch.no_grad():
            for item, enc in zip(examples, encoded):
                ids = torch.tensor([enc.input_ids], dtype=torch.int64, device='cpu')
                mask = torch.tensor([enc.attention_mask], dtype=torch.bool, device='cpu')
                logits = model(ids[:, :-1], mask[:, :-1]).logits[0]
                if logits.shape != (len(enc.input_ids)-1, 259) or not bool(torch.isfinite(logits).all()):
                    raise ContractError('invalid diagnostic logits')
                losses = F.cross_entropy(logits, ids[0, 1:], reduction='none')
                probabilities = torch.softmax(logits, dim=-1)
                item_targets = []
                for position, scored in enumerate(enc.loss_mask):
                    if not scored:
                        continue
                    gold = enc.input_ids[position]
                    vector = logits[position-1]
                    rank = int((vector > vector[gold]).sum()) + int((vector[:gold] == vector[gold]).sum()) + 1
                    record = {'example_identity': item['example_identity'], 'target_position': position,
                              'answer_offset': len(item_targets), 'gold_token': gold,
                              'is_eos': gold == EOS, 'argmax_token': int(vector.argmax()),
                              'correct': int(vector.argmax()) == gold, 'gold_rank': rank,
                              'gold_probability_hex': float(probabilities[position-1, gold]).hex(),
                              'eos_probability_hex': float(probabilities[position-1, EOS]).hex(),
                              'loss_hex': float(losses[position-1]).hex()}
                    item_targets.append(record)
                authoritative, count = target_loss(model, [{'ids': enc.input_ids,
                    'mask': enc.attention_mask, 'loss': enc.loss_mask}])
                if count != len(item_targets):
                    raise ContractError('diagnostic and authoritative target counts differ')
                emitted = greedy(model, item['prompt'], cap)
                row = _generation_row(item, emitted, cap, model.config.context_length)
                row['authoritative_loss_hex'] = float(authoritative).hex()
                row['scored_targets'] = count
                row['first_answer'] = item_targets[0]
                oracle = solve(workload, parse_prompt(workload, item['prompt']))
                if oracle != item['answer']:
                    raise ContractError('admitted answer differs from independent oracle')
                row['non_neural_controls'] = {
                    'training_majority_tokens': [*majority[item['task']].encode('ascii'), EOS],
                    'training_lookup_tokens': [*lookup[item['prompt']].encode('ascii'), EOS],
                    'exact_oracle_tokens': [*oracle.encode('ascii'), EOS]}
                rows.append(row)
                targets.extend(item_targets)
    finally:
        if json_bytes(_state(model)) != json_bytes(before):
            raise ContractError('diagnostic observation changed model, gradients, mode or RNG')
    return {'schema': 'trinite.native-diagnostic-measurements.v1', 'workload': workload,
            'split': 'train', 'model_identity': before['model'], 'generation_cap': cap,
            'rows': rows, 'targets': targets, 'summary': _summary(rows, targets),
            'input_sensitivity': sensitivity(rows), 'prompt_carrier_controls': carrier_controls(rows),
            'non_neural_control_scope': CONTROL_SCOPE,
            'quality_claim': QUALITY_CLAIM}


CONTROL_SCOPE = 'training-derived majority and lookup; independently evaluated exact formal oracle; not model predictions'
QUALITY_CLAIM = 'descriptive training-only diagnostics; no held-out adequacy or release gate'


def _summary(rows, targets):
    first = [r['first_answer'] for r in rows]
    hist = Counter(r['output_tokens'] for r in rows)
    total_authoritative_loss = [float.fromhex(r['authoritative_loss_hex'])*r['scored_targets'] for r in rows]
    return {'examples': len(rows), 'scored_targets': len(targets),
                'teacher_forced_correct': sum(r['correct'] for r in targets),
                'teacher_forced_eos_correct': sum(r['correct'] for r in targets if r['is_eos']),
                'first_answer_correct': sum(r['correct'] for r in first),
                'first_answer_eos_predictions': sum(r['argmax_token'] == EOS for r in first),
                'target_mean_loss_hex': (math.fsum(total_authoritative_loss)/len(targets)).hex(),
                'raw_answer_correct': sum(r['answer_correct'] for r in rows),
                'raw_answer_and_eos_correct': sum(r['answer_and_eos_correct'] for r in rows),
                'raw_malformed': sum(r['malformed_reason'] is not None for r in rows),
                'raw_unfinished': sum(r['stop_reason'] != 'eos' for r in rows),
                'raw_premature_eos': sum(r['premature_eos'] for r in rows),
                'output_length_histogram': {str(k): hist[k] for k in sorted(hist)},
                'training_majority_correct': sum(r['non_neural_controls']['training_majority_tokens'] == r['expected_tokens'] for r in rows),
                'training_lookup_correct': len(rows), 'exact_oracle_correct': len(rows),
                'forward_calls': 2*len(rows)+sum(r['output_tokens'] for r in rows)}


def _hex(value, label, low, high):
    if type(value) is not str:
        raise ContractError('diagnostic '+label+' requires hex string')
    try:
        number = float.fromhex(value)
    except (ValueError, OverflowError) as error:
        raise ContractError('invalid diagnostic '+label) from error
    if not math.isfinite(number) or number.hex() != value or not low <= number <= high:
        raise ContractError('diagnostic '+label+' outside canonical finite bounds')
    return number


def validate_report(report, examples, cap):
    """Check structure, admitted context and recomputable counts, without inference.

    This is not a proof that retained logits/probabilities were produced by a
    model. Custody and external artifact commitments remain the caller's job.
    """
    integer(cap, 'diagnostic generation cap', 1, 32)
    train = admitted_rows(examples, 'arithmetic')
    expected_keys = {'schema','workload','split','model_identity','generation_cap',
        'rows','targets','summary','input_sensitivity','prompt_carrier_controls','non_neural_control_scope','quality_claim'}
    exact_keys(report, expected_keys, 'diagnostic measurements')
    context = {'schema':'trinite.native-diagnostic-measurements.v1','workload':'arithmetic',
        'split':'train','generation_cap':cap,'non_neural_control_scope':CONTROL_SCOPE,
        'quality_claim':QUALITY_CLAIM}
    if any(json_bytes(report[k]) != json_bytes(v) for k,v in context.items()):
        raise ContractError('diagnostic report context mismatch')
    require_identity(report['model_identity'])
    rows, targets = report['rows'], report['targets']
    if type(rows) is not list or len(rows) != len(examples) or type(targets) is not list:
        raise ContractError('incomplete diagnostic rows')
    counts = defaultdict(Counter)
    for row in train:
        counts[row['task']][row['answer']] += 1
    majority = {task:min(values,key=lambda answer:(-values[answer],answer)) for task,values in counts.items()}
    offset = 0
    for item, row in zip(examples, rows):
        exact_keys(row, {'example_identity','family_id','task','semantic_identity','carrier','split','expected_tokens',
            'generated_tokens','answer_correct','answer_and_eos_correct','malformed_reason',
            'stop_reason','premature_eos','output_tokens','context_length','authoritative_loss_hex',
            'scored_targets','first_answer','non_neural_controls'}, 'diagnostic row')
        context_length = integer(row['context_length'],'diagnostic context',2,256)
        enc = ByteTokenizer(context_length).encode(item['text'],answer_start=len(item['prompt'].encode('ascii')))
        tokens = row['generated_tokens']
        if type(tokens) is not list or not 1 <= len(tokens) <= cap:
            raise ContractError('invalid diagnostic generated count')
        for token in tokens:
            integer(token,'diagnostic generated token',0,258)
        if EOS in tokens[:-1] or len(tokens) > context_length-len(item['prompt'].encode('ascii')):
            raise ContractError('diagnostic generation exceeds stop/context boundary')
        if tokens[-1] != EOS and len(tokens) != min(cap, context_length-len(item['prompt'].encode('ascii'))):
            raise ContractError('diagnostic unfinished output ended before its bound')
        expected = _generation_row(item,tokens,cap,context_length)
        if any(json_bytes(row[k]) != json_bytes(v) for k,v in expected.items()):
            raise ContractError('diagnostic generation fields mismatch')
        gold = [*item['answer'].encode('ascii'),EOS]
        controls = {'training_majority_tokens':[*majority[item['task']].encode('ascii'),EOS],
                    'training_lookup_tokens':gold,'exact_oracle_tokens':[
                        *solve('arithmetic',parse_prompt('arithmetic',item['prompt'])).encode('ascii'),EOS]}
        if json_bytes(row['non_neural_controls']) != json_bytes(controls):
            raise ContractError('diagnostic control mismatch')
        integer(row['scored_targets'],'diagnostic target count',1,256)
        positions = [i for i, flag in enumerate(enc.loss_mask) if flag]
        if row['scored_targets'] != len(positions):
            raise ContractError('diagnostic target count mismatch')
        item_targets = targets[offset:offset+len(positions)]
        if len(item_targets) != len(positions):
            raise ContractError('missing diagnostic targets')
        for i,(position,target) in enumerate(zip(positions,item_targets)):
            exact_keys(target, {'example_identity','target_position','answer_offset','gold_token',
                'is_eos','argmax_token','correct','gold_rank','gold_probability_hex',
                'eos_probability_hex','loss_hex'},'diagnostic target')
            token = integer(target['argmax_token'],'diagnostic argmax',0,258)
            rank = integer(target['gold_rank'],'diagnostic rank',1,259)
            g = enc.input_ids[position]
            fields = {'example_identity':item['example_identity'],'target_position':position,
                'answer_offset':i,'gold_token':g,'is_eos':g==EOS,'correct':token==g}
            if any(json_bytes(target[k]) != json_bytes(v) for k,v in fields.items()) or (rank==1) != (token==g):
                raise ContractError('diagnostic teacher target mismatch')
            probability = _hex(target['gold_probability_hex'],'probability',0,1)
            _hex(target['eos_probability_hex'],'EOS probability',0,1)
            if g==EOS and target['gold_probability_hex'] != target['eos_probability_hex']:
                raise ContractError('diagnostic EOS probability mismatch')
            loss = _hex(target['loss_hex'],'loss',0,1e30)
            # Fixed float32 consistency allowance, not an inference parity gate.
            if not math.isclose(probability, math.exp(-loss), rel_tol=2e-5, abs_tol=1e-7):
                raise ContractError('diagnostic probability/loss mismatch')
        if json_bytes(row['first_answer']) != json_bytes(item_targets[0]):
            raise ContractError('diagnostic first-answer mismatch')
        authoritative = _hex(row['authoritative_loss_hex'],'authoritative loss',0,1e30)
        mean = math.fsum(float.fromhex(t['loss_hex']) for t in item_targets)/len(item_targets)
        if not math.isclose(authoritative, mean, rel_tol=1e-5, abs_tol=1e-6):
            raise ContractError('diagnostic authoritative/target mean mismatch')
        offset += len(positions)
    if offset != len(targets):
        raise ContractError('extra diagnostic targets')
    if json_bytes(report['summary']) != json_bytes(_summary(rows,targets)):
        raise ContractError('diagnostic summary mismatch')
    if json_bytes(report['input_sensitivity']) != json_bytes(sensitivity(rows)):
        raise ContractError('diagnostic sensitivity mismatch')
    if json_bytes(report['prompt_carrier_controls']) != json_bytes(carrier_controls(rows)):
        raise ContractError('diagnostic prompt carrier control mismatch')
    return report['summary']
