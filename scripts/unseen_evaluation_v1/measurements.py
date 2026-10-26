"""Historical family-held-out measurements; never a training or tuning path."""
from collections import Counter, defaultdict
import math

import torch
from torch.nn import functional as F

from trinite.comparison import greedy
from trinite.contracts import ContractError, exact_keys, integer, json_bytes, parse_json, require_identity
from trinite.foundations_data import audit_bytes, dataset
from trinite.foundations_oracle import parse_prompt, solve
from trinite.generalization import answer_body
from trinite.tokenizer import ByteTokenizer, EOS
from trinite.training import target_loss
from native_diagnostic_v1.measurements import _state, _hex, carrier_controls


CAP = 5
CONTEXT = 64
EXPOSURE = {
    'checkpoint_exposure': 'validation/test families excluded from this checkpoint updates and training diagnostics',
    'research_exposure': 'historical validation/test families used by prior research; not globally unseen or fresh confirmation',
    'selection': 'all 64 validation and 64 test rows; final step4096 checkpoint selected before evaluation',
    'claim': 'exploratory only; no learning-adequacy, Phase5 or release eligibility',
    'dependence': 'one held-out family per split; two carriers per semantic item; rows are not independent trials',
}


def _corpus():
    raw, manifest = dataset('arithmetic')
    audit_bytes(raw, manifest)
    rows = parse_json(raw, canonical=True)['examples']
    by_split = {split: [r for r in rows if r['split'] == split]
                for split in ('train', 'validation', 'test')}
    if {k: len(v) for k, v in by_split.items()} != {'train': 264, 'validation': 64, 'test': 64}:
        raise ContractError('unseen evaluation fixed corpus counts differ')
    for field in ('example_identity', 'family_id', 'semantic_identity', 'prompt'):
        groups = {split: {r[field] for r in items} for split, items in by_split.items()}
        if any(groups[a] & groups[b] for a, b in
               (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
            raise ContractError('unseen evaluation corpus leaks '+field)
    return by_split


def examples():
    groups = _corpus()
    return groups['validation'] + groups['test']


def train_examples():
    return _corpus()['train']


def _controls(item, train):
    values = Counter(r['answer'] for r in train if r['task'] == item['task'])
    majority = min(values, key=lambda answer: (-values[answer], answer))
    lookup = {r['prompt']: r['answer'] for r in train}
    found = lookup.get(item['prompt'])
    oracle = solve('arithmetic', parse_prompt('arithmetic', item['prompt']))
    if oracle != item['answer']:
        raise ContractError('held-out answer differs from independent formal oracle')
    return {'training_majority_tokens': [*majority.encode('ascii'), EOS],
            'training_lookup_tokens': None if found is None else [*found.encode('ascii'), EOS],
            'constant_eos_tokens': [EOS],
            'exact_oracle_tokens': [*oracle.encode('ascii'), EOS]}


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
            'claim': 'historical held-out descriptive pairs; not independent trials or proof of reasoning'}


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
            'carrier': item['carrier'], 'split': item['split'], 'expected_tokens': expected,
            'generated_tokens': emitted, 'answer_correct': body == item['answer'],
            'answer_and_eos_correct': emitted == expected, 'malformed_reason': malformed,
            'stop_reason': reason, 'premature_eos': has_eos and len(emitted) < len(expected),
            'output_tokens': len(emitted), 'context_length': context_length}


def _finite_raw_logits(module, args, output):
    if not bool(torch.isfinite(output.logits).all()):
        raise ContractError('non-finite raw generation logits')


def measure(model, on_row=None):
    """Score every fixed historical held-out row without changing model or RNG.

    Two teacher-forced forwards use the gold answer prefix (the second is the
    authoritative target_loss cross-check). Raw greedy generation sees only the
    prompt, once, at the fixed grammar-wide cap. No repair, retries or selection.
    An optional progress observer receives copies after each completed row;
    exceptions terminate the attempt without hiding the completed prefix.
    """
    if on_row is not None and not callable(on_row):
        raise ContractError('row observer must be callable')
    if type(model.config.context_length) is not int or model.config.context_length != CONTEXT:
        raise ContractError('unseen evaluation requires context64')
    if model.lane != 'dense':
        raise ContractError('unseen evaluation requires dense lane')
    selected = examples()
    train = train_examples()
    cap = CAP
    workload = 'arithmetic'
    tokenizer = ByteTokenizer(CONTEXT)
    # Validate all encodings before any model invocation.
    encoded = [tokenizer.encode(r['text'], answer_start=len(r['prompt'].encode('ascii')))
               for r in selected]
    controls = [_controls(item, train) for item in selected]
    before = _state(model)
    rows, targets = [], []
    try:
        with torch.no_grad():
            for item, enc, control in zip(selected, encoded, controls):
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
                # Observe the authoritative greedy forwards, without an extra
                # inference or an alternate generation implementation.
                with model.register_forward_hook(_finite_raw_logits):
                    emitted = greedy(model, item['prompt'], cap)
                row = _generation_row(item, emitted, cap, model.config.context_length)
                row['authoritative_loss_hex'] = float(authoritative).hex()
                row['scored_targets'] = count
                row['first_answer'] = item_targets[0]
                row['non_neural_controls'] = control
                rows.append(row)
                targets.extend(item_targets)
                if on_row is not None:
                    # Copies isolate retained progress observers from result data.
                    on_row(parse_json(json_bytes(row), canonical=True),
                           parse_json(json_bytes(item_targets), canonical=True))
    finally:
        if json_bytes(_state(model)) != json_bytes(before):
            raise ContractError('diagnostic observation changed model, gradients, mode or RNG')
    return {'schema': 'trinite.unseen-evaluation-measurements.v1', 'workload': workload,
            'split': 'historical-validation-and-test', 'model_identity': before['model'], 'generation_cap': cap,
            'rows': rows, 'targets': targets, 'summary': _summary(rows, targets),
            'by_split': _by_split(rows, targets), 'exposure': dict(EXPOSURE),
            'input_sensitivity': sensitivity(rows), 'prompt_carrier_controls': carrier_controls(rows),
            'non_neural_control_scope': CONTROL_SCOPE,
            'quality_claim': QUALITY_CLAIM, 'malformed_scope': MALFORMED_SCOPE}


CONTROL_SCOPE = 'training-derived majority and exact-prompt lookup (null on miss), constant EOS, independently evaluated formal oracle; not model predictions'
QUALITY_CLAIM = 'exploratory historical held-out evaluation; not fresh confirmation or a learning/release gate'
MALFORMED_SCOPE = 'special-token, ASCII decoding and EOS termination only; not arithmetic grammar validity'


def _summary(rows, targets):
    first = [r['first_answer'] for r in rows]
    hist = Counter(r['output_tokens'] for r in rows)
    total_authoritative_loss = [float.fromhex(r['authoritative_loss_hex'])*r['scored_targets'] for r in rows]
    constant_counts = Counter(tuple(r['expected_tokens']) for r in rows)
    best_constant = min(constant_counts, key=lambda tokens: (-constant_counts[tokens], tokens))
    return {'examples': len(rows), 'scored_targets': len(targets),
                'families': len({r['family_id'] for r in rows}),
                'semantic_items': len({r['semantic_identity'] for r in rows}),
                'distinct_generated_outputs': len({tuple(r['generated_tokens']) for r in rows}),
                'best_constant_ceiling': {'tokens': list(best_constant),
                    'correct': constant_counts[best_constant],
                    'scope': 'ex-post gold-label constant ceiling, not a trained baseline or model selection'},
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
                'training_lookup_correct': sum(r['non_neural_controls']['training_lookup_tokens'] == r['expected_tokens'] for r in rows),
                'training_lookup_misses': sum(r['non_neural_controls']['training_lookup_tokens'] is None for r in rows),
                'constant_eos_correct': sum(r['non_neural_controls']['constant_eos_tokens'] == r['expected_tokens'] for r in rows),
                'exact_oracle_correct': sum(r['non_neural_controls']['exact_oracle_tokens'] == r['expected_tokens'] for r in rows),
                'forward_calls': 2*len(rows)+sum(r['output_tokens'] for r in rows)}


def _by_split(rows, targets):
    def score(group):
        identities = {r['example_identity'] for r in group}
        return _summary(group, [t for t in targets if t['example_identity'] in identities])
    result = {}
    for split in ('validation', 'test'):
        group = [r for r in rows if r['split'] == split]
        result[split] = {'summary': score(group),
            'by_task': {task: score([r for r in group if r['task'] == task])
                        for task in sorted({r['task'] for r in group})},
            'by_family': {family: score([r for r in group if r['family_id'] == family])
                          for family in sorted({r['family_id'] for r in group})}}
    return result


def validate_report(report):
    """Check structure, admitted context and recomputable counts, without inference.

    This is not a proof that retained logits/probabilities were produced by a
    model. Custody and external artifact commitments remain the caller's job.
    """
    cap = CAP
    selected = examples()
    train = train_examples()
    expected_keys = {'schema','workload','split','model_identity','generation_cap',
        'rows','targets','summary','by_split','exposure','input_sensitivity','prompt_carrier_controls','non_neural_control_scope','quality_claim','malformed_scope'}
    exact_keys(report, expected_keys, 'diagnostic measurements')
    context = {'schema':'trinite.unseen-evaluation-measurements.v1','workload':'arithmetic',
        'split':'historical-validation-and-test','generation_cap':cap,'exposure':EXPOSURE,'non_neural_control_scope':CONTROL_SCOPE,
        'quality_claim':QUALITY_CLAIM, 'malformed_scope':MALFORMED_SCOPE}
    if any(json_bytes(report[k]) != json_bytes(v) for k,v in context.items()):
        raise ContractError('diagnostic report context mismatch')
    require_identity(report['model_identity'])
    rows, targets = report['rows'], report['targets']
    if type(rows) is not list or len(rows) != len(selected) or type(targets) is not list:
        raise ContractError('incomplete diagnostic rows')
    offset = 0
    for item, row in zip(selected, rows):
        exact_keys(row, {'example_identity','family_id','task','semantic_identity','carrier','split','expected_tokens',
            'generated_tokens','answer_correct','answer_and_eos_correct','malformed_reason',
            'stop_reason','premature_eos','output_tokens','context_length','authoritative_loss_hex',
            'scored_targets','first_answer','non_neural_controls'}, 'diagnostic row')
        context_length = integer(row['context_length'],'evaluation context',CONTEXT,CONTEXT)
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
        controls = _controls(item, train)
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
    if json_bytes(report['by_split']) != json_bytes(_by_split(rows, targets)):
        raise ContractError('evaluation split/task/family summary mismatch')
    if json_bytes(report['input_sensitivity']) != json_bytes(sensitivity(rows)):
        raise ContractError('diagnostic sensitivity mismatch')
    if json_bytes(report['prompt_carrier_controls']) != json_bytes(carrier_controls(rows)):
        raise ContractError('diagnostic prompt carrier control mismatch')
    return report['summary']
