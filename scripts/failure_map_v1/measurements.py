"""Exhaustive admitted-domain raw observations, not a generalisation certificate.

The original admission, scorer, baseline rule and greedy decoder are authoritative.
Only selection views and descriptive accounting are new. No answers enter forwards.
"""
import copy
import itertools
import re
from collections import Counter

import torch

from trinite.contracts import ContractError, exact_keys, json_bytes, parse_json, require_identity
from trinite.foundations import score_examples, check_prediction_examples
from trinite.tokenizer import EOS, VOCABULARY_SIZE
from native_diagnostic_v1.measurements import _state
from addition_development_v1.data import dataset, audit_bytes, examples, parse_prompt, CARRIERS
from addition_development_v1.numerical import check_score
from addition_development_v1.protocol import STRATA
from addition_development_v1.request import cell_name
from .protocol import PROTOCOL

CATEGORIES = ('exact_correct', 'wrong_value', 'immediate_eos', 'premature_eos',
              'malformed_output', 'missing_eos')
CLAIMS = {'training_updates': 0, 'fresh_confirmation': False, 'learning_adequate': False,
          'phase5_ready': False, 'release_ready': False, 'optimizer_replayed': False}
BOUNDARY_INPUTS = ('A:-1,2=', 'A:13,0=', 'A:01,2=', 'A:1;2=', 'A:1,2',
                   'A:1,2=3', 'A:１,2=', 'What is 1+2?', 'A:'+('1'*65)+',0=')


def admitted():
    raw, manifest = dataset(); audit_bytes(raw, manifest)
    return parse_json(raw, canonical=True)['examples']


def boundary_controls():
    """Parser-only checks of the harness boundary, NEVER model robustness tests."""
    rows = []
    for prompt in BOUNDARY_INPUTS:
        try:
            parse_prompt(prompt)
        except ContractError:
            rows.append({'prompt': prompt, 'outcome': 'rejected_input', 'model_executed': False})
        else:
            raise ContractError('failure-map boundary accepted an excluded input')
    return {'scope': 'parser-only harness boundary; not model instruction following', 'rows': rows}


def classify(tokens, expected):
    """Disjoint primary failure; exactness includes the EOS and canonical decimal."""
    if type(tokens) is not list or not 1 <= len(tokens) <= PROTOCOL['generation_cap']:
        raise ContractError('invalid failure-map token count')
    if any(type(t) is not int or not 0 <= t < VOCABULARY_SIZE for t in tokens):
        raise ContractError('invalid failure-map token')
    if EOS in tokens[:-1] or (EOS not in tokens and len(tokens) != PROTOCOL['generation_cap']):
        raise ContractError('impossible bounded greedy termination')
    if tokens == expected:
        return 'exact_correct'
    if tokens == [EOS]:
        return 'immediate_eos'
    if tokens[-1] != EOS:
        return 'missing_eos'
    body = tokens[:-1]
    if any(t > 255 for t in body):
        return 'malformed_output'
    text = bytes(body).decode('ascii', errors='replace')
    if re.fullmatch(r'(0|[1-9][0-9]*)', text, re.ASCII) is None:
        return 'malformed_output'
    if len(body) < len(expected)-1 and body == expected[:len(body)]:
        return 'premature_eos'
    return 'wrong_value'


def _view(item, training):
    # `test` is ONLY the unchanged scorer's selection switch. The original
    # admission identity/split are retained separately; seen inputs stay anchors.
    return training + [{**item, 'split': 'test'}]


def _original(arm, model, scores, training):
    exact_keys(scores, set(STRATA), 'failure-map original scores')
    records = {'train': parse_json(training, canonical=True),
               **{s: parse_json(scores[s], canonical=True) for s in STRATA}}
    result = {}
    for selector, record in records.items():
        check_score(record, arm, selector, model)
        for prediction in record['predictions']:
            key = prediction['example_identity']
            if key in result:
                raise ContractError('duplicate original failure-map prediction')
            result[key] = prediction['generated_tokens']
    return result


def _context(item, arm):
    a, b = item['formal']['a'], item['formal']['b']
    seen = arm in item['training_arms']
    return {'example_identity': item['example_identity'], 'family_id': item['family_id'],
            'prompt': item['prompt'], 'answer': item['answer'], 'a': a, 'b': b,
            'carrier': item['carrier'], 'original_split': item['split'],
            'scientific_exposure': 'seen_training_anchor_not_generalisation' if seen else
                ('other_arm_training_family_not_seen_in_this_arm' if item['training_arms'] else
                 'historically_exposed_development_not_fresh_confirmation'),
            'seen_in_this_arm': seen, 'answer_familiar': item['coverage']['answer_seen_in_both_training_arms'],
            'joint_range_digit_length_shift': max(a, b) >= 10,
            'order': 'ascending' if a < b else 'descending' if a > b else 'diagonal',
            'zero_operand': a == 0 or b == 0, 'zero_answer': a == b == 0}


def _row(item, prediction, arm, original):
    key = item['example_identity']; prior = original.get(key)
    match = None if prior is None else prediction['generated_tokens'] == prior
    category = classify(prediction['generated_tokens'], prediction['expected_tokens'])
    return {**_context(item, arm), 'prediction': prediction, 'category': category,
            'historical_generated_tokens': prior,
            'reproducibility': 'not_previously_scored' if prior is None else 'match' if match else 'mismatch',
            'status': 'untested' if match is False else 'pass' if prediction['correct'] else 'fail',
            'status_scope': 'same-context replay mismatch invalidates interpretation' if match is False else
                'exact finite case only; never a region guarantee'}


def _counts(rows):
    return {'rows': len(rows), 'families': len({r['family_id'] for r in rows}),
            'raw_correct': sum(r['prediction']['correct'] for r in rows),
            'pass': sum(r['status'] == 'pass' for r in rows),
            'fail': sum(r['status'] == 'fail' for r in rows),
            'untested': sum(r['status'] == 'untested' for r in rows),
            'status': ('untested' if not rows or any(r['status'] == 'untested' for r in rows)
                       else 'pass' if all(r['status'] == 'pass' for r in rows) else 'fail'),
            'categories': {k: sum(r['category'] == k for r in rows) for k in CATEGORIES},
            'counterexample_identities': [r['example_identity'] for r in rows if r['status'] == 'fail'],
            'reproducibility_mismatch_identities': [r['example_identity'] for r in rows
                                                  if r['reproducibility'] == 'mismatch']}


def _pairs(rows):
    by = {(r['a'], r['b'], r['carrier']): r for r in rows}
    groups = {'carrier': [], 'operand_order': []}
    def pair(left, right):
        lp, rp = left['prediction'], right['prediction']
        same = lp['generated_tokens'] == rp['generated_tokens']
        both = lp['correct'] and rp['correct']
        category = ('both_correct' if both else 'both_wrong_same' if same else
                    'both_wrong_different' if not lp['correct'] and not rp['correct'] else 'one_correct')
        return {'left': left['example_identity'], 'right': right['example_identity'],
                'consistent': same, 'both_correct': both, 'category': category,
                'interpretation_valid': left['reproducibility'] != 'mismatch' and right['reproducibility'] != 'mismatch'}
    for a in range(13):
        for b in range(13):
            groups['carrier'].append(pair(by[a,b,'infix'], by[a,b,'fields']))
            if a < b:
                for carrier in CARRIERS:
                    groups['operand_order'].append(pair(by[a,b,carrier], by[b,a,carrier]))
    return {name: {'pairs': values, 'pair_count': len(values),
                   'consistent': sum(p['consistent'] for p in values),
                   'both_correct': sum(p['both_correct'] for p in values),
                   'interpretation_valid_pairs': sum(p['interpretation_valid'] for p in values),
                   'categories': dict(sorted(Counter(p['category'] for p in values).items())),
                   'count_scope': 'raw observed pairs; interpretation_valid_pairs excludes replay mismatches; consistency is not correctness'}
            for name, values in groups.items()}


def _summary(rows):
    groups = []
    for seen, familiar, shifted, carrier, order in itertools.product(
            (False, True), (False, True), (False, True), CARRIERS, ('ascending','descending','diagonal')):
        criteria = dict(seen_in_this_arm=seen, answer_familiar=familiar,
                        joint_range_digit_length_shift=shifted, carrier=carrier, order=order)
        selected = [r for r in rows if all(r[k] == v for k,v in criteria.items())]
        groups.append({'criteria': criteria, **_counts(selected),
                       'untested_reason': 'no admitted case in this cross-product cell' if not selected else None})
    return {**_counts(rows), 'coverage_cells': groups, 'metamorphic': _pairs(rows),
            'reproducibility': dict(sorted(Counter(r['reproducibility'] for r in rows).items())),
            'historical_raw_predictions_compared': sum(r['reproducibility'] != 'not_previously_scored' for r in rows),
            'historical_raw_predictions_matched': sum(r['reproducibility'] == 'match' for r in rows),
            'seen_anchors': _counts([r for r in rows if r['seen_in_this_arm']]),
            'zero_answer_controls': _counts([r for r in rows if r['zero_answer']]),
            'interpretation': 'finite exhaustive admitted domain, not unbounded arithmetic; correlated pairs, no confidence interval; range and digit length are confounded'}


def _build_report(predictions, arm, seed, model_identity, original_scores, original_training):
    cell_name(arm, seed); require_identity(model_identity)
    items = admitted(); training = examples(arm)
    if type(predictions) is not list or len(predictions) != len(items):
        raise ContractError('failure-map predictions incomplete')
    original = _original(arm, model_identity, original_scores, original_training)
    rows = []
    for item, prediction in zip(items, predictions):
        check_prediction_examples(_view(item, training), 'test', PROTOCOL['generation_cap'],
                                  json_bytes({'predictions': [prediction]}))
        rows.append(_row(item, prediction, arm, original))
    return {'schema': 'trinite.failure-map-measurements.v1', 'cell': cell_name(arm, seed),
            'model_identity': model_identity, 'checkpoint_step': 4000,
            'target_inference_executed': True, 'audit_replays_inference': False,
            'scorer_selection_label': 'test is a technical selector only; see each scientific_exposure',
            'rows': rows, 'summary': _summary(rows), 'boundary_controls': boundary_controls(), **CLAIMS}


def validate_report(record, arm, seed, expected_model, original_scores, original_training):
    """Retained consistency/custody only. Does NOT independently rerun inference."""
    if type(record) is not dict or type(record.get('rows')) is not list:
        raise ContractError('failure-map report required')
    try:
        expected = _build_report([r['prediction'] for r in record['rows']], arm, seed,
                                 expected_model, original_scores, original_training)
    except (KeyError, TypeError) as error:
        raise ContractError('invalid failure-map rows') from error
    if json_bytes(record) != json_bytes(expected):
        raise ContractError('failure-map context, map, metrics or replay comparison differs')
    return expected['summary']


def measure(model, arm, seed, original_scores, original_training, progress=None):
    items = admitted(); training = examples(arm); cell_name(arm, seed)
    if model.config.context_length != 64 or getattr(model, 'lane', None) != 'dense':
        raise ContractError('failure-map requires dense context-64 model')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ContractError('failure-map requires CPU float32')
    if any(1+len(item['prompt'].encode('ascii'))+PROTOCOL['generation_cap']-1 > model.config.context_length
           for item in items):
        raise ContractError('failure-map context would truncate generation')
    before = _state(model)
    original = _original(arm, before['model'], original_scores, original_training)
    predictions = []; calls = 0
    def finite_output(_module, args, output):
        nonlocal calls
        calls += 1
        logits = output.logits
        if (calls > PROTOCOL['max_forward_calls_per_cell'] or
            logits.shape != (*args[0].shape, VOCABULARY_SIZE) or logits.dtype != torch.float32 or
            logits.device.type != 'cpu' or not bool(torch.isfinite(logits).all())):
            raise ContractError('invalid or excessive failure-map forwards')
    hook = model.register_forward_hook(finite_output)
    try:
        for item in items:
            _, values = score_examples(model, _view(item, training), 'test', PROTOCOL['generation_cap'])
            prediction = values[0]
            row = _row(item, prediction, arm, original)
            predictions.append(prediction)
            if progress is not None:
                progress(copy.deepcopy(row))
    finally:
        hook.remove()
        if json_bytes(_state(model)) != json_bytes(before):
            raise ContractError('failure-map changed model, gradients, mode or RNG')
    return _build_report(predictions, arm, seed, before['model'], original_scores, original_training)


def untested_report(arm, seed, expected_model, reason):
    cell_name(arm, seed); require_identity(expected_model)
    return {'cell': cell_name(arm, seed), 'model_identity': expected_model, 'status': 'untested',
            'reason': reason, 'rows': [{**_context(r, arm), 'status': 'untested', 'reason': reason}
                                     for r in admitted()],
            'counts': {'pass': 0, 'fail': 0, 'untested': 338},
            'scope': 'worker unavailable, not a model accuracy result', **CLAIMS}
