"""Descriptive musical metrics over complete retained predictions, no inference.

Native whole-answer+EOS scoring remains authoritative. Components distinguish
count/timing structure from meter/group readout and never repair generations.
"""
from collections import Counter, defaultdict
from trinite.contracts import ContractError, exact_keys
from trinite.tokenizer import EOS
from .codec import TASKS, component_counts
from .numerical import examples, GENERATION_CAP
from .protocol import PROTOCOL

READOUT = ('bar_boundaries', 'group_onsets', 'duration')


def trivial_baselines(rows):
    """Modes learned only from unique TRAIN rows; tie-break is lexical, frozen."""
    counts = defaultdict(Counter)
    for row in rows:
        if row['split'] == 'train':
            counts[row['task'], None][row['answer']] += 1
            counts[row['task'], row['meter']][row['answer']] += 1
    return {key: min(values, key=lambda answer: (-values[answer], answer))
            for key, values in counts.items()}


def _text(tokens):
    # Only a terminal EOS is removed. Internal EOS/BOS/PAD are not repaired.
    body = tokens[:-1] if tokens and tokens[-1] == EOS else tokens
    if any(type(t) is not int or not 0 <= t <= 127 for t in body):
        return ''
    return bytes(body).decode('ascii')


def annotate(selector, record):
    exact_keys(record, {'selector', 'model_identity', 'summary', 'predictions'}, 'music record')
    if record['selector'] != selector:
        raise ContractError('music descriptive split mismatch')
    all_rows = examples()
    selected = {r['example_identity']: r for r in all_rows if r['split'] == selector}
    baselines = trivial_baselines(all_rows)
    predictions = record['predictions']
    if type(predictions) is not list or len(predictions) != len(selected):
        raise ContractError('incomplete music descriptive predictions')
    result, seen = [], set()
    for prediction in predictions:
        exact_keys(prediction, {'example_identity', 'family_id', 'task', 'split',
            'expected_tokens', 'generated_tokens', 'correct', 'baseline_tokens',
            'baseline_correct'}, 'music prediction')
        key = prediction['example_identity']
        if type(key) is not str or key not in selected or key in seen:
            raise ContractError('unknown or duplicate music prediction')
        seen.add(key)
        item = selected[key]
        expected = [*item['answer'].encode('ascii'), EOS]
        emitted = prediction['generated_tokens']
        if (type(emitted) is not list or not 1 <= len(emitted) <= GENERATION_CAP
                or any(type(t) is not int or not 0 <= t <= 258 for t in emitted)):
            raise ContractError('invalid musical prediction tokens')
        constant = [*baselines[item['task'], None].encode('ascii'), EOS]
        correct = emitted == expected
        native_split = 'train' if selector == 'train' else 'test'
        expected_fields = {'family_id': item['family_id'], 'task': item['task'],
            'split': native_split, 'expected_tokens': expected, 'correct': correct,
            'baseline_tokens': constant, 'baseline_correct': constant == expected}
        for name, value in expected_fields.items():
            if type(prediction[name]) is not type(value) or prediction[name] != value:
                raise ContractError('music retained prediction context differs')
        meter_constant = [*baselines[item['task'], item['meter']].encode('ascii'), EOS]
        text = _text(emitted)
        result.append({'example_identity': key, 'family_id': item['family_id'],
            'task': item['task'], 'meter': item['meter'], 'original_split': selector,
            'skill_group': 'signature_group_readout' if item['task'] in READOUT else 'event_structure',
            'prediction': prediction, 'correct': correct,
            'components': component_counts(item['task'], text, item['answer']),
            'task_mode_correct': constant == expected,
            'task_meter_mode_tokens': meter_constant,
            'task_meter_mode_correct': meter_constant == expected,
            'terminal_eos': emitted[-1] == EOS,
            'status': 'correct' if correct else 'missing_eos' if emitted[-1] != EOS else 'wrong_answer'})
    return result


def counts(rows):
    n = len(rows)
    correct = sum(r['correct'] for r in rows)
    components = sum(r['components']['total'] for r in rows)
    component_correct = sum(r['components']['correct'] for r in rows)
    return {'examples': n, 'correct': correct,
        'whole_answer_eos_accuracy': None if not n else [correct, n],
        'component_correct': component_correct, 'components': components,
        'component_accuracy': None if not components else [component_correct, components],
        'task_mode_correct': sum(r['task_mode_correct'] for r in rows),
        'task_meter_mode_correct': sum(r['task_meter_mode_correct'] for r in rows),
        'terminal_eos': sum(r['terminal_eos'] for r in rows),
        'status': 'observed' if n else 'untested_no_cases'}


def group(rows):
    return {'total': counts(rows), **{
        'by_' + field: {value: counts([r for r in rows if r[field] == value])
                        for value in sorted({r[field] for r in rows})}
        for field in ('task', 'meter', 'family_id', 'skill_group')}}


def criterion_flags(rows, split):
    """Frozen descriptive checks, not powered success or downstream gate clearance."""
    criteria = PROTOCOL['criteria']
    fraction = criteria['train_each_task_fraction' if split == 'train' else 'heldout_each_task_fraction']
    margin = criteria['heldout_each_task_majority_margin']
    values = {}
    for task in TASKS:
        selected = [r for r in rows if r['task'] == task]
        n = len(selected)
        correct = sum(r['correct'] for r in selected)
        baseline = sum(r['task_mode_correct'] for r in selected)
        values[task] = {'accuracy_met': bool(n) and correct * fraction[1] >= n * fraction[0],
            'train_mode_margin_met': None if split == 'train' else
                bool(n) and (correct - baseline) * margin[1] >= n * margin[0]}
    return {'per_task': values, 'selection_permitted': False,
            'scope': 'exploratory fixed criteria; all task rows and failures retained'}


def describe(records):
    exact_keys(records, {'train', 'validation', 'test'}, 'music split records')
    if len({r['model_identity'] for r in records.values()}) != 1:
        raise ContractError('music split models differ')
    rows = {split: annotate(split, records[split]) for split in ('train', 'validation', 'test')}
    criteria = {split: criterion_flags(values, split) for split, values in rows.items()}
    train_fit_met = all(value['accuracy_met'] for value in criteria['train']['per_task'].values())
    return {'schema': 'trinite.music-learning-description.v1',
        'train_fit': group(rows['train']),
        'held_out_validation': group(rows['validation']),
        'held_out_test': group(rows['test']), 'rows': rows,
        'criteria': criteria,
        'train_fit_prerequisite_met': train_fit_met,
        'heldout_interpretation': 'exploratory_only' if train_fit_met else 'training_fit_insufficient',
        'scope': 'eight correlated public parent families; exploratory, not powered or fresh confirmation',
        'baseline_scope': 'lexical tie-broken modes from unique training rows only; task and task/meter',
        'readout_scope': 'bar boundaries, group onsets and duration are directly recoverable from signature/groups/bar count',
        'additional_inference_executed': False, 'learning_adequate': False}
