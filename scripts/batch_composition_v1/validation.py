"""Strict recorded-report semantics, recomputed from raw vectors and frozen inputs."""
from trinite.contracts import ContractError, exact_keys, identity, json_bytes, require_identity
from .spec import CELLS, LAYOUTS, workload, references, reference_report, schedule
from .comparison import comparisons, decision


def same(actual, expected, label):
    if json_bytes(actual) != json_bytes(expected):
        raise ContractError(label + ' mismatch')


def calls_for(rows, records, layout, lane):
    calls = []
    for group in schedule(rows, layout):
        carrier = [[256, *rows[i]['prompt'].encode()] for i in group]
        for position in range(2):
            active = list(range(len(group))) if lane == 'forced' else [j for j, i in enumerate(group) if len(records[i]['steps']) > position]
            if not active:
                break
            for j in active:
                carrier[j] = records[group[j]]['steps'][position]['prefix_ids']
            length = max(map(len, carrier)) if layout.endswith('_min') else 32
            pattern = (0, 257, 255) if layout == 'payload8_32' else (258,)
            ids, mask = [], []
            for prefix in carrier:
                extra = length-len(prefix)
                ids.append(prefix + [pattern[k % len(pattern)] for k in range(extra)])
                mask.append([True]*len(prefix) + [False]*extra)
            calls.append({'row_indices': group, 'position': position, 'shape': [len(group), length],
                          'live_lengths': [len(p) for p in carrier], 'active_local_rows': active,
                          'input_identity': identity(json_bytes({'ids': ids, 'mask': mask}))})
    return calls


def validate_layout(layout, name, rows, refs):
    exact_keys(layout, {'forced', 'generated', 'calls'}, 'layout')
    exact_keys(layout['generated'], {'greedy', 'syntax'}, 'generation modes')
    exact_keys(layout['calls'], {'forced', 'greedy', 'syntax'}, 'call lanes')
    if type(layout['forced']) is not list or len(layout['forced']) != len(rows):
        raise ContractError('forced row inventory mismatch')
    for item, row, tokens in zip(layout['forced'], rows, refs):
        exact_keys(item, {'item_id', 'steps'}, 'forced row')
        same(item['item_id'], row['item_id'], 'forced item identity')
        if type(item['steps']) is not list or len(item['steps']) != 2:
            raise ContractError('forced step count mismatch')
        for position, step in enumerate(item['steps']):
            exact_keys(step, {'prefix_ids', 'logits', 'reference_next_token', 'raw_decision', 'syntax_decision'}, 'forced step')
            same(step['prefix_ids'], [256, *row['prompt'].encode(), *tokens[:position]], 'fixed prefix')
            same(step['reference_next_token'], tokens[position], 'reference continuation')
            same(step['raw_decision'], decision(step['logits'], 'greedy', position), 'raw decision')
            same(step['syntax_decision'], decision(step['logits'], 'syntax', position), 'syntax decision')
    for mode in ('greedy', 'syntax'):
        items = layout['generated'][mode]
        if type(items) is not list or len(items) != len(rows):
            raise ContractError('generated row inventory mismatch')
        for item, row in zip(items, rows):
            exact_keys(item, {'item_id', 'tokens', 'stop_reason', 'steps'}, 'generated row')
            same(item['item_id'], row['item_id'], 'generated item identity')
            if type(item['steps']) is not list or not 1 <= len(item['steps']) <= 2:
                raise ContractError('generated step count mismatch')
            prefix = [256, *row['prompt'].encode()]
            emitted = []
            stopped = None
            for position, step in enumerate(item['steps']):
                if stopped is not None:
                    raise ContractError('extra generated step after stop')
                exact_keys(step, {'prefix_ids', 'logits', 'decision'}, 'generated step')
                same(step['prefix_ids'], prefix, 'independently evolving prefix')
                expected = decision(step['logits'], mode, position)
                same(step['decision'], expected, 'generated decision')
                token = expected['token']
                emitted.append(token)
                if token == 257:
                    stopped = 'eos'
                elif position == 1:
                    stopped = 'token_cap'
                elif len(prefix) == 32:
                    stopped = 'context_cap'
                else:
                    prefix = [*prefix, token]
            if stopped is None:
                raise ContractError('generated trajectory stops prematurely')
            same(item['tokens'], emitted, 'emitted tokens')
            same(item['stop_reason'], stopped, 'stop reason')
    counts = {'physical_forward_calls': 0, 'row_evaluations': 0, 'active_decisions': 0}
    for lane in ('forced', 'greedy', 'syntax'):
        records = layout['forced'] if lane == 'forced' else layout['generated'][lane]
        expected = calls_for(rows, records, name, lane)
        same(layout['calls'][lane], expected, 'actual call metadata')
        counts['physical_forward_calls'] += len(expected)
        counts['row_evaluations'] += sum(len(c['row_indices']) for c in expected)
        counts['active_decisions'] += sum(len(c['active_local_rows']) for c in expected)
    return counts


def validate_report(value, binding):
    exact_keys(value, {'schema', 'request', 'cells', 'conforms', 'budget', 'training_updates',
        'test_evaluated', 'sampling_claim', 'speed_claim', 'scope'}, 'report')
    same(value['schema'], 'trinite.batch-composition-report.v1', 'report schema')
    same(value['request'], binding, 'report request')
    rows = workload()
    same(binding['workload'], rows, 'bound workload')
    same(binding['schedules'], {name: schedule(rows, name) for name in LAYOUTS}, 'bound schedules')
    refs = references(rows)
    old = reference_report()
    if type(value['cells']) is not list or len(value['cells']) != 4:
        raise ContractError('four exact endpoint cells required')
    total = {'physical_forward_calls': 0, 'row_evaluations': 0, 'active_decisions': 0}
    for index, (cell, key, historical) in enumerate(zip(value['cells'], CELLS, old['cells'])):
        exact_keys(cell, {'seed', 'arm', 'checkpoint_identity', 'model_identity', 'layouts', 'comparisons',
            'historical_singleton_parity', 'numeric_pass', 'decision_pass', 'conforms', 'budget', 'model_and_rng_unchanged'}, 'endpoint cell')
        same([cell['seed'], cell['arm']], list(key), 'endpoint order')
        same(cell['checkpoint_identity'], binding['checkpoint_identities'][index], 'checkpoint identity')
        require_identity(cell['model_identity'])
        same(cell['model_identity'], historical['model_identity'], 'historical frozen weight identity')
        exact_keys(cell['layouts'], set(LAYOUTS), 'layout inventory')
        budget = {k: 0 for k in total}
        for name in LAYOUTS:
            count = validate_layout(cell['layouts'][name], name, rows, refs[key])
            for k in total:
                budget[k] += count[k]
                total[k] += count[k]
        same(cell['budget'], budget, 'endpoint compute accounting')
        expected_comparisons = comparisons(cell['layouts'], [r['item_id'] for r in rows])
        same(cell['comparisons'], expected_comparisons, 'recomputed comparisons')
        numeric = all(c['numeric_pass'] for c in expected_comparisons)
        decisions = all(c['decision_pass'] for c in expected_comparisons)
        parity = {}
        for mode in ('greedy', 'syntax'):
            actual = cell['layouts']['singleton_min']['generated'][mode]
            expected = historical['modes'][mode]['predictions']
            parity[mode] = len(actual) == len(expected) and all(r['item_id'] == e['example_identity']
                and r['tokens'] == e['generated_tokens'] and r['stop_reason'] == e['stop_reason'] for r, e in zip(actual, expected))
        same(cell['historical_singleton_parity'], parity, 'historical parity')
        same(cell['numeric_pass'], numeric, 'numeric gate')
        same(cell['decision_pass'], decisions, 'decision gate')
        same(cell['conforms'], numeric and decisions and all(parity.values()), 'cell conformance')
        same(cell['model_and_rng_unchanged'], True, 'immutability flag')
    same(value['budget'], total, 'complete compute accounting')
    if total['physical_forward_calls'] > 3000 or total['row_evaluations'] > 8064 or total['active_decisions'] > total['row_evaluations']:
        raise ContractError('report exceeds frozen compute bound')
    same(value['conforms'], all(c['conforms'] for c in value['cells']), 'overall conformance')
    for key, expected in (('training_updates', 0), ('test_evaluated', False), ('sampling_claim', False), ('speed_claim', False),
        ('scope', 'frozen native CPU engineering conformance on exposed development fixtures; no learning or release gate')):
        same(value[key], expected, key)
    return {'conforms': value['conforms'], 'budget': total}
