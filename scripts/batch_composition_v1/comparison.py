"""Finite-vector diagnostics and separate exact decision gates; no Torch dependency."""
import math
import struct
from trinite.contracts import ContractError

ATOL, RTOL = 0.000002, 0.00001
VOCABULARY = 259


def values(vector):
    if type(vector) is not list or len(vector) != VOCABULARY:
        raise ContractError('logits require exactly 259 canonical finite hex values')
    result = []
    for value in vector:
        if type(value) is not str:
            raise ContractError('logit must be canonical hex')
        try:
            parsed = float.fromhex(value)
        except (ValueError, OverflowError) as error:
            raise ContractError('invalid hex logit') from error
        if not math.isfinite(parsed) or parsed.hex() != value:
            raise ContractError('nonfinite/noncanonical logit')
        try:
            rounded = struct.unpack('!f', struct.pack('!f', parsed))[0]
        except (OverflowError, struct.error) as error:
            raise ContractError('logit is outside float32') from error
        if not math.isfinite(rounded) or rounded.hex() != value:
            raise ContractError('logit is not exact finite float32')
        result.append(parsed)
    return result


def decision(vector, mode, position):
    numbers = values(vector)
    if mode not in ('greedy', 'syntax') or type(position) is not int or position not in (0, 1):
        raise ContractError('invalid mode or decision position')
    allowed = list(range(VOCABULARY)) if mode == 'greedy' else list(range(48, 58)) if position == 0 else [257]
    ranked = sorted(allowed, key=lambda i: (-numbers[i], i))
    best = ranked[0]
    margin = numbers[best] - numbers[ranked[1]] if len(ranked) > 1 else None
    return {'token': best, 'top_two_margin_hex': None if margin is None else margin.hex(),
            'exact_tie': margin == 0.0 if margin is not None else False,
            'near_tie_diagnostic': margin <= 2*(ATOL+RTOL*max(abs(numbers[best]), abs(numbers[ranked[1]]))) if margin is not None else False}


def compare_vectors(reference, candidate):
    a, b = values(reference), values(candidate)
    differences = [abs(y-x) for x, y in zip(a, b)]
    scales = [ATOL + RTOL*abs(x) for x in a]
    violations = [i for i, (d, scale) in enumerate(zip(differences, scales)) if d > scale]
    return {'numeric_pass': not violations, 'violating_token_ids': violations,
            'max_absolute_hex': max(differences).hex(),
            'max_relative_diagnostic_hex': max(d/max(abs(x), ATOL) for d, x in zip(differences, a)).hex(),
            'max_normalized_error_hex': max(d/s for d, s in zip(differences, scales)).hex(),
            'exact_coordinate_count': sum(x == y for x, y in zip(a, b))}


def comparisons(layouts, item_ids):
    from .spec import LAYOUTS
    if type(layouts) is not dict or set(layouts) != set(LAYOUTS):
        raise ContractError('exact eight-layout inventory required')
    if type(item_ids) is not list or len(set(item_ids)) != len(item_ids):
        raise ContractError('unique expected item identities required')
    expected = set(item_ids)
    for name, layout in layouts.items():
        if (type(layout) is not dict or set(layout) != {'forced', 'generated', 'calls'}
                or type(layout['generated']) is not dict or set(layout['generated']) != {'greedy', 'syntax'}
                or type(layout['calls']) is not dict or set(layout['calls']) != {'forced', 'greedy', 'syntax'}):
            raise ContractError('exact layout lane/call inventory required')
        for rows in [layout['forced'], *layout['generated'].values()]:
            if len(rows) != len(item_ids) or [r['item_id'] for r in rows] != item_ids:
                raise ContractError('result item inventory/order mismatch')
            if set(r['item_id'] for r in rows) != expected:
                raise ContractError('result rows duplicate or missing')
    edges = [('singleton_min', n) for n in LAYOUTS if n != 'singleton_min']
    edges += [('singleton_32', 'batch8_32')]
    edges += [('batch8_32', n) for n in ('partition4_32', 'reverse8_32', 'recompose8_32', 'batch8_min', 'payload8_32')]
    results = []
    for source, target in edges:
        forced, generated = [], {}
        for a, b in zip(layouts[source]['forced'], layouts[target]['forced']):
            if len(a['steps']) != 2 or len(b['steps']) != 2:
                raise ContractError('forced trajectory requires two stages')
            for position, (x, y) in enumerate(zip(a['steps'], b['steps'])):
                if x['prefix_ids'] != y['prefix_ids']:
                    raise ContractError('forced comparisons require identical prefixes')
                forced.append({'item_id': a['item_id'], 'position': position,
                               **compare_vectors(x['logits'], y['logits']),
                               'decision_pass': all(decision(x['logits'], mode, position)['token'] == decision(y['logits'], mode, position)['token'] for mode in ('greedy', 'syntax'))})
        for mode in ('greedy', 'syntax'):
            rows = []
            for a, b in zip(layouts[source]['generated'][mode], layouts[target]['generated'][mode]):
                comparable = []
                unaligned = abs(len(a['steps']) - len(b['steps']))
                for position, (x, y) in enumerate(zip(a['steps'], b['steps'])):
                    if x['prefix_ids'] == y['prefix_ids']:
                        comparable.append({'position': position, **compare_vectors(x['logits'], y['logits'])})
                    else:
                        unaligned += 1
                rows.append({'item_id': a['item_id'], 'decision_pass': a['tokens'] == b['tokens'] and a['stop_reason'] == b['stop_reason'],
                             'same_prefix_steps': comparable, 'unaligned_steps': unaligned})
            generated[mode] = rows
        numeric_pass = all(r['numeric_pass'] for r in forced) and all(s['numeric_pass'] for rows in generated.values() for r in rows for s in r['same_prefix_steps'])
        decision_pass = all(r['decision_pass'] for r in forced) and all(r['decision_pass'] for rows in generated.values() for r in rows)
        results.append({'reference_layout': source, 'candidate_layout': target, 'forced': forced,
                        'independent_generation': generated, 'numeric_pass': numeric_pass, 'decision_pass': decision_pass})
    return results
