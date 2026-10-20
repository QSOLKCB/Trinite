"""Actual frozen-endpoint comparisons. Direct APIs are internal; use the bounded CLI."""
import random
import time
import numpy as np
import torch
from trinite.contracts import ContractError, identity, json_bytes
from trinite.tokenizer import BOS, EOS, PAD
from trinite.training import model_identity
from frozen_decoding_v1.experiment import load_model, original
from .spec import LAYOUTS, CELLS, MAX_SECONDS, workload, references, reference_report, request, schedule, validate_rows
from .comparison import decision, comparisons


class Budget:
    def __init__(self):
        self.started = time.monotonic()
        self.physical_calls = self.row_evaluations = self.active_decisions = 0

    def charge(self, rows, active):
        if type(rows) is not int or type(active) is not int or not 1 <= active <= rows <= 8:
            raise ContractError('invalid row/call accounting')
        if (self.physical_calls + 1 > 3000 or self.row_evaluations + rows > 8064
                or time.monotonic() - self.started > MAX_SECONDS):
            raise ContractError('fixed conformance budget exceeded')
        self.physical_calls += 1
        self.row_evaluations += rows
        self.active_decisions += active

    def snapshot(self):
        return {'physical_forward_calls': self.physical_calls, 'row_evaluations': self.row_evaluations,
                'active_decisions': self.active_decisions}


def tensors(prefixes, layout):
    if layout not in LAYOUTS or type(prefixes) is not list or not 1 <= len(prefixes) <= 8:
        raise ContractError('invalid prefixes/layout')
    for prefix in prefixes:
        if (type(prefix) is not list or not 1 <= len(prefix) <= 32
                or any(type(t) is not int or not 0 <= t <= PAD for t in prefix)):
            raise ContractError('prefix must be bounded in-vocabulary integer IDs')
    length = max(map(len, prefixes)) if layout.endswith('_min') else 32
    pattern = (0, EOS, 255) if layout == 'payload8_32' else (PAD,)
    ids, mask = [], []
    for prefix in prefixes:
        extra = length-len(prefix)
        ids.append(prefix + [pattern[i % len(pattern)] for i in range(extra)])
        mask.append([True]*len(prefix) + [False]*extra)
    return (torch.tensor(ids, dtype=torch.int64, device='cpu'),
            torch.tensor(mask, dtype=torch.bool, device='cpu'))


def last_live(output, mask):
    if (not isinstance(mask, torch.Tensor) or mask.ndim != 2
            or not isinstance(output, torch.Tensor) or output.dtype != torch.float32 or output.device.type != 'cpu'
            or output.ndim != 3 or tuple(output.shape) != (*mask.shape, 259)
            or mask.dtype != torch.bool or mask.device.type != 'cpu'
            or not bool(mask[:, 0].all()) or bool((~mask[:, :-1] & mask[:, 1:]).any())
            or not bool(torch.isfinite(output).all())):
        raise ContractError('model output/mask is invalid or nonfinite')
    if int(torch.count_nonzero(output[~mask])):
        raise ContractError('masked output logits must be zero')
    positions = mask.sum(dim=1)-1
    selected = output[torch.arange(len(mask)), positions]
    return [[float(v).hex() for v in row] for row in selected.tolist()]


def observe(model, prefixes, layout, budget, active):
    ids, mask = tensors(prefixes, layout)
    budget.charge(len(prefixes), len(active))
    with torch.no_grad():
        vectors = last_live(model(ids, mask).logits, mask)
    return vectors, {'shape': list(ids.shape), 'live_lengths': [len(p) for p in prefixes],
                     'active_local_rows': list(active),
                     'input_identity': identity(json_bytes({'ids': ids.tolist(), 'mask': mask.tolist()}))}


def forced(model, rows, reference_tokens, layout, budget):
    validate_rows(rows)
    if len(reference_tokens) != len(rows):
        raise ContractError('reference trajectory count mismatch')
    for tokens in reference_tokens:
        if type(tokens) is not list or len(tokens) != 2 or type(tokens[0]) is not int or not 48 <= tokens[0] <= 57 or tokens[1] != EOS:
            raise ContractError('invalid fixed reference trajectory')
    records = [{'item_id': r['item_id'], 'steps': []} for r in rows]
    calls = []
    for group in schedule(rows, layout):
        for position in range(2):
            prefixes = [[BOS, *rows[i]['prompt'].encode(), *reference_tokens[i][:position]] for i in group]
            vectors, call = observe(model, prefixes, layout, budget, list(range(len(group))))
            calls.append({'row_indices': group, 'position': position, **call})
            for i, prefix, vector in zip(group, prefixes, vectors):
                records[i]['steps'].append({'prefix_ids': prefix, 'logits': vector,
                    'reference_next_token': reference_tokens[i][position],
                    'raw_decision': decision(vector, 'greedy', position),
                    'syntax_decision': decision(vector, 'syntax', position)})
    return records, calls


def generate(model, rows, layout, mode, budget):
    validate_rows(rows)
    if mode not in ('greedy', 'syntax'):
        raise ContractError('invalid independent generation mode')
    records = [{'item_id': r['item_id'], 'tokens': [], 'stop_reason': None, 'steps': []} for r in rows]
    calls = []
    for group in schedule(rows, layout):
        prefixes = [[BOS, *rows[i]['prompt'].encode()] for i in group]
        for position in range(2):
            active = [j for j, i in enumerate(group) if records[i]['stop_reason'] is None]
            if not active:
                break
            vectors, call = observe(model, prefixes, layout, budget, active)
            calls.append({'row_indices': group, 'position': position, **call})
            for local in active:
                index = group[local]
                record, vector = records[index], vectors[local]
                selected = decision(vector, mode, position)
                record['steps'].append({'prefix_ids': list(prefixes[local]), 'logits': vector, 'decision': selected})
                token = selected['token']
                record['tokens'].append(token)
                if token == EOS:
                    record['stop_reason'] = 'eos'
                elif position == 1:
                    record['stop_reason'] = 'token_cap'
                elif len(prefixes[local]) == 32:
                    record['stop_reason'] = 'context_cap'
                else:
                    prefixes[local].append(token)
            # Finished rows remain unchanged carriers; no EOS or out-of-context append.
    if any(r['stop_reason'] is None or len(r['steps']) != len(r['tokens']) for r in records):
        raise ContractError('unfinished or misaccounted independent generation')
    return records, calls


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def unchanged_rng(before):
    a, b, c = rng_state()
    old_a, old_b, old_c = before
    return a == old_a and b[0] == old_b[0] and np.array_equal(b[1], old_b[1]) and b[2:] == old_b[2:] and torch.equal(c, old_c)


def assert_immutable(model, before, rng):
    if (model_identity(model) != before or any(p.grad is not None for p in model.parameters())
            or any(m.training for m in model.modules()) or model.output_weight is not model.token_embedding
            or not unchanged_rng(rng)):
        raise ContractError('frozen model/eval/gradient/sharing/RNG state changed')


def report(on_cell=None, on_layout=None):
    budget = Budget()
    binding = request()
    rows = workload()
    refs = references(rows)
    old_reference = reference_report()
    cells = []
    old = original()
    if [(c['seed'], c['arm']) for c in old['cells']] != list(CELLS):
        raise ContractError('endpoint inventory changed')
    for cell, historical in zip(old['cells'], old_reference['cells']):
        rng = rng_state()
        model = load_model(cell)
        before = model_identity(model)
        layouts = {}
        initial = budget.snapshot()
        for layout in LAYOUTS:
            fixed, fixed_calls = forced(model, rows, refs[(cell['seed'], cell['arm'])], layout, budget)
            generations, generation_calls = {}, {}
            for mode in ('greedy', 'syntax'):
                generations[mode], generation_calls[mode] = generate(model, rows, layout, mode, budget)
            layouts[layout] = {'forced': fixed, 'generated': generations,
                               'calls': {'forced': fixed_calls, **generation_calls}}
            assert_immutable(model, before, rng)
            if on_layout is not None:
                on_layout({'seed': cell['seed'], 'arm': cell['arm'], 'layout': layout, 'result': layouts[layout]})
            assert_immutable(model, before, rng)
        compared = comparisons(layouts, [r['item_id'] for r in rows])
        parity = {}
        for mode in ('greedy', 'syntax'):
            expected = historical['modes'][mode]['predictions']
            actual = layouts['singleton_min']['generated'][mode]
            parity[mode] = len(actual) == len(expected) and all(
                r['item_id'] == e['example_identity'] and r['tokens'] == e['generated_tokens'] and r['stop_reason'] == e['stop_reason']
                for r, e in zip(actual, expected))
        after = budget.snapshot()
        numeric_pass = all(c['numeric_pass'] for c in compared)
        decision_pass = all(c['decision_pass'] for c in compared)
        record = {'seed': cell['seed'], 'arm': cell['arm'],
                  'checkpoint_identity': identity(json_bytes(cell['final_checkpoint'])), 'model_identity': before,
                  'layouts': layouts, 'comparisons': compared, 'historical_singleton_parity': parity,
                  'numeric_pass': numeric_pass, 'decision_pass': decision_pass,
                  'conforms': numeric_pass and decision_pass and all(parity.values()),
                  'budget': {k: after[k]-initial[k] for k in after}, 'model_and_rng_unchanged': True}
        cells.append(record)
        if on_cell is not None:
            on_cell(record)
        assert_immutable(model, before, rng)
    if request() != binding:
        raise ContractError('source/input/environment bindings changed')
    return {'schema': 'trinite.batch-composition-report.v2', 'request': binding, 'cells': cells,
            'conforms': all(c['conforms'] for c in cells), 'budget': budget.snapshot(),
            'training_updates': 0, 'test_evaluated': False, 'sampling_claim': False, 'speed_claim': False,
            'scope': 'frozen native CPU engineering conformance on exposed development fixtures; no learning or release gate'}
