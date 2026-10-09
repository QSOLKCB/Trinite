"""Scoped scalar curriculum preparation, preserving all parent family splits."""
from pathlib import Path

from .contracts import ContractError, identity, json_bytes, parse_json
from .learning_data import dataset as parent_dataset, audit_bytes as parent_audit, source_receipt as parent_sources
from .curriculum_oracle import solve, parse_prompt
from .tokenizer import ByteTokenizer

POLICY = 'trinite.scalar-curriculum-data.v1'
WORKLOADS = ('arithmetic', 'fold', 'lattice')


def source_receipt():
    root = Path(__file__).parent
    return {**parent_sources(), **{name: identity((root/name).read_bytes()) for name in
        ('curriculum_data.py', 'curriculum_oracle.py')}}


def projections(workload, original):
    """Procedural labels independently checked by the histogram/modular oracle."""
    spec = original['formal']
    if workload == 'fold':
        count = total = squares = 0
        for value in spec['values']:
            count += 1; total += value
            for _ in range(abs(value)): squares += abs(value)
        return [({'task': task, 'values': spec['values']}, str(answer))
                for task, answer in zip(('count', 'sum', 'squares'), (count, total, squares))]
    if workload == 'lattice' and spec['task'] == 'traversal':
        value = 0
        for _ in range(spec['index']):
            value += 17
            while value >= 27: value -= 27
        x = y = 0; remainder = value
        while remainder >= 9: x += 1; remainder -= 9
        while remainder >= 3: y += 1; remainder -= 3
        return [({'task': task, 'index': spec['index']}, str(answer)) for task, answer in zip(
            ('traversal-value', 'traversal-x', 'traversal-y', 'traversal-z'), (value, x, y, remainder))]
    return [(spec, original['answer'])]


def render(workload, spec, carrier, original_prompt):
    if workload == 'arithmetic' or (workload == 'lattice' and not spec['task'].startswith('traversal-')):
        return original_prompt
    if workload == 'fold':
        tag = {'count': 'C', 'sum': 'S', 'squares': 'Q'}[spec['task']]
        body = ','.join(map(str, spec['values']))
    else:
        tag = {'traversal-value': 'Q', 'traversal-x': 'X', 'traversal-y': 'Y', 'traversal-z': 'Z'}[spec['task']]
        body = str(spec['index'])
    if carrier == 'infix': return tag+':'+body+'='
    if carrier == 'fields': return tag+'('+body+'):'
    raise ContractError('invalid scalar carrier')


def dataset(workload):
    if workload not in WORKLOADS: raise ContractError('unknown scalar workload')
    parent_raw, parent_manifest = parent_dataset(workload)
    parent = parent_audit(parent_raw, parent_manifest)
    source = source_receipt(); revision = identity(json_bytes(source)); rows = []
    for original in parse_json(parent_raw, canonical=True)['examples']:
        for spec, answer in projections(workload, original):
            prompt = render(workload, spec, original['carrier'], original['prompt'])
            if solve(workload, spec) != answer or parse_prompt(workload, prompt) != spec:
                raise ContractError('scalar label/carrier failed independent oracle')
            semantic = identity(json_bytes({'workload': workload, 'formal': spec}))
            core = {'formal': spec, 'answer': answer, 'task': spec['task'], 'prompt': prompt,
                'text': prompt+answer, 'family_id': original['family_id'], 'split': original['split'],
                'carrier': original['carrier'], 'semantic_identity': semantic,
                'source_ids': [*original['source_ids'], POLICY+':'+workload],
                'generator_revision': revision, 'seed': 31, 'verifier_outcome': 'verified',
                'transformation_lineage': {'operation': 'scalar-projection.v1',
                    'parent_example_identity': original['example_identity'],
                    'parent_semantic_identity': original['semantic_identity'], 'projection': spec['task']}}
            ByteTokenizer(64).encode(core['text'], answer_start=len(prompt.encode()))
            rows.append({**core, 'example_identity': identity(json_bytes(core))})
    if len({r['text'] for r in rows}) != len(rows): raise ContractError('duplicate scalar text')
    raw = json_bytes({'schema': POLICY, 'workload': workload,
                      'examples': sorted(rows, key=lambda r: r['example_identity'])})
    evidence = {'dataset_identity': identity(raw), 'parent_dataset_identity': identity(parent_raw),
                'parent_manifest_identity': identity(parent_manifest), 'generator_revision': revision}
    manifest = {'schema': 'trinite.scalar-curriculum-admission.v1', 'workload': workload,
        **evidence, 'family_splits': parent['family_splits'],
        'counts': {s: sum(r['split'] == s for r in rows) for s in ('train', 'validation', 'test')},
        'admission': {'source_id': POLICY+':'+workload,
            'origin': 'bounded scalar projections of independently admitted local numeric facts',
            'author': 'Codex generator commissioned by Trent Slade / QSOL-IMC',
            'generation': 'procedural fold sums and traversal loops; independent histogram/modular oracle and prompt parser; unchanged parent family/split assignment',
            'rights_basis': 'new numeric/formal facts and minimal symbolic carriers; no third-party text or model output',
            'supporting_reference': {'implementation': source, 'evidence_identity': identity(json_bytes(evidence))},
            'scope': 'this exact finite scalar preparation only; not admitted prose, books or future generated content',
            'reviewer': 'automated source, oracle, lineage and split audit under commissioned scope',
            'reviewed_on': '2026-10-09', 'outcome': 'admitted',
            'limitations': 'visible development data; no training, learning adequacy, legal certification or generalization claim'}}
    return raw, json_bytes(manifest)


def audit_bytes(raw, manifest):
    if type(raw) is not bytes or type(manifest) is not bytes or len(raw) > 16*1024*1024 or len(manifest) > 2*1024*1024:
        raise ContractError('scalar admission requires bounded bytes')
    record = parse_json(raw, canonical=True)
    workload = record.get('workload') if type(record) is dict else None
    if workload not in WORKLOADS or (raw, manifest) != dataset(workload):
        raise ContractError('scalar admission differs from verified source')
    return parse_json(manifest, canonical=True)
