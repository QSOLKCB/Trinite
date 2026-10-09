"""Bounded finite mathematics and exact algebra; formal facts, no imported text."""
from itertools import product
from pathlib import Path

from .contracts import ContractError, identity, json_bytes, parse_json
from .maths_oracle import TASKS, TAGS, solve, parse_prompt, validate
from .tokenizer import ByteTokenizer

POLICY = 'trinite.maths-data.v2'
SPLIT_SEED = 73


def source_receipt():
    root = Path(__file__).resolve().parent
    return {n: identity((root/n).read_bytes()) for n in
            ('maths_data.py', 'maths_oracle.py', 'contracts.py', 'tokenizer.py')}


def orbit(matrix):
    """Matrix row/column permutations, transpose and global sign stay together."""
    a = [matrix[:2], matrix[2:]]
    variants = []
    for rs, cs, transpose, sign in product((0, 1), (0, 1), (0, 1), (-1, 1)):
        variants.append(tuple(sign*a[(j^cs) if transpose else (i^rs)][(i^rs) if transpose else (j^cs)]
                              for i, j in product(range(2), repeat=2)))
    return min(variants)


def specs():
    for a, b in product(range(8), repeat=2):
        for task in TASKS[:2]: yield {'task': task, 'values': [a, b]}
    for n in range(9):
        for k in range(-1, n+2): yield {'task': 'choose', 'values': [n, k]}
    for m in range(2, 7):
        for a, b in product(range(m), repeat=2): yield {'task': 'mod-add', 'values': [a, b, m]}
    for matrix in product(range(-1, 2), repeat=4):
        for task in TASKS[4:8]: yield {'task': task, 'values': list(matrix)}
        for vector in ((-1, 0), (0, 1), (1, 1)):
            for task in TASKS[8:]: yield {'task': task, 'values': list(matrix+vector)}


def family(spec):
    task, v = validate(spec)
    if task in TASKS[:2]:
        # Set relabelling (bit permutations), set exchange and complementation.
        from itertools import permutations
        variants = []
        for p in permutations(range(3)):
            a, b = [sum(((x >> i) & 1) << p[i] for i in range(3)) for x in v]
            variants.extend((min(a, b), max(a, b)) for a, b in ((a, b), (a^7, b^7)))
        core = ['sets', list(min(variants))]
    elif task == 'choose': core = ['choose', v[0]]
    elif task == 'mod-add': core = ['modulus', v[2]]
    else:
        # All projections, RHS/vectors and tasks for the same matrix orbit.
        # Vector pairs use this same conservative algebra grouping.
        core = ['algebra', list(orbit(v[:4]))]
    return core[0]+':'+identity(json_bytes(core))


def label(spec):
    task, v = validate(spec)
    def multiply(a, b):
        total = 0
        for _ in range(abs(b)): total += a
        return -total if b < 0 else total
    if task in TASKS[:2]:
        a, b = v; count = 0
        for _ in range(3):
            count += int(bool(a % 2 or b % 2) if task == TASKS[0] else bool(a % 2 and b % 2))
            a //= 2; b //= 2
        return str(count)
    if task == 'choose':
        n, k = v
        if not 0 <= k <= n: return '0'
        row = [1]
        for _ in range(n): row = [1]+[row[i-1]+row[i] for i in range(1, len(row))]+[1]
        return str(row[k])
    if task == 'mod-add':
        value = v[0]+v[1]
        while value >= v[2]: value -= v[2]
        return str(value)
    a, b, c, d = v[:4]
    if task == 'dot': return str(multiply(a, c)+multiply(b, d))
    if task.startswith('vector-add'):
        i = int(task[-1]); return str(v[i]+v[i+2])
    det = multiply(a, d)-multiply(b, c)
    if task == 'determinant': return str(det)
    if task.startswith('matvec'):
        i = 2*int(task[-1]); return str(multiply(v[i], v[4])+multiply(v[i+1], v[5]))
    if not det: return 'ERR'  # No unique solution, irrespective of RHS consistency.
    numerator = multiply(d, v[4])-multiply(b, v[5]) if task == 'solve-0' else multiply(a, v[5])-multiply(c, v[4])
    left, right = abs(numerator), abs(det)
    while right: left, right = right, left % right
    numerator, denominator = numerator//left, det//left
    if denominator < 0: numerator, denominator = -numerator, -denominator
    return str(numerator) if denominator == 1 else f'{numerator}/{denominator}'


def render(spec, carrier):
    validate(spec); body = ','.join(map(str, spec['values'])); tag = TAGS[spec['task']]
    if carrier == 'infix': return tag+':'+body+'='
    if carrier == 'fields': return tag+'('+body+'):'
    raise ContractError('unknown maths carrier')


def dataset():
    items = []
    for spec in specs():
        answer = label(spec)
        if answer != solve(spec): raise ContractError('maths independent label disagreement')
        items.append({'formal': spec, 'answer': answer, 'family_id': family(spec),
                      'semantic_identity': identity(json_bytes(spec))})
    if len({i['semantic_identity'] for i in items}) != len(items): raise ContractError('duplicate maths fact')
    assignments = {}
    groups = {}
    for item in items:
        f = item['family_id']; kind = f.split(':')[0]
        if kind == 'algebra':
            a, b, c, d = item['formal']['values'][:4]
            determinant = a*d-b*c
            kind = 'algebra-singular' if not determinant else 'algebra-unique'
            # The sole |det|=2 orbit supplies rational-answer training vocabulary.
            # It cannot appear across splits; rational generalization is not tested.
            if abs(determinant) == 2:
                assignments[f] = 'train'; continue
        groups.setdefault(kind, set()).add(f)
    for kind, group in sorted(groups.items()):
        families = sorted(group, key=lambda f: identity(json_bytes({'policy': POLICY, 'seed': SPLIT_SEED, 'family': f})))
        n = len(families); train = n*7//10; validation = max(1, n//10)
        if n-train-validation < 1: raise ContractError('empty maths split stratum')
        assignments.update({f: 'train' if j < train else 'validation' if j < train+validation else 'test'
                            for j, f in enumerate(families)})
    revision = identity(json_bytes(source_receipt())); examples = []
    for item, carrier in product(items, ('infix', 'fields')):
        prompt = render(item['formal'], carrier)
        if parse_prompt(prompt) != item['formal']: raise ContractError('maths rendered semantics mismatch')
        core = {**item, 'task': item['formal']['task'], 'carrier': carrier, 'prompt': prompt,
                'text': prompt+item['answer'], 'split': assignments[item['family_id']],
                'source_ids': [POLICY], 'generator_revision': revision, 'split_seed': SPLIT_SEED,
                'verifier_outcome': 'verified', 'transformation_lineage': {'formal_identity': item['semantic_identity'], 'carrier': carrier}}
        ByteTokenizer(64).encode(core['text'], answer_start=len(prompt.encode()))
        examples.append({**core, 'example_identity': identity(json_bytes(core))})
    if len({e['text'] for e in examples}) != len(examples): raise ContractError('duplicate maths text')
    raw = json_bytes({'schema': POLICY, 'examples': sorted(examples, key=lambda e: e['example_identity'])})
    evidence = {'formal_items': items, 'dataset_identity': identity(raw),
                'symbols': 'bounded integers, reduced rationals, ERR, unique task tags and delimiters'}
    admission = {'source_id': POLICY, 'origin': 'local exhaustive bounded formal enumeration',
        'author': 'Codex deterministic generator commissioned by Trent Slade / QSOL-IMC',
        'generation': 'procedural set/Pascal/repeated-addition/adjugate labels; independent set/comb/Leibniz/Gaussian oracle; separate prompt parser; rank-stratified orbit family split',
        'rights_basis': 'generated numeric/formal facts and minimal symbolic carriers; no textbook prose, external source examples, pretrained or teacher outputs',
        'supporting_reference': {'implementation': source_receipt(), 'functions': ['specs', 'label', 'render', 'maths_oracle.solve', 'maths_oracle.parse_prompt'], 'evidence_identity': identity(json_bytes(evidence))},
        'evidence': evidence, 'scope': 'this exact bounded generated maths corpus only',
        'reviewer': 'automated commissioned source/rights/oracle/split audit', 'reviewed_on': '2026-10-09',
        'outcome': 'admitted', 'limitations': 'numeric source admission, not human legal certification; visible finite-domain exploratory evaluation, no textbook or universal reasoning claim'}
    manifest = {'schema': 'trinite.maths-admission.v2', 'dataset_identity': identity(raw), 'admission': admission,
                'family_splits': assignments, 'split_policy': 'rank-stratified algebra; sole absolute-determinant-2 orbit train-only; other categories grouped hash rank.v2', 'counts': {s: sum(e['split'] == s for e in examples) for s in ('train', 'validation', 'test')},
                'task_counts': {t: {s: sum(e['task'] == t and e['split'] == s for e in examples) for s in ('train', 'validation', 'test')} for t in TASKS}}
    if any(not c[s] for c in manifest['task_counts'].values() for s in ('train', 'validation', 'test')):
        raise ContractError('each maths task requires every split')
    coverage = {s: {'unique_systems': 0, 'singular_systems': 0, 'rational_targets': 0} for s in ('train', 'validation', 'test')}
    for e in examples:
        if e['task'].startswith('solve'):
            a, b, c, d = e['formal']['values'][:4]
            coverage[e['split']]['unique_systems' if a*d-b*c else 'singular_systems'] += 1
            coverage[e['split']]['rational_targets'] += int('/' in e['answer'])
    if any(not c['unique_systems'] or not c['singular_systems'] for c in coverage.values()) or not coverage['train']['rational_targets']:
        raise ContractError('maths split lacks unique/singular systems or rational training targets')
    manifest['linear_system_coverage'] = coverage
    return raw, json_bytes(manifest)


def audit_bytes(raw, manifest):
    if type(raw) is not bytes or type(manifest) is not bytes or len(raw) > 8*1024*1024 or len(manifest) > 4*1024*1024:
        raise ContractError('bounded maths admission bytes required')
    if (raw, manifest) != dataset(): raise ContractError('maths corpus/admission differs from independently verified generation')
    return parse_json(manifest, canonical=True)
