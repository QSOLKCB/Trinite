"""Numeric QEC admission, independently checked labels and support-family splits."""
from itertools import product
from pathlib import Path

from .contracts import ContractError, identity, json_bytes, parse_json
from .qec_source import oracle, verify_pin
from .tokenizer import ByteTokenizer

KINDS = ('qutrit', 'ququart')


def source_receipt():
    root = Path(__file__).resolve().parent
    return {n: identity((root/n).read_bytes()) for n in (
        'comparison_data.py', 'qec_source.py', 'contracts.py', 'tokenizer.py', 'qec-pin.json')}


def scalar_syndrome(kind, code, error):
    """Independent scalar oracle; no upstream syndrome or field helpers."""
    if kind == 'qutrit':
        return tuple(sum(row[i]*error.z[i]-row[5+i]*error.x[i]
                         for i in range(5)) % 3 for row in code.stabilizers)
    result = []
    for x, z in ((error.lane0_x, error.lane0_z), (error.lane1_x, error.lane1_z)):
        result.extend(sum(row[i]*z[i]+row[5+i]*x[i] for i in range(5)) % 2
                      for row in code.base_stabilizers)
    return tuple(result)


def _digits(values):
    return ''.join(str(x) for x in values)


def numeric_items(kind):
    if kind not in KINDS:
        raise ContractError('unknown QEC task source')
    code, decoder, algebra = oracle(kind)
    items = []
    for error in algebra.paulis_of_weight(5, 1):
        syndrome = code.syndrome(error)
        if syndrome != scalar_syndrome(kind, code, error):
            raise ContractError('QEC syndrome failed independent scalar check')
        result = decoder.correct(error)
        if not result.success:
            raise ContractError('certified single-site error failed exact correction')
        if kind == 'qutrit':
            vectors = (error.x, error.z)
            site = next(i for i in range(5) if error.x[i] or error.z[i])
            correction = (site, (-error.x[site]) % 3, (-error.z[site]) % 3)
            got = (site, result.correction.x[site], result.correction.z[site])
            symbol = _digits(syndrome)
        else:
            vectors = (error.lane0_x, error.lane0_z, error.lane1_x, error.lane1_z)
            site = next(i for i in range(5) if any(v[i] for v in vectors))
            correction = (site, *(v[site] for v in vectors))
            got = (site, result.correction.lane0_x[site], result.correction.lane0_z[site],
                   result.correction.lane1_x[site], result.correction.lane1_z[site])
            symbol = _digits(2*syndrome[i]+syndrome[4+i] for i in range(4))
        if got != correction:
            raise ContractError('QEC correction failed independent inverse check')
        spec = {'kind': kind, 'site': site, 'vectors': [list(v) for v in vectors],
                'syndrome': list(syndrome), 'symbol': symbol, 'correction': list(correction)}
        items.append({**spec, 'identity': identity(json_bytes(spec))})
    if len(items) != (40 if kind == 'qutrit' else 75):
        raise ContractError('incomplete certified QEC error set')
    return sorted(items, key=lambda item: item['identity'])


def admission(kind, items):
    evidence = {'formal_items': items, 'minimal_symbols': ['3', '4', 'S', 'C', ':', '/', '=', '(', ')', ',']}
    return {'source_id': 'trinite.qec-numeric.'+kind+'.v1',
            'origin': 'local enumeration of exact single-site mathematical errors',
            'author': 'Codex deterministic generator commissioned by Trent Slade / QSOL-IMC',
            'generation': 'enumerate every nonidentity weight-one error; pinned QEC labels; independent scalar syndrome and inverse checks; two symbolic carriers per task',
            'rights_basis': 'numeric mathematical facts rendered with minimal symbols; no upstream prose consumed',
            'supporting_reference': {'implementation': source_receipt(), 'qec_dependency': verify_pin(),
                                     'evidence_identity': identity(json_bytes(evidence))},
            'evidence': evidence, 'scope': 'exact generated symbolic syndrome/correction examples only',
            'reviewer': 'automated source/rights/split audit under Trent Slade commissioned protocol',
            'reviewed_on': '2026-10-09', 'outcome': 'admitted',
            'limitations': 'software-generated numeric admission; not legal certification, prose admission, hardware evidence or a fresh capability benchmark'}


def dataset(kind):
    items = numeric_items(kind)
    order = sorted(range(5), key=lambda site: identity(json_bytes({'policy': 'qec-support-rank.v1', 'site': site})))
    splits = {site: 'train' if i < 3 else 'validation' if i == 3 else 'test'
              for i, site in enumerate(order)}
    examples = []
    prefix = '3' if kind == 'qutrit' else '4'
    for item in items:
        for task, carrier in product(('syndrome', 'correction'), ('infix', 'fields')):
            if task == 'syndrome':
                operands = [_digits(v) for v in item['vectors']]
                answer = item['symbol']; tag = prefix+'S'
            else:
                operands = [item['symbol']]
                answer = _digits(item['correction']); tag = prefix+'C'
            prompt = (tag+':'+('/'.join(operands))+'=' if carrier == 'infix'
                      else tag+'('+(','.join(operands))+'):')
            core = {'family_id': kind+':site:'+str(item['site']), 'split': splits[item['site']],
                    'semantic_identity': identity(json_bytes({'error': item['identity'], 'task': task})),
                    'formal_identity': item['identity'], 'task': task, 'carrier': carrier,
                    'prompt': prompt, 'answer': answer, 'text': prompt+answer}
            ByteTokenizer(64).encode(core['text'], answer_start=len(prompt.encode()))
            examples.append({**core, 'example_identity': identity(json_bytes(core))})
    if len({e['text'] for e in examples}) != len(examples):
        raise ContractError('duplicate QEC text')
    data = {'schema': 'trinite.qec-dataset.v1', 'kind': kind,
            'examples': sorted(examples, key=lambda e: e['example_identity'])}
    raw = json_bytes(data)
    manifest = {'schema': 'trinite.qec-admission.v1', 'dataset_identity': identity(raw),
                'admission': admission(kind, items),
                'counts': {s: sum(e['split'] == s for e in examples) for s in ('train', 'validation', 'test')},
                'family_splits': {kind+':site:'+str(site): split for site, split in splits.items()}}
    return raw, json_bytes(manifest)


def audit_bytes(raw, manifest):
    if type(raw) is not bytes or type(manifest) is not bytes or len(raw) > 512*1024 or len(manifest) > 128*1024:
        raise ContractError('QEC admission requires bounded bytes')
    data = parse_json(raw, canonical=True)
    kind = data.get('kind') if type(data) is dict else None
    if kind not in KINDS or (raw, manifest) != dataset(kind):
        raise ContractError('QEC dataset/admission differs from independently verified source')
    return parse_json(manifest, canonical=True)
