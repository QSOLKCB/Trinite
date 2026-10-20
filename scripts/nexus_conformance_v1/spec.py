# SPDX-License-Identifier: MPL-2.0
"""Frozen, deliberately visible conformance cases and per-item lineage."""
import hashlib
import json
from pathlib import Path
from .reference import evolve, plan

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'nexus-z4-formal-conformance-v1'
SEEDS = (0, 7, 0x4e455853, 0xffffffff)
SPLIT = 'conformance-only'


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)+'\n').encode()


def identity(raw):
    return hashlib.sha256(raw).hexdigest()


def sources():
    paths = sorted([ROOT/'scripts/check_nexus_conformance.py'] +
                   list((ROOT/'scripts/nexus_conformance_v1').glob('*.py')) +
                   list((ROOT/'scripts/nexus_conformance_v1/adapters').glob('*')) +
                   [p for p in (ROOT/'fixtures/nexus-donor-v1').rglob('*') if p.is_file()])
    return {p.relative_to(ROOT).as_posix(): identity(p.read_bytes()) for p in paths}


def cases():
    result = []
    for cells in (2, 6):
        n = cells*12
        bank = [[0]*n, [3]*n, [i % 4 for i in range(n)]]
        bank += [[int(i == j) for i in range(n)] for j in range(n)]
        # Independent state generation, not donor seed_register.
        bank += [[hashlib.sha256(f'nexus-state-v1:{k}:{i}'.encode()).digest()[0] % 4
                  for i in range(n)] for k in range(3)]
        for state_index, state in enumerate(bank):
            family = identity(canonical({'cells':cells, 'state':state}))
            for seed in SEEDS:
                for rounds in (0, 1, 2, 8):
                    result.append({'family_id':family, 'state_index':state_index,
                                   'seed':seed, 'rounds':rounds, 'cells':cells, 'state':state})
    for seed in SEEDS:
        state = [0]*24
        result.append({'family_id':identity(canonical({'cells':2,'state':state})),
                       'state_index':0,'seed':seed,'rounds':128,'cells':2,'state':state})
    return result


def request():
    rows = cases()
    coverage = sorted({plan(c['seed'], r) for c in rows for r in range(c['rounds'])})
    if len(coverage) != 64:
        raise ValueError('all 64 plan tuples must be covered')
    return {'schema':'trinite.nexus-conformance-request.v1', 'sources':sources(),
            'cases':rows, 'plan_coverage':coverage, 'split':SPLIT,
            'bounds':{'case_count':len(rows),'max_rounds':128,'cells':[2,6]},
            'source_admission':{'source_id':SOURCE,'origin':'QSOLKCB/NEXUS exact discrete Z4 model',
                'author':'QSOLKCB/NEXUS contributors; Trinite generator commissioned by Trent Slade / QSOL-IMC',
                'rights_basis':'Generated bounded integer state-transition facts and minimal formal templates; no donor prose or model output in items.',
                'software_license':'MPL-2.0; see fixtures/nexus-donor-v1/NOTICE.md',
                'acquisition':'Exact commit-pinned public donor source files retained with Git blob and SHA-256 identities.',
                'supporting_reference':'fixtures/nexus-donor-v1/source-lock.json',
                'content_identity':identity(canonical(rows)),
                'reviewer':'Trinite automated source-specific conformance policy v1',
                'review_date':'2026-10-21',
                'outcome':'conformance-only; not admitted to native model training',
                'limitations':'Fully visible engineering suite. No copyright/legal certification, heldout benchmark, pretrained input, learning or physical claim.'}}


def input_bytes(rows):
    return ''.join(' '.join(map(str, [c['seed'],c['rounds'],c['cells'],*c['state']]))+'\n' for c in rows).encode()


def expected_states(c):
    f = evolve(c['seed'], c['rounds'], c['state'])
    inv = evolve(c['seed'], c['rounds'], c['state'], True)
    return [f, evolve(c['seed'],c['rounds'],f,True), inv,
            evolve(c['seed'],c['rounds'],inv)]


def expected_bytes(rows):
    output=[]
    for c in rows:
        for tag, state in zip(('forward','restored','inverse','recovered'), expected_states(c)):
            output.append(','.join(map(str,[tag,c['seed'],c['rounds'],c['cells'],*state]))+'\n')
    return ''.join(output).encode()


def items(req):
    """Candidate items. Live Rust/Lean parity is certified only by the runner."""
    revision = identity(canonical(req['sources']))
    for c in req['cases']:
        states = expected_states(c)
        if states[1] != c['state'] or states[3] != c['state']:
            raise ValueError('reference roundtrip failed')
        parent = identity(canonical(c))
        for direction, answer in (('forward',states[0]),('inverse',states[2])):
            for target in ('state','register'):
                query = (c['state_index']+c['seed']+c['rounds']) % len(c['state'])
                formal = {'seed':c['seed'],'rounds':c['rounds'],'cells':c['cells'],
                          'input':c['state'],'direction':direction,'target':target,
                          'index':query if target == 'register' else None}
                row = {'source_id':SOURCE,'generator_revision':revision,'seed':c['seed'],
                       'seed_procedure':'fixed [0,7,1313167443,4294967295]; state bank independently generated',
                       'split':SPLIT,'split_procedure':'All cases and correlated directions/projections remain in one visible conformance partition; no heldout split.',
                       'family_id':c['family_id'],'formal_specification':formal,
                       'transformation_lineage':{'parent_case':parent,'state_index':c['state_index']},
                       'rendered_text':canonical(formal).decode().rstrip('\n'),
                       'expected_answer':answer if target == 'state' else answer[query],
                       'verifier_outcome':'exact-python-both-roundtrips; live-rust-and-lean-parity-required'}
                row['item_id'] = identity(canonical(row))
                yield row
