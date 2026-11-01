"""Fresh public construction, whole table-orbit families, train-only admission."""
from copy import deepcopy
from functools import lru_cache
from itertools import permutations

import queue_evaluation_v1 as gate
from procedural_dataset_v1 import cases
from procedural_dataset_v1.admission import prompt as public_prompt, supervised
from trinite.contracts import json_bytes


def orbit(table):
    """Exact orbit under independent context/symbol/meaning bijections.

    Row sorting removes context permutations; exhaustive column and meaning
    permutations cover the remainder. Masks/query are intentionally ignored.
    """
    meanings = max(max(row) for row in table) + 1
    candidates = []
    for columns in permutations(range(len(table[0]))):
        for names in permutations(range(meanings)):
            candidates.append(tuple(sorted(tuple(names[row[s]] for s in columns) for row in table)))
    return min(candidates)


def parents(table):
    changed = deepcopy(table)
    changed[0][0] = (changed[0][0] + 1) % 3
    return [table, changed]


def family_tables(namespace, excluded=()):
    # Fixed hash stream, no label, prediction or difficulty selection. Reject
    # orbit aliases before assigning splits; retain deterministic rejection count.
    tables, occupied = [], set(excluded)
    for index in range(4096):
        raw = bytes.fromhex(gate.digest(f'{namespace}/{index}'.encode('ascii')))
        table = [[raw[c*3+s] % 3 for s in range(3)] for c in range(4)]
        variants = parents(table)
        if any(set(sum(t, [])) != {0, 1, 2} for t in variants):
            continue
        keys = {orbit(t) for t in variants}
        if len(keys) != 2 or keys & occupied:
            continue
        tables.append((index, table))
        occupied |= keys
        if len(tables) == 9:
            break
    gate.require(len(tables) == 9, 'fixed family stream exhausted; no redraw')
    return tables


def exposed_orbits():
    # An early implementation-only metric test derived labels on this original
    # development inventory. Exclude its ENTIRE table families from the fresh
    # scientific inventory; the exclusion is independent of label values.
    return {orbit(parent) for _,table in family_tables('context-symbol-family-v1') for parent in parents(table)}


@lru_cache(maxsize=1)
def _inventory_bytes():
    tables = family_tables('context-symbol-scientific-family-v1',exposed_orbits())
    ranked = sorted(range(9), key=lambda i: gate.digest(json_bytes(tables[i][1])))
    splits = {i: ('train' if rank < 5 else 'validation' if rank < 7 else 'test')
              for rank, i in enumerate(ranked)}
    rows = []
    for family, (index, table) in enumerate(tables):
        unique = {}
        for mutation, parent in enumerate(parents(table)):
            for cm in ([True, False, False, False], [False, True, False, False], [True]*4):
                for sm in ([True, False, False], [True]*3):
                    for query in range(3):
                        request = dict(schema=cases.REQUEST_SCHEMAS['context'], contexts=4, symbols=3,
                            meanings=3, table=parent, context_mask=cm, symbol_mask=sm, query=query)
                        for rename in (False, True):
                            r = deepcopy(request)
                            if rename:
                                r.update(table=[[(v+1) % 3 for v in reversed(row)] for row in reversed(parent)],
                                    context_mask=list(reversed(cm)), symbol_mask=list(reversed(sm)), query=(query+1) % 3)
                            cases.validate_request('context', r)
                            unique[json_bytes(r)] = (r, mutation, rename)
        for request, mutation, rename in unique.values():
            rows.append(dict(id=f'c{len(rows):04d}', family=f'f{family}', split=splits[family],
                generator_index=index, mutation=mutation, renamed=rename, request=request))
    return json_bytes(rows)


def inventory():
    from trinite.contracts import parse_json
    return parse_json(_inventory_bytes(), canonical=True)


def observation(row):
    return cases.public_observation('context', row['request'])


def prompt(obs):
    text = public_prompt('context', obs)
    gate.require(1+len(text.encode('ascii'))+3 <= 256, 'full unrestricted generation allowance exceeds context')
    return text


def encode(obs, answer):
    gate.require(type(answer) is str and answer in ('-1', '0', '1'), 'noncanonical relation target')
    enc = supervised('context', obs, answer)
    return dict(ids=list(enc.input_ids), mask=list(enc.attention_mask), loss=list(enc.loss_mask))


def audit(rows):
    exact, semantic, counts = set(), {}, dict(train=0, validation=0, test=0)
    maximum = 0
    for row in rows:
        r, obs = row['request'], observation(row)
        key = json_bytes(obs)
        gate.require(key not in exact, 'duplicate public input')
        exact.add(key)
        # Dropping masks, query and all names groups every declared related
        # variant. The mutation-parent lineage is separately assigned wholesale.
        key = (r['contexts'], r['symbols'], r['meanings'], orbit(r['table']))
        group = (row['family'], row['split'])
        gate.require(key not in semantic or semantic[key] == group, 'semantic orbit crosses family/split')
        semantic[key] = group
        maximum = max(maximum, len(encode(obs, '-1')['ids']))
        counts[row['split']] += 1
    # #119 covered only square n/n/n dimensions 2,3,4 for every seed.
    # Our 4/3/3 dimensions cannot be equivalent under these bijections.
    gate.require(all((r['request']['contexts'], r['request']['symbols'], r['request']['meanings']) == (4,3,3)
                     for r in rows), 'historical dimension exclusion changed')
    gate.require(len({r['family'] for r in rows}) == 9 and maximum <= 256, 'family/context bound')
    gate.require(not {key[-1] for key in semantic} & exposed_orbits(), 'development-exposed table orbit reused')
    return dict(counts=counts, families=9, semantic_orbits=len(semantic), examples=len(rows),
        max_complete_tokens=maximum, historical_exact_overlap=0, historical_declared_semantic_overlap=0,
        development_orbit_overlap=0,
        requests_sha256=gate.digest(json_bytes(rows)), truncated=0)


def training_rows(rows, oracle):
    result = []
    for row in rows:
        if row['split'] != 'train':
            continue  # This filter must precede ANY privileged oracle call.
        answer = oracle(row['request'])
        gate.require(answer == cases.expected_target('context', row['request']), 'independent label mismatch')
        result.append(dict(identity=row['id'], answer=answer, **encode(observation(row), answer)))
    result.sort(key=lambda r: gate.digest(r['identity'].encode('ascii')))
    gate.require({r['answer'] for r in result} == {'-1','0','1'}, 'train classes incomplete')
    return result


def fit_rows(train):
    # Fixed training-only stratification; no heldout or model outcome selection.
    selected = []
    for answer, count in (('-1',3), ('0',2), ('1',3)):
        selected.extend([r for r in train if r['answer'] == answer][:count])
    gate.require(len(selected) == 8 and len({tuple(r['ids'][:-len(r['answer'])-1]) for r in selected}) == 8,
                 'eight unique consistent fit prefixes required')
    return selected


def rights(rows, source):
    return dict(schema='trinite.context-symbol-admission.v1', outcome='admitted',
        source_id='commissioned-context-symbol-family-v1', author='Codex commissioned by Trent Slade / QSOLKCB',
        origin='original finite numeric tables and masks; no corpus, teacher, book, images or private chat',
        rights_basis='formal numeric facts and minimal symbolic notation; software licensing is separate',
        generation='fixed SHA-256 table stream; two declared mutation parents; whole orbits own splits',
        scope='only exact regenerated context-symbol-scientific-family-v1 inventory',
        verification='actual Rust observe/train evaluate and independent finite Cartesian-world oracle; lossless native tokens',
        reviewer='automated source-specific audit commissioned by owner', reviewed_on='2026-11-02',
        sources=source, inventory_sha256=gate.digest(json_bytes(rows)),
        supporting_reference='scripts/context_symbol_learning_v1/data.py',
        limitations='not legal certification, operator blindness or independent optimization replay')
