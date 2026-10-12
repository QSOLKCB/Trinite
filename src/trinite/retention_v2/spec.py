"""Explicit v2 experiment/schedule specification over unchanged v1 numerical inputs."""
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from ..contracts import ContractError, identity, json_bytes, parse_json
from ..maths_learning import corpora as admitted_v1_corpora
from ..maths_retention import state_for as v1_state, sources as v1_sources
from ..maths_plan import MathsPlan
from .evidence import read_bytes

PROTOCOL_IDENTITY = 'sha256:d4160712380a354fd0ac0272a1b576bbad74696e74190b457b23b4b86e5760e6'
PROFILES = ('pooled', 'rehearsal')


def protocol():
    raw = read_bytes(Path(__file__).parent/'protocol.json', 16384)
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError('v2 protocol changed; freeze a new version')
    return parse_json(raw, canonical=True)


def sources():
    package = Path(__file__).resolve().parent; root = package.parents[2]
    paths = [*package.glob('*.py'), *package.glob('*.json'), root/'AGENTS.md', root/'scripts/run_maths_retention_v2.py']
    return {**v1_sources(), **{str(p.relative_to(root)): identity(read_bytes(p, 1024*1024)) for p in sorted(paths)}}


def cells():
    return tuple((seed, profile) for seed in (0, 1) for profile in (PROFILES if seed == 0 else PROFILES[::-1]))


def cell_name(seed, profile):
    if type(seed) is not int or type(profile) is not str or (seed, profile) not in cells():
        raise ContractError('unknown fixed v2 retention cell')
    return f'maths-retention-v2-{seed}-{profile}-dense'


def profile_settings(run_profile):
    if type(run_profile) is not str or run_profile not in ('research', 'smoke'):
        raise ContractError('unknown fixed v2 run profile')
    return protocol()['run_profiles'][run_profile]


def plan_for(seed, run_profile):
    return replace(MathsPlan.from_dict(profile_settings(run_profile)['training']), seed=seed)


def state_for(seed, profile, run_profile):
    cell_name(seed, profile)
    # The materialized schedule, initial model, optimizer and numerical diagnostics
    # remain owned by the frozen v1 construction; v2 supplies only an explicit plan.
    return v1_state(seed, profile, plan_for(seed, run_profile))


def diagnostic_rows(state):
    rows = state.data['validation']  # historical updater boundary, training-only
    if len({r['identity'] for r in rows}) != len(rows):
        raise ContractError('diagnostic view must contain unique training rows')
    return rows


def immutable(value):
    if type(value) is dict: return MappingProxyType({k: immutable(v) for k, v in value.items()})
    if type(value) is list: return tuple(immutable(v) for v in value)
    return value


@dataclass(frozen=True)
class Corpus:
    raw: bytes
    manifest: bytes
    examples: tuple


def corpora():
    # A fresh full admission audit for every independent invocation. There is no
    # retained success cache. Parsed examples are immutable within this invocation.
    return MappingProxyType({kind: Corpus(raw, manifest, immutable(examples))
                             for kind, (raw, manifest, examples) in admitted_v1_corpora().items()})


def scoring_examples(pack, run_profile):
    profile_settings(run_profile)
    if run_profile == 'research': return pack.examples
    first = {}
    for e in pack.examples:
        key = (e['task'], e['split'])
        first[key] = min(first.get(key, e['semantic_identity']), e['semantic_identity'])
    return tuple(e for e in pack.examples if e['semantic_identity'] == first[e['task'], e['split']])


def expected_counts(examples, split):
    return dict(Counter(e['task'] for e in examples if e['split'] == split))


def inputs(request_raw, packs):
    result = {'request.json': request_raw}
    for kind, pack in packs.items():
        result.update({kind+'-dataset.json': pack.raw, kind+'-manifest.json': pack.manifest})
    return result


def drift(rows, mapping):
    pairs = {}
    for row in rows:
        pairs.setdefault(mapping[row['example_identity']]['semantic_identity'], []).append(row)
    return {'missing_eos': sum(not r['generated_tokens'] or r['generated_tokens'][-1] != 257 for r in rows),
            'wrong_answer_with_eos': sum(not r['correct'] and r['generated_tokens'][-1] == 257 for r in rows),
            'carrier_pairs': len(pairs),
            'carrier_disagreement': sum(len({tuple(r['generated_tokens']) for r in p}) != 1 for p in pairs.values()),
            'both_carriers_exact': sum(all(r['correct'] for r in p) for p in pairs.values())}
