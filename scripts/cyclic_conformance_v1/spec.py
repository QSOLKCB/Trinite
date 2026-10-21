"""Bounded finite cases and source-bound lineage; no native training admission."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ID = 'trinite.exact-cyclic-formal-conformance.v1'
MODULI = tuple(range(1, 13)) + (60, 61, 100, 360)
PAIRS = ((60, 100), (60, 61), (60, 360), (4, 60), (360, 60), (100, 60))
LIMIT = 8 * 1024 * 1024


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def identity(raw):
    return hashlib.sha256(raw).hexdigest()


def source_map():
    paths = ['scripts/check_cyclic_conformance.py',
             'fixtures/cyclic-conformance-v1/input-provenance.json',
             'docs/CYCLIC-CONFORMANCE.md', '.github/workflows/cyclic-conformance.yml']
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'scripts/cyclic_conformance_v1').glob('*.py'))]
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests/cyclic_conformance_v1').glob('*.py'))]
    return {p: identity((ROOT/p).read_bytes()) for p in sorted(paths)}


def cases():
    result = []
    for n in MODULI:
        for k in range(n):
            result.append({'operation': 'element', 'n': n, 'k': k})
    for n, m in PAIRS:
        for k in range(n):
            result.append({'operation': 'projection', 'n': n, 'm': m, 'k': k})
    return result


def request():
    sources = source_map()
    return {'schema': 'trinite.cyclic-request.v1', 'source_id': SOURCE_ID,
            'sources': sources, 'generator_revision': identity(canonical(sources)),
            'procedure': 'ascending explicit moduli/elements then ordered grid pairs/elements; no RNG',
            'seed': None, 'split': 'conformance-only; no train/validation/test partition',
            'cases': cases(), 'limits': {'max_modulus': 360, 'max_cases': 2000,
                                        'max_input_bytes': LIMIT}}
