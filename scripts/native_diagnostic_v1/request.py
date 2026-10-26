"""Fixed arithmetic diagnostic protocol and complete checkout source binding."""
from pathlib import Path

from trinite.contracts import ContractError, identity, integer, json_bytes, parse_json, require_identity
from trinite.budget_training import plan_for
from trinite.foundations_data import dataset, audit_bytes
from trinite.training import require_environment

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'trinite.native-diagnostic-request.v1'
MILESTONES = (0, 1024, 2048, 3072, 4096)
PROTOCOL = {
    'schema': 'trinite.native-diagnostic-protocol.v1',
    'workload': 'arithmetic', 'lane': 'dense', 'candidate': 'extended',
    'partition': 'train only; validation diagnostics alias train; no held-out scoring',
    'milestones': list(MILESTONES), 'generation_cap': 5,
    'worker_seconds': 900, 'update_seconds_total': 480,
    'max_artifact_bytes': 16 * 1024 * 1024, 'max_retained_payload_bytes': 32 * 1024 * 1024,
    'max_worker_log_bytes': 1024 * 1024,
    'update_time_enforcement': 'sampled after each milestone; worker deadline is the hard outer limit',
    'selection': 'none; diagnostic reproduction of historical dense arithmetic anchor',
    'cache_mode': 'unsupported', 'geometry_loss': '0',
    'scope': 'diagnostic engineering workflow; no learning, Phase5 or release eligibility',
}


def read(path, limit=16 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ContractError('missing or unsafe diagnostic artifact')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ContractError('diagnostic artifact exceeds byte bound')
    return raw


def sources():
    """Broad closure deliberately includes every checked-in src runtime asset.

    No imports from tests, global installed Trinite, or unbound checkout helpers.
    Package caches are not source. Addition/removal also changes this map.
    """
    paths = [p for p in (ROOT / 'src').rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and p.suffix != '.pyc']
    paths += list((ROOT / 'scripts/native_diagnostic_v1').glob('*.py'))
    paths += [ROOT / name for name in ('scripts/run_native_diagnostic.py',
              'scripts/run_budget.py', 'requirements.lock', 'AGENTS.md')]
    return {p.relative_to(ROOT).as_posix(): identity(read(p)) for p in sorted(paths)}


def corpus():
    raw, manifest = dataset('arithmetic')
    audit_bytes(raw, manifest)
    return raw, manifest


def training_examples():
    raw, _ = corpus()
    return [row for row in parse_json(raw, canonical=True)['examples'] if row['split'] == 'train']


def core(seed):
    integer(seed, 'diagnostic seed', 0, 2)
    raw, manifest = corpus()
    return {'schema': SCHEMA, 'protocol': PROTOCOL, 'protocol_identity': identity(json_bytes(PROTOCOL)),
            'seed': seed, 'plan': plan_for(seed, 'extended').to_dict(),
            'source': sources(), 'environment': require_environment(),
            'dataset_identity': identity(raw), 'manifest_identity': identity(manifest)}


def validate(raw, expected_identity):
    require_identity(expected_identity)
    if identity(raw) != expected_identity:
        raise ContractError('diagnostic request identity mismatch')
    record = parse_json(raw, canonical=True)
    if type(record) is not dict:
        raise ContractError('diagnostic request must be an object')
    expected = core(record.get('seed'))
    if raw != json_bytes(expected):
        raise ContractError('diagnostic request source/environment/data/protocol mismatch')
    return record
