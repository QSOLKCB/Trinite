"""Bounded shared byte binding and fresh closed-custody index reading for v2."""
from datetime import datetime, timezone
from pathlib import Path

from ..contracts import ContractError, exact_keys, identity, json_bytes, parse_json, require_identity
from ..observation import BundleObserver, verify_observation, verify_pin

OUTPUTS = ('report.json', 'predictions.json', 'steps.json', 'tensors.safetensors', 'metadata.json')
ARTIFACTS = (*OUTPUTS, 'verification.json', 'provenance/manifest.json')
ROOT_ARTIFACTS = ('request.json', 'worker-errors.json')


def safe_path(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ContractError('v2 paths cannot traverse symlinks')
    return path


def read_bytes(path, limit=32*1024*1024):
    path = safe_path(path)
    if not path.is_file(): raise ContractError('missing v2 artifact: '+path.name)
    with path.open('rb') as stream: raw = stream.read(limit+1)
    if len(raw) > limit: raise ContractError('v2 artifact exceeds byte budget')
    return raw


def write(path, value):
    path = safe_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream: stream.write(json_bytes(value))


def freeze(path, core):
    raw = {**core, 'created_at': datetime.now(timezone.utc).isoformat()}
    write(path, raw)
    return identity(json_bytes(raw))


def read_frozen(path, expected_identity, core):
    require_identity(expected_identity); raw = read_bytes(path, 128*1024)
    if identity(raw) != expected_identity: raise ContractError('v2 immutable request identity mismatch')
    value = parse_json(raw, canonical=True)
    exact_keys(value, set(core)|{'created_at'}, 'v2 frozen request')
    if any(json_bytes(value[k]) != json_bytes(v) for k, v in core.items()):
        raise ContractError('v2 request/source/data/environment/profile mismatch')
    try:
        if datetime.fromisoformat(value['created_at']).utcoffset() is None: raise ValueError('timezone required')
    except (TypeError, ValueError) as error: raise ContractError('invalid v2 request clock') from error
    return value, raw


def seal(root, inputs):
    root = safe_path(root); observer = BundleObserver(root/'provenance')
    observer.record('maths-retention-v2', inputs=inputs, outputs={n: read_bytes(root/n) for n in OUTPUTS})
    receipt = observer.finalize()
    write(root/'verification.json', receipt)
    verify_observation(root/'provenance')


def read_bound(root, expected_inputs):
    """One bounded reader for the exact v2 input/output names and root byte copies."""
    root = safe_path(root)
    receipt = verify_observation(root/'provenance')  # always fresh, never a cached verdict
    if read_bytes(root/'verification.json') != json_bytes(receipt):
        raise ContractError('v2 custody receipt copy mismatch')
    manifest = parse_json(read_bytes(root/'provenance/manifest.json'), canonical=True)
    members = {a['content_identity'] for a in manifest['core']['artifacts']}
    artifact_root = root/'provenance/artifacts/sha256'
    paths = list(artifact_root.iterdir())
    if len(paths) > 64: raise ContractError('v2 custody artifact count exceeded')
    indices = []
    for p in paths:
        if p.stat().st_size <= 128*1024:
            raw = read_bytes(p, 128*1024)
            if raw.startswith(b'{"artifacts":'):
                indices.append(parse_json(raw, canonical=True))
    if len(indices) != 1: raise ContractError('v2 needs exactly one custody name index')
    index = indices[0]
    exact_keys(index, {'schema', 'artifacts', 'pin'}, 'v2 name index')
    names = index['artifacts']
    if (index['schema'] != 'trinite.observer-index.v1' or index['pin'] != verify_pin() or type(names) is not dict
            or set(names) != set(expected_inputs)|set(OUTPUTS)):
        raise ContractError('v2 custody names mismatch')
    bound = {}
    for name, key in names.items():
        require_identity(key)
        if key not in members: raise ContractError('v2 index points outside retained manifest')
        raw = read_bytes(artifact_root/key.split(':')[1])
        if identity(raw) != key: raise ContractError('v2 indexed bytes mismatch')
        if name in expected_inputs and raw != expected_inputs[name]:
            raise ContractError('v2 admitted input bytes mismatch')
        if name in OUTPUTS:
            if raw != read_bytes(root/name): raise ContractError('v2 root output copy mismatch')
            bound[name] = raw
    return bound


def commitments(root, names):
    root = safe_path(root)
    return {n: identity(read_bytes(root/n)) for n in names if (root/n).exists()}


def check_commitments(root, expected, names):
    if type(expected) is not dict or set(expected) != set(names):
        raise ContractError('v2 verification request has missing expected artifacts')
    for name in names:
        require_identity(expected[name])
        if identity(read_bytes(Path(root)/name)) != expected[name]:
            raise ContractError('v2 external artifact commitment mismatch: '+name)
