"""Pure-integer, frozen recursive 3x3x3 Menger-style reference geometry.

This is a geometric address space, not a learned activation projection or loss.
"""
from itertools import product
from pathlib import Path

from .contracts import ContractError, identity, integer, json_bytes, parse_json

PROTOCOL_IDENTITY = 'sha256:3fb56ed2e0ce54872b8e8e63e5e22d044913ff891c742379ba07ae1b3b21f4f0'
CHILDREN = tuple(c for c in product(range(3), repeat=3) if sum(d == 1 for d in c) <= 1)
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
SOURCE_FILES = ('src/trinite/contracts.py', 'src/trinite/lattice_scaffold.py',
                'src/trinite/lattice-scaffold-protocol.json', 'scripts/generate_lattice_scaffold.py')


def protocol():
    raw = (Path(__file__).parent / 'lattice-scaffold-protocol.json').read_bytes()
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError('lattice protocol changed; use a new protocol version')
    value = parse_json(raw, canonical=True)
    if (value['schema'] != 'trinite.lattice-scaffold-protocol.v1'
            or value['grid_base'] != 3 or value['depth_min'] != 0
            or value['depth_max'] != 3 or value['leaf_ceiling'] != 8000
            or len(CHILDREN) != 20):
        raise ContractError('unsupported lattice protocol')
    return value


def source_receipt():
    root = Path(__file__).resolve().parents[2]
    return {name: identity((root / name).read_bytes()) for name in SOURCE_FILES}


def construct(depth):
    """Return exact level ancestry and face adjacency; no randomness or floats."""
    rules = protocol()
    integer(depth, 'lattice depth', rules['depth_min'], rules['depth_max'])
    levels = []
    nodes = [{'coordinate': [0, 0, 0], 'parent_index': None, 'child_rank': None}]
    for level in range(depth + 1):
        if len(nodes) != 20 ** level or len(nodes) > rules['leaf_ceiling']:
            raise ContractError('lattice level exceeded frozen occupancy bound')
        levels.append({'depth': level, 'side': 3 ** level, 'count': len(nodes),
                       'occupancy_identity': identity(json_bytes([n['coordinate'] for n in nodes])),
                       'nodes': nodes})
        if level != depth:
            nodes = [{'coordinate': [3 * a + b for a, b in zip(node['coordinate'], child)],
                      'parent_index': index, 'child_rank': rank}
                     for index, node in enumerate(nodes) for rank, child in enumerate(CHILDREN)]
    leaves = levels[-1]['nodes']
    positions = {tuple(n['coordinate']): i for i, n in enumerate(leaves)}
    if len(positions) != len(leaves):
        raise ContractError('duplicate lattice leaf address')
    edges = []
    degrees = [0] * len(leaves)
    for index, node in enumerate(leaves):
        x, y, z = node['coordinate']
        for neighbor in ((x + 1, y, z), (x, y + 1, z), (x, y, z + 1)):
            other = positions.get(neighbor)
            if other is not None:
                edges.append([index, other])
                degrees[index] += 1
                degrees[other] += 1
    edges.sort()
    return {'schema': 'trinite.lattice-scaffold.v1', 'protocol_identity': PROTOCOL_IDENTITY,
            'source': source_receipt(), 'depth': depth, 'leaf_count': len(leaves),
            'side': 3 ** depth, 'levels': levels, 'edges': edges,
            'edge_identity': identity(json_bytes(edges)),
            'degree_histogram': {str(k): degrees.count(k) for k in range(7) if k in degrees},
            'geometry_loss': '0', 'neural_projections_fitted': False,
            'phase5_ready': False, 'learning_adequacy': False}


def verify(raw):
    """Read-only, byte-exact regeneration; rejects stale source/protocol and edits."""
    if type(raw) is not bytes or len(raw) > MAX_ARTIFACT_BYTES:
        raise ContractError('lattice artifact exceeds frozen byte budget')
    value = parse_json(raw, canonical=True)
    if type(value) is not dict or type(value.get('depth')) is not int:
        raise ContractError('invalid lattice artifact or depth')
    expected = json_bytes(construct(value['depth']))
    if raw != expected:
        raise ContractError('lattice artifact differs from fresh frozen construction')
    return {'schema': 'trinite.lattice-scaffold-verification.v1',
            'artifact_identity': identity(raw), 'protocol_identity': PROTOCOL_IDENTITY,
            'leaf_count': value['leaf_count'], 'edge_count': len(value['edges']),
            'outcome': 'passed', 'phase5_ready': False}
