"""Read-only, source-bound observation of previously retained trained activations."""
from collections import Counter, deque
import hashlib
import io
import math
from pathlib import Path
import zipfile

from ..contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json
from ..lattice.scaffold import construct, source_receipt as scaffold_source_receipt
from ..reference_evidence import audit
from ..reference_geometry import decode_vector

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_IDENTITY = 'sha256:5c92baba23f0c2fa41a61d65ea5e74d088262608907127b2318351dc0cd1f59b'
ARCHIVE_IDENTITY = 'sha256:666e8f4d5b49da60aada4a26acbfa6001a5a8b8b62447c1ddfd3a5613cfd8721'
PACKET = 'fixtures/reference-geometry-v1/Trinite-reference-geometry-v1.zip.part000'
SOURCES = ('src/trinite/contracts.py', 'src/trinite/reference_evidence.py',
           'src/trinite/reference_geometry.py', 'src/trinite/reference_inputs.py',
           'src/trinite/lattice_observation/__init__.py',
           'src/trinite/lattice_observation/experiment.py',
           'src/trinite/lattice_observation/protocol.json',
           'scripts/run_lattice_observation.py')
MAX_INPUT = 2 * 1024 * 1024


def safe(path):
    p = Path(path)
    if any(x.is_symlink() for x in (p, *p.parents)):
        raise ContractError('observation refuses symlink traversal')
    return p


def bounded(path, limit=MAX_INPUT):
    with safe(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ContractError('observation input byte budget exceeded')
    return raw


def protocol():
    raw = bounded(Path(__file__).parent / 'protocol.json', 16384)
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError('observation protocol differs from frozen v1')
    r = parse_json(raw, canonical=True)
    if (r['schema'] != 'trinite.activation-lattice-protocol.v1'
            or r['controls'] != ['menger', 'nonrecursive', 'shuffled-topology', 'random-anchors']
            or r['calibration_indices'] != list(range(8))
            or r['evaluation_indices'] != list(range(8, 16))
            or r['capture_layer_order'] != ['block.0', 'final']
            or r['anchor_depth'] != 1 or r['anchor_count'] != 20
            or r['control_seed'] != 29 or r['geometry_loss'] != '0'
            or r['no_training'] is not True or r['phase5_ready'] is not False):
        raise ContractError('observation protocol has unsupported semantics')
    return r


def source_receipt():
    return {name: identity(bounded(ROOT / name, 256 * 1024)) for name in SOURCES}


def packet():
    raw = bounded(ROOT / PACKET)
    if identity(raw) != ARCHIVE_IDENTITY:
        raise ContractError('historical reference packet identity changed')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        wanted = ('run/request.json', 'run/native/capture.json')
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise ContractError('duplicate archived names')
        try:
            if any(archive.getinfo(name).file_size > MAX_INPUT for name in wanted):
                raise ContractError('oversized archived observation input')
            request, capture = (archive.read(name) for name in wanted)
        except KeyError as error:
            raise ContractError('historical packet lacks native capture') from error
    if len(request) > MAX_INPUT or len(capture) > MAX_INPUT:
        raise ContractError('observation archive member exceeded limit')
    return request, capture


def inputs():
    rules = protocol()
    reference, capture = packet()
    if identity(reference) != rules['original_reference_request_identity']:
        raise ContractError('unexpected historical research request')
    req = parse_json(reference, canonical=True)
    r = audit(capture, req, 'native')  # actual original capture auditor, no fakes
    if (len(r['rows']) != rules['capture_rows']
            or r['request_identity'] != identity(reference)
            or len({row['example_identity'] for row in r['rows']}) != 16
            or len({req['probes'][i]['family_id'] for i in range(8)}) != 1
            or len({req['probes'][i]['family_id'] for i in range(8, 16)}) != 1
            or req['probes'][0]['family_id'] == req['probes'][8]['family_id']):
        raise ContractError('probe split is not family-isolated or complete')
    return req, r, reference, capture


def core(reference, capture):
    return {'schema': 'trinite.activation-lattice-request.v1',
            'protocol': protocol(), 'protocol_identity': PROTOCOL_IDENTITY,
            'source': source_receipt(), 'scaffold_source': source_receipt_scaffold(),
            'scaffold_identity': identity(json_bytes(construct(1))),
            'reference_request_identity': identity(reference),
            'native_capture_identity': identity(capture),
            'historical_packet_identity': ARCHIVE_IDENTITY,
            'capture_status': 'previously published trained native arithmetic; training-exposed probes'}


def source_receipt_scaffold():
    return scaffold_source_receipt()


def freeze(path):
    _, _, raw_req, capture = inputs()
    raw = json_bytes(core(raw_req, capture))
    with safe(path).open('xb') as stream:
        stream.write(raw)
    return {'request_identity': identity(raw), 'outcome': 'frozen', 'research_class': 'exploratory'}


def request_at(path, expected):
    raw = bounded(path, 256 * 1024)
    if type(expected) is not str or identity(raw) != expected:
        raise ContractError('observation request external identity mismatch')
    _, _, ref, capture = inputs()
    value = parse_json(raw, canonical=True)
    if raw != json_bytes(core(ref, capture)):
        raise ContractError('observation source, protocol or historical input differs')
    return value


def ranked_grid(seed):
    items = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]
    if seed is None:
        return items[:20]
    return sorted(items, key=lambda c: identity(json_bytes({'seed': seed, 'anchor': list(c)})))[:20]


def graph_with_budget(coords, edges):
    pairs = [(sum((a-b)**2 for a, b in zip(coords[i], coords[j])), i, j)
             for i in range(20) for j in range(i + 1, 20)]
    return [[i, j] for _, i, j in sorted(pairs)[:edges]]


def conditions():
    lattice = construct(1)
    coords = [tuple(n['coordinate']) for n in lattice['levels'][-1]['nodes']]
    base_edges = lattice['edges']
    n, m = len(coords), len(base_edges)
    if n != 20 or not (0 < m < 190):
        raise ContractError('unexpected fixed depth-one lattice')
    order = sorted(range(n), key=lambda i: identity(json_bytes({'seed': 29, 'node': i})))
    shuffle = [sorted([order[a], order[b]]) for a, b in base_edges]
    records = {'menger': (coords, base_edges),
               'nonrecursive': (ranked_grid(None), graph_with_budget(ranked_grid(None), m)),
               'shuffled-topology': (coords, sorted(shuffle)),
               'random-anchors': (ranked_grid(29), graph_with_budget(ranked_grid(29), m))}
    for name, (points, graph) in records.items():
        if len(points) != 20 or len(set(points)) != 20 or len(graph) != m or len({tuple(x) for x in graph}) != m:
            raise ContractError('matched-control graph budget changed: ' + name)
    return records


def signed_projection(rows, seed):
    if len(rows) != 16 or len({len(row) for row in rows}) != 1:
        raise ContractError('capture projection requires 16 same-width vectors')
    width = len(rows[0])
    if width != 64 or any(type(x) is not float or not math.isfinite(x) for r in rows for x in r):
        raise ContractError('native trained capture requires finite width 64')
    signs = [[1 if hashlib.sha256(json_bytes({'seed': seed, 'axis': a, 'dimension': j})).digest()[0] & 1
              else -1 for j in range(width)] for a in range(3)]
    projected = [tuple(math.fsum(s * v for s, v in zip(sign, row)) / math.sqrt(width)
                       for sign in signs) for row in rows]
    calibration = projected[:8]
    extrema = [(min(row[a] for row in calibration), max(row[a] for row in calibration)) for a in range(3)]
    normalized = []
    for row in projected:
        coords = []
        for axis, x in enumerate(row):
            lo, hi = extrema[axis]
            if hi == lo:
                coords.append(1.0)
            else:
                v = 2 * (x - lo) / (hi - lo)
                coords.append(max(0.0, min(2.0, v)))
        normalized.append(tuple(coords))
    return normalized, {'axis_range_hex': [[a.hex(), b.hex()] for a, b in extrema],
                        'degenerate_axes': [i for i, (a, b) in enumerate(extrema) if a == b],
                        'evaluation_clipped_axes': sum(
                            int(2 * (row[i] - lo) / (hi - lo) < 0 or 2 * (row[i] - lo) / (hi - lo) > 2)
                            for row in projected[8:] for i, (lo, hi) in enumerate(extrema) if hi != lo)}


def anchor_map(points, anchors):
    return [min(range(len(anchors)), key=lambda i: (
        math.fsum((x - y)**2 for x, y in zip(row, anchors[i])), i)) for row in points]


def hops(edges, count=20):
    neighbors = [[] for _ in range(count)]
    for i, j in edges:
        neighbors[i].append(j); neighbors[j].append(i)
    result = []
    for source in range(count):
        distances = [count + 1] * count
        distances[source] = 0
        todo = deque([source])
        while todo:
            u = todo.popleft()
            for v in neighbors[u]:
                if distances[v] == count + 1:
                    distances[v] = distances[u] + 1
                    todo.append(v)
        result.append(distances)
    return result


def score(rows, normalized, anchors, edges):
    subset = rows[8:]
    mapped = anchor_map(normalized[8:], anchors)
    graph = hops(edges)
    nearest = [min((j for j in range(8) if j != i),
                   key=lambda j: (math.fsum((x-y)**2 for x, y in zip(subset[i], subset[j])), j))
               for i in range(8)]
    distances = [graph[mapped[i]][mapped[j]] for i, j in enumerate(nearest)]
    return {'evaluation_count': 8, 'mapped_anchor_indices': mapped,
            'raw_nearest_neighbor_indices': nearest, 'graph_hops': distances,
            'graph_hops_sum': sum(distances),
            'same_or_adjacent': sum(d <= 1 for d in distances),
            'unique_anchors': len(set(mapped)),
            'anchor_collisions': 8-len(set(mapped)),
            'unreachable': sum(d > 20 for d in distances)}


def analyze(request_path, expected_identity):
    request_at(request_path, expected_identity)
    req, capture, _, raw = inputs()
    records = conditions()
    result = {}
    for layer in protocol()['capture_layer_order']:
        values = [decode_vector(r['anchors'][-1]['vectors'][layer]) for r in capture['rows']]
        normalized, fit = signed_projection(values, protocol()['control_seed'])
        arms = {}
        for name, (anchors, edges) in records.items():
            arms[name] = {**score(values, normalized, anchors, edges),
                          'anchor_identity': identity(json_bytes(anchors)),
                          'edge_identity': identity(json_bytes(edges)),
                          'anchor_count': len(anchors), 'edge_count': len(edges)}
        result[layer] = {'calibration': fit, 'arms': arms,
                         'capture_vector_identities': [
                             r['anchors'][-1]['vectors'][layer]['bytes_identity'] for r in capture['rows']]}
    return {'schema': 'trinite.activation-lattice-analysis.v1',
            'request_identity': expected_identity, 'native_capture_identity': identity(raw),
            'historical_reference_identity': protocol()['original_reference_request_identity'],
            'research_class': 'exploratory-retrospective-training-exposed',
            'selection': {'calibration_indices': list(range(8)), 'evaluation_indices': list(range(8, 16)),
                          'model_held_out': False, 'probe_family_isolation': True,
                          'layer': protocol()['capture_layer_order']},
            'layers': result, 'training_or_fitting_of_model': False,
            'geometry_loss': '0', 'held_out_learning_adequacy': False, 'phase5_ready': False,
            'claim': 'descriptive quantization/topology diagnostic, not improved reasoning or causal evidence'}


def run(request, request_identity, output):
    raw = json_bytes(analyze(request, request_identity))
    with safe(output).open('xb') as stream:
        stream.write(raw)
    return {'schema': 'trinite.activation-lattice-run.v1', 'artifact_identity': identity(raw),
            'outcome': 'completed', 'phase5_ready': False}


def verify(request, request_identity, output, artifact_identity):
    raw = bounded(output)
    if identity(raw) != artifact_identity:
        raise ContractError('external analysis artifact commitment mismatch')
    if raw != json_bytes(analyze(request, request_identity)):
        raise ContractError('observation analysis not reproducible from bound evidence')
    return {'schema': 'trinite.activation-lattice-verification.v1', 'outcome': 'passed',
            'analysis_identity': artifact_identity, 'request_identity': request_identity,
            'read_only': True, 'phase5_ready': False}
