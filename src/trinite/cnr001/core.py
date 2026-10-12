"""CNR-001: frozen, descriptive post-v0.1.0 topology comparisons.

Exact trained native baseline states and scorer are reused without model calls.
MaleCNS source Parquet is separately admitted and hashed; PyArrow is only a
bounded input decoder and never a numerical learning backend.
"""
from collections import deque
import hashlib
import json
from pathlib import Path

from ..contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json
from ..e8_reference import etq
from ..lattice.scaffold import construct
from ..lattice_observation import experiment as baseline
from ..reference_geometry import decode_vector

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "fixtures/cnr-001-source"
PROTOCOL_PATH = Path(__file__).parent / "protocol.json"
BASELINE_SHA = "a1cd03ba8d29f335bea29179663931bbd9c35927"
PROTOCOL_GIT_BLOB = "36a63b401a06811f669f538c343cef572f98c402"
SOURCE_PATHS = (
    "src/trinite/__init__.py", "src/trinite/contracts.py",
    "src/trinite/cnr001/__init__.py", "src/trinite/cnr001/core.py",
    "src/trinite/cnr001/protocol.json", "scripts/run_cnr001.py",
    "src/trinite/lattice_observation/experiment.py",
    "src/trinite/lattice/scaffold.py", "src/trinite/e8_reference/etq.py",
    "src/trinite/reference_geometry.py", "src/trinite/reference_evidence.py",
    "src/trinite/reference_inputs.py", "src/trinite/lattice_observation/protocol.json",
    "src/trinite/lattice/protocol.json", "fixtures/cnr-001-source/metadata.json",
    "requirements.connectome.lock",
)
BOUND = 2 * 1024 * 1024
ARCHIVE = "sha256:666e8f4d5b49da60aada4a26acbfa6001a5a8b8b62447c1ddfd3a5613cfd8721"


def safe_output(path):
    path = Path(path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ContractError("unsafe CNR output path")
    return path


def read(path, limit=BOUND):
    path = Path(path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ContractError("unsafe CNR path")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ContractError("CNR input exceeds bound")
    return raw


def protocol():
    raw = read(PROTOCOL_PATH, 8192)
    if git_blob(raw) != PROTOCOL_GIT_BLOB:
        raise ContractError("CNR protocol differs from preregistered producer")
    p = parse_json(raw, canonical=True)
    if (p["schema"] != "trinite.cnr-001-protocol.v1"
            or p["baseline_commit"] != BASELINE_SHA
            or p["baseline_reference_archive_sha256"] != ARCHIVE
            or p["seed"] != 29 or p["receiver_nodes"] != 20
            or p["edge_budget"] != 24 or p["geometry_loss"] != "0"
            or p["phase5_ready"] is not False or p["release_ready"] is not False
            or p["comparison_arms"] != [
                "baseline-menger", "male-cns20", "etq101-20", "etq303-20",
                "male-cns20-degree-rewired", "etq101-20-degree-rewired",
                "etq303-20-degree-rewired", "hash-random20"]):
        raise ContractError("unsupported frozen CNR-001 protocol")
    return p, identity(raw)


def source_receipt():
    return {name: identity(read(ROOT / name)) for name in SOURCE_PATHS}


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw).hexdigest()


def source_data():
    p, _ = protocol()
    result = {}
    for name, expected in p["source_git_blobs"].items():
        raw = read(DATA / name)
        if git_blob(raw) != expected:
            raise ContractError("MaleCNS source blob changed: " + name)
        result[name] = raw
    # Third-party JSON includes a finite min_confidence decimal (0.5).
    # The upstream bytes are already pinned by Git blob. Preserve decimals as
    # exact strings rather than weakening Trinite's integer-only JSON contract.
    def unique_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ContractError("duplicate upstream metadata key")
            out[key] = value
        return out
    try:
        meta = json.loads(result["metadata.json"].decode("utf-8"),
                          object_pairs_hook=unique_pairs,
                          parse_float=lambda value: value,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ContractError("nonfinite upstream metadata")))
    except (ValueError, UnicodeDecodeError) as error:
        raise ContractError("invalid upstream MaleCNS metadata") from error
    if (meta["schema_version"] != 1 or meta["num_nodes"] != 100
            or meta["num_edges"] != 3630
            or meta["provenance"]["dataset"] != "male-cns:v1.0"
            or meta["provenance"]["license"] != "CC-BY-4.0"
            or meta["provenance"]["filters"]["min_synapses"] != 5
            or meta["provenance"]["filters"]["min_confidence"] != "0.5"):
        raise ContractError("foreign MaleCNS source metadata")
    return result, meta


def biological_graph():
    raw, meta = source_data()
    try:
        import importlib.metadata
        if importlib.metadata.version("pyarrow") != "21.0.0":
            raise ContractError("CNR Parquet reader requires pinned pyarrow 21.0.0")
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError as error:
        raise ContractError("CNR biological ingestion requires separately hash-locked PyArrow") from error
    import io
    nodes = pq.read_table(pa.BufferReader(raw["nodes.parquet"]))
    edges = pq.read_table(pa.BufferReader(raw["edges.parquet"]))
    if ("node_id" not in nodes.column_names or
            not {"source_index", "target_index", "synapse_count"} <= set(edges.column_names)):
        raise ContractError("unexpected MaleCNS parquet schema")
    identifiers = nodes.column("node_id").to_pylist()
    s, t, w = (edges.column(key).to_pylist() for key in ("source_index", "target_index", "synapse_count"))
    if len(identifiers) != 100 or len(set(identifiers)) != 100 or len(s) != 3630:
        raise ContractError("changed MaleCNS body/connection counts")
    if any(type(n) is not int or n <= 0 for n in identifiers):
        raise ContractError("invalid biological body identity")
    if any(type(a) is not int or type(b) is not int or not (0 <= a < 100 and 0 <= b < 100)
           or a == b or type(v) is not int or v < 5 for a, b, v in zip(s, t, w)):
        raise ContractError("invalid MaleCNS directed synapse graph")
    if len({(a, b) for a, b in zip(s, t)}) != 3630:
        raise ContractError("duplicate biological connection")
    return identifiers, list(zip(s, t, w)), meta


def symmetric_weights(size, weighted_edges):
    pairs = {}
    for u, v, weight in weighted_edges:
        if type(u) is not int or type(v) is not int or not (0 <= u < size and 0 <= v < size):
            raise ContractError("invalid source graph edge")
        if u == v or type(weight) is not int or weight <= 0:
            raise ContractError("invalid source graph weight")
        key = (min(u, v), max(u, v))
        pairs[key] = pairs.get(key, 0) + weight
    return pairs


def select20(size, pairs, ids):
    if len(ids) != size or size < 20:
        raise ContractError("graph cannot supply twenty nodes")
    weighted_degree = [0] * size
    degree = [0] * size
    for (a, b), weight in pairs.items():
        weighted_degree[a] += weight; weighted_degree[b] += weight
        degree[a] += 1; degree[b] += 1
    first = min(range(size), key=lambda k: (-weighted_degree[k], ids[k]))
    selected = [first]
    while len(selected) < 20:
        current = set(selected)
        def ranking(i):
            into = sum(weight for (a, b), weight in pairs.items()
                       if (a == i and b in current) or (b == i and a in current))
            return (-into, -weighted_degree[i], ids[i])
        nxt = min((i for i in range(size) if i not in current), key=ranking)
        selected.append(nxt)
    return selected


def reduce_graph(size, pairs, ids):
    selected = select20(size, pairs, ids)
    mapping = {old: new for new, old in enumerate(selected)}
    local = [(weight, mapping[a], mapping[b])
             for (a, b), weight in pairs.items() if a in mapping and b in mapping]
    local.sort(key=lambda x: (-x[0], min(x[1], x[2]), max(x[1], x[2])))
    if len(local) < 24:
        raise ContractError("cannot match fixed 24-edge control budget")
    edges = sorted([sorted([a, b]) for _, a, b in local[:24]])
    return {"selected_source_ids": [ids[i] for i in selected],
            "edges": edges, "available_induced_edges": len(local),
            "retained_pair_weights": [w for w, _, _ in local[:24]]}


def etq_graphs():
    base = etq.base_adjacency()
    base_pairs = {(a, b): 1 for a in range(101) for b in range(a+1, 101) if base[a][b]}
    lifted = etq.lifted_graph()
    if (len(base_pairs) != 1687 or lifted["edgeCount"] != 5364
            or lifted["vertexCount"] != 303 or lifted["connected"] is not True):
        raise ContractError("pinned ETQ graph cardinality changed")
    lifted_pairs = {(edge["source"], edge["target"]): 1 for edge in lifted["edges"]}
    return reduce_graph(101, base_pairs, list(range(101))), \
           reduce_graph(303, lifted_pairs, list(range(303))), lifted


def rewired(edges, seed=29):
    """Deterministic simple-graph degree-preserving swaps with exact receipts."""
    original = {tuple(x) for x in edges}
    selected = set(original)
    swaps = 0
    for k in range(120):
        order = sorted(selected, key=lambda e: identity(json_bytes({"seed": seed, "pass": k, "edge": list(e)})))
        if len(order) < 2: break
        a, b = order[0], order[1]
        # a,b each ordered u<v; compare both symmetric partner swaps.
        candidate = sorted([tuple(sorted((a[0], b[1]))), tuple(sorted((b[0], a[1])))])
        if (all(x[0] != x[1] for x in candidate)
                and len(set(candidate)) == 2
                and not any(x in selected - {a, b} for x in candidate)):
            selected.difference_update((a, b))
            selected.update(candidate); swaps += 1
    result = sorted([list(x) for x in selected])
    if len(result) != len(edges) or degrees(result) != degrees(edges):
        raise ContractError("rewiring broke edge count or degree multiset")
    return result, swaps


def degrees(edges):
    out = [0] * 20
    for a, b in edges:
        out[a] += 1; out[b] += 1
    return out


def random_edges(seed=29):
    pairs = [[a, b] for a in range(20) for b in range(a+1, 20)]
    pairs.sort(key=lambda x: identity(json_bytes({"seed": seed, "pair": x})))
    return sorted(pairs[:24])


def etq_diagnostics(lifted):
    neighbours = [[] for _ in range(303)]
    for edge in lifted["edges"]:
        a, b = edge["source"], edge["target"]
        neighbours[a].append(b); neighbours[b].append(a)
    if any(not ns for ns in neighbours):
        raise ContractError("ETQ lifted graph has isolated states")
    event_non_edges = 0
    for n in range(303):
        j, a = etq.event_address(n)
        j2, a2, _ = etq.monomial_power(j, a, 1)
        if etq.tensor_index(j2, a2) not in neighbours[etq.tensor_index(j, a)]:
            event_non_edges += 1
    position, walk = 0, [0]
    for step in range(303):
        choices = sorted(neighbours[position], key=lambda x: identity(
            json_bytes({"seed": 29, "step": step, "source": position, "target": x})))
        position = choices[0]
        walk.append(position)
    if any(y not in neighbours[x] for x, y in zip(walk, walk[1:])):
        raise ContractError("ETQ graph walk traversed a nonedge")
    if etq.monomial_power(0, 0, 303) != (0, 0, 0):
        raise ContractError("ETQ 303 monomial cycle changed")
    return {"full_graph_nodes": 303, "full_graph_edges": len(lifted["edges"]),
            "event_cycle_steps": 303, "event_transition_nonedges": event_non_edges,
            "event_cycle_returns_to_origin": True, "event_phase_exponent_mod4": 0,
            "genuine_graph_walk_steps": 303, "genuine_graph_walk_visited": len(set(walk)),
            "genuine_graph_walk_ends_at": walk[-1],
            "separate_event_and_walk": True}


def candidate_graphs(bio=None):
    ids, directed, meta = biological_graph() if bio is None else bio
    if len(ids) != 100 or len(directed) != 3630:
        raise ContractError("CNR real sample count required")
    male = reduce_graph(100, symmetric_weights(100, directed), ids)
    base, lifted20, full = etq_graphs()
    control = construct(1)
    original = control["edges"]
    if len(original) != 24:
        raise ContractError("v0.1.0 lattice edge count changed")
    graphs = {"baseline-menger": original, "male-cns20": male["edges"],
              "etq101-20": base["edges"], "etq303-20": lifted20["edges"]}
    swaps = {}
    for arm in ("male-cns20", "etq101-20", "etq303-20"):
        edges, count = rewired(graphs[arm])
        graphs[arm + "-degree-rewired"] = edges
        swaps[arm] = count
    graphs["hash-random20"] = random_edges()
    for name, edges in graphs.items():
        if len(edges) != 24 or len({tuple(pair) for pair in edges}) != 24:
            raise ContractError("CNR graph budget mismatch: " + name)
        if any(type(a) is not int or type(b) is not int or not 0 <= a < b < 20 for a,b in edges):
            raise ContractError("CNR invalid simple control graph")
    return graphs, {"male_cns": male, "etq101": base, "etq303": lifted20,
                    "rewire_swaps": swaps, "etq_diagnostics": etq_diagnostics(full),
                    "biological_provenance": meta["provenance"]}


def baseline_vectors():
    original_request, records, raw_request, raw_capture = baseline.inputs()
    if identity(raw_request) != baseline.protocol()["original_reference_request_identity"]:
        raise ContractError("v0.1.0 request identity mismatch")
    if baseline.ARCHIVE_IDENTITY != ARCHIVE:
        raise ContractError("historical capture archive mismatch")
    layers = {}
    for layer in ("block.0", "final"):
        vectors = [decode_vector(row["anchors"][-1]["vectors"][layer]) for row in records["rows"]]
        normalized, normalization = baseline.signed_projection(vectors, 29)
        layers[layer] = (vectors, normalized, normalization)
    return layers, {"request_identity":identity(raw_request),
                    "capture_identity":identity(raw_capture),
                    "original_baseline_commit":BASELINE_SHA}


def analyze(bio=None):
    protocol_record, protocol_sha = protocol()
    graphs, provenance = candidate_graphs(bio)
    layers, baseline_input = baseline_vectors()
    positions = [tuple(n["coordinate"]) for n in construct(1)["levels"][-1]["nodes"]]
    scores = {}
    for layer, (vectors, normalized, fit) in layers.items():
        arms = {}
        for name in protocol_record["comparison_arms"]:
            e = graphs[name]
            metrics = baseline.score(vectors, normalized, positions, e)
            arms[name] = {**metrics, "edge_identity": identity(json_bytes(e)),
                          "edge_count":len(e), "degree_sequence":degrees(e),
                          "receiver_identity":identity(json_bytes([list(p) for p in positions]))}
        # Recompute v0.1.0 baseline using exactly its unchanged scoring owner.
        reference_graph = baseline.conditions()["menger"][1]
        if arms["baseline-menger"]["graph_hops_sum"] != baseline.score(
                vectors, normalized, positions, reference_graph)["graph_hops_sum"]:
            raise ContractError("baseline metric is not reproduced")
        scores[layer] = {"baseline_from_v0_1_0": arms["baseline-menger"],
                         "arms": arms, "baseline_calibration": fit,
                         "no_unseen_model_evaluation": True}
    return {"schema":"trinite.cnr-001-analysis.v1",
            "protocol_identity": protocol_sha,"baseline":baseline_input,
            "biological_data_source_git_blobs":protocol_record["source_git_blobs"],
            "source":source_receipt(),"graph_provenance":provenance,"layers":scores,
            "comparability":{"receiver_nodes":20,"receiver_coordinates":"identical",
                             "graph_edges":24,"graph_reduction":"experimental, not full connectomes",
                             "biological_direction_ignored_for_primary":True,
                             "native_training_probes_exposed":True,
                             "no_historical_cnr_outcome_in_v0_1_0":True},
            "geometry_loss":"0","phase5_ready":False,"release_ready":False,
            "claim":"retrospective graph-topology diagnostic; no causal improvement, neural learning or biological homology"}
