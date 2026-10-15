"""Non-displacing, historical-ranked eight-plus-eight paired input schedule.

This is a data-only composition of two consecutive supported native updates.
It changes neither the authoritative optimizer nor per-update token-normalized
loss. Removing every additional batch recovers the original historical batch
in exact slot order; intervening updates do not preserve old model trajectories.
"""
from collections import Counter, defaultdict
import hashlib

from ..contracts import ContractError, identity, integer, json_bytes
from .inputs import MAX_ROUNDS, normalized, revalidate

ARMS = ("extra-final", "extra-procedural")
SEEDS = (0, 1)
BATCH_SIZE = 8
GROUPS = (("arithmetic", "final"), ("fold-sum", "final"), ("gf3", "final"),
          ("fold-sum", "prefix"), ("fold-sum", "trace"),
          ("gf3", "prefix"), ("gf3", "trace"))


def _cell(seed, arm, macro_rounds):
    integer(seed, "auxiliary seed", 0, 1)
    integer(macro_rounds, "auxiliary macro-round budget", 1, MAX_ROUNDS)
    if type(arm) is not str or arm not in ARMS:
        raise ContractError("unsupported non-displacing auxiliary arm")


def auxiliary_stream(macro_round):
    integer(macro_round, "auxiliary macro-round", 0, MAX_ROUNDS - 1)
    return ("fold-sum" if macro_round % 2 == 0 else "gf3",
            "prefix" if (macro_round // 2) % 2 == 0 else "trace")


def advance_batch(groups, counters, key):
    """Advance only the selected stream by eight, including across wraps.

    This pure indexing primitive does not admit rows. The public scheduler
    validates source-bound materialization before constructing its groups.
    """
    if (type(groups) is not dict or type(counters) is not dict
            or type(key) is not tuple or len(key) != 2
            or any(type(part) is not str for part in key) or key not in GROUPS
            or key not in groups or type(groups[key]) is not list or not groups[key]):
        raise ContractError("known nonempty auxiliary stream and cursor map required")
    cursor = counters.get(key, 0)
    integer(cursor, "auxiliary stream cursor", 0, MAX_ROUNDS * BATCH_SIZE - BATCH_SIZE)
    group = groups[key]
    selected = [group[(cursor + offset) % len(group)] for offset in range(BATCH_SIZE)]
    counters[key] = cursor + BATCH_SIZE
    return selected


def _groups(inputs, seed, items):
    original = inputs.historical_items
    mapping = inputs.receipt["mapping"]
    prospective = {entry["historical_identity"]: items[entry["prospective_index"]]
                   for entry in mapping}
    result = {}
    for key in GROUPS:
        selected = [row for row in original if (row["kind"], row["mode"]) == key]
        if not selected:
            raise ContractError("empty historical stream")
        selected.sort(key=lambda row: identity(json_bytes(
            {"seed": seed, "item": row["example_identity"]})))
        result[key] = [prospective[row["example_identity"]] for row in selected]
    return result


def _schedule(inputs, seed, arm, macro_rounds):
    """Internal composition after the public boundary has revalidated inputs."""
    items = inputs.items
    groups = _groups(inputs, seed, items)
    primary = inputs.primary_indices(seed)
    counters = {}
    selected = []
    for macro in range(macro_rounds):
        selected.extend(items[index] for index in primary[macro * 8:(macro + 1) * 8])
        kind, mode = auxiliary_stream(macro)
        key = (kind, "final") if arm == "extra-final" else (kind, mode)
        selected.extend(advance_batch(groups, counters, key))
    if len(selected) != macro_rounds * 16:
        raise ContractError("non-displacing schedule must retain two eight-row updates")
    return selected


def schedule(inputs, seed, arm, macro_rounds=MAX_ROUNDS):
    """Flat 8-row primary / 8-row additional updates, each admitted and train-only."""
    _cell(seed, arm, macro_rounds)
    revalidate(inputs)
    return _schedule(inputs, seed, arm, macro_rounds)


def _sequence_identity(encoded):
    """Exact canonical JSON-list encoding without allocating repeated full rows."""
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, raw in enumerate(encoded):
        if index:
            digest.update(b",")
        digest.update(raw)
    digest.update(b"]\n")
    return "sha256:" + digest.hexdigest()


def _count(rows, originals, normalized_bytes):
    groups = defaultdict(lambda: {"visits": 0, "target_tokens": 0, "input_tokens": 0})
    for row in rows:
        length = len(row["text"].encode("utf-8")) + 2
        if row["split"] != "train" or length > 64:
            raise ContractError("nontraining or overlength auxiliary row")
        count = groups[row["kind"] + "/" + row["mode"]]
        count["visits"] += 1
        count["target_tokens"] += len(row["answer"].encode("utf-8")) + 1
        count["input_tokens"] += length
    return {"by_kind_and_mode": dict(groups),
            "totals": {key: sum(group[key] for group in groups.values())
                       for key in ("visits", "target_tokens", "input_tokens")},
            "original_row_sequence_identity": _sequence_identity(
                json_bytes(originals[row["example_identity"]])[:-1] for row in rows),
            "prospective_row_sequence_identity": _sequence_identity(
                json_bytes(row["example_identity"])[:-1] for row in rows),
            "normalized_row_sequence_identity": _sequence_identity(
                normalized_bytes[row["example_identity"]] for row in rows)}


def schedule_receipt(inputs, seed, arm, macro_rounds=MAX_ROUNDS):
    """Actual ordered-input commitments and target/exposure counts; no learning."""
    _cell(seed, arm, macro_rounds)
    revalidate(inputs)
    rows = _schedule(inputs, seed, arm, macro_rounds)
    originals = {entry["prospective_identity"]: entry["historical_identity"]
                 for entry in inputs.receipt["mapping"]}
    normalized_bytes = {row["example_identity"]: json_bytes(normalized(row))[:-1]
                        for row in inputs.items}
    primary = [row for start in range(0, len(rows), 16) for row in rows[start:start + 8]]
    extra = [row for start in range(0, len(rows), 16) for row in rows[start + 8:start + 16]]
    primary_count = _count(primary, originals, normalized_bytes)
    extra_count = _count(extra, originals, normalized_bytes)
    combined = _count(rows, originals, normalized_bytes)
    full_identity = inputs.receipt["historical_primary_sequence_identities"][str(seed)]
    selected_identity = identity(json_bytes([[row["prompt"], row["answer"]] for row in primary]))
    if macro_rounds == MAX_ROUNDS and selected_identity != full_identity:
        raise ContractError("historical primary sequence was displaced or reordered")
    if combined["totals"]["target_tokens"] > 1048576:
        raise ContractError("auxiliary scored-target bound exceeded")
    return {"schema": "trinite.procedural-qec-auxiliary-schedule.v1",
            "seed": seed, "arm": arm, "macro_rounds": macro_rounds,
            "input_mapping_identity": inputs.content_identity,
            "native_updates": macro_rounds * 2,
            "primary": primary_count, "extra": extra_count, "combined": combined,
            "original_audit_primary_sequence_identity": full_identity,
            "selected_primary_sequence_identity": selected_identity,
            "schedule_identity": combined["prospective_row_sequence_identity"],
            "padded_encoded_positions": macro_rounds * 2 * 8 * 64,
            "model_input_positions": macro_rounds * 2 * 8 * 63,
            "within_existing_target_ceiling": True}


def coverage(inputs, scheduled, arm):
    """Unique training exposure, separate from scheduled repeated visits."""
    if type(arm) is not str or arm not in ARMS:
        raise ContractError("invalid auxiliary exposure arm")
    revalidate(inputs)
    if type(scheduled) is not list or not scheduled or len(scheduled) > MAX_ROUNDS * 16:
        raise ContractError("bounded auxiliary schedule required")
    items = inputs.items
    admitted = {row["example_identity"]: json_bytes(row) for row in items}
    groups = {key: {row["example_identity"] for row in items
                    if (row["kind"], row["mode"]) == key} for key in GROUPS}
    used = {key: Counter() for key in GROUPS}
    for row in scheduled:
        if (type(row) is not dict or row.get("example_identity") not in admitted
                or json_bytes(row) != admitted[row["example_identity"]]):
            raise ContractError("scheduled row is altered or unadmitted")
        used[row["kind"], row["mode"]][row["example_identity"]] += 1
    result = {}
    for kind, mode in GROUPS:
        available = groups[kind, mode]
        visited = set(used[kind, mode])
        visits = sum(used[kind, mode].values())
        if arm == "extra-procedural" and mode != "final" and len(visited) != min(visits, len(available)):
            raise ContractError("independent auxiliary cursor excludes reachable rows")
        if arm == "extra-final" and mode != "final" and visits:
            raise ContractError("procedural rows cannot enter extra-final control")
        result[kind + "/" + mode] = {
            "available": len(available), "visits": visits,
            "distinct_visited": len(visited), "distinct_unvisited": len(available - visited),
            "visited_identity": identity(json_bytes(sorted(visited))),
            "unvisited_identity": identity(json_bytes(sorted(available - visited)))}
    return result
