"""v3 independently advancing QEC auxiliary streams. Pure integer scheduling.

Historical v2 scheduler and all its evidence are intentionally untouched.
"""
from ..contracts import ContractError, identity, integer, json_bytes

KINDS = ("fold-sum", "gf3")
MODES = ("prefix", "trace")
ARMS = ("final-only", "procedural")
GROUPS = (("arithmetic", "final"), ("fold-sum", "final"),
          ("gf3", "final"), ("fold-sum", "prefix"),
          ("fold-sum", "trace"), ("gf3", "prefix"), ("gf3", "trace"))


def ranked(items, seed):
    integer(seed, "procedural v3 seed", 0, 2**32 - 1)
    if type(items) is not list or not items:
        raise ContractError("empty procedural v3 stream")
    if len({e["example_identity"] for e in items}) != len(items):
        raise ContractError("duplicate procedure stream identities")
    return sorted(items, key=lambda e: identity(json_bytes(
        {"seed": seed, "item": e["example_identity"]})))


def auxiliary_stream(step):
    integer(step, "procedural v3 update", 0, 8191)
    return ("fold-sum" if step % 2 == 0 else "gf3",
            "prefix" if (step // 2) % 2 == 0 else "trace")


def advance(groups, counters, kind, mode):
    """Select the next row only on a visit to this EXACT stream."""
    key = kind, mode
    if key not in groups or type(groups[key]) is not list or not groups[key]:
        raise ContractError("unknown or empty procedural auxiliary stream")
    n = counters.get(key, 0)
    integer(n, "procedural stream cursor", 0, 2**32 - 1)
    counters[key] = n + 1
    return groups[key][n % len(groups[key])]


def schedule(items, seed, arm, steps):
    """One fixed 8-row batch/update; only v3 auxiliary row selection differs."""
    integer(steps, "v3 update budget", 1, 8192)
    if type(arm) is not str or arm not in ARMS:
        raise ContractError("unsupported procedural v3 arm")
    if type(items) is not list:
        raise ContractError("procedural v3 requires admitted rows")
    groups = {key: ranked([e for e in items
                            if (e["kind"], e["mode"]) == key], seed)
              for key in GROUPS}
    counters = {key: 0 for key in (("fold-sum", "prefix"), ("fold-sum", "trace"),
                                    ("gf3", "prefix"), ("gf3", "trace"))}
    rows = []
    for step in range(steps):
        rows.extend(groups["arithmetic", "final"][(step * 4 + j)
                    % len(groups["arithmetic", "final"])] for j in range(4))
        rows.extend(groups["fold-sum", "final"][(step * 2 + j)
                    % len(groups["fold-sum", "final"])] for j in range(2))
        rows.append(groups["gf3", "final"][step % len(groups["gf3", "final"])])
        kind, mode = auxiliary_stream(step)
        if arm == "final-only":
            # Preserve the old complete control schedule exactly.
            selected = groups[kind, "final"]
            rows.append(selected[(step // 2) % len(selected)])
        else:
            rows.append(advance(groups, counters, kind, mode))
    if len(rows) != steps * 8:
        raise ContractError("v3 schedule must retain eight examples per update")
    return rows


def coverage(items, scheduled, arm):
    """Audit unique exposed rows, including unvisited DEVELOPMENT examples."""
    if arm not in ARMS:
        raise ContractError("invalid v3 exposure arm")
    groups = {key: {e["example_identity"] for e in items
                     if (e["kind"], e["mode"]) == key} for key in GROUPS}
    used = {key: {} for key in GROUPS}
    for row in scheduled:
        key = row["kind"], row["mode"]
        if key not in groups or row["example_identity"] not in groups[key]:
            raise ContractError("scheduled row is not an admitted v3 example")
        counts = used[key]
        ident = row["example_identity"]
        counts[ident] = counts.get(ident, 0) + 1
    result = {}
    for kind, mode in GROUPS:
        key = (kind, mode)
        available = groups[key]
        visited = set(used[key])
        # Strict enough to catch the old parity trap. A stream cannot
        # cover more distinct rows than its total visits.
        visits = sum(used[key].values())
        if arm == "procedural" and mode != "final" and len(visited) != min(visits, len(available)):
            raise ContractError("v3 auxiliary cursor excludes reachable training rows")
        result[kind + "/" + mode] = {
            "available": len(available), "visits": visits,
            "distinct_visited": len(visited), "distinct_unvisited": len(available - visited),
            "visited_identity": identity(json_bytes(sorted(visited))),
            "unvisited_identity": identity(json_bytes(sorted(available - visited)))}
    return result
