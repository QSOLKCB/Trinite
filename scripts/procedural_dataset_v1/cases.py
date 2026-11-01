"""Versioned synthetic families, public-only lossless carrier, independent targets.

Families, not row seeds, own splits. This public inventory is engineering-exposed;
it is not a fresh evaluation population or authorization to train.
"""
from __future__ import annotations

import copy
import json
import re

MODULES = ("ternary", "tombstone", "context")
FAMILIES = {
    f"{module}-{size}": {"module": module, "split": split}
    for module in MODULES
    for size, split in ((2, "train"), (3, "validation"), (4, "test"))
}
REQUEST_SCHEMAS = dict(zip(MODULES, ("trinite.ternary-evidence-request.v1",
    "trinite.tombstone-request.v1", "trinite.context-symbol-request.v1")))
OBSERVATION_SCHEMAS = dict(REQUEST_SCHEMAS, tombstone="trinite.tombstone-observation.v1",
                          context="trinite.context-symbol-observation.v1")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer outside bounds")
    return value


def fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("missing or unknown fields")


def array(value, low, high):
    if type(value) is not list or not low <= len(value) <= high:
        raise ValueError("invalid array length")
    return value


def mask(value, size, nonempty=False):
    array(value, size, size)
    if any(type(x) is not bool for x in value) or (nonempty and not any(value)):
        raise ValueError("invalid boolean mask")


def validate_request(module, request):
    if module not in MODULES:
        raise ValueError("unknown module")
    if module == "ternary":
        fields(request, ("schema", "input_kind", "weights", "entries"))
        if request["input_kind"] != "generic-public-trits":
            raise ValueError("unknown input kind")
        weights = array(request["weights"], 1, 64)
        for w in weights:
            integer(w, 0, 10**9)
        if not sum(weights):
            raise ValueError("zero total weight")
        ids = set()
        for e in array(request["entries"], 1, 64):
            fields(e, ("id", "active", "trits", "mask"))
            if type(e["id"]) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", e["id"]) or e["id"] in ids:
                raise ValueError("invalid or duplicate entry identity")
            ids.add(e["id"])
            if type(e["active"]) is not bool:
                raise ValueError("invalid active flag")
            mask(e["mask"], len(weights))
            for t, seen in zip(array(e["trits"], len(weights), len(weights)), e["mask"]):
                integer(t, -1, 1)
                if not seen and t != 0:
                    raise ValueError("noncanonical missing value")
    elif module == "tombstone":
        fields(request, ("schema", "before", "removed", "after", "offset"))
        for key in ("before", "removed", "after"):
            for value in array(request[key], 0, 16):
                integer(value, 0, 255)
        integer(request["offset"], 0, 16)
    else:
        fields(request, ("schema", "contexts", "symbols", "meanings", "table", "context_mask", "symbol_mask", "query"))
        c, s, m = (integer(request[k], 1, 8) for k in ("contexts", "symbols", "meanings"))
        for row in array(request["table"], c, c):
            for value in array(row, s, s):
                integer(value, 0, m - 1)
        mask(request["context_mask"], c, True)
        mask(request["symbol_mask"], s, True)
        integer(request["query"], 0, m - 1)
    if request["schema"] != REQUEST_SCHEMAS[module]:
        raise ValueError("unknown request schema")


def public_observation(module, request):
    validate_request(module, request)
    return dict(copy.deepcopy(request), schema=OBSERVATION_SCHEMAS[module])


def encode_public(module, observation):
    if module not in MODULES or type(observation) is not dict or observation.get("schema") != OBSERVATION_SCHEMAS[module]:
        raise ValueError("unknown public observation")
    r = dict(observation, schema=REQUEST_SCHEMAS[module])
    validate_request(module, r)
    if module == "ternary":
        payload = ["E1", r["weights"], [[e[k] for k in ("id", "active", "trits", "mask")] for e in r["entries"]]]
    elif module == "tombstone":
        payload = ["T1", *[r[k] for k in ("before", "removed", "after", "offset")]]
    else:
        payload = ["C1", *[r[k] for k in ("contexts", "symbols", "meanings", "table", "context_mask", "symbol_mask", "query")]]
    return canonical(payload)


def decode_public(text):
    if type(text) is not str or len(text) > 65536:
        raise ValueError("invalid carrier")
    try:
        p = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise ValueError("invalid carrier JSON") from exc
    array(p, 1, 8)
    tag = p[0]
    if tag == "E1" and len(p) == 3:
        module = "ternary"
        entries = []
        for row in array(p[2], 1, 64):
            array(row, 4, 4)
            entries.append(dict(zip(("id", "active", "trits", "mask"), row)))
        r = dict(input_kind="generic-public-trits", weights=p[1], entries=entries)
    elif tag == "T1" and len(p) == 5:
        module = "tombstone"
        r = dict(zip(("before", "removed", "after", "offset"), p[1:]))
    elif tag == "C1" and len(p) == 8:
        module = "context"
        r = dict(zip(("contexts", "symbols", "meanings", "table", "context_mask", "symbol_mask", "query"), p[1:]))
    else:
        raise ValueError("unknown carrier version or shape")
    r["schema"] = OBSERVATION_SCHEMAS[module]
    # Canonical equality rejects alternate lexical forms, floats, duplicate keys,
    # escaped identifiers and ignored whitespace rather than silently repairing.
    if encode_public(module, r) != text:
        raise ValueError("noncanonical carrier")
    return module, r


def expected_target(module, request):
    """Independent Python arithmetic/finite-set/slicing oracle, never a provider fallback."""
    validate_request(module, request)
    if module == "context":
        truths = {request["table"][c][s] == request["query"]
                  for c, cm in enumerate(request["context_mask"]) if cm
                  for s, sm in enumerate(request["symbol_mask"]) if sm}
        return str(0 if len(truths) == 2 else (1 if True in truths else -1))
    if module == "tombstone":
        b, r, a = (request[k] for k in ("before", "removed", "after"))
        o = request["offset"]
        if not r:
            return "argument"
        if len(r) > len(b) or o > len(b) - len(r):
            return "length"
        return "valid-splice" if b[o:o+len(r)] == r and b[:o] + b[o+len(r):] == a else "mismatch"
    rows = []
    w = request["weights"]
    for e in request["entries"]:
        d = sum(x for x, seen in zip(w, e["mask"]) if seen)
        s = sum(x*t for x, t, seen in zip(w, e["trits"], e["mask"]) if seen)
        status = "inactive" if not e["active"] else ("scored" if d else "abstain")
        rows.append([status, s if status == "scored" else None, d, sum(w), sum(e["mask"]), len(w)])
    return canonical(rows)


def target_from_outcome(module, outcome):
    """Project verified provider outcomes onto meaningful targets, not hash receipts.

    Full source/provider receipt validation belongs to admission; this projection
    does not certify an outcome or replace comparison with expected_target.
    """
    if module == "context":
        fields(outcome, ("schema", "relation"))
        if outcome["schema"] != "trinite.context-symbol-outcome.v1":
            raise ValueError("unknown outcome schema")
        return str(integer(outcome["relation"], -1, 1))
    if module == "tombstone":
        fields(outcome, ("schema", "label", "native_code", "receipt"))
        if outcome["schema"] != "trinite.tombstone-outcome.v1" or type(outcome["native_code"]) is not int:
            raise ValueError("unknown outcome schema/code")
        if {0: "valid-splice", 1: "argument", 2: "length", 6: "mismatch"}.get(outcome["native_code"]) != outcome["label"]:
            raise ValueError("invalid semantic outcome")
        if (outcome["receipt"] is None) != (outcome["native_code"] != 0):
            raise ValueError("invalid receipt presence")
        return outcome["label"]
    if module != "ternary":
        raise ValueError("unknown module")
    fields(outcome, ("schema", "entries"))
    if outcome["schema"] != "trinite.ternary-evidence-result.v1":
        raise ValueError("unknown outcome schema")
    rows = []
    for e in array(outcome["entries"], 1, 64):
        fields(e, ("id", "active", "status", "score", "coverage", "observed_count", "factor_count"))
        fields(e["coverage"], ("numerator", "denominator"))
        d = integer(e["coverage"]["numerator"], 0, 64*10**9)
        w = integer(e["coverage"]["denominator"], 1, 64*10**9)
        if type(e["active"]) is not bool or d > w:
            raise ValueError("invalid coverage/active")
        status = "inactive" if not e["active"] else ("scored" if d else "abstain")
        if e["status"] != status:
            raise ValueError("invalid score status")
        s = None
        if status == "scored":
            fields(e["score"], ("numerator", "denominator"))
            s = integer(e["score"]["numerator"], -d, d)
            if type(e["score"]["denominator"]) is not int or e["score"]["denominator"] != d:
                raise ValueError("invalid denominator")
        elif e["score"] is not None:
            raise ValueError("undefined score must be null")
        n = integer(e["factor_count"], 1, 64)
        count = integer(e["observed_count"], 0, n)
        rows.append([status, s, d, w, count, n])
    return canonical(rows)


def generate(family, seed):
    if type(family) is not str or family not in FAMILIES:
        raise ValueError("unknown structural family")
    integer(seed, 0, 65535)
    module = FAMILIES[family]["module"]
    n = int(family.rsplit("-", 1)[1])
    cases = []
    if module == "context":
        table = [[(c+s+seed) % n for s in range(n)] for c in range(n)]
        parents = [table, [row[:] for row in table]]
        parents[1][0][0] = (parents[1][0][0]+1) % n
        for table in parents:
            for cm in ([True]+[False]*(n-1), [False, True]+[False]*(n-2), [True]*n):
                for q in range(n):
                    r = dict(contexts=n, symbols=n, meanings=n, table=table,
                             context_mask=cm, symbol_mask=[True]+[False]*(n-1), query=q)
                    cases.append(r)
                    cases.append(dict(r, table=[list(reversed(row)) for row in reversed(table)],
                                      context_mask=list(reversed(cm)), symbol_mask=list(reversed(r["symbol_mask"]))))
    elif module == "ternary":
        weights = [1+(seed % 3), 0] + list(range(2, n))
        for active in (True, False):
            for mode in range(4):
                seen = [[True]*n, [False]*n, [False, True]+[False]*(n-2), [True]+[False]*(n-1)][mode]
                trits = [((i+seed) % 3)-1 if m else 0 for i, m in enumerate(seen)]
                row = dict(id="e", active=active, trits=trits, mask=seen)
                cases.append(dict(input_kind="generic-public-trits", weights=weights, entries=[row]))
                cases.append(dict(input_kind="generic-public-trits", weights=weights, entries=[dict(row, trits=[0]*n)]))
                changed = dict(row, trits=[-t for t in trits])
                cases.append(dict(input_kind="generic-public-trits", weights=weights, entries=[changed]))
                cases.append(dict(input_kind="generic-public-trits", weights=list(reversed(weights)), entries=[dict(row, id="r", trits=list(reversed(trits)), mask=list(reversed(seen)))]))
    else:
        before = [(seed+17*i) % 256 for i in range(2*n)]
        # All byte renamings, offset/length counterfactuals and parent mutations
        # remain in the length-defined family, never another split.
        for b in (before, [(x+1) % 256 for x in before]):
            for o in (0, n-1):
                removed = b[o:o+2]
                after = b[:o]+b[o+2:]
                cases.extend([dict(before=b, removed=removed, after=after, offset=o),
                    dict(before=b, removed=[], after=after, offset=o),
                    dict(before=b, removed=removed, after=after, offset=16),
                    dict(before=b, removed=[(removed[0]+1)%256, removed[1]], after=after, offset=o),
                    dict(before=b, removed=removed, after=[(after[0]+1)%256]+after[1:], offset=o)])
    unique = {}
    for r in cases:
        r = dict(copy.deepcopy(r), schema=REQUEST_SCHEMAS[module])
        validate_request(module, r)
        unique[canonical(r)] = r
    return list(unique.values())


def verify_outcome(module, request, outcome):
    """Verify every provider field against an independently reconstructed result."""
    import hashlib

    expected = expected_target(module, request)
    actual = target_from_outcome(module, outcome)
    if actual != expected:
        raise ValueError("module oracle disagreement")
    if module == "context":
        wanted = dict(schema="trinite.context-symbol-outcome.v1", relation=int(expected))
    elif module == "ternary":
        rows = []
        for source, target in zip(request["entries"], json.loads(expected)):
            status, s, d, w, count, n = target
            rows.append(dict(id=source["id"], active=source["active"], status=status,
                             score=None if s is None else dict(numerator=s, denominator=d),
                             coverage=dict(numerator=d, denominator=w), observed_count=count, factor_count=n))
        wanted = dict(schema="trinite.ternary-evidence-result.v1", entries=rows)
    else:
        code = {"valid-splice": 0, "argument": 1, "length": 2, "mismatch": 6}[expected]
        receipt = None
        if code == 0:
            receipt = dict(format="tombstone.receipt", version=1, kind="explicit-delete", previous_receipt_hash=None)
            for source, destination in (("before", "before"), ("after", "after"), ("removed", "deleted")):
                value = bytes(request[source])
                receipt[destination] = dict(sha256=hashlib.sha256(value).hexdigest(), size=len(value))
                if destination == "deleted":
                    receipt[destination]["offset"] = request["offset"]
            receipt["receipt_hash"] = hashlib.sha256(b"TOMBSTONE-RECEIPT-V1\n" + canonical(receipt).encode("ascii")).hexdigest()
        wanted = dict(schema="trinite.tombstone-outcome.v1", label=expected, native_code=code, receipt=receipt)
    # Type-sensitive canonical equality also rejects Python bool/int equivalence.
    if canonical(outcome) != canonical(wanted):
        raise ValueError("module outcome or receipt disagreement")
    return actual
