"""Independent finite-set oracle; public synthetic conformance, not unseen data."""
import itertools
import json
import subprocess
import sys

binary = sys.argv[1]
checks = 0


def run(operation, raw, valid=True):
    global checks
    p = subprocess.run([binary, operation], input=raw, capture_output=True, timeout=3)
    checks += 1
    if not valid:
        assert p.returncode == 2 and not p.stdout and p.stderr, (raw, p)
        return
    assert p.returncode == 0 and not p.stderr, (raw, p)
    return json.loads(p.stdout)


def encode(r):
    return json.dumps(r, separators=(",", ":")).encode()


def masks(n):
    return list(itertools.product((False, True), repeat=n))[1:]


def request(c, s, m, table, cm, sm, q):
    return dict(schema="trinite.context-symbol-request.v1", contexts=c,
                symbols=s, meanings=m, table=table, context_mask=list(cm),
                symbol_mask=list(sm), query=q)


def check(r):
    possible = {r["table"][c][s] == r["query"]
                for c in range(r["contexts"]) if r["context_mask"][c]
                for s in range(r["symbols"]) if r["symbol_mask"][s]}
    expected = 0 if len(possible) == 2 else (1 if True in possible else -1)
    result = run("evaluate-privileged", encode(r))
    assert result == dict(schema="trinite.context-symbol-outcome.v1", relation=expected)
    observed = run("observe", encode(r))
    assert observed == dict(r, schema="trinite.context-symbol-observation.v1")
    # Joint context/symbol permutations preserve semantics, not arbitrary table-only swaps.
    permuted = dict(r, table=[list(reversed(row)) for row in reversed(r["table"])],
                    context_mask=list(reversed(r["context_mask"])),
                    symbol_mask=list(reversed(r["symbol_mask"])))
    assert run("evaluate-privileged", encode(permuted)) == result
    # Canonical output independent of request key ordering and whitespace.
    reordered = json.dumps(dict(reversed(list(r.items()))), indent=1).encode()
    p1 = subprocess.check_output([binary, "observe"], input=encode(r), timeout=3)
    p2 = subprocess.check_output([binary, "observe"], input=reordered, timeout=3)
    assert p1 == p2


for c, s, m in itertools.product((1, 2), repeat=3):
    for cells in itertools.product(range(m), repeat=c * s):
        table = [list(cells[i * s:(i + 1) * s]) for i in range(c)]
        for cm, sm, q in itertools.product(masks(c), masks(s), range(m)):
            check(request(c, s, m, table, cm, sm, q))

base = request(2, 2, 2, [[0, 1], [1, 1]], [True, False], [True, False], 0)
# Max admitted dimensions and output; no precision/float behavior.
check(request(8, 8, 8, [[(c+s) % 8 for s in range(8)] for c in range(8)],
              [True]*8, [True]*8, 7))
for key, value in [
    ("contexts", 0), ("contexts", 9), ("symbols", 9), ("meanings", 0),
    ("meanings", 9), ("query", 2), ("query", -1), ("query", True),
    ("table", [[0], [1, 1]]), ("table", [[0, 1]]),
    ("table", [[0, 2], [1, 1]]), ("table", [[0, 0, 0]]),
    ("context_mask", [False, False]), ("symbol_mask", [False, False]),
    ("context_mask", [True]), ("symbol_mask", [True, True, True]),
    ("context_mask", [1, 0]), ("target", 1), ("family_id", 7),
    ("seed", 1), ("hidden_state", 0),
]:
    for op in ("observe", "evaluate-privileged"):
        run(op, encode(dict(base, **{key: value})), False)
for key in base:
    missing = dict(base)
    del missing[key]
    run("observe", encode(missing), False)
raw = encode(base)
for malformed in [
    raw.replace(b'"query":0', b'"query":-0'),
    raw.replace(b'"query":0', b'"query":00'),
    raw.replace(b'"query":0', b'"query":0.0'),
    raw.replace(b'"query":0', b'"query":1e0'),
    raw.replace(b'"query":0', b'"query":9223372036854775808'),
    raw.replace(b'"query":0', b'"query":-9223372036854775809'),
    raw.replace(b'"query":0', b'"query":0,"query":1'),
    raw.replace(b'"query"', b'"qu\\u0065ry"'),
    raw + b" {}", b'{"x":"\xc3\xa9"}', b"[" * 10 + b"0" + b"]" * 10,
    raw + b" " * 16385, b"", b"null", b"[]", b"{}",
]:
    run("observe", malformed, False)
assert run("invalid", raw, False) is None
assert run(b"\xff", raw, False) is None
print(f"PASS: {checks} CLI checks; exhaustive all 1..2 dimensions/tables/nonempty masks/queries")
