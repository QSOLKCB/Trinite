#!/usr/bin/env python3
"""Independent, public-synthetic conformance only; never production labels.

Runs the actual C99-backed binary. The finite cases are not a learning benchmark,
proof of historical deletion, secure erasure, or an exhaustive byte-domain proof.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

REQUEST_SCHEMA = "trinite.tombstone-request.v1"
OBSERVATION_SCHEMA = "trinite.tombstone-observation.v1"
OUTCOME_SCHEMA = "trinite.tombstone-outcome.v1"
LABELS = {0: "valid-splice", 1: "argument", 2: "length", 6: "mismatch"}


def request(before, removed, after, offset):
    return dict(schema=REQUEST_SCHEMA, before=list(before), removed=list(removed),
                after=list(after), offset=offset)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def expected_code(value):
    """Independent slicing/equality oracle, preserving native error precedence."""
    before, removed, after = (bytes(value[key]) for key in ("before", "removed", "after"))
    offset = value["offset"]
    if not removed:
        return 1
    if len(removed) > len(before) or offset > len(before) - len(removed):
        return 2
    if before[offset:offset + len(removed)] != removed:
        return 6
    if before[:offset] + before[offset + len(removed):] != after:
        return 6
    return 0


def verify_receipt(receipt, value):
    assert set(receipt) == {"after", "before", "deleted", "format", "kind",
                            "previous_receipt_hash", "receipt_hash", "version"}
    assert receipt["format"] == "tombstone.receipt"
    assert receipt["version"] == 1
    assert receipt["kind"] == "explicit-delete"
    assert receipt["previous_receipt_hash"] is None
    for source, field in (("before", "before"), ("after", "after"), ("removed", "deleted")):
        expected = {"sha256": hashlib.sha256(bytes(value[source])).hexdigest(),
                    "size": len(value[source])}
        if field == "deleted":
            expected["offset"] = value["offset"]
        assert receipt[field] == expected, (field, receipt[field], expected)
    body = {key: data for key, data in receipt.items() if key != "receipt_hash"}
    # Fixed TOMBSTONE v1 ASCII/integer profile only; not arbitrary JSON JCS.
    expected = hashlib.sha256(b"TOMBSTONE-RECEIPT-V1\n" + encode(body)).hexdigest()
    assert receipt["receipt_hash"] == expected


class Harness:
    def __init__(self, binary):
        self.binary = str(Path(binary).resolve(strict=True))
        self.calls = 0
        self.cases = 0
        self.invalid = 0
        self.labels = dict.fromkeys(LABELS.values(), 0)
        self.digest = hashlib.sha256()

    def run(self, operation, raw, expected_exit=0):
        self.calls += 1
        result = subprocess.run([self.binary, operation], input=raw, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=3, check=False)
        assert result.returncode == expected_exit, (operation, result.returncode, result.stderr)
        assert len(result.stdout) + len(result.stderr) <= 4096
        if expected_exit:
            assert result.stdout == b"", result.stdout
            assert result.stderr == b"invalid request\n", result.stderr
            return None
        assert result.stderr == b"", result.stderr
        assert result.stdout.endswith(b"\n") and result.stdout.count(b"\n") == 1
        return json.loads(result.stdout)

    def check(self, value, observe=False):
        raw = encode(value)
        expected = expected_code(value)
        outcome = self.run("evaluate-privileged", raw)
        assert set(outcome) == {"schema", "label", "native_code", "receipt"}
        assert outcome["schema"] == OUTCOME_SCHEMA
        assert type(outcome["native_code"]) is int and outcome["native_code"] == expected, (value, outcome, expected)
        assert outcome["label"] == LABELS[expected], (value, outcome)
        if expected:
            assert outcome["receipt"] is None
        else:
            verify_receipt(outcome["receipt"], value)
        if observe:
            actual = self.run("observe", raw)
            public = dict(value, schema=OBSERVATION_SCHEMA)
            assert actual == public, (actual, public)
            assert set(actual) == {"schema", "before", "removed", "after", "offset"}
        self.cases += 1
        self.labels[LABELS[expected]] += 1
        self.digest.update(raw + b"\n" + encode(outcome) + b"\n")

    def reject(self, raw):
        for operation in ("observe", "evaluate-privileged"):
            self.run(operation, raw, 2)
        self.invalid += 1


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: oracle.py PATH_TO_BUILT_ADAPTER")
    h = Harness(sys.argv[1])
    explicit = [
        request([], [], [], 0),
        request([0], [], [255], 16),  # argument wins over all other semantic failures
        request([0], [0, 1], [255], 16),  # length wins over mismatch
        request([0], [0], [], 0),  # full removal, empty after is valid
        request([0, 255, 128, 10], [255, 128], [0, 10], 1),
        request([0, 255, 128, 10], [255, 127], [0, 10], 1),
        request([0, 255, 128, 10], [255, 128], [0, 11], 1),
        request([0, 255, 128, 10], [255, 128], [0], 1),
        request(list(range(16)), list(range(16)), [], 0),
        request(list(range(16)), [15], list(range(15)), 15),
        request(list(range(16)), [15], list(range(15)), 16),
    ]
    # All four offsets are valid witnesses, not a unique inferred history.
    explicit += [request(b"aaaa", b"a", b"aaa", offset) for offset in range(4)]
    for value in explicit:
        h.check(value, observe=True)
    # Exhaustive length <= 3 words over {0,255}, every offset 0..4.
    words = [list(word) for size in range(4) for word in itertools.product((0, 255), repeat=size)]
    for before, removed, after in itertools.product(words, repeat=3):
        for offset in range(5):
            h.check(request(before, removed, after, offset))
    rng = random.Random(1072026)
    for _ in range(256):
        before = [rng.randrange(256) for _ in range(rng.randrange(17))]
        removed = [rng.randrange(256) for _ in range(rng.randrange(17))]
        after = [rng.randrange(256) for _ in range(rng.randrange(17))]
        h.check(request(before, removed, after, rng.randrange(17)), observe=True)
        if before:
            offset = rng.randrange(len(before))
            end = rng.randrange(offset + 1, len(before) + 1)
            h.check(request(before, before[offset:end], before[:offset] + before[end:], offset), observe=True)
    good = request([0, 255], [255], [0], 1)
    bad_values = []
    for field in good:
        bad_values.append({key: value for key, value in good.items() if key != field})
    bad_values += [dict(good, extra=0), dict(good, schema="unknown"), [], None]
    for key in ("before", "removed", "after"):
        for value in (None, False, 0, "", [False], [1.0], [-1], [256], [0] * 17, [[0]]):
            bad_values.append(dict(good, **{key: value}))
    for value in (None, False, "1", -1, 17, 1.0, 2**64):
        bad_values.append(dict(good, offset=value))
    for value in bad_values:
        h.reject(encode(value))
    raw = encode(good)
    for malformed in (
        b"", b"{}", raw + b"{}", raw + b"x", b" " * 1025,
        raw.replace(b'"offset":1', b'"offset":01'),
        raw.replace(b'"offset":1', b'"offset":-0'),
        raw.replace(b'"offset":1', b'"offset":1e0'),
        raw.replace(b'"offset":1', b'"offset":1,"offset":1'),
        raw.replace(b'"schema"', b'"schem\\u0061"'),
        raw.replace(b'"schema"', b'"schem\xc3\xa1"'),
        b'{"nest":' + b'[' * 10 + b'0' + b']' * 10 + b'}',
        raw[:-1] + b',}',
    ):
        h.reject(malformed)
    # Exact transport cap is accepted; one extra byte is rejected.
    h.run("observe", raw + b" " * (1024 - len(raw)))
    h.reject(raw + b" " * (1025 - len(raw)))
    h.run("unknown-operation", raw, 2)
    print(json.dumps({"schema": "trinite.tombstone-conformance.v1", "semantic_cases": h.cases,
                      "malformed_cases": h.invalid, "process_calls": h.calls,
                      "labels": h.labels, "case_outcome_sha256": h.digest.hexdigest(),
                      "scope": "public synthetic actual-C99 conformance; no learning or deletion-history claim"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
