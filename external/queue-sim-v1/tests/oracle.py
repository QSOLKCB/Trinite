#!/usr/bin/env python3
"""Independent job-record oracle for queue-v1 software conformance only.

Uses unbounded Python integers and per-job completion records rather than the
production FIFO implementation. All schedules here are exposed synthetic test
inputs, not a held-out benchmark, native training data or learning evidence.
Run: python3 tests/oracle.py target/release/trinite-queue-sim-v1
"""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest


SUMMARY_KEYS = ("offered", "admitted", "dropped", "served", "wait_sum", "wait_max")


def request(arrivals=None, service=None, *, capacity=4, initial_queue=0,
            outages=None, observe_ticks=1, observed=None):
    arrivals = [2, 3, 1, 0] if arrivals is None else arrivals
    service = [1] * len(arrivals) if service is None else service
    return {
        "schema": "trinite.queue-request.v1",
        "capacity": capacity,
        "initial_queue": initial_queue,
        "arrivals": arrivals,
        "service": service,
        "outages": [False] * len(arrivals) if outages is None else outages,
        "observe_ticks": observe_ticks,
        "observed": [True] * 6 if observed is None else observed,
    }


def reference(spec):
    # Each immutable admission has its own birth/completion record. Recompute
    # pending jobs from those records; do not reproduce a FIFO deque algorithm.
    jobs = [[0, None] for _ in range(spec["initial_queue"])]
    rows = []
    for tick, offered in enumerate(spec["arrivals"]):
        pending = [i for i, (_, done) in enumerate(jobs) if done is None]
        admitted = min(offered, spec["capacity"] - len(pending))
        jobs.extend([[tick, None] for _ in range(admitted)])
        eligible = [i for i, (_, done) in enumerate(jobs) if done is None]
        nominal = spec["service"][tick]
        effective = 0 if spec["outages"][tick] else nominal
        finished = eligible[:effective]
        waits = [tick - jobs[i][0] for i in finished]
        for index in finished:
            jobs[index][1] = tick
        ages = [tick + 1 - born for born, done in jobs if done is None]
        rows.append({
            "tick": tick, "offered": offered, "admitted": admitted,
            "dropped": offered - admitted, "nominal_service": nominal,
            "effective_service": effective, "outage": spec["outages"][tick],
            "served": len(finished), "queue": len(ages),
            "wait_sum": sum(waits), "wait_max": max(waits, default=0),
            "oldest_pending_age": max(ages, default=0),
        })

    def summarize(selected):
        return {key: (max((row[key] for row in selected), default=0)
                      if key == "wait_max" else sum(row[key] for row in selected))
                for key in SUMMARY_KEYS}

    return {
        "schema": "trinite.queue-ground-truth.v1", "privileged": True,
        "capacity": spec["capacity"], "observe_ticks": spec["observe_ticks"],
        "steps": len(spec["arrivals"]), "initial_queue": spec["initial_queue"],
        "rows": rows, "summary": summarize(rows),
        "forecast": summarize(rows[spec["observe_ticks"]:]),
        "pending_ages": [len(rows) - born for born, done in jobs if done is None],
    }


def projection(spec, truth):
    rows = []
    for row in truth["rows"][:spec["observe_ticks"]]:
        values = [row[key] for key in (
            "queue", "offered", "effective_service", "served", "dropped",
            "oldest_pending_age")]
        rows.append({"tick": row["tick"], "outage": row["outage"],
                     "values": [value if seen else 0
                                for value, seen in zip(values, spec["observed"])],
                     "observed": spec["observed"]})
    return {"schema": "trinite.queue-observation.v1",
            "capacity": spec["capacity"], "observe_ticks": spec["observe_ticks"],
            "horizon": len(spec["arrivals"]) - spec["observe_ticks"], "rows": rows}


class QueueOracleTests(unittest.TestCase):
    binary = None

    def run_cli(self, mode, spec=None, raw=None):
        if raw is None:
            raw = json.dumps(spec, separators=(",", ":")).encode("ascii")
        return subprocess.run([self.binary, mode], input=raw,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=False, timeout=5)

    def success(self, mode, spec):
        run = self.run_cli(mode, spec)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stderr, b"")
        self.assertEqual(run.stdout.count(b"\n"), 1)
        self.assertTrue(run.stdout.endswith(b"\n"))
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                self.assertNotIn(key, result, "duplicate output key")
                result[key] = value
            return result
        return json.loads(run.stdout, object_pairs_hook=unique_object)

    def assert_exact(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_exact(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_exact(left, right)
        else:
            self.assertEqual(actual, expected)

    def check_case(self, spec):
        truth = reference(spec)
        actual = self.success("simulate-privileged", spec)
        self.assert_exact(actual, truth)
        self.assert_exact(self.success("observe", spec), projection(spec, truth))
        # Conservation is separately asserted rather than only trusting oracle
        # byte/field equality. Initially present jobs are not new offered jobs.
        pending = spec["initial_queue"]
        for row in actual["rows"]:
            self.assertEqual(row["offered"], row["admitted"] + row["dropped"])
            pending += row["admitted"] - row["served"]
            self.assertEqual(pending, row["queue"])
            self.assertGreaterEqual(pending, 0)
            self.assertLessEqual(pending, spec["capacity"])
        self.assertEqual(spec["initial_queue"] + actual["summary"]["admitted"],
                         actual["summary"]["served"] + len(actual["pending_ages"]))
        return actual

    def test_schedule_families_and_selected_cutoffs(self):
        # Forty-eight visible cases: steady, burst, alternating, ramp,
        # sparse arrivals, variable service, outage and initial-backlog contrasts.
        families = [[2] * 8, [8, 0, 0, 8, 0, 0, 8, 0],
                    [0, 5] * 4, list(range(8)), [0, 0, 7, 0, 0, 1, 0, 0],
                    [6, 5, 4, 3, 2, 1, 0, 0]]
        for family, initial, outage in itertools.product(range(6), [0, 3], [False, True]):
            for cutoff in [1, 7]:
                with self.subTest(family=family, initial=initial, outage=outage, cutoff=cutoff):
                    self.check_case(request(families[family],
                                            [0, 1, 3, 2, 7, 1, 0, 4], capacity=9,
                                            initial_queue=initial, observe_ticks=cutoff,
                                            outages=[outage and i in [2, 3, 6] for i in range(8)]))

    def test_admission_precedes_service(self):
        truth = self.check_case(request([3, 0], [3, 3], capacity=3, initial_queue=3))
        self.assertEqual(truth["rows"][0]["dropped"], 3)
        self.assertEqual(truth["rows"][0]["served"], 3)
        self.assertEqual(truth["rows"][0]["queue"], 0)

    def test_fifo_waits_and_censored_pending_ages(self):
        truth = self.check_case(request([2, 1, 1, 0], [0, 1, 1, 0],
                                        capacity=8, initial_queue=1, observe_ticks=2))
        self.assertEqual([r["wait_sum"] for r in truth["rows"]], [0, 1, 2, 0])
        self.assertEqual(truth["pending_ages"], [4, 3, 2])
        self.assertEqual(truth["summary"]["wait_sum"], 3)
        self.assertEqual(truth["forecast"]["wait_sum"], 2)
        self.assertEqual(truth["forecast"]["served"], 1)

    def test_zero_service_is_not_outage(self):
        normal = request([1, 1], [0, 0])
        outage = copy.deepcopy(normal)
        outage["outages"] = [True, True]
        a = self.check_case(normal)
        b = self.check_case(outage)
        self.assertEqual(a["summary"], b["summary"])
        self.assertNotEqual(self.success("observe", normal), self.success("observe", outage))

    def test_masks_preserve_dynamics_and_zero_is_not_missing(self):
        full = request([0, 3, 2], [2, 1, 1], observe_ticks=2)
        truth = self.success("simulate-privileged", full)
        for bits in itertools.product([False, True], repeat=6):
            masked = copy.deepcopy(full)
            masked["observed"] = list(bits)
            self.assert_exact(self.success("simulate-privileged", masked), truth)
            self.assert_exact(self.success("observe", masked), projection(masked, truth))
        missing = copy.deepcopy(full)
        missing["observed"] = [False] * 6
        known_zero = self.success("observe", full)["rows"][0]
        unknown = self.success("observe", missing)["rows"][0]
        self.assertEqual(known_zero["values"][0], unknown["values"][0])
        self.assertTrue(known_zero["observed"][0])
        self.assertFalse(unknown["observed"][0])

    def test_suffix_cannot_change_observation_bytes(self):
        a = request([1, 2, 3, 4, 5, 6], [2, 1, 0, 0, 0, 0], observe_ticks=2)
        b = copy.deepcopy(a)
        b["arrivals"][2:] = [32, 0, 32, 32]
        b["service"][2:] = [32, 32, 32, 32]
        b["outages"][2:] = [True, False, True, True]
        self.assertEqual(self.run_cli("observe", a).stdout, self.run_cli("observe", b).stdout)
        self.assertNotEqual(self.success("simulate-privileged", a),
                            self.success("simulate-privileged", b))

    def test_resource_bounds(self):
        for outage in [False, True]:
            self.check_case(request([32] * 256, [32] * 256, capacity=128,
                                    initial_queue=128, observe_ticks=255,
                                    outages=[outage] * 256))
        self.check_case(request([0, 0], [0, 0], capacity=1))

    def test_exact_replay_and_object_field_order(self):
        spec = request()
        first = self.run_cli("simulate-privileged", spec)
        self.assertEqual(first.returncode, 0)
        self.assertEqual(first.stdout, self.run_cli("simulate-privileged", spec).stdout)
        reordered = dict(reversed(list(spec.items())))
        self.assertEqual(first.stdout, self.run_cli("simulate-privileged", reordered).stdout)

    def assert_rejected(self, spec=None, raw=None, mode="observe"):
        result = self.run_cli(mode, spec, raw)
        self.assertEqual(result.returncode, 2, (spec, raw, result.stderr))
        self.assertEqual(result.stdout, b"")
        self.assertTrue(result.stderr)

    def test_invalid_request_fields(self):
        for key in request():
            spec = request()
            del spec[key]
            self.assert_rejected(spec)
        cases = {
            "schema": ["wrong", 1], "capacity": [0, 129, True, 1.5],
            "initial_queue": [-1, 5, False], "observe_ticks": [0, 4, 5, True],
            "arrivals": [[], [1], [1] * 257, [33] * 4, [-1] * 4, [True] * 4],
            "service": [[1], [33] * 4, [-1] * 4, [False] * 4],
            "outages": [[], [False], [0] * 4],
            "observed": [[], [True] * 5, [True] * 7, [1] * 6],
        }
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    spec = request()
                    spec[key] = value
                    self.assert_rejected(spec)
        extra = request()
        extra["unexpected"] = 0
        self.assert_rejected(extra)

    def test_strict_wire_rejections(self):
        valid = json.dumps(request(), separators=(",", ":")).encode()
        malformed = [b"", b"[]", b"{}", b"null", valid + b"{}",
                     valid.replace(b'"capacity":4', b'"capacity":04'),
                     valid.replace(b'"initial_queue":0', b'"initial_queue":-0'),
                     valid.replace(b'"capacity":4', b'"capacity":4,"capacity":4'),
                     valid.replace(b'"capacity":4', b'"capacity":9223372036854775808'),
                     valid.replace(b'"schema"', b'"sche\\u006da"'),
                     valid.replace(b'"schema"', b'"sch\xc3\xa9ma"'),
                     b"[" * 10 + b"0" + b"]" * 10, b" " * 16385]
        for raw in malformed:
            with self.subTest(raw=raw[:100]):
                self.assert_rejected(raw=raw)
        for mode in ["", "simulate", "unknown", "observe-before"]:
            self.assert_rejected(request(), mode=mode)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: oracle.py /path/to/trinite-queue-sim-v1")
    QueueOracleTests.binary = str(Path(sys.argv.pop()).resolve(strict=True))
    unittest.main(verbosity=2)
