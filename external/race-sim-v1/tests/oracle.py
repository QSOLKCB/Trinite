"""Independent Python-integer conformance oracle; never a producer/trainer.

All cases are exposed engineering checks, not held-out prediction evidence.
"""
import copy
import json
import pathlib
import subprocess
import sys
import unittest

BINARY = pathlib.Path(sys.argv.pop(1)).resolve()
EXAMPLE = json.loads((pathlib.Path(__file__).parents[1] / "example-request.json").read_text())
MASK64 = (1 << 64) - 1


def word(seed, identifier, stream, step):
    counter = identifier * 2**32 + stream * 2**16 + step
    x = (seed + counter * 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def oracle(request):
    field = []
    for r in sorted(request["runners"], key=lambda r: r["id"]):
        i = r["id"]
        if i in request["scratches"]:
            continue
        latent = dict(speed=40 + word(request["seed"], i, 0, 0) % 41,
                      stamina=32 + word(request["seed"], i, 1, 0) % 65,
                      fatigue=word(request["seed"], i, 2, 0) % 17,
                      track_fit=[word(request["seed"], i, k, 0) % 17 - 8 for k in (3, 4, 5)])
        field.append(dict(id=i, distance=0, latent=latent))
    pace = sum(r["latent"]["speed"] for r in field) // len(field)
    for lane, r in enumerate(field):
        state = r["latent"]
        # Deliberately runner-major, whereas the Rust producer is step-major.
        increments = []
        for tick in range(request["steps"]):
            fatigue = min(60, state["fatigue"] + tick * (100-state["stamina"]) // 256)
            advance = (state["speed"] + state["track_fit"][request["track"]] - fatigue
                       - max(0, pace-state["stamina"]) // 8 - len(field) // 4
                       - (lane if tick == 0 else 0)
                       + word(request["seed"], r["id"], 6, tick) % 7 - 3)
            increments.append(min(100, max(1, advance)))
        r["distance"] = sum(increments)
    return sorted(field, key=lambda r: (-r["distance"], r["id"]))


def call(mode, request):
    result = subprocess.run([str(BINARY), mode], input=json.dumps(request).encode(),
                            capture_output=True, timeout=5, check=True)
    if result.stderr:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


class Conformance(unittest.TestCase):
    def test_equations_across_bounded_cases(self):
        for seed in (0, 7, 99, 2**32-1):
            for track in (0, 1, 2):
                for steps in (1, 32, 256):
                    r = copy.deepcopy(EXAMPLE)
                    r.update(seed=seed, track=track, steps=steps)
                    with self.subTest(seed=seed, track=track, steps=steps):
                        self.assertEqual(call("simulate-privileged", r)["ranking"], oracle(r))

    def test_scratch_counterfactual(self):
        r = copy.deepcopy(EXAMPLE)
        before = call("observe-before", r)
        r["scratches"] = [2]
        self.assertEqual(before, call("observe-before", r))
        self.assertEqual(call("simulate-privileged", r)["ranking"], oracle(r))
        view = call("observe-after", r)
        withdrawn = next(x for x in view["runners"] if x["id"] == 2)
        self.assertFalse(withdrawn["active"])
        self.assertFalse(withdrawn["observed"][4])
        self.assertNotIn(2, [x["id"] for x in call("simulate-privileged", r)["ranking"]])

    def test_observation_projection_and_neutral(self):
        r = copy.deepcopy(EXAMPLE)
        r["runners"][0]["observed"] = [False]*5
        view = call("observe-before", r)
        self.assertEqual(set(view), {"schema", "phase", "steps", "track", "runners"})
        first = view["runners"][0]
        self.assertTrue(first["active"])
        self.assertEqual(first["values"], [0]*5)
        self.assertEqual(first["observed"], [False]*5)
        for row in view["runners"]:
            self.assertEqual(set(row), {"id", "active", "values", "observed"})
            self.assertTrue(all(v in (-1, 0, 1) for v in row["values"]))
        self.assertEqual(call("simulate-privileged", r)["ranking"], oracle(EXAMPLE))

    def test_malformed_wire_rejection(self):
        for data in (b"[", b"{}x", b"[]", b" "*16385, b'{"seed":1,"seed":2}', b'[[[[[[[[[[0]]]]]]]]]]'):
            result = subprocess.run([str(BINARY), "observe-before"], input=data, capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, b"")

    def test_input_order_invariance(self):
        r = copy.deepcopy(EXAMPLE)
        r["runners"].reverse()
        for mode in ("observe-before", "observe-after", "simulate-privileged"):
            self.assertEqual(call(mode, r), call(mode, EXAMPLE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
