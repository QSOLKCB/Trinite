"""Independent carrier/generator boundaries; no optimizer or provider fakes."""
import copy
import json
import unittest
from procedural_dataset_v1 import cases as c
from trinite.tokenizer import ByteTokenizer


class CasesTests(unittest.TestCase):
    def test_generator_reproduction_budget_and_roundtrip(self):
        for seed in (0, 17, 65535):
            global_inputs = {}
            prefixes = {}
            for family, meta in c.FAMILIES.items():
                rows = c.generate(family, seed)
                self.assertEqual(rows, c.generate(family, seed))
                self.assertLessEqual(len(rows), 64)
                self.assertEqual(len(rows), len({c.canonical(r) for r in rows}))
                for r in rows:
                    module = meta["module"]
                    public = c.public_observation(module, r)
                    prompt = c.encode_public(module, public)
                    self.assertEqual(c.decode_public(prompt), (module, public))
                    target = c.expected_target(module, r)
                    encoded = ByteTokenizer().encode(prompt+"\n"+target, answer_start=len(prompt)+1)
                    self.assertLessEqual(len(encoded.input_ids), 256)
                    self.assertNotIn("family", prompt)
                    self.assertNotIn("seed", prompt)
                    key = c.canonical(public)
                    self.assertEqual(global_inputs.setdefault(key, meta["split"]), meta["split"])
                    prefix = dict(public)
                    prefix.pop("query", None)
                    key = c.canonical(prefix)
                    self.assertEqual(prefixes.setdefault(key, meta["split"]), meta["split"])

    def test_generator_limits(self):
        for family, seed in (("unknown", 0), ("context-2", -1), ("context-2", 65536), ("context-2", True), ([], 0)):
            with self.assertRaises(ValueError):
                c.generate(family, seed)

    def test_strict_requests(self):
        for module in c.MODULES:
            r = c.generate(module+"-2", 17)[0]
            for field in r:
                bad = copy.deepcopy(r)
                del bad[field]
                with self.assertRaises(ValueError):
                    c.validate_request(module, bad)
            for key, value in (("target", 1), ("seed", 1), ("family", "x"), ("schema", "wrong")):
                with self.assertRaises(ValueError):
                    c.validate_request(module, dict(r, **{key: value}))
        r = c.generate("context-2", 17)[0]
        for key, value in (("query", True), ("query", 2), ("query", -1), ("contexts", 9), ("context_mask", [False]*2), ("symbol_mask", [1,0]), ("table", [[0]])):
            with self.assertRaises(ValueError):
                c.validate_request("context", dict(r, **{key: value}))
        r = c.generate("tombstone-2", 17)[0]
        for key, value in (("offset", True), ("offset", 17), ("before", [256]), ("removed", [1.0]), ("after", [0]*17)):
            with self.assertRaises(ValueError):
                c.validate_request("tombstone", dict(r, **{key: value}))
        r = c.generate("ternary-2", 17)[0]
        for weights in ([], [0,0], [True,1], [-1,1], [10**9+1,0]):
            with self.assertRaises(ValueError):
                c.validate_request("ternary", dict(r, weights=weights))
        for row in (dict(r["entries"][0], id="bad space"), dict(r["entries"][0], active=1), dict(r["entries"][0], mask=[False,False], trits=[1,0]), dict(r["entries"][0], mask=[1,1])):
            with self.assertRaises(ValueError):
                c.validate_request("ternary", dict(r, entries=[row]))
        with self.assertRaises(ValueError):
            c.validate_request("ternary", dict(r, entries=r["entries"]*2))

    def test_carrier_rejects_noncanonical_and_unknown(self):
        p = c.encode_public("context", c.public_observation("context", c.generate("context-2",17)[0]))
        for malformed in ("", "null", "{}", "[]", p+" ", p+"{}", p.replace('"C1"', '"C2"'), p.replace('"C1"', '"C\\u0031"'), p.replace(",2,", ",2.0,",1), "["*2000+"0"+"]"*2000, "x"*65537):
            with self.assertRaises(ValueError):
                c.decode_public(malformed)

    def test_unknown_neutral_inactive_and_exact_fractions(self):
        outcomes = [json.loads(c.expected_target("ternary", r))[0] for r in c.generate("ternary-2",17)]
        self.assertTrue(any(t[0] == "scored" and t[1] == 0 and t[2] > 0 for t in outcomes))
        self.assertTrue(any(t[0] == "abstain" and t[1] is None and t[2] == 0 for t in outcomes))
        self.assertTrue(any(t[0] == "inactive" and t[1] is None for t in outcomes))
        r = dict(schema=c.REQUEST_SCHEMAS["ternary"], input_kind="generic-public-trits", weights=[2,4], entries=[dict(id="e", active=True, trits=[1,0], mask=[True,True])])
        self.assertEqual(json.loads(c.expected_target("ternary",r)), [["scored",2,6,6,2,2]])
        for n in (2, 3, 4):
            self.assertEqual({c.expected_target("context",r) for r in c.generate(f"context-{n}",17)}, {"-1","0","1"})
        self.assertEqual({c.expected_target("tombstone",r) for r in c.generate("tombstone-2",17)}, {"argument","length","mismatch","valid-splice"})

    def test_full_provider_field_tamper_rejected(self):
        r = c.generate("context-2",17)[0]
        out = dict(schema="trinite.context-symbol-outcome.v1", relation=int(c.expected_target("context",r)))
        self.assertEqual(c.verify_outcome("context",r,out), str(out["relation"]))
        for bad in (dict(out, extra=1), dict(out, relation=True), dict(out, relation=17)):
            with self.assertRaises(ValueError):
                c.verify_outcome("context",r,bad)
        r = dict(schema=c.REQUEST_SCHEMAS["ternary"], input_kind="generic-public-trits", weights=[2,4], entries=[dict(id="e", active=True, trits=[1,0], mask=[True,True])])
        row = dict(id="e", active=True, status="scored", score=dict(numerator=2,denominator=6), coverage=dict(numerator=6,denominator=6), observed_count=2, factor_count=2)
        out = dict(schema="trinite.ternary-evidence-result.v1", entries=[row])
        self.assertEqual(c.verify_outcome("ternary",r,out), c.expected_target("ternary",r))
        for key, value in (("id","x"), ("active",1), ("factor_count",True), ("score",dict(numerator=1,denominator=3))):
            with self.assertRaises(ValueError):
                c.verify_outcome("ternary", r, dict(out, entries=[dict(row, **{key:value})]))


if __name__ == "__main__":
    unittest.main()
