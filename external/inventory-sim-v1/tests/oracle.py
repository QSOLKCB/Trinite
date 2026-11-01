#!/usr/bin/env python3
"""Independent delivery-event ledger oracle; exposed software tests, not learning.

Python unbounded arithmetic and precomputed receipt times differ from the Rust
pending-batch transition. All inputs are original synthetic conformance cases.
"""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest

KEYS = ("ordered", "delivered", "received", "discarded", "demand", "sales", "lost", "stockout_ticks")

def request(demand=None, orders=None, *, capacity=8, initial_stock=2,
            lead_times=None, supplier_available=None, observe_ticks=1, observed=None):
    demand = [2, 4, 3, 1] if demand is None else demand
    n = len(demand)
    return {"schema": "trinite.inventory-request.v1", "capacity": capacity,
            "initial_stock": initial_stock, "demand": demand,
            "orders": [3] * n if orders is None else orders,
            "lead_times": [1] * n if lead_times is None else lead_times,
            "supplier_available": [True] * n if supplier_available is None else supplier_available,
            "observe_ticks": observe_ticks, "observed": [True] * 8 if observed is None else observed}

def reference(spec):
    n = len(spec["demand"])
    # Independently precompute each batch's first available receipt event.
    # No mutable pending list or repeated due-batch removal as in production.
    batches = []
    for tick, quantity in enumerate(spec["orders"]):
        if quantity:
            due = tick + spec["lead_times"][tick]
            receipt = next((t for t in range(due, n) if spec["supplier_available"][t]), None)
            batches.append((tick, due, quantity, receipt))
    received_total = sold_total = 0
    rows = []
    for tick, demand in enumerate(spec["demand"]):
        opening_stock = spec["initial_stock"] + received_total - sold_total
        delivered = sum(q for _, _, q, receipt in batches if receipt == tick)
        received = min(delivered, spec["capacity"] - opening_stock)
        sales = min(demand, opening_stock + received)
        received_total += received
        sold_total += sales
        pending_units = sum(q for born, _, q, receipt in batches
                            if born <= tick and (receipt is None or receipt > tick))
        rows.append({"tick": tick, "ordered": spec["orders"][tick],
                     "delivered": delivered, "received": received, "discarded": delivered-received,
                     "demand": demand, "sales": sales, "lost": demand-sales,
                     "stock": spec["initial_stock"] + received_total - sold_total,
                     "pending_units": pending_units,
                     "supplier_available": spec["supplier_available"][tick]})
    def summarize(selected):
        return {k: sum((r["lost"] > 0) if k == "stockout_ticks" else r[k] for r in selected)
                for k in KEYS}
    return {"schema": "trinite.inventory-ground-truth.v1", "privileged": True,
            "capacity": spec["capacity"], "observe_ticks": spec["observe_ticks"], "steps": n,
            "initial_stock": spec["initial_stock"], "rows": rows,
            "summary": summarize(rows), "forecast": summarize(rows[spec["observe_ticks"]:]),
            "final_stock": rows[-1]["stock"],
            "pending": [{"ordered_tick": born, "due_tick": due, "quantity": q}
                        for born, due, q, receipt in batches if receipt is None]}

def projection(spec, truth):
    rows = []
    for row in truth["rows"][:spec["observe_ticks"]]:
        values = [row[k] for k in ("stock", "demand", "sales", "lost", "received", "discarded", "ordered", "pending_units")]
        rows.append({"tick": row["tick"], "supplier_available": row["supplier_available"],
                     "values": [v if known else 0 for v, known in zip(values, spec["observed"])],
                     "observed": spec["observed"]})
    return {"schema": "trinite.inventory-observation.v1", "capacity": spec["capacity"],
            "observe_ticks": spec["observe_ticks"], "horizon": len(spec["demand"])-spec["observe_ticks"], "rows": rows}

class InventoryOracleTests(unittest.TestCase):
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
        expected = reference(spec)
        actual = self.success("simulate-privileged", spec)
        self.assert_exact(actual, expected)
        self.assert_exact(self.success("observe", spec), projection(spec, expected))
        stock = spec["initial_stock"]
        pending = 0
        for row in actual["rows"]:
            stock += row["received"] - row["sales"]
            pending += row["ordered"] - row["delivered"]
            self.assertEqual(stock, row["stock"])
            self.assertEqual(pending, row["pending_units"])
            self.assertTrue(0 <= stock <= spec["capacity"])
            self.assertEqual(row["demand"], row["sales"] + row["lost"])
            self.assertEqual(row["delivered"], row["received"] + row["discarded"])
        totals = actual["summary"]
        self.assertEqual(spec["initial_stock"] + totals["ordered"],
                         totals["sales"] + totals["discarded"] + actual["final_stock"]
                         + sum(b["quantity"] for b in actual["pending"]))
        return actual

    def test_schedule_families_and_cutoffs(self):
        families = [[2]*8, [8,0,0,8,0,0,8,0], [0,5]*4, list(range(8)),
                    [0,0,7,0,0,1,0,0], [6,5,4,3,2,1,0,0]]
        for family, initial, disruption, cutoff in itertools.product(range(6), [0,5], [False,True], [1,7]):
            self.check_case(request(families[family], [2,0,5,3,0,7,1,4], capacity=9,
                                    initial_stock=initial, lead_times=[0,2,1,0,4,1,3,0],
                                    supplier_available=[not(disruption and i in [1,2,5]) for i in range(8)],
                                    observe_ticks=cutoff))

    def test_receipts_precede_demand_and_overflow_is_discarded(self):
        truth = self.check_case(request([3,1], [3,1], capacity=3, initial_stock=3, lead_times=[0,0]))
        self.assertEqual(truth["rows"][0]["discarded"], 3)
        self.assertEqual(truth["summary"]["sales"], 4)
        self.assertEqual(truth["summary"]["stockout_ticks"], 0)
        self.assertEqual(truth["final_stock"], 0)

    def test_delayed_receipts_release_all_due_and_not_future(self):
        truth = self.check_case(request([0]*5, [2,3,4,0,0], capacity=20, initial_stock=0,
                                        lead_times=[0,0,4,0,0], supplier_available=[False,False,False,True,True]))
        self.assertEqual([r["delivered"] for r in truth["rows"]], [0,0,0,5,0])
        self.assertEqual(truth["pending"], [{"ordered_tick":2,"due_tick":6,"quantity":4}])

    def test_lost_sales_no_backorders_and_stockout_definition(self):
        truth = self.check_case(request([0,4,0,1], [0,0,3,0], initial_stock=0, lead_times=[0]*4))
        self.assertEqual(truth["summary"]["lost"], 4)
        self.assertEqual(truth["summary"]["stockout_ticks"], 1)
        self.assertEqual(truth["final_stock"], 2)

    def test_due_at_horizon_and_overdue_batches_preserved(self):
        truth = self.check_case(request([0]*3, [2,3,4], initial_stock=0,
                                        lead_times=[3,0,32], supplier_available=[False]*3))
        self.assertEqual(truth["pending"], [
            {"ordered_tick":0,"due_tick":3,"quantity":2},
            {"ordered_tick":1,"due_tick":1,"quantity":3},
            {"ordered_tick":2,"due_tick":34,"quantity":4}])
        self.assertEqual(truth["summary"]["delivered"], 0)

    def test_all_masks_and_zero_unknown_supplier_distinction(self):
        full = request([0,3,2], [0,2,0], initial_stock=0, observe_ticks=2)
        truth = self.success("simulate-privileged", full)
        for bits in itertools.product([False,True], repeat=8):
            masked = copy.deepcopy(full); masked["observed"] = list(bits)
            self.assert_exact(self.success("simulate-privileged", masked), truth)
            self.assert_exact(self.success("observe", masked), projection(masked, truth))
        missing = copy.deepcopy(full); missing["observed"] = [False]*8
        unavailable = copy.deepcopy(missing); unavailable["supplier_available"] = [False]*3
        a = self.success("observe", full)["rows"][0]
        b = self.success("observe", missing)["rows"][0]
        c = self.success("observe", unavailable)["rows"][0]
        self.assertEqual(a["values"][0],b["values"][0]); self.assertNotEqual(a["observed"],b["observed"])
        self.assertTrue(b["supplier_available"]); self.assertFalse(c["supplier_available"])

    def test_future_suffix_cannot_change_observation_bytes(self):
        a = request([1,2,3,4,5,6], observe_ticks=2)
        b = copy.deepcopy(a)
        for key in ["demand","orders","lead_times"]: b[key][2:] = [32]*4
        b["supplier_available"][2:] = [False]*4
        self.assertEqual(self.run_cli("observe",a).stdout,self.run_cli("observe",b).stdout)
        self.assertNotEqual(self.success("simulate-privileged",a),self.success("simulate-privileged",b))

    def test_maximum_bounds_and_mass_delivery(self):
        for available in [[False]*256,[False]*255+[True],[True]*256]:
            self.check_case(request([32]*256,[32]*256,capacity=128,initial_stock=128,
                                    lead_times=[0]*256,supplier_available=available,observe_ticks=255))
        self.check_case(request([0,0],[0,0],capacity=1,initial_stock=0))

    def test_exact_replay_and_field_order(self):
        spec=request(); first=self.run_cli("simulate-privileged",spec)
        self.assertEqual(first.returncode,0)
        self.assertEqual(first.stdout,self.run_cli("simulate-privileged",spec).stdout)
        self.assertEqual(first.stdout,self.run_cli("simulate-privileged",dict(reversed(list(spec.items())))).stdout)

    def assert_rejected(self,spec=None,raw=None,mode="observe"):
        result=self.run_cli(mode,spec,raw)
        self.assertEqual(result.returncode,2,(spec,raw,result.stderr)); self.assertEqual(result.stdout,b"")
        self.assertTrue(result.stderr)

    def test_invalid_request_fields(self):
        for key in request():
            spec=request(); del spec[key]; self.assert_rejected(spec)
        cases={"schema":["wrong",1],"capacity":[0,129,True,1.5],"initial_stock":[-1,9,False],
               "observe_ticks":[0,4,5,True],"demand":[[],[1],[1]*257,[33]*4,[-1]*4,[True]*4],
               "orders":[[1],[33]*4,[-1]*4,[False]*4],"lead_times":[[1],[33]*4,[-1]*4,[False]*4],
               "supplier_available":[[],[False],[0]*4],"observed":[[],[True]*7,[True]*9,[1]*8]}
        for key,values in cases.items():
            for value in values:
                spec=request();spec[key]=value;self.assert_rejected(spec)
        extra=request();extra["unexpected"]=0;self.assert_rejected(extra)

    def test_strict_wire_rejections(self):
        valid=json.dumps(request(),separators=(",",":")).encode()
        malformed=[b"",b"[]",b"{}",b"null",valid+b"{}",
                   valid.replace(b'"capacity":8',b'"capacity":08'),
                   valid.replace(b'"initial_stock":2',b'"initial_stock":-0'),
                   valid.replace(b'"capacity":8',b'"capacity":8,"capacity":8'),
                   valid.replace(b'"capacity":8',b'"capacity":9223372036854775808'),
                   valid.replace(b'"schema"',b'"sche\\u006da"'),
                   b"["*10+b"0"+b"]"*10,b" "*16385]
        for raw in malformed:self.assert_rejected(raw=raw)
        for mode in ["","simulate","unknown"]:self.assert_rejected(request(),mode=mode)

if __name__ == "__main__":
    if len(sys.argv)!=2:raise SystemExit("usage: oracle.py /path/to/trinite-inventory-sim-v1")
    InventoryOracleTests.binary=str(Path(sys.argv.pop()).resolve(strict=True))
    unittest.main(verbosity=2)
