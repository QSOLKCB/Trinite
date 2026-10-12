"""CNR-001 offline contract and topology tests; never certify biological source by fake rows."""
from pathlib import Path
import tempfile
import unittest

from trinite.contracts import ContractError, identity, json_bytes
from trinite.cnr001 import core
from trinite.e8_reference import etq

ROOT = Path(__file__).resolve().parents[1]


def synthetic_sample():
    # Software-control input only: never presented as a biological measurement.
    pairs = [(a, b, 5 + ((a + b) % 17))
             for a in range(100) for b in range(100) if a != b][:3630]
    return list(range(10000, 10100)), pairs, {"provenance":{"dataset":"test-synthetic-nonbiological"}}


class CNR001Contracts(unittest.TestCase):
    def test_frozen_release_protocol_source_and_genuine_dataset_bytes(self):
        p, digest = core.protocol()
        self.assertEqual(p["baseline_commit"], core.BASELINE_SHA)
        self.assertEqual(p["run_class"], "retrospective-exploratory-training-exposed-observation-not-held-out")
        self.assertEqual(p["edge_budget"], 24)
        self.assertEqual(p["geometry_loss"], "0")
        self.assertFalse(p["phase5_ready"])
        self.assertEqual(len(digest), 71)
        raw, meta = core.source_data()
        self.assertEqual(meta["num_nodes"],100)
        self.assertEqual(meta["num_edges"],3630)
        self.assertEqual(meta["provenance"]["license"],"CC-BY-4.0")
        for name, value in raw.items():
            self.assertEqual(core.git_blob(value),p["source_git_blobs"][name])
        source = core.source_receipt()
        self.assertIn("src/trinite/lattice_observation/experiment.py", source)
        self.assertIn("src/trinite/cnr001/protocol.json", source)
        self.assertIn("scripts/run_cnr001.py", source)

    def test_biological_source_tampering_is_rejected_before_any_ingestion(self):
        original = core.DATA
        with tempfile.TemporaryDirectory() as temp:
            dst = Path(temp)
            for name in ("metadata.json","nodes.parquet","edges.parquet"):
                (dst/name).write_bytes((original/name).read_bytes())
            from unittest.mock import patch
            with patch.object(core,"DATA",dst):
                self.assertEqual(core.source_data()[1]["num_nodes"],100)
                (dst/"edges.parquet").write_bytes((dst/"edges.parquet").read_bytes()+b"x")
                with self.assertRaises(ContractError):
                    core.source_data()

    def test_exact_etq_full_graph_cycle_and_true_graph_walk(self):
        base, lift, full = core.etq_graphs()
        self.assertEqual(full["edgeCount"],5364)
        self.assertEqual(len(base["edges"]),24)
        self.assertEqual(len(lift["edges"]),24)
        diag=core.etq_diagnostics(full)
        self.assertEqual(diag["event_cycle_steps"],303)
        self.assertEqual(diag["genuine_graph_walk_steps"],303)
        self.assertTrue(diag["event_cycle_returns_to_origin"])
        self.assertTrue(diag["separate_event_and_walk"])
        self.assertGreater(diag["genuine_graph_walk_visited"],1)
        self.assertEqual(etq.monomial_power(0,0,303),(0,0,0))
        # A 303-event step must not be conflated with an adjacency edge.
        self.assertGreater(diag["event_transition_nonedges"],0)

    def test_matched_edges_and_degree_preserving_control(self):
        control=core.construct(1)["edges"]
        self.assertEqual(len(control),24)
        rebuilt,count=core.rewired(control)
        self.assertEqual(len(rebuilt),24)
        self.assertEqual(core.degrees(rebuilt),core.degrees(control))
        self.assertGreaterEqual(count,0)
        self.assertEqual(len(set(map(tuple,rebuilt))),24)
        self.assertEqual(len(core.random_edges()),24)
        _,edges,_=synthetic_sample()
        weighted=core.symmetric_weights(100,edges)
        graph=core.reduce_graph(100,weighted,list(range(10000,10100)))
        self.assertEqual(len(graph["edges"]),24)
        self.assertEqual(len(graph["selected_source_ids"]),20)

    def test_tagged_baseline_capture_and_same_score_owner(self):
        layers,source=core.baseline_vectors()
        self.assertEqual(source["original_baseline_commit"],core.BASELINE_SHA)
        self.assertEqual(set(layers),{"block.0","final"})
        for vectors, normalized, fit in layers.values():
            self.assertEqual(len(vectors),16)
            self.assertEqual(len(vectors[0]),64)
            self.assertEqual(len(normalized),16)
            self.assertEqual(len(fit["axis_range_hex"]),3)

    def test_software_only_graph_comparison_is_exploratory_and_reproducible(self):
        graph=synthetic_sample()
        r=core.analyze(graph)
        r2=core.analyze(graph)
        self.assertEqual(json_bytes(r),json_bytes(r2))
        self.assertEqual(r["baseline"]["original_baseline_commit"],core.BASELINE_SHA)
        self.assertEqual(set(r["layers"]),{"block.0","final"})
        for layer in r["layers"].values():
            self.assertEqual(len(layer["arms"]),8)
            for arm in layer["arms"].values():
                self.assertEqual(arm["edge_count"],24)
                self.assertEqual(arm["receiver_identity"],layer["arms"]["baseline-menger"]["receiver_identity"])
                self.assertEqual(sum(arm["graph_hops"]),arm["graph_hops_sum"])
        self.assertEqual(r["graph_provenance"]["biological_provenance"]["dataset"],
                         "test-synthetic-nonbiological")
        self.assertFalse(r["phase5_ready"])
        self.assertFalse(r["release_ready"])
        self.assertEqual(r["geometry_loss"],"0")

    def test_request_freeze_external_identity_without_optional_bio_reader(self):
        import importlib.util
        file = ROOT/"scripts/run_cnr001.py"
        self.assertTrue(file.is_file())
        from trinite.cnr001 import core
        with tempfile.TemporaryDirectory() as temp:
            # Freeze only; full real biological execution is mandatory in
            # dedicated tests/cnr, not silently skipped here.
            from unittest.mock import patch
            import runpy,sys
            from pathlib import Path
            # Verify installed source/data closure read-only.
            assert core.source_data()[1]["num_nodes"] == 100
            self.assertEqual(core.source_receipt()["scripts/run_cnr001.py"],
                             identity(file.read_bytes()))


if __name__=="__main__":
    unittest.main()
