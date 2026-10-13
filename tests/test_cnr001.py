"""CNR-001 offline contract and topology tests; never certify biological source by fake rows."""
from pathlib import Path
import io
import hashlib
import json
import tarfile
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
        self.assertEqual(p["schema"], "trinite.cnr-001-protocol.v4")
        self.assertEqual(p["evidence_revision"].split(";")[0], "v4-installed-pins-and-authentic-input-boundary")
        for path, expected in p["baseline_git_blobs"].items():
            self.assertEqual(core.git_blob(core.read(core.ROOT/path)), expected)
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
        self.assertIn("src/trinite/cnr001/protocol-v2.json", source)
        self.assertIn("src/trinite/cnr001/protocol-v3.json", source)
        self.assertIn("src/trinite/cnr001/protocol-v4.json", source)
        self.assertIn("src/trinite/e8_reference/__init__.py", source)
        self.assertIn("src/trinite/lattice_observation/experiment.py", source)
        self.assertIn(".github/workflows/cnr-001.yml", source)
        self.assertIn("scripts/run_cnr001.py", source)
        self.assertEqual(source["src/trinite/packed_ternary/__init__.py"],
                         identity((ROOT / "src/trinite/packed_ternary/__init__.py").read_bytes()))

    def test_permanent_v1_v2_archive_and_external_commitments(self):
        packet = ROOT / "fixtures/cnr-001-evidence/cnr001-v1-v2.tar.gz"
        raw = packet.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         "be7ec8ade4c19c05bff5f11741a2b48b4a7760e8ec8fa811c263a021354f18df")
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            entries = archive.getmembers()
            self.assertEqual(len({e.name for e in entries}), len(entries))
            self.assertTrue(all(e.isfile() and e.size < 256 * 1024
                                and ".." not in e.name.split("/")
                                and not e.name.startswith("/") for e in entries))
            files = {e.name: archive.extractfile(e).read() for e in entries}
        manifest = json.loads(files["MANIFEST.json"])
        self.assertEqual(manifest["schema"], "trinite.cnr-001-evidence-archive.v1")
        self.assertEqual(set(manifest["entries"]), set(files)-{"MANIFEST.json"})
        for name, record in manifest["entries"].items():
            self.assertEqual(len(files[name]), record["bytes"])
            self.assertEqual(hashlib.sha256(files[name]).hexdigest(), record["sha256"])
        self.assertEqual(hashlib.sha256(files["v1/cnr-request.json"]).hexdigest(),
                         "5363b6e5548579e1581434756e7c73a02729ee6d0e7056e4776f319c07d80b86")
        self.assertEqual(hashlib.sha256(files["v1/cnr-analysis.json"]).hexdigest(),
                         "0753d5d538aa2c18715b659148190e67d76b460996490def7c6e8c4b1230a11b")
        self.assertEqual(hashlib.sha256(files["v2/cnr-request.json"]).hexdigest(),
                         "2d402c2786968e8ce9b844ae9b8e8aafca293500ee2bb2088b3fc9ddc7bcc6ab")
        self.assertEqual(hashlib.sha256(files["v2/cnr-analysis.json"]).hexdigest(),
                         "58477fb9ac31eb39063c17953164afc230b0142aa5a264fed9e67fc0169a2711")
        self.assertEqual(json.loads(files["v1/cnr-request.json"])["protocol"]["schema"],
                         "trinite.cnr-001-protocol.v1")
        self.assertEqual(json.loads(files["v2/cnr-request.json"])["protocol"]["schema"],
                         "trinite.cnr-001-protocol.v2")

    def test_separate_immutable_v3_packet_and_versioned_schema(self):
        packet = ROOT / "fixtures/cnr-001-evidence/cnr001-v3.tar.gz"
        raw = packet.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         "fcec1f268a3985aeb921899d2cfcd8886ec91d53598faa6184761eb800a5ef5a")
        with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as archive:
            entries=archive.getmembers()
            self.assertEqual(len({item.name for item in entries}),len(entries))
            self.assertTrue(all(item.isfile() and item.size < 256*1024 and
                                not item.name.startswith("/") and
                                ".." not in item.name.split("/") for item in entries))
            contents={item.name:archive.extractfile(item).read() for item in entries}
        manifest=json.loads(contents["MANIFEST.json"])
        self.assertEqual(manifest["schema"],"trinite.cnr-001-evidence-archive.v3")
        self.assertEqual(manifest["historical_v1_v2_sha256"],
                         "be7ec8ade4c19c05bff5f11741a2b48b4a7760e8ec8fa811c263a021354f18df")
        self.assertEqual(set(manifest["entries"]),set(contents)-{"MANIFEST.json"})
        for name, entry in manifest["entries"].items():
            self.assertEqual(len(contents[name]),entry["bytes"])
            self.assertEqual(hashlib.sha256(contents[name]).hexdigest(),entry["sha256"])
        self.assertEqual(hashlib.sha256(contents["v3/cnr-request.json"]).hexdigest(),
                         "30ec9f8eb2e8ef2b0c4b4e0dc7276337a589003c47bcbcd7367816d078c5cf6f")
        self.assertEqual(hashlib.sha256(contents["v3/cnr-analysis.json"]).hexdigest(),
                         "6d6b8574ab89720d0b3124e4333e2b26be8d0ccae15539a9533075da7b1a9de1")
        self.assertEqual(json.loads(contents["v3/cnr-request.json"])["schema"],
                         "trinite.cnr-001-request.v3")
        self.assertEqual(json.loads(contents["v3/cnr-analysis.json"])["schema"],
                         "trinite.cnr-001-analysis.v3")

    def test_permanent_v4_packet_has_independent_commitments(self):
        raw = (ROOT / "fixtures/cnr-001-evidence/cnr001-v4.tar.gz").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         "95d16ee195270aa9d41d08be1b63382ae9db804f9306cb2791740f81c0b733a4")
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            members = archive.getmembers()
            self.assertEqual(len({m.name for m in members}), len(members))
            self.assertTrue(all(m.isfile() and m.size < 256 * 1024 for m in members))
            files = {m.name: archive.extractfile(m).read() for m in members}
        manifest = json.loads(files["MANIFEST.json"])
        self.assertEqual(manifest["schema"], "trinite.cnr-001-evidence-archive.v4")
        self.assertEqual(manifest["producer_run"], "38295236828")
        self.assertEqual(set(manifest["entries"]), set(files) - {"MANIFEST.json"})
        for name, entry in manifest["entries"].items():
            self.assertEqual(len(files[name]), entry["bytes"])
            self.assertEqual(hashlib.sha256(files[name]).hexdigest(), entry["sha256"])
        request = json.loads(files["v4/cnr-request.json"])
        report = json.loads(files["v4/cnr-analysis.json"])
        self.assertEqual(request["schema"], "trinite.cnr-001-request.v4")
        self.assertEqual(report["schema"], "trinite.cnr-001-analysis.v4")
        self.assertEqual(report["request_identity"], identity(files["v4/cnr-request.json"]))

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
        self.assertEqual(bool(count), rebuilt != control)
        # A pair of edges sharing their source cannot be counted as a swap
        # when the only proposed output is the same pair of edges.
        rigid = [[0, i] for i in range(1, 20)]
        rigid += [[1, j] for j in range(2, 7)]
        self.assertEqual(len(rigid),24)
        rigid_after, rigid_count = core.rewired(sorted(rigid))
        self.assertEqual(rigid_count, 0)
        self.assertEqual(rigid_after, sorted(rigid))
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

    def test_injected_samples_cannot_claim_authentic_data_commitments(self):
        from unittest.mock import patch
        sample = synthetic_sample()
        # Even authentic-looking metadata must not make injected rows official.
        for dataset in ("test-synthetic-nonbiological", "male-cns:v1.0"):
            sample[2]["provenance"]["dataset"] = dataset
            with self.subTest(dataset=dataset), patch.object(core, "candidate_graphs") as graphs:
                with self.assertRaisesRegex(ContractError, "rejects injected biological samples"):
                    core.analyze(sample)
                graphs.assert_not_called()

    def test_software_only_graph_controls_remain_reproducible(self):
        graph = synthetic_sample()
        graphs, provenance = core.candidate_graphs(graph)
        self.assertEqual((graphs, provenance), core.candidate_graphs(graph))
        self.assertEqual(len(graphs), 8)
        self.assertTrue(all(len(edges) == 24 for edges in graphs.values()))
        for arm in ("male-cns20", "etq101-20", "etq303-20"):
            receipt = provenance["rewire_swaps"][arm]
            self.assertEqual(receipt["eligible_control"],
                             receipt["effective_swaps"] > 0 and receipt["distinct_final_topology"])
        self.assertEqual(provenance["biological_provenance"]["dataset"],
                         "test-synthetic-nonbiological")

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
