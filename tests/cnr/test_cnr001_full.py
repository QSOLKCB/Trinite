"""Mandatory authentic MaleCNS-v1.0 Parquet execution: separate pinned CI lane."""
from pathlib import Path
import json
import tarfile
import tempfile
import unittest

from trinite.cnr001 import core
from trinite.contracts import identity, json_bytes, ContractError

ROOT=Path(__file__).resolve().parents[2]


class CNR001FullBiology(unittest.TestCase):
    def test_real_connectome_ingestion_and_baseline_measurement(self):
        ids, edges, metadata=core.biological_graph()  # requires real pyarrow v21
        self.assertEqual(len(ids),100)
        self.assertEqual(len(edges),3630)
        self.assertEqual(metadata["provenance"]["dataset"],"male-cns:v1.0")
        report=core.analyze()
        self.assertEqual(report["graph_provenance"]["biological_provenance"]["dataset"],"male-cns:v1.0")
        self.assertEqual(report["biological_data_source_git_blobs"],
                         core.protocol()[0]["source_git_blobs"])
        self.assertEqual(report["baseline"]["original_baseline_commit"],core.BASELINE_SHA)
        for layer in report["layers"].values():
            self.assertEqual(layer["baseline_from_v0_1_0"],
                             layer["arms"]["baseline-menger"])
            self.assertEqual({x["edge_count"] for x in layer["arms"].values()},{24})
            self.assertEqual(len({x["receiver_identity"] for x in layer["arms"].values()}),1)
        self.assertEqual(json_bytes(report),json_bytes(core.analyze()))

        self.assertEqual(core.protocol()[0]["schema"], "trinite.cnr-001-protocol.v4")
        self.assertEqual(report["schema"], "trinite.cnr-001-analysis.v4")
        self.assertEqual(core.attest_reader()[0].__name__, "pyarrow")

    def test_fresh_request_run_and_read_only_verification(self):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("cnr001_runner", ROOT / "scripts/run_cnr001.py")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            rpath = Path(tmp) / "request.json"
            apath = Path(tmp) / "analysis.json"
            frozen = module.freeze(rpath)
            req_id = frozen["request_identity"]
            ran = module.run(rpath, req_id, apath)
            before = (rpath.read_bytes(), apath.read_bytes())
            got = module.verify(rpath, req_id, apath, ran["analysis_identity"])
            self.assertEqual(got["status"], "passed")
            self.assertTrue(got["read_only"])
            self.assertEqual(before, (rpath.read_bytes(), apath.read_bytes()))
            # The historical v5 packet belongs to its original published
            # source closure; importing unrelated future modules changes the
            # fresh request identity. Never reseal v5 or pretend it is current.
            original_v4 = ROOT / "fixtures/cnr-001-evidence/cnr001-v4.tar.gz"
            self.assertEqual(identity(original_v4.read_bytes()),
                             "sha256:95d16ee195270aa9d41d08be1b63382ae9db804f9306cb2791740f81c0b733a4")
            historical = ROOT / "fixtures/cnr-001-evidence/cnr001-v5.tar.gz"
            self.assertEqual(
                identity(historical.read_bytes()),
                "sha256:03e735f1101d26f935134ff9fe3f95fce43d6e40b2a03e0f9de0136305ee2144")
            with tarfile.open(historical, "r:gz") as old_archive:
                old_request = old_archive.extractfile("v5/cnr-request.json").read()
                previous = json.loads(old_request)
            current = json.loads(before[0])
            self.assertNotEqual(before[0], old_request)
            self.assertEqual(
                {k: v for k, v in previous.items() if k != "source"},
                {k: v for k, v in current.items() if k != "source"})
            # A newly versioned packet must bind every byte of this producer.
            # No selective equality exceptions or regenerated old manifests.
            new_archive = ROOT / "fixtures/cnr-001-evidence/cnr001-v6.tar.gz"
            digest_file = ROOT / "fixtures/cnr-001-evidence/cnr001-v6.sha256"
            self.assertTrue(new_archive.is_file(), "missing source-bound v6 evidence packet")
            self.assertTrue(digest_file.is_file(), "missing externally held v6 digest")
            recorded = digest_file.read_text("ascii").strip().split()
            self.assertEqual(len(recorded), 2)
            self.assertEqual(recorded[1], "cnr001-v6.tar.gz")
            # Independent reviewed release commitment, not a value learned
            # from the packet being verified in this test.
            self.assertRegex(recorded[0], r"^[0-9a-f]{64}$")  # Pin the independent digest after v6 capture
            self.assertEqual(identity(new_archive.read_bytes()), "sha256:" + recorded[0])
            with tarfile.open(new_archive, "r:gz") as archive:
                names = archive.getnames()
                self.assertEqual(len(names), len(set(names)))
                self.assertEqual(
                    set(names), {"MANIFEST.json", "v6/cnr-request.json",
                                "v6/cnr-analysis.json", "v6/cnr-freeze-receipt.json",
                                "v6/cnr-run-receipt.json", "v6/cnr-verification.json"})
                manifest = json.loads(archive.extractfile("MANIFEST.json").read())
                self.assertEqual(manifest["schema"], "trinite.cnr-001-evidence-archive.v6")
                self.assertEqual(set(manifest["entries"]), set(names) - {"MANIFEST.json"})
                self.assertEqual(json_bytes(manifest), archive.extractfile("MANIFEST.json").read())
                self.assertEqual(manifest["parent_v5_sha256"], "03e735f1101d26f935134ff9fe3f95fce43d6e40b2a03e0f9de0136305ee2144")
                for member, attrs in manifest["entries"].items():
                    raw_member = archive.extractfile(member).read()
                    self.assertEqual(attrs, {"bytes": len(raw_member),
                                             "sha256": identity(raw_member)[7:]})
                self.assertEqual(before[0], archive.extractfile("v6/cnr-request.json").read())
                self.assertEqual(before[1], archive.extractfile("v6/cnr-analysis.json").read())
                self.assertEqual(json_bytes(got), archive.extractfile("v6/cnr-verification.json").read())
            with self.assertRaises(FileExistsError):
                module.freeze(rpath)
            apath.write_bytes(before[1] + b" ")
            with self.assertRaisesRegex(ContractError, "external CNR analysis commitment differs"):
                module.verify(rpath, req_id, apath, ran["analysis_identity"])
            # Even a refreshed external identity cannot bless unrelated bytes.
            with self.assertRaisesRegex(ContractError, "replay mismatch"):
                module.verify(rpath, req_id, apath, identity(apath.read_bytes()))


if __name__ == "__main__":
    unittest.main()
