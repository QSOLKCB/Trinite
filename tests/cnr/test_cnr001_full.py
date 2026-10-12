"""Mandatory authentic MaleCNS-v1.0 Parquet execution: separate pinned CI lane."""
from pathlib import Path
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
