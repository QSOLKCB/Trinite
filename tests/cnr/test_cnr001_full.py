"""Mandatory authentic MaleCNS-v1.0 Parquet execution: separate pinned CI lane."""
from pathlib import Path
import io
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
        from trinite.contracts import parse_json
        from importlib.util import spec_from_file_location, module_from_spec
        spec=spec_from_file_location("cnr001_runner",ROOT/"scripts/run_cnr001.py")
        module=module_from_spec(spec)
        spec.loader.exec_module(module)
        packet=ROOT/"fixtures/cnr-001-evidence/cnr001-v1-v2.tar.gz"
        with tarfile.open(packet,"r:gz") as archive:
            request_bytes=archive.extractfile("v2/cnr-request.json").read()
            analysis_bytes=archive.extractfile("v2/cnr-analysis.json").read()
            verification_bytes=archive.extractfile("v2/cnr-verification.json").read()
        self.assertEqual(request_bytes,json_bytes(module.expected_request()))
        import hashlib
        request_id="sha256:"+hashlib.sha256(request_bytes).hexdigest()
        report["request_identity"]=request_id
        self.assertEqual(analysis_bytes,json_bytes(report))
        with tempfile.TemporaryDirectory() as tmp:
            rfile=Path(tmp)/"request.json"; afile=Path(tmp)/"analysis.json"
            rfile.write_bytes(request_bytes);afile.write_bytes(analysis_bytes)
            external="sha256:"+hashlib.sha256(analysis_bytes).hexdigest()
            self.assertEqual(json_bytes(module.verify(rfile,request_id,afile,external)),
                             verification_bytes)



if __name__=="__main__":
    unittest.main()
