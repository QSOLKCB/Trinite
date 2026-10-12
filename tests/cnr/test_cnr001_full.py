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

        # Historical v2 packet is retained for its *own* archived producer;
        # active v3 requests are separately source-bound and must not be
        # mistaken for a replay of the v2 numerical implementation.
        self.assertEqual(core.protocol()[0]["schema"],"trinite.cnr-001-protocol.v3")
        self.assertEqual(report["schema"],"trinite.cnr-001-analysis.v3")
        self.assertEqual(core.attest_reader()[0].__name__,"pyarrow")
        # The newly frozen v3 archive must regenerate bit-for-bit under the
        # current true hash-locked reader and complete source contract.
        from importlib.util import spec_from_file_location, module_from_spec
        from trinite.contracts import json_bytes
        import hashlib
        spec=spec_from_file_location("cnr001_runner",ROOT/"scripts/run_cnr001.py")
        module=module_from_spec(spec);spec.loader.exec_module(module)
        with tarfile.open(ROOT/"fixtures/cnr-001-evidence/cnr001-v3.tar.gz","r:gz") as archived:
            request=archived.extractfile("v3/cnr-request.json").read()
            analysis=archived.extractfile("v3/cnr-analysis.json").read()
            receipt=archived.extractfile("v3/cnr-verification.json").read()
        self.assertEqual(request,json_bytes(module.expected_request()))
        req_id="sha256:"+hashlib.sha256(request).hexdigest()
        expected=core.analyze();expected["request_identity"]=req_id
        self.assertEqual(analysis,json_bytes(expected))
        with tempfile.TemporaryDirectory() as tmp:
            rpath=Path(tmp)/"request.json"; apath=Path(tmp)/"analysis.json"
            rpath.write_bytes(request); apath.write_bytes(analysis)
            got=module.verify(rpath,req_id,apath,"sha256:"+hashlib.sha256(analysis).hexdigest())
            self.assertEqual(json_bytes(got),receipt)



if __name__=="__main__":
    unittest.main()
