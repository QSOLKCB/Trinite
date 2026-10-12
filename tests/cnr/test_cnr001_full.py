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


if __name__=="__main__":
    unittest.main()
