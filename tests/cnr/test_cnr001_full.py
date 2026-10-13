"""Mandatory authentic MaleCNS-v1.0 Parquet execution: separate pinned CI lane."""
from pathlib import Path
import json
import tarfile
import tempfile
import unittest

from trinite.cnr001 import core
from trinite.contracts import identity, json_bytes, ContractError

ROOT=Path(__file__).resolve().parents[2]
V7_SHA256 = "03327a365b76474bfaf0956eb06766cabdf9de02e5ca1162c0c1474f9d893288"
V8_SHA256 = "718e5fec4674ab615b4162a956500b0bc245e25f15153770a5bfa4e6f7d72b42"
PACKET_FILES = ("cnr-request.json", "cnr-analysis.json", "cnr-freeze-receipt.json",
                "cnr-run-receipt.json", "cnr-verification.json")


class CNR001FullBiology(unittest.TestCase):
    def assert_retained_sha256(self, archive, sidecar, expected):
        """Require the independently retained checksum as well as archive bytes."""
        self.assertTrue(sidecar.is_file(), "missing retained sidecar: " + sidecar.name)
        self.assertEqual(sidecar.read_text(encoding="ascii"),
                         expected + "  " + archive.name + "\n")
        self.assertEqual(identity(archive.read_bytes()), "sha256:" + expected)

    def assert_packet_members(self, members, version, parent_version, parent_digest):
        """Archive structure only; actual numerical replay is separately required."""
        expected = {"MANIFEST.json"} | {version + "/" + name for name in PACKET_FILES}
        self.assertEqual(set(members), expected)
        manifest_bytes = members["MANIFEST.json"]
        manifest = json.loads(manifest_bytes)
        self.assertEqual(json_bytes(manifest), manifest_bytes)
        self.assertEqual(manifest["schema"], "trinite.cnr-001-evidence-archive." + version)
        self.assertEqual(manifest["parent_" + parent_version + "_sha256"], parent_digest)
        self.assertEqual(set(manifest["entries"]), expected - {"MANIFEST.json"})
        for name, attr in manifest["entries"].items():
            raw = members[name]
            self.assertEqual(attr, {"bytes": len(raw), "sha256": identity(raw)[7:]})
        self.assertEqual(manifest["request_identity"],
                         identity(members[version + "/cnr-request.json"]))
        self.assertEqual(manifest["analysis_identity"],
                         identity(members[version + "/cnr-analysis.json"]))
        return manifest

    def read_packet(self, path):
        with tarfile.open(path, "r:gz") as archive:
            entries = archive.getmembers()
            self.assertTrue(all(m.isfile() for m in entries))
            self.assertEqual(len(entries), len({m.name for m in entries}))
            return {m.name: archive.extractfile(m).read() for m in entries}

    def test_sidecar_deletion_and_rewrite_fail_closed(self):
        for version in ("v7", "v8", "v9"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                archive = Path(temporary) / ("cnr001-" + version + ".tar.gz")
                sidecar = Path(temporary) / ("cnr001-" + version + ".sha256")
                archive.write_bytes(b"synthetic archive contents")
                digest = identity(archive.read_bytes())[7:]
                sidecar.write_text(digest + "  " + archive.name + "\n", encoding="ascii")
                self.assert_retained_sha256(archive, sidecar, digest)
                sidecar.unlink()
                with self.assertRaises(AssertionError):
                    self.assert_retained_sha256(archive, sidecar, digest)
                for modified in (("0" * 64) + "  " + archive.name + "\n",
                                 digest + "  cnr001-v6.tar.gz\n",
                                 digest + " " + archive.name + "\n"):
                    sidecar.write_text(modified, encoding="ascii")
                    with self.subTest(sidecar=modified), self.assertRaises(AssertionError):
                        self.assert_retained_sha256(archive, sidecar, digest)

    def test_immutable_v8_packet_sidecar_and_v7_ancestry(self):
        root = ROOT / "fixtures/cnr-001-evidence"
        self.assert_retained_sha256(root / "cnr001-v7.tar.gz",
                                    root / "cnr001-v7.sha256", V7_SHA256)
        self.assert_retained_sha256(root / "cnr001-v8.tar.gz",
                                    root / "cnr001-v8.sha256", V8_SHA256)
        members = self.read_packet(root / "cnr001-v8.tar.gz")
        self.assert_packet_members(members, "v8", "v7", V7_SHA256)

    def test_packet_wrong_parent_or_member_fails_closed(self):
        members = self.read_packet(ROOT / "fixtures/cnr-001-evidence/cnr001-v8.tar.gz")
        with self.assertRaises(AssertionError):
            self.assert_packet_members(members, "v8", "v7", "0" * 64)
        # In-memory corruption tests do not replace or regenerate any packet.
        for name in PACKET_FILES:
            changed = dict(members)
            changed["v8/" + name] += b" "
            with self.subTest(member=name), self.assertRaises(AssertionError):
                self.assert_packet_members(changed, "v8", "v7", V7_SHA256)

    def retained_v9_members(self):
        root = ROOT / "fixtures/cnr-001-evidence"
        self.assert_retained_sha256(root / "cnr001-v8.tar.gz",
                                    root / "cnr001-v8.sha256", V8_SHA256)
        archive, sidecar = root / "cnr001-v9.tar.gz", root / "cnr001-v9.sha256"
        self.assertTrue(archive.is_file(), "missing newly published v9 packet")
        self.assertTrue(sidecar.is_file(), "missing independently retained v9 digest")
        words = sidecar.read_text("ascii").strip().split()
        self.assertEqual(len(words), 2)
        self.assertRegex(words[0], r"^[a-f0-9]{64}$")
        self.assert_retained_sha256(archive, sidecar, words[0])
        members = self.read_packet(archive)
        self.assert_packet_members(members, "v9", "v8", V8_SHA256)
        return members

    def test_retained_v9_packet_has_exact_v8_ancestry(self):
        self.retained_v9_members()

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
            # V6 and v7 belong to their own exact producing revisions;
            # comparing away changed source receipts is never acceptable.
            v6_archive = ROOT / "fixtures/cnr-001-evidence/cnr001-v6.tar.gz"
            self.assertEqual(identity(v6_archive.read_bytes()),
                             "sha256:86cb1be634f41cceabb7f1b49c5b5d4803b5a182ca04167b27ff96aea36a0aa4")
            v7_archive = ROOT / "fixtures/cnr-001-evidence/cnr001-v7.tar.gz"
            v7_sidecar = ROOT / "fixtures/cnr-001-evidence/cnr001-v7.sha256"
            self.assert_retained_sha256(
                v7_archive, v7_sidecar,
                "03327a365b76474bfaf0956eb06766cabdf9de02e5ca1162c0c1474f9d893288")
            # V8 also belongs to its immutable historical producer. The new
            # codec and fixed workflow require a separately committed v9 packet.
            v8_archive = ROOT / "fixtures/cnr-001-evidence/cnr001-v8.tar.gz"
            self.assert_retained_sha256(
                v8_archive, ROOT / "fixtures/cnr-001-evidence/cnr001-v8.sha256",
                V8_SHA256)
            members = self.retained_v9_members()
            manifest = json.loads(members["MANIFEST.json"])
            self.assertEqual(manifest["request_identity"], req_id)
            self.assertEqual(manifest["analysis_identity"], ran["analysis_identity"])
            self.assertEqual(members["v9/cnr-request.json"], before[0])
            self.assertEqual(members["v9/cnr-analysis.json"], before[1])
            self.assertEqual(members["v9/cnr-freeze-receipt.json"], json_bytes(frozen))
            self.assertEqual(members["v9/cnr-run-receipt.json"], json_bytes(ran))
            self.assertEqual(members["v9/cnr-verification.json"], json_bytes(got))
            self.assertNotEqual(before[0], old_request)
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
