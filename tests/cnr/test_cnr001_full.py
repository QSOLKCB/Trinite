"""Mandatory authentic MaleCNS-v1.0 Parquet execution: separate pinned CI lane."""
from pathlib import Path
import json
import gzip
import tarfile
import tempfile
import unittest

from trinite.cnr001 import core
from trinite.contracts import identity, json_bytes, parse_json, ContractError

ROOT=Path(__file__).resolve().parents[2]
V7_SHA256 = "03327a365b76474bfaf0956eb06766cabdf9de02e5ca1162c0c1474f9d893288"
V8_SHA256 = "718e5fec4674ab615b4162a956500b0bc245e25f15153770a5bfa4e6f7d72b42"
V9_SHA256 = "44c5dc8906c9a747498288a027aab907998b03ca0d508437b6a53b1b8d2aa59d"
# Independently captured from hosted run 38422757597, job 115384119235.
V10_SHA256 = "6059a5f97f7d6783e7d489c6787205d7c0402647173d2e96e5cc7321a630fc38"
V10_BYTES = 12184
# Independently captured from hosted run 38642825390, job 116122306581.
# Exact archive also reproduced by a separate local unchanged CNR replay.
V11_SHA256 = "7d52dc711da37a5889810199910384c287a7d8f91f4f286b81bb2343de39b647"
V11_BYTES = 12496
AUXILIARY_SOURCE_DELTA = {
    "added": ["src/trinite/procedural_qec_auxiliary/__init__.py",
              "src/trinite/procedural_qec_auxiliary/archive.py",
              "src/trinite/procedural_qec_auxiliary/experiment.py",
              "src/trinite/procedural_qec_auxiliary/inputs.py",
              "src/trinite/procedural_qec_auxiliary/scheduler.py",
              "src/trinite/procedural_qec_auxiliary/spec.py"],
    "removed": [], "modified": [".github/workflows/cnr-001.yml"],
}
EXPORT_SOURCE_DELTA = {
    "added": ["src/trinite/forward_export/__init__.py",
              "src/trinite/forward_export/encoder.py",
              "src/trinite/forward_export/source.py"],
    "removed": [], "modified": [".github/workflows/cnr-001.yml"],
}
PACKET_FILES = ("cnr-request.json", "cnr-analysis.json", "cnr-freeze-receipt.json",
                "cnr-run-receipt.json", "cnr-verification.json")


class CNR001FullBiology(unittest.TestCase):
    def assert_retained_sha256(self, archive, sidecar, expected):
        """Require the independently retained checksum as well as archive bytes."""
        self.assertTrue(archive.is_file(), "missing retained archive: " + archive.name)
        self.assertTrue(sidecar.is_file(), "missing retained sidecar: " + sidecar.name)
        self.assertRegex(expected, r"^[a-f0-9]{64}$")
        self.assertEqual(sidecar.read_bytes(),
                         (expected + "  " + archive.name + "\n").encode("ascii"))
        self.assertEqual(identity(archive.read_bytes()), "sha256:" + expected)

    def assert_packet_members(self, members, version, parent_version, parent_digest):
        """Archive structure only; actual numerical replay is separately required."""
        expected = {"MANIFEST.json"} | {version + "/" + name for name in PACKET_FILES}
        self.assertEqual(set(members), expected)
        manifest_bytes = members["MANIFEST.json"]
        manifest = parse_json(manifest_bytes, canonical=True)
        self.assertEqual(json_bytes(manifest), manifest_bytes)
        self.assertEqual(manifest["schema"], "trinite.cnr-001-evidence-archive." + version)
        self.assertEqual(manifest["parent_" + parent_version + "_sha256"], parent_digest)
        self.assertEqual(set(manifest["entries"]), expected - {"MANIFEST.json"})
        for name, attr in manifest["entries"].items():
            raw = members[name]
            self.assertEqual(attr, {"bytes": len(raw), "sha256": identity(raw)[7:]})
            parse_json(raw, canonical=True)
        self.assertEqual(manifest["request_identity"],
                         identity(members[version + "/cnr-request.json"]))
        self.assertEqual(manifest["analysis_identity"],
                         identity(members[version + "/cnr-analysis.json"]))
        return manifest

    def read_packet(self, path):
        with tarfile.open(path, "r:gz") as archive:
            entries = archive.getmembers()
            self.assertEqual(len(entries), 6)
            self.assertTrue(all(m.isfile() and m.size < 512 * 1024
                                and not m.name.startswith("/") and ".." not in m.name.split("/")
                                for m in entries))
            self.assertEqual(len(entries), len({m.name for m in entries}))
            return {m.name: archive.extractfile(m).read() for m in entries}

    def test_sidecar_deletion_and_rewrite_fail_closed(self):
        for version in ("v7", "v8", "v9", "v10", "v11"):
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
        self.assert_retained_sha256(archive, sidecar, V9_SHA256)
        members = self.read_packet(archive)
        self.assert_packet_members(members, "v9", "v8", V8_SHA256)
        return members

    def test_retained_v9_packet_has_exact_v8_ancestry(self):
        self.retained_v9_members()

    def assert_source_delta(self, request):
        old_request = parse_json(self.retained_v10_members()["v10/cnr-request.json"], canonical=True)
        self.assertEqual(set(request), set(old_request))
        for key in old_request:
            if key != "source":
                self.assertEqual(request[key], old_request[key], key)
        old, new = old_request["source"], request["source"]
        self.assertEqual(new, core.source_receipt())
        self.assertEqual({"added": sorted(set(new) - set(old)),
                          "removed": sorted(set(old) - set(new)),
                          "modified": sorted(key for key in set(old) & set(new)
                                             if old[key] != new[key])}, AUXILIARY_SOURCE_DELTA)

    def retained_v10_members(self, root=None):
        self.retained_v9_members()
        root = ROOT / "fixtures/cnr-001-evidence" if root is None else root
        archive, sidecar = root / "cnr001-v10.tar.gz", root / "cnr001-v10.sha256"
        self.assertTrue(archive.is_file(), "missing newly published v10 packet")
        self.assertTrue(sidecar.is_file(), "missing independently retained v10 digest")
        # The sidecar must agree with the captured identity, never define it.
        self.assert_retained_sha256(archive, sidecar, V10_SHA256)
        self.assertEqual(len(archive.read_bytes()), V10_BYTES)
        members = self.read_packet(archive)
        manifest = self.assert_packet_members(members, "v10", "v9", V9_SHA256)
        self.assertEqual(manifest["source_delta"], EXPORT_SOURCE_DELTA)
        previous = parse_json(self.retained_v9_members()["v9/cnr-request.json"], canonical=True)
        request = parse_json(members["v10/cnr-request.json"], canonical=True)
        self.assertEqual(set(request), set(previous))
        for key in previous:
            if key != "source":
                self.assertEqual(request[key], previous[key], key)
        old, new = previous["source"], request["source"]
        self.assertEqual({"added": sorted(set(new) - set(old)),
                          "removed": sorted(set(old) - set(new)),
                          "modified": sorted(key for key in set(old) & set(new)
                                             if old[key] != new[key])}, EXPORT_SOURCE_DELTA)
        return members

    def test_retained_v10_packet_has_exact_v9_ancestry(self):
        self.retained_v10_members()

    def retained_v11_members(self, root=None):
        self.retained_v10_members()
        root = ROOT / "fixtures/cnr-001-evidence" if root is None else root
        archive, sidecar = root / "cnr001-v11.tar.gz", root / "cnr001-v11.sha256"
        self.assertTrue(archive.is_file(), "missing newly published v11 packet")
        self.assertTrue(sidecar.is_file(), "missing independently retained v11 digest")
        self.assertIsNotNone(V11_SHA256, "v11 archive identity has not been independently captured")
        self.assertIsNotNone(V11_BYTES, "v11 archive length has not been independently captured")
        self.assert_retained_sha256(archive, sidecar, V11_SHA256)
        self.assertEqual(len(archive.read_bytes()), V11_BYTES)
        members = self.read_packet(archive)
        manifest = self.assert_packet_members(members, "v11", "v10", V10_SHA256)
        self.assertEqual(manifest["source_delta"], AUXILIARY_SOURCE_DELTA)
        self.assert_source_delta(parse_json(members["v11/cnr-request.json"], canonical=True))
        return members

    def test_retained_v11_requires_complete_source_and_exact_v10_ancestry(self):
        self.retained_v11_members()

    def test_coordinated_v11_gzip_metadata_and_sidecar_reseal_is_rejected(self):
        original = ROOT / "fixtures/cnr-001-evidence/cnr001-v11.tar.gz"
        expected_members = self.retained_v11_members()
        raw = original.read_bytes()
        self.assertEqual(raw[:10], b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff")
        changed = raw[:4] + (1).to_bytes(4, "little") + raw[8:]
        self.assertEqual(len(changed), V11_BYTES)
        self.assertEqual(gzip.decompress(changed), gzip.decompress(raw))
        digest = identity(changed)[7:]
        self.assertNotEqual(digest, V11_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, sidecar = root / "cnr001-v11.tar.gz", root / "cnr001-v11.sha256"
            archive.write_bytes(changed)
            sidecar.write_text(digest + "  " + archive.name + "\n", encoding="ascii")
            self.assertEqual(self.read_packet(archive), expected_members)
            self.assert_retained_sha256(archive, sidecar, digest)
            with self.assertRaises(AssertionError):
                self.retained_v11_members(root)
        self.assertEqual(original.read_bytes(), raw)

    def test_coordinated_v10_gzip_metadata_and_sidecar_reseal_is_rejected(self):
        original = ROOT / "fixtures/cnr-001-evidence/cnr001-v10.tar.gz"
        raw = original.read_bytes()
        expected_members = self.retained_v10_members()
        # Verify the captured gzip magic, deflate method, absent optional
        # fields, zero mtime, maximum-compression flag and portable OS marker.
        self.assertEqual(raw[:10], b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff")
        # Changing only mtime guarantees equal length without depending on a
        # local zlib compressor's output. Never change the retained original.
        changed = raw[:4] + (1).to_bytes(4, "little") + raw[8:]
        self.assertEqual(len(changed), V10_BYTES)
        self.assertEqual(gzip.decompress(changed), gzip.decompress(raw))
        digest = identity(changed)[7:]
        self.assertNotEqual(digest, V10_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, sidecar = root / "cnr001-v10.tar.gz", root / "cnr001-v10.sha256"
            archive.write_bytes(changed)
            sidecar.write_text(digest + "  " + archive.name + "\n", encoding="ascii")
            self.assertEqual(self.read_packet(archive), expected_members)
            # A coordinated replacement is internally consistent but has lost
            # the independently captured hosted archive identity.
            self.assert_retained_sha256(archive, sidecar, digest)
            with self.assertRaises(AssertionError):
                self.retained_v10_members(root)
        self.assertEqual(original.read_bytes(), raw)

    def test_actual_reader_rejects_modified_source_and_external_request_bytes(self):
        from scripts import run_cnr001 as runner
        original = json_bytes(runner.expected_request())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "request.json"
            path.write_bytes(original)
            self.assertEqual(runner.request_at(path, identity(original)),
                             parse_json(original, canonical=True))
            for mutation in ("added", "removed", "modified", "noncanonical"):
                changed = parse_json(original, canonical=True)
                if mutation == "added":
                    changed["source"]["src/trinite/unrequested.py"] = "sha256:" + "0" * 64
                elif mutation == "removed":
                    del changed["source"][AUXILIARY_SOURCE_DELTA["added"][0]]
                elif mutation == "modified":
                    changed["source"][AUXILIARY_SOURCE_DELTA["added"][0]] = "sha256:" + "0" * 64
                raw = json_bytes(changed) + (b" " if mutation == "noncanonical" else b"")
                path.write_bytes(raw)
                for commitment in (identity(original), identity(raw)):
                    with self.subTest(mutation=mutation, commitment=commitment):
                        with self.assertRaisesRegex(ContractError, "request/source/data commitment differs"):
                            runner.request_at(path, commitment)
                        self.assertEqual(path.read_bytes(), raw)

    def test_actual_verify_rejects_wrong_analysis_bytes_and_refreshed_identity(self):
        from scripts import run_cnr001 as runner
        with tempfile.TemporaryDirectory() as temporary:
            request, analysis = Path(temporary) / "request.json", Path(temporary) / "analysis.json"
            frozen = runner.freeze(request)
            ran = runner.run(request, frozen["request_identity"], analysis)
            original = (request.read_bytes(), analysis.read_bytes())
            analysis.write_bytes(original[1] + b" ")
            with self.assertRaisesRegex(ContractError, "external CNR analysis commitment differs"):
                runner.verify(request, frozen["request_identity"], analysis, ran["analysis_identity"])
            with self.assertRaisesRegex(ContractError, "replay mismatch"):
                runner.verify(request, frozen["request_identity"], analysis, identity(analysis.read_bytes()))
            self.assertEqual(request.read_bytes(), original[0])
            self.assertEqual(analysis.read_bytes(), original[1] + b" ")

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
            self.assertEqual(set(previous), set(current))
            for key in previous:
                if key != "source":
                    self.assertEqual(current[key], previous[key], key)
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
            # V8, v9 and v10 belong to their immutable historical producers.
            # The additive auxiliary package requires a separate v11 packet.
            v8_archive = ROOT / "fixtures/cnr-001-evidence/cnr001-v8.tar.gz"
            self.assert_retained_sha256(
                v8_archive, ROOT / "fixtures/cnr-001-evidence/cnr001-v8.sha256",
                V8_SHA256)
            self.assert_source_delta(current)
            members = self.retained_v11_members()
            manifest = json.loads(members["MANIFEST.json"])
            self.assertEqual(manifest["request_identity"], req_id)
            self.assertEqual(manifest["analysis_identity"], ran["analysis_identity"])
            self.assertEqual(members["v11/cnr-request.json"], before[0])
            self.assertEqual(members["v11/cnr-analysis.json"], before[1])
            self.assertEqual(members["v11/cnr-freeze-receipt.json"], json_bytes(frozen))
            self.assertEqual(members["v11/cnr-run-receipt.json"], json_bytes(ran))
            self.assertEqual(members["v11/cnr-verification.json"], json_bytes(got))
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
