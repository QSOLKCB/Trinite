"""Standard-library-only adversarial transport checks (synthetic packets).

These fixtures do NOT prove native optimizer replay, PROVENANCE custody, or
learning quality. The separate real CPU smoke must provide those assertions.
"""
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from procedural_qec_v3_archive import (ArchiveError, CELL_COMMITTED, ROOT_COMMITTED,
    SCHEMA, TOOLS, canonical, identity, safe_name, verify_archive, verify_contract)


def synthetic_packet():
    # Construct a *structurally* closed mock, not a fake scientific result.
    members = {}
    request = {"schema": "trinite.procedural-qec-request.v3",
               "run_profile": "smoke", "protocol_identity": "sha256:" + "1" * 64,
               "source": {"src/trinite/mock.py": identity(b"offline fixture source\n")}}
    request_raw = canonical(request)
    members["run/request.json"] = request_raw
    members["run/worker-errors.json"] = canonical({})
    members["external/request.json"] = request_raw
    for name, raw in request["source"].items():
        members["source/" + name] = b"offline fixture source\n"
    archival_tools = {}
    for name in TOOLS:
        raw = ("fixture source:" + name + "\n").encode()
        members["archival-tool/" + name] = raw
        archival_tools[name] = identity(raw)
    cell_records, committed_cells = [], {}
    for seed in (0, 1):
        for arm in ("final-only", "procedural"):
            cell = f"procedural-qec-{seed}-{arm}"
            report = {"schema": "trinite.procedural-qec-report.v3",
                      "seed": seed, "arm": arm,
                      "held_out_scored": False, "learning_adequacy": False,
                      "phase5_ready": False}
            bytes_report = canonical(report)
            checks = {name: True for name in (
                "closed_custody", "request_and_source_binding",
                "safe_tensor_structure", "scheduled_visits_and_targets",
                "final_predictions_and_loss", "complete_optimizer_replay")}
            cell_records.append({"seed": seed, "arm": arm, "outcome": "completed",
                                 "report_identity": identity(bytes_report),
                                 "report": report, "checks": checks})
            committed_cells[cell] = {}
            for name in CELL_COMMITTED:
                raw = bytes_report if name == "report.json" else (
                    cell + "/" + name + "\n").encode()
                key = "run/" + cell + "/" + name
                members[key] = raw
                committed_cells[cell][name] = identity(raw)
    verification = {
        "schema": "trinite.procedural-qec-verification-request.v3",
        "request_identity": identity(request_raw),
        "root_artifacts": {name: identity(members["run/" + name])
                           for name in ROOT_COMMITTED},
        "cells": committed_cells}
    verify_raw = canonical(verification)
    members["external/verification-request.json"] = verify_raw
    summary = {
        "schema": "trinite.procedural-qec-summary.v3",
        "run_profile": "smoke", "outcome": "completed",
        "request_identity": identity(request_raw),
        "verification_request_identity": identity(verify_raw),
        "protocol_identity": request["protocol_identity"],
        "cells": cell_records, "held_out_scored": False,
        "phase5_ready": False, "learning_adequacy": False,
        "training_gate_pass": False, "selected_arm": None}
    members["run/summary.json"] = canonical(summary)
    members["audit/replay.json"] = canonical({
        "outcome": "completed", "replayed_cells": 4,
        "summary_identity": identity(members["run/summary.json"]),
        "verification_request_identity": identity(verify_raw),
        "learning_adequacy": False})
    inventory = {"schema": SCHEMA,
                 "request_identity": identity(request_raw),
                 "verification_request_identity": identity(verify_raw),
                 "summary_identity": identity(members["run/summary.json"]),
                 "run_profile": "smoke", "evidence_class": "integration-smoke-only",
                 "source": request["source"], "archival_tools": archival_tools,
                 "members": {name: {"identity": identity(raw), "bytes": len(raw)}
                             for name, raw in sorted(members.items())}}
    return members, inventory


def zip_packet(path, members, inventory):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted({**members, "inventory.json": canonical(inventory)}.items()):
            z.writestr(name, raw)
    return identity(path.read_bytes()), identity(canonical(inventory))


class ProceduralEvidenceArchiveTests(unittest.TestCase):
    def test_structural_closure_and_exact_external_digest(self):
        members, inventory = synthetic_packet()
        self.assertEqual(verify_contract(members, inventory)["outcome"], "passed")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.zip"
            archive_id, inventory_id = zip_packet(path, members, inventory)
            self.assertEqual(verify_archive(path, archive_id, inventory_id)["member_count"],
                             len(members))
            with self.assertRaisesRegex(ArchiveError, "independent SHA-256"):
                verify_archive(path, "sha256:" + "0" * 64, inventory_id)
            with self.assertRaisesRegex(ArchiveError, "independently held"):
                verify_archive(path, archive_id, "sha256:" + "0" * 64)

    def test_reseal_one_cell_cannot_override_independent_verifier(self):
        members, inventory = synthetic_packet()
        target = "run/procedural-qec-0-procedural/report.json"
        members[target] = canonical({"outcome": "fake-success"})
        inventory["members"][target] = {"identity": identity(members[target]),
                                         "bytes": len(members[target])}
        with self.assertRaisesRegex(ArchiveError, "summary does not match|summary.*report"):
            verify_contract(members, inventory)

    def test_missing_committed_artifact_and_forged_summary_fail(self):
        members, inventory = synthetic_packet()
        target = "run/procedural-qec-1-final-only/verification.json"
        del members[target]
        del inventory["members"][target]
        with self.assertRaisesRegex(ArchiveError, "committed run bytes differ"):
            verify_contract(members, inventory)
        members, inventory = synthetic_packet()
        inventory["run_profile"] = "research"
        with self.assertRaisesRegex(ArchiveError, "class/profile mismatch"):
            verify_contract(members, inventory)

    def test_unsafe_and_duplicate_zip_members_fail_with_matching_zip_hash(self):
        for name in ("../escape", "/absolute", "a/../b", "a\\b", "a\x00b",
                     "a//b", "a/\nb"):
            with self.subTest(name=repr(name)), self.assertRaises(ArchiveError):
                safe_name(name)
        members, inventory = synthetic_packet()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.zip"
            with zipfile.ZipFile(path, "x") as z:
                for name, raw in {**members, "inventory.json": canonical(inventory)}.items():
                    z.writestr(name, raw)
                z.writestr("run/request.json", members["run/request.json"])
            with self.assertRaisesRegex(ArchiveError, "unsafe, duplicate"):
                verify_archive(path, identity(path.read_bytes()), identity(canonical(inventory)))

    def test_archive_tamper_detected_even_when_zip_repacked(self):
        members, inventory = synthetic_packet()
        members["run/worker-errors.json"] = canonical({"hidden": "error"})
        inventory["members"]["run/worker-errors.json"] = {
            "identity": identity(members["run/worker-errors.json"]),
            "bytes": len(members["run/worker-errors.json"])}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forged.zip"
            archive_id, inv_id = zip_packet(path, members, inventory)
            with self.assertRaisesRegex(ArchiveError, "committed run bytes differ"):
                verify_archive(path, archive_id, inv_id)


if __name__ == "__main__":
    unittest.main()
