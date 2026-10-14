"""Stdlib transport attacks using synthetic engineering packets, not replay proof.

Custody directories are produced/verified by the real pinned upstream observer.
The miniature Git fixture is explicitly synthetic; these tests never certify
native optimizer replay, historical admission, or scientific learning outcomes.
The separate CPU engineering run exercises production archive-time replay.
"""
from copy import deepcopy
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from trinite.observation import BundleObserver, verify_observation
from trinite.procedural_qec_auxiliary import archive as a

ROOT = Path(__file__).resolve().parents[1]


def miniature_history():
    raw = b"# synthetic transport source only\n"
    blob = a._git_identity("blob", raw)
    src = b"100644 fixture.py\0" + bytes.fromhex(blob)
    src_id = a._git_identity("tree", src)
    tree = b"40000 src\0" + bytes.fromhex(src_id)
    tree_id = a._git_identity("tree", tree)
    commit = (f"tree {tree_id}\nauthor Fixture <fixture@example.invalid> 0 +0000\n"
              "committer Fixture <fixture@example.invalid> 0 +0000\n\nSynthetic transport fixture\n").encode()
    commit_id = a._git_identity("commit", commit)
    objects = {commit_id: ("commit", commit), tree_id: ("tree", tree), src_id: ("tree", src)}
    members = {"history/git/objects/" + oid: value for oid, (_, value) in objects.items()}
    members["history/source/src/fixture.py"] = raw
    snapshot = {"commit": commit_id, "tree": tree_id, "layout": "shallow-sparse-src",
        "objects": {oid: {"kind": kind, "identity": a.identity(value)} for oid, (kind, value) in objects.items()},
        "files": {"src/fixture.py": {"mode": "100644", "git_blob": blob, "identity": a.identity(raw)}}}
    return members, snapshot


def packet(history=None):
    members, snapshot = deepcopy(history or miniature_history())
    historical_data, historical_manifest = {"synthetic": "historical data"}, {"synthetic": "historical manifest"}
    material = {"dataset.json": a.canonical({"synthetic": "current data"}),
        "manifest.json": a.canonical({"synthetic": "current manifest"}),
        "historical-dataset.json": a.canonical(historical_data),
        "historical-manifest.json": a.canonical(historical_manifest),
        "historical-payload.json": a.canonical({"dataset": historical_data, "manifest": historical_manifest,
                                                "schedules": {"synthetic": []}})}
    receipt = {"historical_producer_commit": snapshot["commit"], "historical_git_tree": snapshot["tree"],
        "historical_dataset_identity": a.identity(material["historical-dataset.json"]),
        "historical_manifest_identity": a.identity(material["historical-manifest.json"]),
        "historical_payload_identity": a.identity(material["historical-payload.json"]),
        "prospective_dataset_identity": a.identity(material["dataset.json"]),
        "prospective_manifest_identity": a.identity(material["manifest.json"])}
    material["mapping-receipt.json"] = a.canonical(receipt)
    members.update({"inputs/" + p: raw for p, raw in material.items()})
    specification = {"research_execution_enabled": False}
    source = b"# synthetic prospective source\n"
    request = {"schema": "trinite.procedural-qec-auxiliary-request.v1", "run_profile": "engineering",
        "specification": specification, "specification_identity": a.identity(a.canonical(specification)),
        "source": {"src/trinite/mock.py": a.identity(source)},
        "inputs": {field: a.identity(material[path]) for path, field in a.INPUT_COMMITMENTS.items()}}
    request_raw = a.canonical(request)
    members["source/src/trinite/mock.py"] = source
    members["external/request.json"] = members["run/request.json"] = request_raw
    members["run/inputs.json"] = material["mapping-receipt.json"]
    members["run/worker-errors.json"] = a.canonical({})
    cells, commitments = [], {}
    with tempfile.TemporaryDirectory() as temp:
        for seed in (0, 1):
            for arm in ("extra-final", "extra-procedural"):
                name = f"auxiliary-qec-{seed}-{arm}"
                report = {"schema": "trinite.procedural-qec-auxiliary-report.v1", "seed": seed,
                    "arm": arm, "run_profile": "engineering", "evidence_class": "engineering-conformance-only",
                    **{key: False for key in ("held_out_scored", "training_gate_evaluated", "learning_adequacy",
                                             "phase5_ready", "format_selection_eligible", "release_ready")}}
                root = Path(temp) / name
                root.mkdir()
                for path in a.CELL_COMMITTED - {"verification.json", "provenance/manifest.json"}:
                    (root / path).write_bytes(a.canonical(report) if path == "report.json" else (name + "/" + path).encode())
                observer = BundleObserver(root / "provenance")
                observer.record("synthetic-transport-fixture", inputs={"request.json": request_raw,
                    "inputs.json": material["mapping-receipt.json"], "dataset.json": material["dataset.json"],
                    "manifest.json": material["manifest.json"]}, outputs={path: (root / path).read_bytes()
                        for path in a.CELL_COMMITTED - {"verification.json", "provenance/manifest.json"}})
                (root / "verification.json").write_bytes(a.canonical(observer.finalize()))
                verify_observation(root / "provenance")
                for path, raw in a.tree_bytes(root).items():
                    members["run/" + name + "/" + path] = raw
                commitments[name] = {path: a.identity(members["run/" + name + "/" + path]) for path in a.CELL_COMMITTED}
                cells.append({"seed": seed, "arm": arm, "outcome": "completed", "report": report,
                    "report_identity": a.identity(a.canonical(report)), "checks": dict.fromkeys(a.CHECKS, True)})
    verification = {"schema": "trinite.procedural-qec-auxiliary-verification-request.v1",
        "request_identity": a.identity(request_raw),
        "root_artifacts": {p: a.identity(members["run/" + p]) for p in a.ROOT_COMMITTED}, "cells": commitments}
    verification_raw = a.canonical(verification)
    members["external/verification-request.json"] = verification_raw
    summary = {"schema": "trinite.procedural-qec-auxiliary-summary.v1", "run_profile": "engineering",
        "evidence_class": "engineering-conformance-only", "outcome": "completed", "selected_arm": None,
        "worker_errors": {}, "pairing_errors": [], "request_identity": a.identity(request_raw),
        "verification_request_identity": a.identity(verification_raw), "cells": cells,
        **{key: False for key in ("held_out_scored", "training_gate_evaluated", "training_gate_pass",
            "learning_adequacy", "phase5_ready", "format_selection_eligible", "release_ready")}}
    members["run/summary.json"] = a.canonical(summary)
    members["audit/replay.json"] = a.canonical({"outcome": "completed", "replayed_cells": 4,
        "summary_identity": a.identity(members["run/summary.json"]), "verification_request_identity": a.identity(verification_raw),
        "learning_adequacy": False, "phase5_ready": False, "evidence_class": "engineering-conformance-only"})
    inventory = {"schema": a.SCHEMA, "request_identity": a.identity(request_raw),
        "verification_request_identity": a.identity(verification_raw), "summary_identity": a.identity(members["run/summary.json"]),
        "run_profile": "engineering", "evidence_class": "engineering-conformance-only", "source": request["source"],
        "inputs": {p: a.identity(raw) for p, raw in material.items()}, "historical_source": snapshot,
        "members": {name: {"identity": a.identity(raw), "bytes": len(raw)} for name, raw in members.items()}}
    return members, inventory


def zip_packet(path, members, inventory):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted({**members, "inventory.json": a.canonical(inventory)}.items()):
            archive.writestr(name, raw)
    return a.identity(path.read_bytes()), a.identity(a.canonical(inventory))


def update(members, inventory, name, raw):
    members[name] = raw
    inventory["members"][name] = {"identity": a.identity(raw), "bytes": len(raw)}


class AuxiliaryArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = miniature_history()
        cls.template = packet(cls.history)

    def setUp(self):
        self.commit = patch.object(a, "HISTORICAL_COMMIT", self.history[1]["commit"])
        self.tree = patch.object(a, "HISTORICAL_TREE", self.history[1]["tree"])
        self.commit.start()
        self.tree.start()
        self.addCleanup(self.commit.stop)
        self.addCleanup(self.tree.stop)
        self.members, self.inventory = deepcopy(self.template)

    def test_closed_transport_distinguishes_replay_and_admission(self):
        result = a.verify_contract(self.members, self.inventory)
        self.assertEqual(result["outcome"], "passed")
        self.assertEqual(result["verification_scope"], "byte-transport-only")
        self.assertFalse(result["numerical_replay_performed"])
        self.assertFalse(result["custody_admission_performed"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "archive.zip"
            ids = zip_packet(path, self.members, self.inventory)
            before = path.read_bytes()
            self.assertEqual(a.verify_archive(path, *ids)["member_count"], len(self.members))
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(a.ArchiveError, "independent SHA-256"):
                a.verify_archive(path, "sha256:" + "0" * 64, ids[1])
            with self.assertRaisesRegex(a.ArchiveError, "independently held"):
                a.verify_archive(path, ids[0], "sha256:" + "0" * 64)

    def test_resealed_run_bytes_and_source_rejected(self):
        for name in ("run/worker-errors.json", "run/inputs.json", "source/src/trinite/mock.py",
                     "run/auxiliary-qec-0-extra-final/tensors.safetensors"):
            with self.subTest(name=name):
                members, inventory = deepcopy(self.template)
                update(members, inventory, name, members[name] + b" ")
                with self.assertRaises(a.ArchiveError):
                    a.verify_contract(members, inventory)

    def test_extra_and_missing_members_cannot_be_resealed(self):
        for name in ("unknown.txt", "run/unexpected.log", "run/auxiliary-qec-0-extra-final/unknown.log",
                     "source/extra.py", "history/source/src/extra.py", "inputs/extra.json"):
            with self.subTest(name=name):
                members, inventory = deepcopy(self.template)
                update(members, inventory, name, b"extra")
                with self.assertRaises(a.ArchiveError):
                    a.verify_contract(members, inventory)
        for name in ("run/auxiliary-qec-0-extra-final/verification.json", "inputs/historical-payload.json",
                     "history/source/src/fixture.py"):
            with self.subTest(name=name):
                members, inventory = deepcopy(self.template)
                del members[name]
                del inventory["members"][name]
                with self.assertRaises(a.ArchiveError):
                    a.verify_contract(members, inventory)

    def test_uncommitted_provenance_extra_missing_and_modified_content_rejected(self):
        target = next(n for n in self.members if "/provenance/artifacts/sha256/" in n)
        for action in ("change", "remove"):
            with self.subTest(action=action):
                members, inventory = deepcopy(self.template)
                if action == "remove":
                    del members[target]
                    del inventory["members"][target]
                else:
                    update(members, inventory, target, b"changed custody content")
                with self.assertRaises(a.ArchiveError):
                    a.verify_contract(members, inventory)

    def test_resealed_provenance_metadata_does_not_change_manifest_commitments(self):
        for fragment in ("/provenance/artifact_records/", "/provenance/events/"):
            with self.subTest(fragment=fragment):
                members, inventory = deepcopy(self.template)
                target = next(name for name in members if fragment in name)
                record = a.parse(members[target])
                record["injected"] = "not authorized by original manifest"
                update(members, inventory, target, a.canonical(record))
                with self.assertRaises(a.ArchiveError):
                    a.verify_contract(members, inventory)

    def test_historical_payload_bound_beyond_resealed_inventory(self):
        target = "inputs/historical-payload.json"
        payload = a.parse(self.members[target])
        payload["schedules"] = {"synthetic": [42]}
        update(self.members, self.inventory, target, a.canonical(payload))
        self.inventory["inputs"]["historical-payload.json"] = a.identity(self.members[target])
        with self.assertRaisesRegex(a.ArchiveError, "input receipt differs"):
            a.verify_contract(self.members, self.inventory)

    def test_original_git_object_pin_cannot_be_resealed(self):
        target = "history/source/src/fixture.py"
        update(self.members, self.inventory, target, b"changed historical source")
        self.inventory["historical_source"]["files"]["src/fixture.py"]["identity"] = a.identity(self.members[target])
        with self.assertRaisesRegex(a.ArchiveError, "historical source blob commitment"):
            a.verify_contract(self.members, self.inventory)

    def test_research_profile_and_scientific_claims_rejected(self):
        self.inventory["run_profile"] = "research"
        with self.assertRaisesRegex(a.ArchiveError, "research execution"):
            a.verify_contract(self.members, self.inventory)
        self.members, self.inventory = deepcopy(self.template)
        summary = a.parse(self.members["run/summary.json"])
        summary["learning_adequacy"] = True
        update(self.members, self.inventory, "run/summary.json", a.canonical(summary))
        self.inventory["summary_identity"] = a.identity(self.members["run/summary.json"])
        with self.assertRaisesRegex(a.ArchiveError, "unsupported scientific"):
            a.verify_contract(self.members, self.inventory)

    def test_unsafe_names_duplicate_json_and_noncanonical_json(self):
        for name in ("../escape", "/absolute", "a/../b", "a\\b", "a\0b", "a//b", "a/\nb", "C:drive"):
            with self.subTest(name=repr(name)), self.assertRaises(a.ArchiveError):
                a.safe_name(name)
        for raw in (b'{"a":1,"a":2}\n', b'{ "a":1}\n', b'{"a":1.0}\n', b'{"a":NaN}\n'):
            with self.subTest(raw=raw), self.assertRaises(a.ArchiveError):
                a.parse(raw)

    def test_duplicate_symlink_oversized_zip_and_symlink_input_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            for kind in ("duplicate", "symlink", "directory"):
                path = temp / (kind + ".zip")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with zipfile.ZipFile(path, "x") as archive:
                        for name, raw in {**self.members, "inventory.json": a.canonical(self.inventory)}.items():
                            archive.writestr(name, raw)
                        if kind == "duplicate":
                            archive.writestr("run/request.json", self.members["run/request.json"])
                        else:
                            info = zipfile.ZipInfo("bad" if kind == "symlink" else "bad/")
                            info.create_system = 3
                            info.external_attr = (stat.S_IFLNK | 0o777) << 16
                            archive.writestr(info, b"../escape")
                with self.subTest(kind=kind), self.assertRaises(a.ArchiveError):
                    a.verify_archive(path, a.identity(path.read_bytes()), a.identity(a.canonical(self.inventory)))
            path = temp / "valid.zip"
            ids = zip_packet(path, self.members, self.inventory)
            with patch.object(a, "MAX_MEMBER_BYTES", 10), self.assertRaises(a.ArchiveError):
                a.verify_archive(path, *ids)
            alias = temp / "alias.zip"
            alias.symlink_to(path)
            with self.assertRaisesRegex(a.ArchiveError, "symlinks"):
                a.verify_archive(alias, *ids)

    def test_restore_exact_source_is_exclusive_and_clean(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            path = temp / "archive.zip"
            ids = zip_packet(path, self.members, self.inventory)
            output = temp / "restored"
            result = a.restore_historical_source(path, *ids, output)
            self.assertEqual(result["historical_commit"], a.HISTORICAL_COMMIT)
            self.assertEqual((output / "src/fixture.py").read_bytes(), self.members["history/source/src/fixture.py"])
            self.assertEqual(a._git(output, "status", "--porcelain"), b"")
            with self.assertRaisesRegex(a.ArchiveError, "unused directory"):
                a.restore_historical_source(path, *ids, output)

    def test_restoration_keeps_original_sha1_with_global_sha256_default(self):
        # Temporary HOME only; never read or change the user's Git configuration.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home = root / "home"
            home.mkdir()
            config = home / ".gitconfig"
            configured = b"[init]\n    defaultObjectFormat = sha256\n"
            config.write_bytes(configured)
            path = root / "archive.zip"
            ids = zip_packet(path, self.members, self.inventory)
            with patch.dict(os.environ, {"HOME": str(home), "XDG_CONFIG_HOME": str(home / "xdg")}):
                ordinary = root / "ordinary"
                ordinary.mkdir()
                a._git(ordinary, "init", "--quiet")
                self.assertEqual(a._git(ordinary, "rev-parse", "--show-object-format"), b"sha256\n")
                output = root / "restored"
                result = a.restore_historical_source(path, *ids, output)
                self.assertEqual(result["outcome"], "passed")
                self.assertEqual(a._git(output, "rev-parse", "--show-object-format"), b"sha1\n")
                self.assertEqual(a._git(output, "rev-parse", "HEAD").decode().strip(), a.HISTORICAL_COMMIT)
                self.assertEqual(a._git(output, "status", "--porcelain"), b"")
            self.assertEqual(config.read_bytes(), configured)

    def test_builder_rejects_source_output_before_reading_or_replaying(self):
        # Controller failure injection only: this does not certify native replay.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src/nested").mkdir(parents=True)
            for output in (root / "src/unbound.zip", root / "src/nested/unbound.zip",
                           root / "src/nested/../unbound.zip"):
                with self.subTest(output=output), patch.object(a, "ROOT", root), \
                        patch.object(a, "read") as read, patch.object(a, "tree_bytes") as scan:
                    with self.assertRaisesRegex(a.ArchiveError, "outside current producer source"):
                        a.build_archive(output, root / "run", root / "request.json",
                            "sha256:" + "0" * 64, root / "verification.json",
                            "sha256:" + "1" * 64, root / "historical")
                    read.assert_not_called()
                    scan.assert_not_called()
                    self.assertFalse(output.exists())

    def test_stdlib_only_import_and_cli_rejects_missing_commitments(self):
        program = ("import sys;sys.path.insert(0," + repr(str(ROOT / "src")) + ");"
                   "from trinite.procedural_qec_auxiliary import archive;"
                   "assert not any(n.split('.')[0] in ('torch','numpy') for n in sys.modules)")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", program], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        result = subprocess.run([sys.executable, "-I", "-S", "-B",
            str(ROOT / "scripts/archive_procedural_qec_auxiliary.py"), "--verify-archive"], capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
