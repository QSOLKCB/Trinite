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


def research_fixture(fold_bonus=1):
    """Invented bytes only: no admitted scientific request, model, or training."""
    from trinite.procedural_qec_auxiliary.spec import specification
    candidate = specification()["research_candidate"]
    gates = specification()["gates"]
    totals = {"final": {task: total for task, (_, total) in gates.items()},
              "prefix": {"fold-sum": 1150, "gf3": 54}, "trace": {"fold-sum": 1150, "gf3": 54}}
    items = []
    for mode, tasks in totals.items():
        for task, count in tasks.items():
            for index in range(count):
                items.append({"example_identity": a.identity(f"synthetic/{mode}/{task}/{index}".encode()),
                    "mode": mode, "task": task, "kind": task if task in ("fold-sum", "gf3") else "arithmetic",
                    "split": "train"})
    support = {}
    for seed in (0, 1):
        for arm in ("extra-final", "extra-procedural"):
            name = f"auxiliary-qec-{seed}-{arm}"
            used = [i["example_identity"] for i in items if arm == "extra-procedural" or i["mode"] == "final"]
            visits = [used[i % len(used)] for i in range(65536)]
            history = [{"step": i + 1, "examples": visits[i * 8:(i + 1) * 8],
                "target_tokens": 16, "loss_hex": float(1).hex(), "gradient_norm_hex": float(1).hex()}
                for i in range(8192)]
            diagnostics = [{"step": i, "loss_hex": float(1).hex()} for i in (2048, 4096, 6144, 8192)]
            predictions, scores, observed = {}, {}, {}
            for mode, tasks in totals.items():
                predictions[mode], counters = [], dict.fromkeys(tasks, 0)
                task_scores = {task: {"correct": 0, "examples": 0} for task in tasks}
                for item in [i for i in items if i["mode"] == mode]:
                    task = item["task"]
                    limit = gates[task][0] + (fold_bonus if task == "fold-sum" and arm == "extra-procedural" else 0)
                    correct = counters[task] < limit
                    counters[task] += 1
                    predictions[mode].append({"example_identity": item["example_identity"],
                                             "task": task, "split": "train", "correct": correct})
                    task_scores[task]["examples"] += 1
                    task_scores[task]["correct"] += int(correct)
                    if mode != "final":
                        group = observed.setdefault(item["kind"] + "/" + mode,
                            {"visited": {"correct": 0, "examples": 0}, "unvisited": {"correct": 0, "examples": 0}})
                        part = group["visited" if arm == "extra-procedural" else "unvisited"]
                        part["examples"] += 1
                        part["correct"] += int(correct)
                scores[mode] = {"examples": sum(v["examples"] for v in task_scores.values()),
                                "correct": sum(v["correct"] for v in task_scores.values()), "tasks": task_scores}
            for group in observed.values():
                for part in group.values():
                    part["accuracy_defined"] = part["examples"] > 0
            schedule = {"seed": seed, "arm": arm, "macro_rounds": 4096, "native_updates": 8192,
                "padded_encoded_positions": 4194304, "model_input_positions": 4128768,
                "within_existing_target_ceiling": True, "schedule_identity": a.identity(a.canonical(visits)),
                "combined": {"totals": {"visits": 65536, "target_tokens": 131072}},
                "primary": {key: a.identity(f"synthetic-primary/{seed}/{key}".encode()) for key in
                    ("original_row_sequence_identity", "prospective_row_sequence_identity", "normalized_row_sequence_identity")}}
            support[name] = {"schedule": schedule, "steps": {"steps": history, "train_diagnostics": diagnostics},
                "predictions": {"predictions": predictions}, "scores": scores, "observed": observed,
                "plan": {"steps": 8192, "batch_size": 8, "validation_every": 2048, "seed": seed,
                    "max_seconds": candidate["max_seconds"], "max_target_tokens": candidate["max_target_tokens"],
                    "model": {key: candidate[key] for key in ("blocks", "width", "heads", "feed_forward_width", "context_length")},
                    **specification()["optimizer"]},
                "losses": {mode: {"all_loss_hex": float(1).hex(), "tasks": dict.fromkeys(tasks, float(1).hex())}
                           for mode, tasks in totals.items()}}
    return {"examples": items}, support


def packet(history=None, *, profile="engineering", fold_bonus=1):
    research = profile == "research"
    prefix = "trinite.procedural-qec-auxiliary-" + ("research-" if research else "")
    evidence_class = "training-only-exploratory-candidate" if research else "engineering-conformance-only"
    if research:
        from trinite.procedural_qec_auxiliary.spec import specification as candidate_spec, assess_training_counts
        current_dataset, support = research_fixture(fold_bonus)
    else:
        current_dataset, support = {"synthetic": "current data"}, {}
    members, snapshot = deepcopy(history or miniature_history())
    historical_data, historical_manifest = {"synthetic": "historical data"}, {"synthetic": "historical manifest"}
    material = {"dataset.json": a.canonical(current_dataset),
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
    specification = candidate_spec() if research else {"research_execution_enabled": False}
    source = b"# synthetic prospective source\n"
    request = {"schema": prefix + "request.v1", "run_profile": profile,
        "specification": specification, "specification_identity": a.identity(a.canonical(specification)),
        "source": {"src/trinite/mock.py": a.identity(source)},
        "inputs": {field: a.identity(material[path]) for path, field in a.INPUT_COMMITMENTS.items()}}
    if research:
        request["schedules"] = {name: v["schedule"] for name, v in support.items()}
        request["environment"] = {"synthetic": "transport-only, never native execution"}
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
                report = {"schema": prefix + "report.v1", "seed": seed,
                    "arm": arm, "run_profile": profile, "evidence_class": evidence_class,
                    **{key: False for key in ("held_out_scored", "training_gate_evaluated", "learning_adequacy",
                                             "phase5_ready", "format_selection_eligible", "release_ready")}}
                if research:
                    data = support[name]
                    report.update(steps=8192, plan=data["plan"], schedule=data["schedule"], target_tokens=131072,
                        request_identity=a.identity(request_raw), scores=data["scores"], source=request["source"],
                        inputs=request["inputs"], environment=request["environment"], specification_identity=request["specification_identity"],
                        initialization_identity=a.identity(f"synthetic-init-{seed}".encode()),
                        model_identity=a.identity(f"synthetic-final-{seed}-{arm}".encode()),
                        auxiliary_visited_unvisited=data["observed"], loss_by_mode_and_task=data["losses"],
                        mixed_all_modes_loss_hex=float(1).hex(), final_only_loss_hex=float(1).hex(),
                        training_gate_evaluated=True,
                        training_task_failures=(["fold-sum"] if arm == "extra-procedural" and fold_bonus < 0 else []),
                        training_gate_pass=not (arm == "extra-procedural" and fold_bonus < 0))
                root = Path(temp) / name
                root.mkdir()
                for path in a.CELL_COMMITTED - {"verification.json", "provenance/manifest.json"}:
                    (root / path).write_bytes(a.canonical(report) if path == "report.json" else (name + "/" + path).encode())
                if research:
                    (root / "steps.json").write_bytes(a.canonical(data["steps"]))
                    (root / "predictions.json").write_bytes(a.canonical(data["predictions"]))
                    (root / "metadata.json").write_bytes(a.canonical({"schema": prefix + "checkpoint.v1",
                        "run_profile": profile, "seed": seed, "arm": arm, "step": 8192, "cursor": 65536,
                        "target_tokens": 131072, "history": data["steps"]["steps"],
                        "validation": data["steps"]["train_diagnostics"], "plan": data["plan"],
                        "request_identity": a.identity(request_raw),
                        **{key: report[key] for key in ("source", "inputs", "environment", "specification_identity",
                                                       "initialization_identity", "model_identity", "schedule")}}))
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
    verification = {"schema": prefix + "verification-request.v1",
        "request_identity": a.identity(request_raw),
        "root_artifacts": {p: a.identity(members["run/" + p]) for p in a.ROOT_COMMITTED}, "cells": commitments}
    verification_raw = a.canonical(verification)
    members["external/verification-request.json"] = verification_raw
    summary = {"schema": prefix + "summary.v1", "run_profile": profile,
        "evidence_class": evidence_class, "outcome": "completed", "selected_arm": None,
        "worker_errors": {}, "pairing_errors": [], "request_identity": a.identity(request_raw),
        "verification_request_identity": a.identity(verification_raw), "cells": cells,
        **{key: False for key in ("held_out_scored", "training_gate_evaluated", "training_gate_pass",
            "learning_adequacy", "phase5_ready", "format_selection_eligible", "release_ready")}}
    if research:
        criteria = assess_training_counts({name: data["scores"]["final"]["tasks"] for name, data in support.items()})
        summary.update(training_gate_evaluated=True, training_gate_pass=criteria["treatment_training_gate_pass"],
            positive_fold_contrast_both_seeds=criteria["positive_fold_contrast_both_seeds"], contrasts=criteria["contrasts"])
    members["run/summary.json"] = a.canonical(summary)
    members["audit/replay.json"] = a.canonical({"outcome": "completed", "replayed_cells": 4,
        "summary_identity": a.identity(members["run/summary.json"]), "verification_request_identity": a.identity(verification_raw),
        "learning_adequacy": False, "phase5_ready": False, "evidence_class": evidence_class})
    inventory = {"schema": a.RESEARCH_SCHEMA if research else a.SCHEMA, "request_identity": a.identity(request_raw),
        "verification_request_identity": a.identity(verification_raw), "summary_identity": a.identity(members["run/summary.json"]),
        "run_profile": profile, "evidence_class": evidence_class, "source": request["source"],
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



class DormantResearchArchiveTests(unittest.TestCase):
    """Synthetic shape/count checks only; public scientific execution stays shut."""
    @classmethod
    def setUpClass(cls):
        cls.history = miniature_history()
        cls.template = packet(cls.history, profile="research")

    def setUp(self):
        for name, value in (("HISTORICAL_COMMIT", self.history[1]["commit"]),
                            ("HISTORICAL_TREE", self.history[1]["tree"])):
            patcher = patch.object(a, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.members, self.inventory = deepcopy(self.template)
        self.request = a.parse(self.members["external/request.json"])
        self.summary = a.parse(self.members["run/summary.json"])

    def test_private_full_matrix_contract_is_transport_only(self):
        result = a._verify_profile_contract(self.members, self.inventory, "research")
        self.assertEqual(result["outcome"], "passed")
        self.assertEqual(result["schema"], "trinite.procedural-qec-auxiliary-research-archive-verification.v1")
        self.assertEqual(result["verification_scope"], "byte-transport-only")
        self.assertFalse(result["numerical_replay_performed"])
        self.assertFalse(result["custody_admission_performed"])
        self.assertTrue(self.summary["training_gate_pass"])
        self.assertTrue(self.summary["positive_fold_contrast_both_seeds"])
        self.assertEqual(sum(c["report"]["steps"] for c in self.summary["cells"]), 32768)
        self.assertFalse(self.request["specification"]["research_execution_enabled"])

    def test_zero_negative_contrasts_remain_valid_without_arm_selection(self):
        for bonus in (0, -1):
            with self.subTest(bonus=bonus):
                members, inventory = packet(self.history, profile="research", fold_bonus=bonus)
                result = a._verify_profile_contract(members, inventory, "research")
                summary = a.parse(members["run/summary.json"])
                self.assertEqual(result["outcome"], "passed")
                self.assertEqual(summary["training_gate_pass"], bonus == 0)
                self.assertFalse(summary["positive_fold_contrast_both_seeds"])
                self.assertIsNone(summary["selected_arm"])
                self.assertFalse(summary["learning_adequacy"])

    def test_every_public_archive_path_rejects_research(self):
        with self.assertRaisesRegex(a.ArchiveError, "research execution"):
            a.verify_contract(self.members, self.inventory)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "synthetic.zip"
            ids = zip_packet(path, self.members, self.inventory)
            with self.assertRaisesRegex(a.ArchiveError, "research execution"):
                a.verify_archive(path, *ids)
            with self.assertRaisesRegex(a.ArchiveError, "research execution"):
                a.restore_historical_source(path, *ids, root / "restored")
            self.assertFalse((root / "restored").exists())
            request, verification = root / "request.json", root / "verification.json"
            request.write_bytes(self.members["external/request.json"])
            verification.write_bytes(self.members["external/verification-request.json"])
            with patch.object(a, "tree_bytes") as scan:
                with self.assertRaisesRegex(a.ArchiveError, "research execution"):
                    a.build_archive(root / "forbidden.zip", root / "run", request,
                        a.identity(request.read_bytes()), verification, a.identity(verification.read_bytes()), root / "old")
                scan.assert_not_called()
            self.assertFalse((root / "forbidden.zip").exists())

    def test_incomplete_history_cadence_or_checkpoint_rejected(self):
        prefix = "run/auxiliary-qec-0-extra-final/"
        for field, mutate in (
            ("steps.json", lambda v: v["steps"].pop()),
            ("steps.json", lambda v: v["train_diagnostics"].pop()),
            ("metadata.json", lambda v: v.update(schema="trinite.procedural-qec-auxiliary-checkpoint.v1")),
            ("metadata.json", lambda v: v.update(cursor=32768)),
        ):
            with self.subTest(field=field):
                members = dict(self.members)
                record = a.parse(members[prefix + field])
                mutate(record)
                members[prefix + field] = a.canonical(record)
                with self.assertRaises(a.ArchiveError):
                    a._verify_research_records(members, self.request, self.summary)

    def test_candidate_checkpoint_metadata_bound_is_enforced_before_parse(self):
        from trinite.procedural_qec_auxiliary.spec import metadata_limit
        name = "run/auxiliary-qec-0-extra-final/metadata.json"
        self.assertGreater(len(self.members[name]), metadata_limit("engineering"))
        self.assertLessEqual(len(self.members[name]), metadata_limit("research"))
        oversized = b"x" * (metadata_limit("research") + 1)
        with self.assertRaisesRegex(a.ArchiveError, "candidate byte bound"):
            a._verify_research_records({**self.members, name: oversized}, self.request, self.summary)

    def test_repeated_missing_or_held_out_predictions_rejected(self):
        name = "run/auxiliary-qec-0-extra-final/predictions.json"
        for mutate in (
            lambda v: v["predictions"]["final"].append(v["predictions"]["final"][0]),
            lambda v: v["predictions"]["prefix"].pop(),
            lambda v: v["predictions"]["trace"][0].update(split="test"),
            lambda v: v["predictions"]["final"][1].update(example_identity=v["predictions"]["final"][0]["example_identity"]),
        ):
            value = a.parse(self.members[name])
            mutate(value)
            with self.assertRaises(a.ArchiveError):
                a._verify_research_records({**self.members, name: a.canonical(value)}, self.request, self.summary)

    def test_gate_contrast_auxiliary_counts_and_loss_inventory_rejected(self):
        for mutate in (
            lambda s: s.update(training_gate_pass=False),
            lambda s: s.update(positive_fold_contrast_both_seeds=False),
            lambda s: s["contrasts"]["0"].update(add=True),
            lambda s: s["cells"][0]["report"].update(training_task_failures=["add"]),
            lambda s: s["cells"][0]["report"]["scores"]["final"]["tasks"]["add"].update(examples=132),
            lambda s: s["cells"][0]["report"]["auxiliary_visited_unvisited"]["fold-sum/prefix"]["visited"].update(accuracy_defined=True),
            lambda s: s["cells"][0]["report"]["loss_by_mode_and_task"].pop("trace"),
        ):
            summary = deepcopy(self.summary)
            mutate(summary)
            with self.assertRaises(a.ArchiveError):
                a._verify_research_records(self.members, self.request, summary)

    def test_request_checkpoint_environment_and_paired_bindings_rejected(self):
        for field in ("source", "inputs", "environment", "schedule", "initialization_identity"):
            with self.subTest(field=field):
                name = "run/auxiliary-qec-0-extra-final/metadata.json"
                metadata = a.parse(self.members[name])
                metadata[field] = "changed checkpoint context"
                with self.assertRaises(a.ArchiveError):
                    a._verify_research_records({**self.members, name: a.canonical(metadata)}, self.request, self.summary)
        summary = deepcopy(self.summary)
        summary["cells"][0]["report"]["plan"]["model"]["width"] = 16
        with self.assertRaisesRegex(a.ArchiveError, "model/resource/optimizer"):
            a._verify_research_records(self.members, self.request, summary)
        summary, request, members = deepcopy(self.summary), deepcopy(self.request), dict(self.members)
        treatment = next(c["report"] for c in summary["cells"] if c["seed"] == 0 and c["arm"] == "extra-procedural")
        schedule = treatment["schedule"]
        schedule["primary"]["original_row_sequence_identity"] = "sha256:" + "0" * 64
        request["schedules"]["auxiliary-qec-0-extra-procedural"] = schedule
        name = "run/auxiliary-qec-0-extra-procedural/metadata.json"
        metadata = a.parse(members[name])
        metadata["schedule"] = schedule
        members[name] = a.canonical(metadata)
        with self.assertRaisesRegex(a.ArchiveError, "paired original primary"):
            a._verify_research_records(members, request, summary)

    def test_required_replay_check_and_external_commitment_cannot_be_resealed(self):
        summary = deepcopy(self.summary)
        summary["cells"][0]["checks"]["complete_optimizer_replay"] = False
        update(self.members, self.inventory, "run/summary.json", a.canonical(summary))
        self.inventory["summary_identity"] = a.identity(self.members["run/summary.json"])
        audit = a.parse(self.members["audit/replay.json"])
        audit["summary_identity"] = self.inventory["summary_identity"]
        update(self.members, self.inventory, "audit/replay.json", a.canonical(audit))
        with self.assertRaisesRegex(a.ArchiveError, "replay checks"):
            a._verify_profile_contract(self.members, self.inventory, "research")
        members, inventory = deepcopy(self.template)
        target = "run/auxiliary-qec-1-extra-procedural/tensors.safetensors"
        update(members, inventory, target, b"forged optimizer bytes")
        with self.assertRaisesRegex(a.ArchiveError, "committed run bytes"):
            a._verify_profile_contract(members, inventory, "research")


if __name__ == "__main__":
    unittest.main()
