"""Bounded lossless transport for engineering-only auxiliary conformance.

Import and transport verification use only the standard library. A transport
pass checks retained bytes and externally supplied commitments, NOT numerical
replay or custody admission. Building a packet first calls the production full
native replay. Historical source is a source-only, shallow sparse Git snapshot:
exact original commit, all tree objects and all src blobs, without old unrelated
fixture archives. No original request, source, evidence or digest is rewritten.
"""
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "trinite.procedural-qec-auxiliary-archive.v1"
RESEARCH_SCHEMA = "trinite.procedural-qec-auxiliary-research-archive.v1"
MAX_MEMBERS = 2048
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_ENTRIES = 10000
ROOT_COMMITTED = frozenset(("request.json", "inputs.json", "worker-errors.json"))
CELL_COMMITTED = frozenset(("report.json", "predictions.json", "steps.json",
    "tensors.safetensors", "metadata.json", "verification.json", "provenance/manifest.json"))
CHECKS = frozenset(("closed_custody", "request_and_source_binding", "safe_tensor_structure",
    "scheduled_visits_and_targets", "final_predictions_and_loss", "complete_optimizer_replay"))
CELLS = frozenset(f"auxiliary-qec-{seed}-{arm}" for seed in (0, 1)
                  for arm in ("extra-final", "extra-procedural"))
INPUT_FILES = {"dataset.json": "dataset_raw", "manifest.json": "manifest_raw",
    "historical-dataset.json": "historical_dataset_raw",
    "historical-manifest.json": "historical_manifest_raw",
    "historical-payload.json": "historical_payload_raw", "mapping-receipt.json": "receipt_raw"}
INPUT_COMMITMENTS = {"dataset.json": "dataset_identity", "manifest.json": "manifest_identity",
    "historical-dataset.json": "historical_dataset_identity",
    "historical-manifest.json": "historical_manifest_identity",
    "mapping-receipt.json": "mapping_receipt_identity"}
HISTORICAL_COMMIT = "93522bf8f18fe38970ceafdd2ace792f8922c382"
HISTORICAL_TREE = "c8cc28e560729722539a2a6ee47408f50a7a802e"
ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class ArchiveError(ValueError):
    """Malformed, incomplete or incorrectly committed archival material."""


def require(condition, message):
    if not condition:
        raise ArchiveError(message)


def identity(raw):
    require(type(raw) is bytes, "identity requires exact bytes")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def require_id(value):
    require(type(value) is str and ID_RE.fullmatch(value) is not None,
            "noncanonical external SHA-256 identity")
    return value


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def parse(raw):
    def pairs(entries):
        value = {}
        for key, item in entries:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    def reject(_):
        raise ArchiveError("floating or nonfinite JSON number")
    try:
        require(type(raw) is bytes and len(raw) <= MAX_MEMBER_BYTES, "invalid JSON byte budget")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_float=reject, parse_constant=reject)
        require(canonical(value) == raw, "noncanonical JSON evidence")
        return value
    except (UnicodeError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ArchiveError("invalid canonical JSON evidence") from exc


def safe_path(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            "archival paths must not traverse symlinks")
    return path.resolve()


def safe_name(name):
    require(type(name) is str and 0 < len(name) <= 512 and "\\" not in name
            and ":" not in name and not name.startswith("/")
            and not any(ord(c) < 32 or ord(c) == 127 for c in name)
            and all(p not in ("", ".", "..") for p in name.split("/")),
            "unsafe archive member name")
    return name


def read(path, limit=MAX_MEMBER_BYTES):
    path = safe_path(path)
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "missing regular archival file: " + str(path))
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    require(len(raw) <= limit, "archive member size limit exceeded")
    return raw


def add(members, name, raw):
    safe_name(name)
    require(name not in members and type(raw) is bytes and len(raw) <= MAX_MEMBER_BYTES,
            "duplicate or oversized archive member")
    members[name] = raw
    require(len(members) <= MAX_MEMBERS and sum(map(len, members.values())) <= MAX_TOTAL_BYTES,
            "total archive member budget exceeded")


def tree_bytes(root):
    """Read all regular bytes, rejecting symlinks/specials and bounded traversal."""
    root = safe_path(root)
    require(root.is_dir(), "missing archive input directory")
    result, entries = {}, 0
    def fail(error):
        raise ArchiveError("cannot enumerate archive input") from error
    for directory, directories, files in os.walk(root, followlinks=False, onerror=fail):
        entries += len(directories) + len(files)
        require(entries <= MAX_ENTRIES, "archive traversal budget exceeded")
        for name in directories + files:
            path = safe_path(Path(directory) / name)
            require(path.is_dir() or stat.S_ISREG(path.stat().st_mode), "nonregular archive input")
        directories.sort()
        for name in sorted(files):
            path = Path(directory) / name
            add(result, path.relative_to(root).as_posix(), read(path))
    return result


def _git(root, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(["git", "--no-replace-objects", "--literal-pathspecs",
        "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "core.autocrlf=false",
        "-C", str(root), *args], capture_output=True, env=env, timeout=60, check=False)
    require(result.returncode == 0 and len(result.stdout) <= MAX_MEMBER_BYTES,
            "historical Git operation failed: " + result.stderr.decode(errors="replace")[:1000])
    return result.stdout


def _git_identity(kind, raw):
    return hashlib.sha1(kind.encode("ascii") + b" " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def _tree_entries(raw):
    entries, offset, names = [], 0, set()
    while offset < len(raw):
        space, end = raw.find(b" ", offset), raw.find(b"\0", offset)
        require(offset < space < end and end + 21 <= len(raw), "malformed historical Git tree")
        try:
            mode, name = raw[offset:space].decode("ascii"), raw[space + 1:end].decode("utf-8")
        except UnicodeError as exc:
            raise ArchiveError("malformed historical tree encoding") from exc
        safe_name(name)
        require("/" not in name and name not in names and mode in ("40000", "100644", "100755"),
                "unsupported or duplicate historical tree entry")
        names.add(name)
        entries.append((mode, name, raw[end + 1:end + 21].hex()))
        offset = end + 21
    return entries


def historical_snapshot(root):
    """Capture original Git objects/source, never new commits or resealed pins."""
    root = safe_path(root)
    require(_git(root, "rev-parse", "HEAD").decode().strip() == HISTORICAL_COMMIT,
            "historical source commit mismatch")
    require(not _git(root, "status", "--porcelain=v1", "--untracked-files=all"),
            "historical source checkout is dirty")
    members, objects, files = {}, {}, {}
    def obj(oid, kind):
        if oid in objects:
            return members["history/git/objects/" + oid]
        require(len(objects) < 512, "historical Git object count exceeded")
        raw = _git(root, "cat-file", kind, oid)
        require(len(raw) <= 2 * 1024 * 1024 and _git_identity(kind, raw) == oid,
                "historical Git object differs")
        objects[oid] = {"kind": kind, "identity": identity(raw)}
        add(members, "history/git/objects/" + oid, raw)
        return raw
    obj(HISTORICAL_COMMIT, "commit")
    def walk(oid, prefix="", depth=0):
        require(depth < 32, "historical tree depth exceeded")
        for mode, name, child in _tree_entries(obj(oid, "tree")):
            path = prefix + name
            if mode == "40000":
                walk(child, path + "/", depth + 1)
            elif path.startswith("src/"):
                raw = read(root / path, 2 * 1024 * 1024)
                require(_git_identity("blob", raw) == child, "historical source blob differs")
                files[path] = {"mode": mode, "git_blob": child, "identity": identity(raw)}
                add(members, "history/source/" + path, raw)
    walk(HISTORICAL_TREE)
    require(set(tree_bytes(root / "src")) == {p[4:] for p in files},
            "extra or missing historical source file")
    snapshot = {"commit": HISTORICAL_COMMIT, "tree": HISTORICAL_TREE,
                "objects": objects, "files": files, "layout": "shallow-sparse-src"}
    verify_historical(members, snapshot)
    return members, snapshot


def verify_historical(members, snapshot):
    require(type(snapshot) is dict and set(snapshot) == {"commit", "tree", "objects", "files", "layout"}
            and snapshot["commit"] == HISTORICAL_COMMIT and snapshot["tree"] == HISTORICAL_TREE
            and snapshot["layout"] == "shallow-sparse-src", "historical source pin/layout mismatch")
    objects, files = snapshot["objects"], snapshot["files"]
    require(type(objects) is dict and 1 <= len(objects) <= 512
            and type(files) is dict and 1 <= len(files) <= 512, "historical source closure count differs")
    expected = {"history/git/objects/" + k for k in objects} | {"history/source/" + p for p in files}
    require({n for n in members if n.startswith("history/")} == expected,
            "extra or missing historical source member")
    for oid, record in objects.items():
        require(type(oid) is str and GIT_RE.fullmatch(oid) is not None
                and type(record) is dict and set(record) == {"kind", "identity"}
                and record["kind"] in ("commit", "tree"), "invalid historical object record")
        raw = members["history/git/objects/" + oid]
        require(len(raw) <= 2 * 1024 * 1024 and identity(raw) == require_id(record["identity"])
                and _git_identity(record["kind"], raw) == oid, "historical Git object commitment differs")
    commit = members.get("history/git/objects/" + HISTORICAL_COMMIT, b"")
    require(commit.startswith(("tree " + HISTORICAL_TREE + "\n").encode()),
            "historical commit does not bind original tree")
    visited, expected_files = {HISTORICAL_COMMIT}, {}
    def walk(oid, prefix="", depth=0):
        require(depth < 32 and oid in objects and objects[oid]["kind"] == "tree",
                "missing historical Git tree")
        visited.add(oid)
        for mode, name, child in _tree_entries(members["history/git/objects/" + oid]):
            path = prefix + name
            if mode == "40000":
                walk(child, path + "/", depth + 1)
            elif path.startswith("src/"):
                require(path in files, "missing historical source blob")
                raw = members["history/source/" + path]
                expected_files[path] = {"mode": mode, "git_blob": child, "identity": identity(raw)}
                require(_git_identity("blob", raw) == child, "historical source blob commitment differs")
    walk(HISTORICAL_TREE)
    require(visited == set(objects) and expected_files == files,
            "extra or changed historical source closure")
    return expected


def _run_inventory(members):
    """Enumerate complete declared bundle bytes; native custody is checked elsewhere."""
    expected = {"run/" + p for p in ROOT_COMMITTED | {"summary.json"}}
    for cell in CELLS:
        prefix = "run/" + cell + "/"
        expected.update(prefix + p for p in CELL_COMMITTED)
        manifest = parse(members.get(prefix + "provenance/manifest.json", b""))
        core = manifest.get("core", {}) if type(manifest) is dict else {}
        require(type(core) is dict and core.get("scope") == "closed"
                and type(core.get("artifacts")) is list and 1 <= len(core["artifacts"]) <= 64
                and type(core.get("events")) is list and 1 <= len(core["events"]) <= 32,
                "archive requires closed bounded custody manifest")
        require(set(manifest) == {"core", "manifest_identity", "self_hash_exclusion"}
                and manifest["self_hash_exclusion"] == "manifest_identity"
                and manifest["manifest_identity"] == identity(b"PROVENANCE/MANIFEST/v1\0" + canonical(core)),
                "custody manifest byte commitment differs")
        seen = set()
        for artifact in core["artifacts"]:
            require(type(artifact) is dict and set(artifact) == {"content_identity", "record_identity", "retention"}
                    and artifact["retention"] == "CONTENT_RETAINED", "malformed retained artifact entry")
            content, record = require_id(artifact["content_identity"]), require_id(artifact["record_identity"])
            require(content not in seen, "duplicate custody artifact")
            seen.add(content)
            name = prefix + "provenance/artifacts/sha256/" + content[7:]
            require(name in members and identity(members[name]) == content, "retained custody content differs")
            expected.add(name)
            record_name = prefix + "provenance/artifact_records/sha256/" + record[7:] + ".json"
            require(record_name in members and identity(b"PROVENANCE/ARTIFACT-RECORD/v1\0" + members[record_name]) == record,
                    "custody artifact record commitment differs")
            metadata = parse(members[record_name])
            require(type(metadata) is dict and metadata.get("content_identity") == content
                    and metadata.get("byte_count") == len(members[name]) and type(metadata.get("byte_count")) is int
                    and metadata.get("retention") == "CONTENT_RETAINED", "custody artifact metadata differs")
            expected.add(record_name)
        event_ids = [require_id(e) for e in core["events"]]
        require(len(set(event_ids)) == len(event_ids), "duplicate custody event")
        for event in event_ids:
            name = prefix + "provenance/events/sha256/" + event[7:] + ".json"
            value = parse(members.get(name, b""))
            require(type(value) is dict and set(value) == {"core", "event_identity", "self_hash_exclusion"}
                    and value["self_hash_exclusion"] == "event_identity" and value["event_identity"] == event
                    and identity(b"PROVENANCE/EVENT/v1\0" + canonical(value["core"])) == event,
                    "custody event byte commitment differs")
            expected.add(name)
    require({n for n in members if n.startswith("run/")} == expected,
            "extra or missing run/bundle bytes")
    return expected


def verify_contract(members, inventory):
    """Public engineering-only closed transport boundary; research stays disabled."""
    require(type(inventory) is dict and inventory.get("run_profile") == "engineering",
            "archive research execution/claims disabled")
    return _verify_profile_contract(members, inventory, "engineering")


def _verify_profile_contract(members, inventory, profile):
    """Dormant profile-aware byte contract; no model calls or execution authority.

    The private research branch is exercised only by synthetic structural tests.
    Public builders/verifiers cannot enter it before a separately reviewed,
    source-changing scientific activation and immutable protocol freeze.
    """
    require(profile in ("engineering", "research"), "unknown archival profile")
    research = profile == "research"
    schema = RESEARCH_SCHEMA if research else SCHEMA
    prefix = "trinite.procedural-qec-auxiliary-" + ("research-" if research else "")
    evidence_class = "training-only-exploratory-candidate" if research else "engineering-conformance-only"
    require(type(members) is dict and type(inventory) is dict, "invalid archive structure")
    keys = {"schema", "request_identity", "verification_request_identity", "summary_identity",
            "run_profile", "evidence_class", "source", "inputs", "historical_source", "members"}
    require(set(inventory) == keys and inventory["schema"] == schema, "invalid archive inventory schema")
    require(inventory["run_profile"] == profile and inventory["evidence_class"] == evidence_class,
            "archive research execution/claims disabled")
    require(all(type(raw) is bytes for raw in members.values())
            and 0 < len(members) <= MAX_MEMBERS and sum(map(len, members.values())) <= MAX_TOTAL_BYTES,
            "archive member budget exceeded")
    require(type(inventory["members"]) is dict and set(inventory["members"]) == set(members),
            "archive inventory is not complete")
    for name, value in inventory["members"].items():
        safe_name(name)
        require(type(members[name]) is bytes and len(members[name]) <= MAX_MEMBER_BYTES
                and type(value) is dict and set(value) == {"identity", "bytes"}
                and value["identity"] == identity(members[name]) and type(value["bytes"]) is int
                and value["bytes"] == len(members[name]), "archive member size/identity mismatch: " + name)
    required = {"external/request.json", "external/verification-request.json", "audit/replay.json",
                "run/request.json", "run/inputs.json", "run/worker-errors.json", "run/summary.json"}
    require(required <= set(members), "missing archive requests/run/audit")
    request_id, verification_id = require_id(inventory["request_identity"]), require_id(inventory["verification_request_identity"])
    summary_id = require_id(inventory["summary_identity"])
    require(identity(members["external/request.json"]) == request_id
            and members["run/request.json"] == members["external/request.json"]
            and identity(members["external/verification-request.json"]) == verification_id
            and identity(members["run/summary.json"]) == summary_id,
            "original external request, verification or summary commitments differ")
    request = parse(members["external/request.json"])
    verification = parse(members["external/verification-request.json"])
    summary = parse(members["run/summary.json"])
    audit = parse(members["audit/replay.json"])
    require(all(type(r) is dict for r in (request, verification, summary, audit)), "invalid evidence object")
    require(request.get("schema") == prefix + "request.v1"
            and request.get("run_profile") == profile
            and type(request.get("specification")) is dict
            and request["specification"].get("research_execution_enabled") is False
            and request.get("specification_identity") == identity(canonical(request["specification"])),
            "request is not engineering-only candidate")
    require(summary.get("schema") == prefix + "summary.v1"
            and summary.get("request_identity") == request_id
            and summary.get("verification_request_identity") == verification_id
            and summary.get("run_profile") == profile and summary.get("outcome") == "completed"
            and summary.get("selected_arm") is None and summary.get("worker_errors") == {}
            and summary.get("pairing_errors") == [] and parse(members["run/worker-errors.json"]) == {},
            "archive requires complete four-cell engineering result")
    for record in (summary, audit):
        require(record.get("evidence_class") == evidence_class
                and record.get("learning_adequacy") is False and record.get("phase5_ready") is False,
                "archived receipt claims unsupported scientific evidence")
    require(all(summary.get(k) is False for k in ("held_out_scored", "format_selection_eligible", "release_ready"))
            and summary.get("training_gate_evaluated") is research
            and (research or summary.get("training_gate_pass") is False),
            "archived summary claims an unsupported outcome")
    require(audit.get("outcome") == "completed" and type(audit.get("replayed_cells")) is int
            and audit["replayed_cells"] == 4 and audit.get("summary_identity") == summary_id
            and audit.get("verification_request_identity") == verification_id,
            "retained archive-time native replay receipt differs")
    require(type(summary.get("cells")) is list and len(summary["cells"]) == 4,
            "archive needs all four declared cells")
    records = set()
    for cell in summary["cells"]:
        require(type(cell) is dict and type(cell.get("seed")) is int and cell["seed"] in (0, 1)
                and cell.get("arm") in ("extra-final", "extra-procedural")
                and cell.get("outcome") == "completed" and type(cell.get("checks")) is dict
                and set(cell["checks"]) == CHECKS and all(v is True for v in cell["checks"].values()),
                "cell lacks recorded native replay checks")
        name = f"auxiliary-qec-{cell['seed']}-{cell['arm']}"
        require(name not in records, "duplicate engineering cell")
        records.add(name)
        raw = members.get("run/" + name + "/report.json", b"")
        report = parse(raw)
        require(identity(raw) == cell.get("report_identity") and report == cell.get("report")
                and report.get("schema") == prefix + "report.v1"
                and report.get("seed") == cell["seed"] and report.get("arm") == cell["arm"]
                and report.get("run_profile") == profile
                and report.get("evidence_class") == evidence_class
                and report.get("training_gate_evaluated") is research
                and all(report.get(k) is False for k in ("held_out_scored",
                    "learning_adequacy", "phase5_ready", "format_selection_eligible", "release_ready")),
                "cell summary/report commitments or engineering scope differ")
    require(records == CELLS and set(verification) == {"schema", "request_identity", "root_artifacts", "cells"}
            and verification["schema"] == prefix + "verification-request.v2"
            and verification["request_identity"] == request_id
            and type(verification["cells"]) is dict and set(verification["cells"]) == CELLS
            and type(verification["root_artifacts"]) is dict
            and set(verification["root_artifacts"]) == ROOT_COMMITTED,
            "external verification inventory differs")
    committed = {"run/" + p: digest for p, digest in verification["root_artifacts"].items()}
    for cell, record in verification["cells"].items():
        require(type(record) is dict and set(record) == {"outcome", "artifacts"},
                "external cell inventory differs")
        # Failed v2 records retain externally bound partial bytes for read-only
        # failure verification. They cannot certify completed archive replay.
        require(record["outcome"] == "completed", "archive requires completed cell artifact inventory")
        require(type(record["artifacts"]) is dict and set(record["artifacts"]) == CELL_COMMITTED,
                "external completed cell inventory differs")
        committed.update({"run/" + cell + "/" + p: digest for p, digest in record["artifacts"].items()})
    for name, digest in committed.items():
        require(name in members and identity(members[name]) == require_id(digest),
                "archived committed run bytes differ: " + name)
    expected = _run_inventory(members) | {"external/request.json", "external/verification-request.json", "audit/replay.json"}
    source = request.get("source")
    require(type(source) is dict and 0 < len(source) <= 512 and source == inventory["source"],
            "producer source closure differs")
    for path, digest in source.items():
        name = "source/" + safe_name(path)
        require(name in members and identity(members[name]) == require_id(digest), "frozen producer source differs")
        expected.add(name)
    require(type(inventory["inputs"]) is dict and set(inventory["inputs"]) == set(INPUT_FILES)
            and type(request.get("inputs")) is dict and set(request["inputs"]) == set(INPUT_COMMITMENTS.values()),
            "input byte inventory differs")
    for path in INPUT_FILES:
        name = "inputs/" + path
        require(name in members and identity(members[name]) == require_id(inventory["inputs"][path]),
                "materialized input bytes differ")
        if path in INPUT_COMMITMENTS:
            require(identity(members[name]) == require_id(request["inputs"][INPUT_COMMITMENTS[path]]),
                    "request-bound input bytes differ")
        expected.add(name)
    require(members["run/inputs.json"] == members["inputs/mapping-receipt.json"], "mapping receipt copy differs")
    receipt = parse(members["inputs/mapping-receipt.json"])
    receipt_fields = {"historical-dataset.json": "historical_dataset_identity",
        "historical-manifest.json": "historical_manifest_identity",
        "historical-payload.json": "historical_payload_identity",
        "dataset.json": "prospective_dataset_identity", "manifest.json": "prospective_manifest_identity"}
    require(type(receipt) is dict and all(receipt.get(field) == identity(members["inputs/" + name])
            for name, field in receipt_fields.items())
            and receipt.get("historical_producer_commit") == HISTORICAL_COMMIT
            and receipt.get("historical_git_tree") == HISTORICAL_TREE,
            "request-bound historical/current input receipt differs")
    payload = parse(members["inputs/historical-payload.json"])
    require(type(payload) is dict and canonical(payload.get("dataset")) == members["inputs/historical-dataset.json"]
            and canonical(payload.get("manifest")) == members["inputs/historical-manifest.json"],
            "original historical payload differs from retained data")
    expected.update(verify_historical(members, inventory["historical_source"]))
    require(set(members) == expected, "unexpected archive member outside closed inventory")
    if research:
        _verify_research_records(members, request, summary)
    return {"schema": prefix + "archive-verification.v1",
        "outcome": "passed", "verification_scope": "byte-transport-only",
        "numerical_replay_performed": False, "custody_admission_performed": False,
        "run_profile": profile, "evidence_class": evidence_class,
        "request_identity": request_id, "verification_request_identity": verification_id,
        "summary_identity": summary_id, "member_count": len(members),
        "learning_adequacy": False, "phase5_ready": False}


def _finite_hex(value):
    try:
        number = float.fromhex(value)
    except (TypeError, ValueError) as exc:
        raise ArchiveError("noncanonical retained numerical diagnostic") from exc
    require(type(value) is str and math.isfinite(number) and number >= 0 and number.hex() == value,
            "noncanonical retained numerical diagnostic")


def _verify_research_records(members, request, summary):
    """Dormant count/shape contract, not numerical truth or research authority.

    Complete histories and unique predictions are transport requirements. The
    unchanged production replay must establish their numerical correctness.
    Reuse pure candidate criteria rather than defining another gate scorer.
    """
    from .spec import assess_training_counts, metadata_limit, profile_settings, specification
    candidate = profile_settings("research")
    require(request["specification"].get("research_candidate") == candidate
            and request["specification"].get("batch_size") == 8
            and request["specification"].get("pad_to") == 64, "dormant research request budget differs")
    dataset = parse(members["inputs/dataset.json"])
    require(type(dataset) is dict and type(dataset.get("examples")) is list
            and len(dataset["examples"]) == 3876, "research needs all unique admitted diagnostic rows")
    items = {}
    for item in dataset["examples"]:
        require(type(item) is dict and item.get("split") == "train"
                and item.get("mode") in ("final", "prefix", "trace")
                and type(item.get("task")) is str and type(item.get("kind")) is str,
                "research diagnostic row must be training-only")
        ident = require_id(item.get("example_identity"))
        require(ident not in items, "duplicate research diagnostic identity")
        items[ident] = item
    denominators = {"final": {task: total for task, (_, total) in specification()["gates"].items()},
        "prefix": {"fold-sum": 1150, "gf3": 54}, "trace": {"fold-sum": 1150, "gf3": 54}}
    require(type(request.get("schedules")) is dict and set(request["schedules"]) == CELLS,
            "research requires four complete schedule commitments")
    all_counts = {}
    for cell in summary["cells"]:
        name = f"auxiliary-qec-{cell['seed']}-{cell['arm']}"
        report, prefix, schedule = cell["report"], "run/" + name + "/", request["schedules"][name]
        require(type(schedule) is dict and report.get("schedule") == schedule
                and schedule.get("macro_rounds") == 4096 and type(schedule.get("macro_rounds")) is int
                and schedule.get("native_updates") == 8192 and type(schedule.get("native_updates")) is int
                and schedule.get("seed") == cell["seed"] and schedule.get("arm") == cell["arm"]
                and schedule.get("padded_encoded_positions") == 4194304
                and schedule.get("model_input_positions") == 4128768
                and schedule.get("within_existing_target_ceiling") is True,
                "research schedule is not the complete fixed four-cell budget")
        require(report.get("steps") == 8192 and type(report.get("steps")) is int
                and type(report.get("plan")) is dict and report["plan"].get("steps") == 8192
                and report["plan"].get("batch_size") == 8 and report["plan"].get("validation_every") == 2048,
                "research report does not retain all 8192 native updates")
        plan = report["plan"]
        require(type(plan.get("model")) is dict and all(type(plan["model"].get(key)) is int
                and plan["model"][key] == candidate[key]
                for key in ("blocks", "width", "heads", "feed_forward_width", "context_length"))
                and all(type(plan.get(key)) is int and plan[key] == candidate[key]
                        for key in ("steps", "validation_every", "max_seconds", "max_target_tokens"))
                and plan.get("seed") == cell["seed"]
                and all(plan.get(key) == value for key, value in specification()["optimizer"].items()),
                "research model/resource/optimizer plan differs from candidate")
        require(report.get("source") == request["source"] and report.get("inputs") == request["inputs"]
                and type(request.get("environment")) is dict and report.get("environment") == request["environment"]
                and report.get("specification_identity") == request["specification_identity"],
                "research report source/input/specification commitments differ")
        require_id(report.get("initialization_identity"))
        require_id(report.get("model_identity"))
        require(len(members[prefix + "metadata.json"]) <= metadata_limit("research"),
                "research checkpoint metadata exceeds candidate byte bound")
        steps, metadata = parse(members[prefix + "steps.json"]), parse(members[prefix + "metadata.json"])
        require(type(steps) is dict and set(steps) == {"steps", "train_diagnostics"}
                and type(steps["steps"]) is list and len(steps["steps"]) == 8192
                and type(steps["train_diagnostics"]) is list and type(metadata) is dict
                and metadata.get("schema") == "trinite.procedural-qec-auxiliary-research-checkpoint.v1"
                and metadata.get("run_profile") == "research"
                and metadata.get("seed") == cell["seed"] and metadata.get("arm") == cell["arm"]
                and metadata.get("step") == 8192 and type(metadata.get("step")) is int
                and metadata.get("cursor") == 65536 and type(metadata.get("cursor")) is int
                and metadata.get("history") == steps["steps"]
                and metadata.get("validation") == steps["train_diagnostics"]
                and metadata.get("plan") == report["plan"]
                and metadata.get("schedule") == report["schedule"]
                and all(metadata.get(key) == report[key] for key in ("source", "inputs", "environment", "specification_identity",
                        "initialization_identity", "model_identity"))
                and metadata.get("request_identity") == summary["request_identity"]
                and report.get("request_identity") == summary["request_identity"],
                "research checkpoint/history/request commitments differ")
        visits, target_total = [], 0
        for index, step in enumerate(steps["steps"], 1):
            require(type(step) is dict and step.get("step") == index and type(step.get("step")) is int
                    and type(step.get("examples")) is list and len(step["examples"]) == 8
                    and all(type(e) is str and e in items for e in step["examples"])
                    and type(step.get("target_tokens")) is int and 1 <= step["target_tokens"] <= 512,
                    "research history has missing, reordered or malformed native updates")
            _finite_hex(step.get("loss_hex"))
            _finite_hex(step.get("gradient_norm_hex"))
            visits.extend(step["examples"])
            target_total += step["target_tokens"]
        require([d.get("step") for d in steps["train_diagnostics"] if type(d) is dict]
                == [2048, 4096, 6144, 8192] and len(steps["train_diagnostics"]) == 4,
                "research diagnostic cadence differs")
        for diagnostic in steps["train_diagnostics"]:
            _finite_hex(diagnostic.get("loss_hex"))
        require(type(schedule.get("combined")) is dict and type(schedule["combined"].get("totals")) is dict
                and schedule["combined"]["totals"].get("visits") == 65536
                and schedule["combined"]["totals"].get("target_tokens") == target_total
                and 0 < target_total <= 1048576 and report.get("target_tokens") == target_total
                and metadata.get("target_tokens") == target_total
                and schedule.get("schedule_identity") == identity(canonical(visits)),
                "research history target/visit/schedule commitments differ")
        predictions = parse(members[prefix + "predictions.json"])
        require(type(predictions) is dict and set(predictions) == {"predictions"}
                and type(predictions["predictions"]) is dict and set(predictions["predictions"]) == set(denominators)
                and type(report.get("scores")) is dict and set(report["scores"]) == set(denominators),
                "research requires complete final/prefix/trace diagnostics")
        observed, selected = {}, set(visits)
        for mode, expected_counts in denominators.items():
            rows, expected = predictions["predictions"][mode], {k for k, item in items.items() if item["mode"] == mode}
            require(type(rows) is list and len(rows) == sum(expected_counts.values()),
                    "research diagnostic denominator counts scheduled repetitions")
            seen, counts = set(), {task: {"correct": 0, "examples": 0} for task in expected_counts}
            for row in rows:
                require(type(row) is dict and type(row.get("example_identity")) is str
                        and row["example_identity"] in expected and row["example_identity"] not in seen
                        and row.get("split") == "train" and type(row.get("correct")) is bool,
                        "research predictions omit, repeat or expose held-out examples")
                ident = row["example_identity"]
                seen.add(ident)
                task = items[ident]["task"]
                require(task in counts and row.get("task") == task, "research prediction task differs")
                counts[task]["examples"] += 1
                counts[task]["correct"] += int(row["correct"])
                if mode != "final":
                    group = observed.setdefault(items[ident]["kind"] + "/" + mode,
                        {"visited": {"correct": 0, "examples": 0}, "unvisited": {"correct": 0, "examples": 0}})
                    part = group["visited" if ident in selected else "unvisited"]
                    part["examples"] += 1
                    part["correct"] += int(row["correct"])
            scores = report["scores"][mode]
            require(seen == expected and {task: row["examples"] for task, row in counts.items()} == expected_counts
                    and type(scores) is dict and scores.get("examples") == len(rows)
                    and scores.get("correct") == sum(row["correct"] for row in counts.values())
                    and type(scores.get("tasks")) is dict and set(scores["tasks"]) == set(counts),
                    "research unique diagnostic counts differ")
            for task, count in counts.items():
                actual = scores["tasks"][task]
                require(type(actual) is dict and all(type(actual.get(k)) is int and actual[k] == v for k, v in count.items()),
                        "research per-task score count differs")
            if mode == "final":
                all_counts[name] = counts
        for group in observed.values():
            for part in group.values():
                part["accuracy_defined"] = part["examples"] > 0
        require(canonical(report.get("auxiliary_visited_unvisited")) == canonical(observed),
                "research auxiliary diagnostics must use unique exposed/unexposed rows")
        losses = report.get("loss_by_mode_and_task")
        require(type(losses) is dict and set(losses) == set(denominators)
                and type(losses["final"]) is dict
                and report.get("final_only_loss_hex") == losses["final"].get("all_loss_hex"),
                "research per-mode/final loss inventory differs")
        _finite_hex(report.get("mixed_all_modes_loss_hex"))
        for mode, tasks in denominators.items():
            require(type(losses[mode]) is dict and type(losses[mode].get("tasks")) is dict
                    and set(losses[mode]["tasks"]) == set(tasks), "research task loss inventory differs")
            _finite_hex(losses[mode].get("all_loss_hex"))
            for value in losses[mode]["tasks"].values():
                _finite_hex(value)
    for seed in (0, 1):
        control, treatment = [next(cell["report"] for cell in summary["cells"]
                                  if cell["seed"] == seed and cell["arm"] == arm)
                              for arm in ("extra-final", "extra-procedural")]
        require(all(control[key] == treatment[key] for key in ("initialization_identity", "inputs", "plan")),
                "research paired initialization/input/plan differs")
        for key in ("original_row_sequence_identity", "prospective_row_sequence_identity", "normalized_row_sequence_identity"):
            primary_control, primary_treatment = control["schedule"].get("primary"), treatment["schedule"].get("primary")
            require(type(primary_control) is dict and type(primary_treatment) is dict
                    and require_id(primary_control.get(key)) == require_id(primary_treatment.get(key)),
                    "research paired original primary schedule differs")
    criteria = assess_training_counts(all_counts)
    for cell in summary["cells"]:
        name = f"auxiliary-qec-{cell['seed']}-{cell['arm']}"
        report = cell["report"]
        require(report.get("training_task_failures") == criteria["cells"][name]["training_task_failures"]
                and report.get("training_gate_pass") is criteria["cells"][name]["training_gate_pass"],
                "research cell gate differs from original unique task counts")
    require(summary.get("training_gate_pass") is criteria["treatment_training_gate_pass"]
            and summary.get("positive_fold_contrast_both_seeds") is criteria["positive_fold_contrast_both_seeds"]
            and canonical(summary.get("contrasts")) == canonical(criteria["contrasts"]),
            "research absolute gate and paired fold contrast must remain distinct")


def build_archive(path, run, request, request_identity, verification, verification_identity, historical_source):
    """Fresh materialization and full native replay, followed by exact transport."""
    request_identity, verification_identity = require_id(request_identity), require_id(verification_identity)
    path, run, request, verification = map(safe_path, (path, run, request, verification))
    historical_source = safe_path(historical_source)
    current_source = safe_path(ROOT / "src")
    require(path != current_source and current_source not in path.parents,
            "archive output must stay outside current producer source tree")
    require(not path.exists() and path.parent.is_dir() and path != run and run not in path.parents
            and request != verification and all(p != run and run not in p.parents for p in (request, verification))
            and path != request and path != verification and historical_source not in path.parents,
            "archive and external commitments require distinct unused paths outside run/source")
    request_raw, verification_raw = read(request, 128 * 1024), read(verification, 128 * 1024)
    require(identity(request_raw) == request_identity and identity(verification_raw) == verification_identity,
            "independently supplied external request identities differ")
    record = parse(request_raw)
    require(type(record) is dict and record.get("run_profile") == "engineering", "archive research execution disabled")
    before = tree_bytes(run)
    # No cached pass, receipt-only shortcut, alternate updater or final-forward substitute.
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from . import experiment
    from .inputs import materialize_inputs, revalidate
    inputs = materialize_inputs(historical_source)
    checked = experiment.verify_only(run, request, request_identity, verification, verification_identity, inputs)
    require(checked.get("outcome") == "completed" and checked.get("replayed_cells") == 4,
            "fresh full four-cell native replay is required")
    require(before == tree_bytes(run), "run bytes changed during native replay")
    members = {}
    for name, raw in before.items():
        add(members, "run/" + name, raw)
    add(members, "external/request.json", request_raw)
    add(members, "external/verification-request.json", verification_raw)
    add(members, "audit/replay.json", canonical(checked))
    source = record.get("source")
    require(type(source) is dict and 0 < len(source) <= 512, "missing producer source closure")
    for name, digest in sorted(source.items()):
        raw = read(ROOT / safe_name(name), 2 * 1024 * 1024)
        require(identity(raw) == require_id(digest), "producer differs from original request: " + name)
        add(members, "source/" + name, raw)
    input_ids = {}
    for name, field in INPUT_FILES.items():
        raw = getattr(inputs, field)
        input_ids[name] = identity(raw)
        add(members, "inputs/" + name, raw)
    historical, snapshot = historical_snapshot(historical_source)
    for name, raw in historical.items():
        add(members, name, raw)
    inventory = {"schema": SCHEMA, "request_identity": request_identity,
        "verification_request_identity": verification_identity,
        "summary_identity": identity(members["run/summary.json"]), "run_profile": "engineering",
        "evidence_class": "engineering-conformance-only", "source": source,
        "inputs": input_ids, "historical_source": snapshot,
        "members": {name: {"identity": identity(raw), "bytes": len(raw)} for name, raw in sorted(members.items())}}
    verify_contract(members, inventory)
    revalidate(inputs)
    require(before == tree_bytes(run) and read(request) == request_raw and read(verification) == verification_raw,
            "original run or external commitments changed after replay")
    require(all(read(ROOT / name, 2 * 1024 * 1024) == members["source/" + name] for name in source),
            "producer source changed after replay")
    inventory_raw = canonical(inventory)
    add(members, "inventory.json", inventory_raw)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    archive_raw = read(path, MAX_ARCHIVE_BYTES)
    result = verify_archive(path, identity(archive_raw), identity(inventory_raw))
    return {**result, "archive_bytes": len(archive_raw), "archive_build_native_replay": "completed",
            "archive_build_replayed_cells": 4, "training_gate_pass": False}


def _load_archive(path, archive_identity, inventory_identity):
    require_id(archive_identity)
    require_id(inventory_identity)
    raw = read(path, MAX_ARCHIVE_BYTES)
    require(identity(raw) == archive_identity, "archive differs from independent SHA-256 commitment")
    contents, total = {}, 0
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            infos = archive.infolist()
            require(0 < len(infos) <= MAX_MEMBERS + 1, "ZIP member count exceeded")
            for info in infos:
                safe_name(info.filename)
                require(info.filename not in contents and not info.is_dir() and not info.flag_bits & 1
                        and 0 <= info.file_size <= MAX_MEMBER_BYTES
                        and info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        and (info.create_system != 3 or stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG)),
                        "unsafe, duplicate, encrypted or oversized ZIP member")
                total += info.file_size
                require(total <= MAX_TOTAL_BYTES, "ZIP uncompressed size bound exceeded")
                with archive.open(info) as source:
                    value = source.read(MAX_MEMBER_BYTES + 1)
                require(len(value) == info.file_size, "truncated or oversized ZIP member")
                contents[info.filename] = value
    except (zipfile.BadZipFile, RuntimeError, EOFError, OSError, NotImplementedError) as exc:
        raise ArchiveError("corrupt ZIP transport") from exc
    require("inventory.json" in contents, "missing archive inventory")
    inventory_raw = contents.pop("inventory.json")
    require(identity(inventory_raw) == inventory_identity, "inventory differs from independently held commitment")
    inventory = parse(inventory_raw)
    return contents, inventory


def verify_archive(path, archive_identity, inventory_identity):
    """Read-only, stdlib-only transport check. It does not execute retained code."""
    members, inventory = _load_archive(path, archive_identity, inventory_identity)
    return {**verify_contract(members, inventory), "archive_identity": archive_identity,
            "inventory_identity": inventory_identity}


def restore_historical_source(path, archive_identity, inventory_identity, output):
    """Restore only authenticated historical source to a NEW sparse checkout.

    The checkout deliberately omits unrelated historical fixtures, but retains
    the exact original HEAD/tree/src and a complete sparse index. Git is needed
    only for restoration. The original isolated source admission is rerun by
    materialize_inputs; restoration itself makes no numerical/admission claim.
    """
    members, inventory = _load_archive(path, archive_identity, inventory_identity)
    verify_contract(members, inventory)
    output = safe_path(output)
    require(not output.exists() and output.parent.is_dir(), "historical restore needs an unused directory")
    output.mkdir()
    _git(output, "init", "--quiet", "--object-format=sha1")
    gitdir = output / ".git"
    snapshot = inventory["historical_source"]
    def write_object(oid, kind, raw):
        target = gitdir / "objects" / oid[:2] / oid[2:]
        encoded = kind.encode() + b" " + str(len(raw)).encode() + b"\0" + raw
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(zlib.compress(encoded))
    for oid, record in snapshot["objects"].items():
        write_object(oid, record["kind"], members["history/git/objects/" + oid])
    for name, record in snapshot["files"].items():
        write_object(record["git_blob"], "blob", members["history/source/" + name])
    (gitdir / "shallow").write_text(HISTORICAL_COMMIT + "\n", encoding="ascii")
    _git(output, "config", "core.sparseCheckout", "true")
    _git(output, "config", "core.sparseCheckoutCone", "false")
    (gitdir / "info" / "sparse-checkout").write_text("/src/\n", encoding="ascii")
    _git(output, "checkout", "--detach", HISTORICAL_COMMIT)
    require(not _git(output, "status", "--porcelain=v1", "--untracked-files=all"), "restored checkout is dirty")
    retained, check = historical_snapshot(output)
    require(check == snapshot and all(members[n] == raw for n, raw in retained.items()),
            "restored historical source differs")
    return {"schema": "trinite.procedural-qec-auxiliary-historical-restoration.v1",
        "outcome": "passed", "historical_commit": HISTORICAL_COMMIT,
        "historical_tree": HISTORICAL_TREE, "source_files": len(snapshot["files"]),
        "numerical_replay_performed": False, "source_admission_performed": False}
