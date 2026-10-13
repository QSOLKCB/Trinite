"""Lossless, independently checked transport of QEC procedural research evidence.

This is an *external* archival tool, not part of the frozen PR #34 trainer.
Only the separately frozen v3 native producer can numerically verify its own
request. The archival verifier needs no Torch and makes no capability claim.
"""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "trinite.procedural-qec-archive.v3"
MAX_MEMBERS = 2048
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
ROOT_COMMITTED = frozenset(("request.json", "worker-errors.json"))
CELL_COMMITTED = frozenset(("report.json", "predictions.json", "steps.json",
                           "tensors.safetensors", "metadata.json",
                           "verification.json", "provenance/manifest.json"))
ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
TOOLS = ("scripts/procedural_qec_v3_archive.py",
         "scripts/archive_procedural_qec_v3.py",
         ".github/workflows/procedural-qec-v3-research.yml")


class ArchiveError(ValueError):
    """Malformed, incomplete, or incorrectly committed archival material."""


def identity(raw):
    if type(raw) is not bytes:
        raise ArchiveError("identity requires exact bytes")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def require_id(value):
    if type(value) is not str or ID_RE.fullmatch(value) is None:
        raise ArchiveError("noncanonical external SHA-256 identity")
    return value


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def parse(raw):
    def pairs(entries):
        value = {}
        for key, item in entries:
            if key in value:
                raise ArchiveError("duplicate JSON key")
            value[key] = item
        return value
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_float=lambda _: (_ for _ in ()).throw(ArchiveError("floating JSON number")),
                           parse_constant=lambda _: (_ for _ in ()).throw(ArchiveError("nonfinite JSON")))
        if canonical(value) != raw:
            raise ArchiveError("noncanonical JSON evidence")
    except (UnicodeError, TypeError, ValueError, OverflowError) as exc:
        raise ArchiveError("invalid canonical JSON evidence") from exc
    return value


def safe_path(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ArchiveError("archival paths must not traverse symlinks")
    return path


def safe_name(name):
    if (type(name) is not str or not name or len(name) > 512 or "\\" in name or "\x00" in name or name.startswith("/")
            or any(ord(c) < 32 or ord(c) == 127 for c in name) or any(
                p in ("", ".", "..") for p in name.split("/"))):
        raise ArchiveError("unsafe archive member name")
    return name


def read(path, limit=MAX_MEMBER_BYTES):
    path = safe_path(path)
    if not path.is_file():
        raise ArchiveError("missing archival file: " + str(path))
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ArchiveError("archive member size limit exceeded")
    return raw


def add(members, name, raw):
    safe_name(name)
    if name in members or type(raw) is not bytes or len(raw) > MAX_MEMBER_BYTES:
        raise ArchiveError("duplicate or oversized archive member")
    members[name] = raw
    if len(members) > MAX_MEMBERS or sum(map(len, members.values())) > MAX_TOTAL_BYTES:
        raise ArchiveError("total archive member budget exceeded")


def verify_contract(members, inventory, *, archive_identity=None):
    """Independently validate the entire closed transport without trusting receipts."""
    if type(members) is not dict or type(inventory) is not dict:
        raise ArchiveError("invalid archive structure")
    if (set(inventory) != {"schema", "request_identity", "verification_request_identity",
                           "summary_identity", "run_profile", "source",
                           "archival_tools", "members", "evidence_class"}
            or inventory["schema"] != SCHEMA):
        raise ArchiveError("invalid archive inventory schema")
    request_id = require_id(inventory["request_identity"])
    verify_id = require_id(inventory["verification_request_identity"])
    summary_id = require_id(inventory["summary_identity"])
    if inventory["run_profile"] not in ("smoke", "research"):
        raise ArchiveError("unknown archive experiment profile")
    if inventory["evidence_class"] != ("training-only-exploratory" if inventory["run_profile"] == "research"
                                      else "integration-smoke-only"):
        raise ArchiveError("archive evidence class/profile mismatch")
    expected_members = inventory["members"]
    if type(expected_members) is not dict or set(expected_members) != set(members):
        raise ArchiveError("archive inventory is not complete")
    for name, value in expected_members.items():
        safe_name(name)
        if (type(value) is not dict or set(value) != {"identity", "bytes"}
                or value["identity"] != identity(members[name])
                or type(value["bytes"]) is not int
                or value["bytes"] != len(members[name])):
            raise ArchiveError("archive member size/identity mismatch: " + name)

    request = parse(members["external/request.json"])
    verification = parse(members["external/verification-request.json"])
    summary = parse(members["run/summary.json"])
    audit = parse(members["audit/replay.json"])
    if (identity(members["external/request.json"]) != request_id
            or identity(members["external/verification-request.json"]) != verify_id
            or identity(members["run/summary.json"]) != summary_id
            or members["external/request.json"] != members["run/request.json"]):
        raise ArchiveError("original request, verification or summary commitments differ")
    if (request.get("schema") != "trinite.procedural-qec-request.v3"
            or request.get("run_profile") != inventory["run_profile"]
            or request.get("protocol_identity") != summary.get("protocol_identity")
            or summary.get("schema") != "trinite.procedural-qec-summary.v3"
            or summary.get("request_identity") != request_id
            or summary.get("verification_request_identity") != verify_id
            or summary.get("run_profile") != inventory["run_profile"]
            or summary.get("outcome") != "completed"
            or summary.get("selected_arm") is not None
            or summary.get("held_out_scored") is not False
            or summary.get("learning_adequacy") is not False
            or summary.get("phase5_ready") is not False
            or audit.get("outcome") != "completed"
            or audit.get("replayed_cells") != 4
            or audit.get("summary_identity") != summary_id
            or audit.get("verification_request_identity") != verify_id
            or audit.get("learning_adequacy") is not False):
        raise ArchiveError("archived result does not establish complete training-only replay")
    if type(summary.get("cells")) is not list or len(summary["cells"]) != 4:
        raise ArchiveError("archive needs all four declared experimental cells")
    required_checks = {"closed_custody", "request_and_source_binding",
                       "safe_tensor_structure", "scheduled_visits_and_targets",
                       "final_predictions_and_loss", "complete_optimizer_replay"}
    records = {}
    for cell in summary["cells"]:
        if (type(cell) is not dict or cell.get("outcome") != "completed"
                or cell.get("seed") not in (0, 1) or type(cell.get("seed")) is not int
                or cell.get("arm") not in ("final-only", "procedural")
                or type(cell.get("checks")) is not dict
                or set(cell["checks"]) != required_checks
                or not all(cell["checks"][k] is True for k in required_checks)):
            raise ArchiveError("cell is missing required native replay checks")
        name = "procedural-qec-" + str(cell["seed"]) + "-" + cell["arm"]
        if name in records:
            raise ArchiveError("duplicate experimental cell")
        if (identity(members.get("run/" + name + "/report.json", b"")) != cell.get("report_identity")
                or parse(members["run/" + name + "/report.json"]) != cell.get("report")):
            raise ArchiveError("cell summary does not match retained source-bound report")
        report = cell["report"]
        if (report.get("held_out_scored") is not False or report.get("learning_adequacy") is not False
                or report.get("phase5_ready") is not False):
            raise ArchiveError("archived report incorrectly claims held-out/Phase5 evidence")
        records[name] = cell
    if set(records) != {f"procedural-qec-{seed}-{arm}" for seed in (0, 1)
                       for arm in ("final-only", "procedural")}:
        raise ArchiveError("missing required experiment arm/seed")
    if (type(verification) is not dict
            or set(verification) != {"schema", "request_identity", "root_artifacts", "cells"}
            or verification["schema"] != "trinite.procedural-qec-verification-request.v3"
            or verification["request_identity"] != request_id
            or type(verification["cells"]) is not dict or set(verification["cells"]) != set(records)
            or type(verification["root_artifacts"]) is not dict
            or set(verification["root_artifacts"]) != ROOT_COMMITTED
            or any(type(v) is not dict or set(v) != CELL_COMMITTED
                   for v in verification["cells"].values())):
        raise ArchiveError("closed external verification request differs")
    committed = {"run/" + name: digest for name, digest in verification["root_artifacts"].items()}
    for cell, items in verification["cells"].items():
        if type(items) is not dict:
            raise ArchiveError("malformed external cell commitments")
        for name, digest in items.items():
            committed["run/" + cell + "/" + name] = digest
    for name, digest in committed.items():
        require_id(digest)
        if name not in members or identity(members[name]) != digest:
            raise ArchiveError("archived committed run bytes differ: " + name)
    source = request.get("source")
    if type(source) is not dict or not source or source != inventory["source"]:
        raise ArchiveError("producer source-closure inventory mismatch")
    if set(name for name in members if name.startswith("source/")) != {
            "source/" + safe_name(p) for p in source}:
        raise ArchiveError("missing or extra producer source files")
    for p, digest in source.items():
        require_id(digest)
        if identity(members["source/" + p]) != digest:
            raise ArchiveError("producer source bytes differ from frozen experiment request")
    archival_tools = inventory["archival_tools"]
    if (type(archival_tools) is not dict or set(archival_tools) != set(TOOLS)
            or set(name for name in members if name.startswith("archival-tool/"))
            != {"archival-tool/" + p for p in TOOLS}):
        raise ArchiveError("archival tool source closure mismatch")
    for p, digest in archival_tools.items():
        require_id(digest)
        if identity(members["archival-tool/" + p]) != digest:
            raise ArchiveError("archival tool identity differs")

    return {"schema": "trinite.procedural-qec-archive-verification.v3",
            "outcome": "passed", "profile": inventory["run_profile"],
            "request_identity": request_id, "verification_request_identity": verify_id,
            "summary_identity": summary_id, "member_count": len(members),
            "learning_adequacy": False, "phase5_ready": False}


def build_archive(path, run, request, request_identity, verification, verification_identity):
    """Replay the published producer first; then build a lossless external ZIP."""
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from trinite.procedural_qec_v3 import experiment

    request_identity, verification_identity = require_id(request_identity), require_id(verification_identity)
    path = safe_path(path).absolute()
    run = safe_path(run).resolve()
    request, verification = safe_path(request), safe_path(verification)
    if (path == run or run in path.parents or not path.parent.is_dir()
            or path.exists() or request.resolve() == verification.resolve()
            or any(run == p.resolve() or run in p.resolve().parents for p in (request, verification))):
        raise ArchiveError("archive/external commitments need distinct unused paths outside the run")
    req_raw, verify_raw = read(request, 128 * 1024), read(verification, 128 * 1024)
    if identity(req_raw) != request_identity or identity(verify_raw) != verification_identity:
        raise ArchiveError("independently supplied external request identities differ")
    # This is a FULL numerical check: original native initialization, every
    # AdamW update and expected prediction are replayed, not just final inference.
    checked = experiment.verify_only(run, request, request_identity,
                                     verification, verification_identity)
    if checked["outcome"] != "completed" or checked["replayed_cells"] != 4:
        raise ArchiveError("full four-cell native verification is required")
    if checked["summary_identity"] != identity(read(run / "summary.json", 16 * 1024 * 1024)):
        raise ArchiveError("fresh replay summary differs from original run")
    request_record = parse(req_raw)
    members = {}
    run_paths = []
    for item in sorted(run.rglob("*")):
        safe_path(item)
        if item.is_file():
            name = "run/" + item.relative_to(run).as_posix()
            add(members, name, read(item))
            run_paths.append((item, name))
    add(members, "external/request.json", req_raw)
    add(members, "external/verification-request.json", verify_raw)
    add(members, "audit/replay.json", canonical(checked))
    source = request_record["source"]
    if type(source) is not dict or not source:
        raise ArchiveError("missing original request producer closure")
    for name, digest in sorted(source.items()):
        safe_name(name)
        member = read(ROOT / name, 2 * 1024 * 1024)
        if identity(member) != require_id(digest):
            raise ArchiveError("current producer differs from the frozen original: " + name)
        add(members, "source/" + name, member)
    archive_tool = {}
    for name in TOOLS:
        value = read(ROOT / name, 2 * 1024 * 1024)
        archive_tool[name] = identity(value)
        add(members, "archival-tool/" + name, value)

    summary_raw = members["run/summary.json"]
    inventory = {"schema": SCHEMA, "request_identity": request_identity,
        "verification_request_identity": verification_identity,
        "summary_identity": identity(summary_raw), "run_profile": request_record["run_profile"],
        "evidence_class": ("training-only-exploratory" if request_record["run_profile"] == "research"
                           else "integration-smoke-only"),
        "source": source, "archival_tools": archive_tool,
        "members": {name: {"identity": identity(value), "bytes": len(value)}
                    for name, value in sorted(members.items())}}
    audit = verify_contract(members, inventory)
    for original, name in run_paths:
        if read(original) != members[name]:
            raise ArchiveError("run files changed after native replay: " + name)
    inventory_raw = canonical(inventory)
    add(members, "inventory.json", inventory_raw)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, value in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, value, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    archive_bytes = read(path, MAX_ARCHIVE_BYTES)
    receipt = verify_archive(path, identity(archive_bytes), identity(inventory_raw))
    return {**receipt, "evidence_class": inventory["evidence_class"],
            "training_gate_pass": parse(summary_raw)["training_gate_pass"],
            "archive_bytes": len(archive_bytes), "original_native_replay": audit["outcome"]}


def verify_archive(path, archive_identity, inventory_identity):
    """Read-only independent transport validation, no Torch or model imports."""
    require_id(archive_identity)
    require_id(inventory_identity)
    raw = read(path, MAX_ARCHIVE_BYTES)
    if identity(raw) != archive_identity:
        raise ArchiveError("archive differs from independent SHA-256 commitment")
    contents = {}
    total = 0
    try:
        with zipfile.ZipFile(BytesIO(raw)) as z:
            infos = z.infolist()
            if not infos or len(infos) > MAX_MEMBERS + 1:
                raise ArchiveError("archive inventory size exceeded")
            for info in infos:
                safe_name(info.filename)
                if (info.filename in contents or info.is_dir()
                        or info.file_size > MAX_MEMBER_BYTES
                        or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        or (info.create_system == 3
                            and stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG))):
                    raise ArchiveError("unsafe, duplicate, or oversized ZIP member")
                total += info.file_size
                if total > MAX_TOTAL_BYTES:
                    raise ArchiveError("ZIP uncompressed size bound exceeded")
                value = z.read(info)
                if len(value) != info.file_size:
                    raise ArchiveError("truncated ZIP member")
                contents[info.filename] = value
    except (zipfile.BadZipFile, RuntimeError, EOFError, OSError) as exc:
        raise ArchiveError("corrupt ZIP transport") from exc
    if "inventory.json" not in contents:
        raise ArchiveError("archive has no manifest inventory")
    raw_inventory = contents.pop("inventory.json")
    if identity(raw_inventory) != inventory_identity:
        raise ArchiveError("inventory differs from independently held commitment")
    result = verify_contract(contents, parse(raw_inventory))
    return {**result, "archive_identity": archive_identity,
            "inventory_identity": inventory_identity}
