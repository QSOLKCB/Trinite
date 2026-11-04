# SPDX-License-Identifier: MPL-2.0
"""TT-003-DIAG-001: immutable inputs, raw output taxonomy and evidence custody.

No model import or neural execution in this module. No changing source receipts,
historical TT-003 packets, rights, or trained checkpoint files.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import tempfile

from trinite.contracts import (
    ContractError, identity, json_bytes, parse_json, require_identity,
)
from teacher_lab_v1 import core as lab
from teacher_training_v1 import spec as teacher_spec
from teacher_dataset_v1 import core as dataset_spec

PLAN_PATH = "configs/teacher/tt003-diag-001.json"
SCHEMA = "trinite.teacher-frozen-diagnostic-request.v1"
REPORT_SCHEMA = "trinite.teacher-frozen-diagnostic-report.v1"
MANIFEST_SCHEMA = "trinite.teacher-frozen-diagnostic-manifest.v1"
MAX_BYTES = 4 * 1024 * 1024
MAX_TENSOR_BYTES = 32 * 1024 * 1024
ROW_COUNT = 972
PIN = {
    "dataset_manifest_identity": "sha256:b90287c3549e2a877b4e23325b51f234f3e54eab3a48c0227f34c5f18d065687",
    "teacher_request_identity": "sha256:72e9342f769c957c0d8c690d528bebb7fa3500cb96c7168e8d20aa32123fcd44",
    "teacher_run_manifest_identity": "sha256:a74aec8d610a678e2e2fee0c32ece11dd1e1fcc8fa53d8e8da6db40d3ef73004",
    "teacher_tensor_identity": "sha256:ec0e42658818fb9fb1712f11117eebc3c5adf101aa668e335ab788b5e2f6a551",
    "teacher_metadata_identity": "sha256:04f81e0855949b349e6c50ce0aff5b386c0a95a02fcbf75e7008a96111605948",
    "teacher_model_identity": "sha256:54282d9dfbf9b2e88469e11e84048013dd3d031844b3cf6513009888c7ec3628",
    "initial_model_identity": "sha256:62d2f48f4d3f3d6398ea32a7c8eb0d207f66a97a4ebf3c26ddff2ca196ebc9fa",
}
REQUEST_FILES = ("request.json",)
RESULT_FILES = ("request.json", "predictions.jsonl", "summary.json", "manifest.json")
CLOSURE = (
    "scripts/run_teacher_diag.py",
    "scripts/teacher_diag_v1/__init__.py",
    "scripts/teacher_diag_v1/core.py",
    "scripts/teacher_diag_v1/native.py",
    "configs/teacher/tt003-diag-001.json",
    "docs/TEACHER-LAB-DIAG-001.md", "LICENSE",
)


def plan() -> dict:
    raw = lab.source_bytes(PLAN_PATH)
    data = parse_json(raw, canonical=True)
    # This diagnostic version is pinned to exactly one historical run.
    if (type(data) is not dict or set(data) != {
            "schema", "source_commit", "dataset_manifest_identity",
            "teacher_request_identity", "teacher_run_manifest_identity",
            "teacher_tensor_identity", "teacher_metadata_identity",
            "teacher_model_identity", "initial_model_identity", "scope",
            "cap_generated_tokens", "max_execution_seconds",
            "training_authorized", "optimizer_updates", "heldout_scored"}):
        raise ContractError("diagnostic plan schema mismatch")
    if (data["schema"] != "trinite.teacher-frozen-diagnostic-plan.v1"
            or data["source_commit"] != "6f2ec05b407a3bafe7a5ed947fd9caadc4be65f7"
            or data["scope"] != "all-972-admitted-train-rows-only"
            or type(data["cap_generated_tokens"]) is not int
            or data["cap_generated_tokens"] != 5
            or type(data["max_execution_seconds"]) is not int
            or data["max_execution_seconds"] != 1800
            or data["training_authorized"] is not False
            or type(data["optimizer_updates"]) is not int
            or data["optimizer_updates"] != 0
            or data["heldout_scored"] is not False):
        raise ContractError("diagnostic version, scope or authority changed")
    for k, expected in PIN.items():
        require_identity(data[k])
        if data[k] != expected:
            raise ContractError("historical TT-003-EVIDENCE-001 anchor changed")
    return data


def read_member(directory: Path, name: str, cap: int = MAX_BYTES) -> bytes:
    if name not in {"request.json", "manifest.json", "metadata.json",
                    "tensors.safetensors", "report.json", "weights-rights.json",
                    "approval.json", "train.jsonl", "predictions.jsonl",
                    "summary.json"}:
        raise ContractError("unsupported diagnostic artifact")
    path = Path(directory)
    if path.is_symlink() or not path.is_dir():
        raise ContractError("unsafe artifact parent")
    f = path / name
    info = f.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > cap:
        raise ContractError("unsafe or oversized diagnostic artifact")
    with f.open("rb") as stream:
        raw = stream.read(cap + 1)
    if len(raw) > cap:
        raise ContractError("diagnostic artifact exceeded bound")
    return raw


def check_members(folder: Path, names: tuple[str, ...]) -> None:
    accepted, found = set(names), set()
    with os.scandir(folder) as entries:
        for item in entries:
            if item.name not in accepted or item.name in found:
                raise ContractError("extra or repeated artifact member")
            found.add(item.name)
    if found != accepted:
        raise ContractError("incomplete artifact inventory")


def historical_inputs(dataset: Path, run: Path) -> tuple[dict, dict, dict, dict]:
    """Verify source and exact original TT-003 artifacts without loading Torch."""
    p = plan()
    dataset_spec.verify(dataset, p["dataset_manifest_identity"])
    request_bytes = read_member(run, "request.json")
    manifest_bytes = read_member(run, "manifest.json")
    tensors_bytes = read_member(run, "tensors.safetensors", MAX_TENSOR_BYTES)
    meta_bytes = read_member(run, "metadata.json")
    if (identity(request_bytes) != p["teacher_request_identity"]
            or identity(manifest_bytes) != p["teacher_run_manifest_identity"]
            or identity(tensors_bytes) != p["teacher_tensor_identity"]
            or identity(meta_bytes) != p["teacher_metadata_identity"]):
        raise ContractError("historical checkpoint/request identity mismatch")
    request = parse_json(request_bytes, canonical=True)
    if request != teacher_spec.request_for(dataset, p["dataset_manifest_identity"]):
        raise ContractError("original teacher source/input closure mismatch")
    manifest = parse_json(manifest_bytes, canonical=True)
    if (manifest.get("schema") != "trinite.teacher-run-manifest.v1"
            or manifest.get("status") != "COMPLETED_TRAIN_ONLY"
            or manifest.get("request_identity") != p["teacher_request_identity"]
            or manifest.get("heldout_scored") is not False
            or manifest.get("optimizer_updates") != 32):
        raise ContractError("original teacher manifest is not completed train-only")
    check_members(run, ("approval.json", "request.json",
                        "tensors.safetensors", "metadata.json",
                        "weights-rights.json", "report.json",
                        "manifest.json"))
    actual = []
    for name in ("approval.json", "request.json", "tensors.safetensors",
                 "metadata.json", "weights-rights.json", "report.json"):
        raw = read_member(run, name, MAX_TENSOR_BYTES if name == "tensors.safetensors" else MAX_BYTES)
        actual.append({"name": name, "bytes": len(raw), "identity": identity(raw)})
    if actual != manifest.get("members"):
        raise ContractError("original run manifest member hashes disagree")
    report = parse_json(read_member(run, "report.json"), canonical=True)
    metadata = parse_json(meta_bytes, canonical=True)
    if (report.get("status") != "COMPLETED_TRAIN_ONLY"
            or report.get("optimizer_updates") != 32
            or report.get("heldout_scored") is not False
            or report.get("model_identity") != p["teacher_model_identity"]
            or metadata.get("final_model_identity") != p["teacher_model_identity"]
            or metadata.get("initialization_identity") != p["initial_model_identity"]
            or metadata.get("step") != 32
            or metadata.get("target_tokens") != 430):
        raise ContractError("historical teacher report/checkpoint scope mismatch")
    return p, request, metadata, report


def scheduled_rows(metadata: dict, request: dict, dataset: Path) -> tuple[list[dict], set[str]]:
    """Full TRAIN inventory and actual 128 source-bound scheduled exposures."""
    raw = read_member(dataset, "train.jsonl")
    if identity(raw) != request["dataset_train_identity"]:
        raise ContractError("TRAIN bytes changed")
    rows = [parse_json(line + b"\n", canonical=True) for line in raw.splitlines()]
    if len(rows) != ROW_COUNT:
        raise ContractError("unexpected TRAIN row count")
    lookup = {}
    for row in rows:
        if (set(row) != {"schema", "id", "prompt", "observation", "split",
                         "family", "answer", "supervised_tokens",
                         "complete_context_tokens"}
                or row["split"] != "train" or row["family"] not in ("S1", "S2", "W")
                or row["id"] in lookup):
            raise ContractError("invalid or duplicate TRAIN row")
        lookup[row["id"]] = row
    history = metadata.get("history")
    if type(history) is not list or len(history) != 32:
        raise ContractError("missing original update history")
    visits = []
    for idx, step in enumerate(history, 1):
        if (type(step) is not dict or type(step.get("step")) is not int
                or step["step"] != idx or type(step.get("examples")) is not list
                or len(step["examples"]) != 4):
            raise ContractError("invalid original training schedule")
        visits.extend(step["examples"])
    expected = sorted(rows, key=lambda r: identity(json_bytes(
        {"seed": request["plan"]["seed"], "item": r["id"]})))
    scheduled = [r["id"] for r in expected[:128]]
    if (visits != scheduled
            or identity(json_bytes(visits)) != request["train_schedule_identity"]
            or set(visits) - set(lookup)
            or len(set(visits)) != 128):
        raise ContractError("original visited TRAIN rows mismatch")
    return rows, set(visits)


def build_request(dataset: Path, run: Path) -> dict:
    p, teacher, metadata, _ = historical_inputs(dataset, run)
    rows, visited = scheduled_rows(metadata, teacher, dataset)
    paths = sorted(set(CLOSURE) | {source["path"] for source in teacher["source_closure"]})
    closure = [{"path": path, "identity": identity(lab.source_bytes(path))}
               for path in paths]
    return {
        "schema": SCHEMA,
        "status": "FROZEN_READ_ONLY_DIAGNOSTIC",
        "plan": p,
        "teacher_source_closure": teacher["source_closure"],
        "diagnostic_source_closure": closure,
        "teacher_request_identity": p["teacher_request_identity"],
        "teacher_run_manifest_identity": p["teacher_run_manifest_identity"],
        "dataset_manifest_identity": p["dataset_manifest_identity"],
        "train_rows": len(rows), "visited_rows": len(visited),
        "unvisited_rows": len(rows) - len(visited),
        "candidate_evaluations": 2 * len(rows),
        "cap_generated_tokens": 5,
        "training_authorized": False, "optimizer_updates": 0,
        "heldout_scored": False, "checkpoint_modified": False,
    }


def freeze(destination: Path, dataset: Path, run: Path) -> str:
    req = build_request(dataset, run)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ContractError("fresh diagnostic request directory required")
    stage = Path(tempfile.mkdtemp(prefix=".tt003diag-request-", dir=destination.parent))
    try:
        with (stage / "request.json").open("xb") as f:
            f.write(json_bytes(req)); f.flush(); os.fsync(f.fileno())
        if destination.exists() or destination.is_symlink():
            raise ContractError("request destination was replaced")
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return identity(json_bytes(req))


def verify_request(request_dir: Path, external_anchor: str, dataset: Path, run: Path) -> dict:
    require_identity(external_anchor)
    check_members(request_dir, REQUEST_FILES)
    raw = read_member(request_dir, "request.json")
    if identity(raw) != external_anchor:
        raise ContractError("independent diagnostic request anchor differs")
    given = parse_json(raw, canonical=True)
    if given != build_request(dataset, run):
        raise ContractError("diagnostic request/source/checkpoint changed")
    return given


def classify(generated: list[int], expected: str, task: str) -> dict:
    """Strict unconstrained greedy prediction taxonomy; no answer repair."""
    if (type(generated) is not list or len(generated) > 5
            or any(type(v) is not int or v < 0 or v > 258 for v in generated)
            or type(expected) is not str or task not in ("check", "apply", "inverse")):
        raise ContractError("invalid diagnostic prediction")
    expected_len = 1 if task == "check" else 4
    if len(expected) != expected_len or any(
            c not in ("01" if task == "check" else "012") for c in expected):
        raise ContractError("invalid expected training label")
    eos = generated[-1:] == [257]
    body = generated[:-1] if eos else generated
    special = any(token >= 256 for token in body)
    ascii_only = all(0 <= token < 128 for token in body)
    decoded = bytes(body).decode("ascii") if ascii_only else None
    permitted = "01" if task == "check" else "012"
    valid = (eos and not special and decoded is not None
             and len(decoded) == expected_len
             and all(char in permitted for char in decoded))
    exact = generated == [*expected.encode("ascii"), 257]
    if exact:
        why = "exact"
    elif special or decoded is None or any(
            char not in permitted for char in (decoded or "")):
        why = "invalid_symbol"
    elif not eos:
        why = "missing_eos"
    elif len(body) < expected_len:
        why = "premature_eos"
    elif len(body) > expected_len:
        why = "excess_length"
    else:
        why = "wrong_answer"
    return {"tokens": generated, "decoded": decoded, "eos": eos,
            "answer_length": len(body), "expected_length": expected_len,
            "format_valid": valid, "exact": exact, "classification": why}


def analysis_summary(results: list[dict]) -> dict:
    """Pure recomputable metrics; count outputs, never assert learning."""
    if type(results) is not list or len(results) != ROW_COUNT:
        raise ContractError("full TRAIN diagnostic requires exactly 972 rows")
    identifiers = set()
    groups: dict[str, list[dict]] = {"all": []}
    improvements = {"improved": 0, "regressed": 0, "unchanged_exact": 0,
                    "both_incorrect": 0, "raw_changed": 0}
    for row in results:
        if type(row) is not dict or set(row) != {
                "id", "family", "task", "expected", "visited", "initial", "final"}:
            raise ContractError("diagnostic row schema mismatch")
        if row["id"] in identifiers:
            raise ContractError("duplicate scored TRAIN identity")
        identifiers.add(row["id"])
        for stage in ("initial", "final"):
            expected_score = classify(row[stage]["tokens"], row["expected"], row["task"])
            if row[stage] != expected_score:
                raise ContractError("diagnostic classification mismatch")
        a, b = row["initial"], row["final"]
        if not a["exact"] and b["exact"]:
            improvements["improved"] += 1
        elif a["exact"] and not b["exact"]:
            improvements["regressed"] += 1
        elif a["exact"] and b["exact"]:
            improvements["unchanged_exact"] += 1
        else:
            improvements["both_incorrect"] += 1
        if a["tokens"] != b["tokens"]:
            improvements["raw_changed"] += 1
        label = "visited" if row["visited"] is True else "unvisited"
        kind = "scalar_check" if row["task"] == "check" else "vector_" + row["task"]
        for key in ("all", label, "kind:" + kind,
                    label + "/" + kind,
                    "circuit:" + row["family"],
                    "label:" + row["expected"] if kind == "scalar_check" else "vector_length:4"):
            groups.setdefault(key, []).append(row)
    summary = {"schema": REPORT_SCHEMA, "train_rows": len(results),
               "visited_rows": len(groups.get("visited", [])),
               "unvisited_rows": len(groups.get("unvisited", [])),
               "pairwise_changes": improvements,
               "groups": {}}
    for name in sorted(groups):
        members = groups[name]
        stages = {}
        for stage in ("initial", "final"):
            counts = {s: 0 for s in ("exact", "wrong_answer", "missing_eos",
                                     "premature_eos", "excess_length", "invalid_symbol")}
            hist: dict[str, int] = {}
            for r in members:
                scored = r[stage]
                counts[scored["classification"]] += 1
                sequence = ",".join(map(str, scored["tokens"]))
                hist[sequence] = hist.get(sequence, 0) + 1
            stages[stage] = {
                "exact_count": counts["exact"],
                "format_valid_count": sum(r[stage]["format_valid"] for r in members),
                "eos_count": sum(r[stage]["eos"] for r in members),
                "classification_counts": counts,
                "distinct_raw_outputs": len(hist),
                "top_raw_outputs": [
                    {"tokens_csv": key, "count": count}
                    for key, count in sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
                ],
            }
        summary["groups"][name] = {"count": len(members), **stages}
    if summary["visited_rows"] + summary["unvisited_rows"] != ROW_COUNT:
        raise ContractError("visited/unvisited diagnostic partition incomplete")
    return summary
