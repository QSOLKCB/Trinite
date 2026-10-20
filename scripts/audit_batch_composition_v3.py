"""Bounded read-only audit v3 of retained batch-v2 artifacts; no model execution.

Supersedes the historical, byte-preserved fixture auditor. Validation remains
active under Python optimization. This is an offline audit, not inference replay.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

MAX_FILE = 64 * 1024 * 1024
MAX_METADATA = 1024 * 1024
MAX_FILES = 64
MAX_TOTAL = 512 * 1024 * 1024


class AuditError(ValueError):
    """The input cannot support a successful offline audit receipt."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def read(path, limit=MAX_FILE):
    path = Path(path)
    require(not path.is_symlink(), "input symlink is not permitted: " + str(path))
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode), "regular input file required")
        require(info.st_size <= limit, "input exceeds byte bound")
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "input exceeds byte bound")
    return raw


def inventory(root):
    result, total = {}, 0
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "artifact symlink is not permitted")
        if path.is_dir():
            continue
        require(len(result) < MAX_FILES, "artifact file count exceeds bound")
        require(path.is_file(), "artifact entries must be regular files/directories")
        total += path.stat().st_size
        require(total <= MAX_TOTAL, "artifact total bytes exceed bound")
        result[path.relative_to(root).as_posix()] = digest(read(path))
    require(bool(result), "empty artifact directory")
    return result


def output_destination(root, output):
    """Reject lexical and resolved aliases into the tree before any audit/write."""
    root = Path(root).resolve(strict=True)
    require(root.is_dir(), "artifact directory required")
    output = Path(output)
    require(not output.is_symlink(), "output symlink is not permitted")
    resolved = output.resolve()
    require(not resolved.is_relative_to(root), "output must be outside artifact directory")
    require(not resolved.exists(), "output already exists")
    require(resolved.parent.is_dir(), "output parent directory must exist")
    return resolved


def validate_status(status, conforms):
    require(type(status) is dict and set(status) == {
        "schema", "outcome", "exit_code", "wall_seconds_hex"}, "invalid status fields")
    require(status["schema"] == "trinite.batch-composition-status.v2", "invalid status schema")
    require(type(conforms) is bool, "report conforms must be boolean")
    code, outcome = status["exit_code"], status["outcome"]
    require(type(code) is int and type(outcome) is str, "invalid status types")
    pairings = {"completed": 0, "completed_nonconformant": 2, "timeout": 124,
                "launch_failed": 1, "controller_failed": 1, "interrupted": 130}
    require(outcome in pairings or outcome == "failed", "unknown status outcome")
    require(code == pairings[outcome] if outcome in pairings else code not in (0, 2),
            "inconsistent status outcome/exit-code pairing")
    wall = status["wall_seconds_hex"]
    require(type(wall) is str, "invalid status wall time")
    try:
        seconds = float.fromhex(wall)
    except (ValueError, OverflowError) as error:
        raise AuditError("invalid status wall time") from error
    require(math.isfinite(seconds) and seconds >= 0 and seconds.hex() == wall,
            "invalid status wall time")
    completed = outcome in ("completed", "completed_nonconformant")
    if completed:
        require(code == (0 if conforms else 2), "completed status contradicts report")
    return completed


def validate_replay_receipt(receipt, invocation, request_id, report_id, conforms):
    require(type(receipt) is dict and set(receipt) == {
        "schema", "request_identity", "expected_identity", "actual_identity",
        "exact_match", "conforms", "scope"}, "invalid replay receipt fields")
    require(receipt["schema"] == "trinite.batch-composition-replay-receipt.v2",
            "invalid replay receipt schema")
    require(invocation["expected_identity"] == report_id, "invocation expected identity mismatch")
    require(receipt["request_identity"] == request_id, "replay request identity mismatch")
    require(receipt["expected_identity"] == report_id == receipt["actual_identity"],
            "replay report identity mismatch")
    require(receipt["exact_match"] is True, "replay must match exactly")
    require(type(conforms) is bool and receipt["conforms"] is conforms, "replay conforms mismatch")
    require(receipt["scope"] == "same-environment complete inference replay, not optimizer replay",
            "invalid replay scope")


def audit(artifact_dir, source_root, approved_review, output):
    root = Path(artifact_dir).resolve(strict=True)
    destination = output_destination(root, output)
    auditor_raw = read(Path(__file__))
    source_root = Path(source_root).resolve(strict=True)
    # Do not write import caches into either an input tree or its source closure.
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(source_root / "src"), str(source_root / "scripts")]
    from trinite.contracts import json_bytes, parse_json
    from batch_composition_v1.validation import validate_report
    from batch_composition_v1.spec import LAYOUTS

    approved_raw = read(approved_review, MAX_METADATA)
    approved = parse_json(approved_raw, canonical=True)
    require(type(approved.get("source")) is dict and bool(approved["source"]),
            "approved source map required")
    for name, expected in approved["source"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "unsafe source path")
        require(digest(read(source_root / path)) == expected, "source identity mismatch: " + name)
    before = inventory(root)
    candidates = [root / name for name in before if Path(name).name == "batch-composition-request.json"]
    require(len(candidates) == 1, "exactly one request required")
    request_path = candidates[0]
    report_path = request_path.with_name("batch-composition-report.json")
    request_raw, report_raw = read(request_path, MAX_METADATA), read(report_path)
    request = parse_json(request_raw, canonical=True)
    report = parse_json(report_raw, canonical=True)
    require(request["schema"] == "trinite.batch-composition-request.v2", "invalid request schema")
    require(report["schema"] == "trinite.batch-composition-report.v2", "invalid report schema")
    require(type(report["conforms"]) is bool, "report conforms must be boolean")
    require(request["source"] == approved["source"], "request source mismatch")
    require(digest(json_bytes(request["protocol"])) == approved["protocol_identity"], "protocol mismatch")
    report_id, request_id = digest(report_raw), digest(request_raw)
    attempts = []
    expected_layouts = [
        {"seed": cell["seed"], "arm": cell["arm"], "layout": layout, "result": cell["layouts"][layout]}
        for cell in report["cells"] for layout in LAYOUTS]
    for name in sorted(before):
        if Path(name).name != "invocation.json":
            continue
        invocation_path = root / name
        invocation = parse_json(read(invocation_path, MAX_METADATA), canonical=True)
        require(type(invocation) is dict and set(invocation) == {
            "schema", "action", "request_identity", "expected_identity",
            "deadline_seconds", "final_name"}, "invalid invocation fields")
        require(invocation["schema"] == "trinite.batch-composition-invocation.v2",
                "invalid invocation schema")
        require(invocation["action"] in ("run", "verify"), "invalid invocation action")
        require(invocation["request_identity"] == request_id, "invocation request mismatch")
        require(type(invocation["deadline_seconds"]) is int and invocation["deadline_seconds"] == 300,
                "invocation deadline mismatch")
        require(invocation["final_name"] == report_path.name, "invocation final name mismatch")
        expected_id = report_id if invocation["action"] == "verify" else None
        require(invocation["expected_identity"] == expected_id, "invocation expected identity mismatch")
        attempt = invocation_path.parent
        status = parse_json(read(attempt / "status.json", MAX_METADATA), canonical=True)
        completed = validate_status(status, report["conforms"])
        item = {"directory": attempt.relative_to(root).as_posix(),
                "action": invocation["action"], "outcome": status["outcome"], "exit_code": status["exit_code"]}
        if completed:
            # Receipt guards precede expensive recorded-vector validation.
            if invocation["action"] == "verify":
                receipt = parse_json(read(attempt / "replay-receipt.json", MAX_METADATA), canonical=True)
                validate_replay_receipt(receipt, invocation, request_id, report_id, report["conforms"])
            computed = read(attempt / "computed-report.json")
            require(computed == report_raw, "computed report differs")
            for filename, expected in (("completed-cells.jsonl", report["cells"]),
                                       ("completed-layouts.jsonl", expected_layouts)):
                lines = read(attempt / filename).splitlines(keepends=True)
                require(len(lines) == len(expected), "partial stream length mismatch: " + filename)
                for line, value in zip(lines, expected):
                    require(parse_json(line, canonical=True) == value, "partial stream mismatch: " + filename)
            item.update(computed_report_identity=digest(computed), completed_cells=4, completed_layouts=32)
            if invocation["action"] == "verify":
                item["exact_replay"] = True
        attempts.append(item)
    require(sum(a["action"] == "run" and a["outcome"] in ("completed", "completed_nonconformant")
                for a in attempts) == 1, "exactly one completed run required")
    require(sum(a["action"] == "verify" and a.get("exact_replay", False)
                for a in attempts) == 1, "exactly one completed exact replay required")
    checked = validate_report(report, request)
    vectors = []
    for cell in report["cells"]:
        for comparison in cell["comparisons"]:
            vectors.extend(comparison["forced"])
            vectors.extend(step for mode in comparison["independent_generation"].values()
                           for row in mode for step in row["same_prefix_steps"])
    require(inventory(root) == before, "artifact files changed during read-only audit")
    require(read(approved_review, MAX_METADATA) == approved_raw, "approved review changed during audit")
    require(read(Path(__file__)) == auditor_raw, "auditor changed during audit")
    for name, expected in approved["source"].items():
        require(digest(read(source_root / name)) == expected, "source changed during audit: " + name)
    receipt = {
        "schema": "trinite.independent-batch-ci-audit.v3",
        "auditor_identity": digest(auditor_raw),
        "approved_review_identity": digest(approved_raw),
        "request_identity": request_id, "report_identity": report_id, "report_bytes": len(report_raw),
        "protocol_identity": approved["protocol_identity"], "source_files_verified": len(approved["source"]),
        "strict_recorded_report_validation": checked, "environment": request["environment"], "attempts": attempts,
        "same_prefix_vector_comparisons": len(vectors),
        "max_absolute_error_hex": max(float.fromhex(v["max_absolute_hex"]) for v in vectors).hex(),
        "max_normalized_error_hex": max(float.fromhex(v["max_normalized_error_hex"]) for v in vectors).hex(),
        "violating_coordinates": sum(len(v["violating_token_ids"]) for v in vectors),
        "cells_identity": digest(json_bytes(report["cells"])), "all_artifact_file_hashes": before,
        "read_only_inputs_unchanged": True, "local_inference_executed": False,
        "scope": "Offline source/schema/recorded-vector/call/receipt audit of hosted CPU run and exact replay; not a local model replay, optimizer replay, or hardware attestation.",
    }
    raw = json_bytes(receipt)
    # Re-resolve the caller's path immediately before exclusive publication.
    require(output_destination(root, output) == destination, "output destination changed during audit")
    with destination.open("xb") as stream:
        stream.write(raw)
    return {"output": str(destination), "receipt_identity": digest(raw),
            "report_identity": report_id, "conforms": report["conforms"],
            "same_prefix_vector_comparisons": len(vectors)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("artifact-dir", "source-root", "approved-review", "output"):
        parser.add_argument("--" + option, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(audit(args.artifact_dir, args.source_root, args.approved_review, args.output)))
        return 0
    except (AuditError, OSError, ValueError, KeyError, TypeError) as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
