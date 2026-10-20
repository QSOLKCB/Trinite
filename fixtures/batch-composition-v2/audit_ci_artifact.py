"""Read-only independent v2 CI-artifact audit; never executes a model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--approved-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    source_root = args.source_root.resolve()
    sys.path[:0] = [str(source_root / "src"), str(source_root / "scripts")]
    from trinite.contracts import json_bytes, parse_json
    from batch_composition_v1.validation import validate_report
    from batch_composition_v1.spec import LAYOUTS

    approved = parse_json(args.approved_review.read_bytes(), canonical=True)
    for name, expected in approved["source"].items():
        path = source_root / name
        assert path.is_file() and not path.is_symlink(), name
        assert digest(path.read_bytes()) == expected, name
    inventory_before = {p.relative_to(root).as_posix(): digest(p.read_bytes())
                        for p in root.rglob("*") if p.is_file()}
    candidates = list(root.rglob("batch-composition-request.json"))
    assert len(candidates) == 1, candidates
    request_path = candidates[0]
    report_path = request_path.with_name("batch-composition-report.json")
    request_raw, report_raw = request_path.read_bytes(), report_path.read_bytes()
    assert len(request_raw) <= 1024 * 1024
    assert len(report_raw) <= 64 * 1024 * 1024
    request = parse_json(request_raw, canonical=True)
    report = parse_json(report_raw, canonical=True)
    assert request["schema"] == "trinite.batch-composition-request.v2"
    assert report["schema"] == "trinite.batch-composition-report.v2"
    assert request["source"] == approved["source"]
    assert digest(json_bytes(request["protocol"])) == approved["protocol_identity"]
    checked = validate_report(report, request)
    report_id, request_id = digest(report_raw), digest(request_raw)

    attempts = []
    for invocation_path in sorted(root.rglob("invocation.json")):
        invocation = parse_json(invocation_path.read_bytes(), canonical=True)
        if invocation.get("schema") != "trinite.batch-composition-invocation.v2":
            continue
        attempt = invocation_path.parent
        status = parse_json((attempt / "status.json").read_bytes(), canonical=True)
        assert status["schema"] == "trinite.batch-composition-status.v2"
        assert invocation["request_identity"] == request_id
        assert invocation["deadline_seconds"] == 300
        item = {"directory": attempt.relative_to(root).as_posix(),
                "action": invocation["action"], "outcome": status["outcome"],
                "exit_code": status["exit_code"]}
        if status["exit_code"] in (0, 2):
            assert status["exit_code"] == (0 if report["conforms"] else 2)
            computed = (attempt / "computed-report.json").read_bytes()
            assert computed == report_raw
            count = 0
            with (attempt / "completed-cells.jsonl").open("rb") as stream:
                for count, line in enumerate(stream, 1):
                    assert count <= 4
                    assert parse_json(line, canonical=True) == report["cells"][count - 1]
            assert count == 4
            count = 0
            expected_layouts = [
                {"seed": cell["seed"], "arm": cell["arm"], "layout": layout,
                 "result": cell["layouts"][layout]}
                for cell in report["cells"] for layout in LAYOUTS]
            with (attempt / "completed-layouts.jsonl").open("rb") as stream:
                for count, line in enumerate(stream, 1):
                    assert count <= 32
                    assert parse_json(line, canonical=True) == expected_layouts[count - 1]
            assert count == 32
            item["computed_report_identity"] = digest(computed)
            item["completed_cells"] = 4
            item["completed_layouts"] = 32
            if invocation["action"] == "verify":
                receipt = parse_json((attempt / "replay-receipt.json").read_bytes(), canonical=True)
                assert receipt["schema"] == "trinite.batch-composition-replay-receipt.v2"
                assert invocation["expected_identity"] == report_id
                assert receipt["request_identity"] == request_id
                assert receipt["expected_identity"] == report_id == receipt["actual_identity"]
                assert receipt["exact_match"] is True
                assert receipt["conforms"] is report["conforms"]
                item["exact_replay"] = True
        attempts.append(item)
    assert len([a for a in attempts if a["action"] == "run" and a["exit_code"] in (0, 2)]) == 1
    assert len([a for a in attempts if a["action"] == "verify" and a.get("exact_replay")]) == 1
    vectors = []
    for cell in report["cells"]:
        for comparison in cell["comparisons"]:
            vectors.extend(comparison["forced"])
            vectors.extend(step for mode in comparison["independent_generation"].values()
                           for item in mode for step in item["same_prefix_steps"])
    inventory_after = {p.relative_to(root).as_posix(): digest(p.read_bytes())
                       for p in root.rglob("*") if p.is_file()}
    assert inventory_after == inventory_before, "Artifact files changed during read-only audit."
    receipt = {
        "schema": "trinite.independent-batch-ci-audit.v2",
        "request_identity": request_id,
        "report_identity": report_id,
        "report_bytes": len(report_raw),
        "protocol_identity": approved["protocol_identity"],
        "source_files_verified": len(approved["source"]),
        "strict_recorded_report_validation": checked,
        "environment": request["environment"],
        "attempts": attempts,
        "same_prefix_vector_comparisons": len(vectors),
        "max_absolute_error_hex": max(float.fromhex(v["max_absolute_hex"]) for v in vectors).hex(),
        "max_normalized_error_hex": max(float.fromhex(v["max_normalized_error_hex"]) for v in vectors).hex(),
        "violating_coordinates": sum(len(v["violating_token_ids"]) for v in vectors),
        "cells_identity": digest(json_bytes(report["cells"])),
        "all_artifact_file_hashes": inventory_before,
        "read_only_inputs_unchanged": True,
        "local_inference_executed": False,
        "scope": "Independent offline source/schema/recorded-vector/call/receipt audit of hosted CPU run and exact replay; not a local model replay, optimizer replay, or hardware attestation.",
    }
    raw = json_bytes(receipt)
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(args.output), "receipt_identity": digest(raw),
                      "report_identity": report_id, "conforms": report["conforms"],
                      "same_prefix_vector_comparisons": len(vectors)}))


if __name__ == "__main__":
    main()
