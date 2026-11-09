#!/usr/bin/env python3
"""MTC-PRINCIPAL-R2-001: immutable second-pass feedback from verified revisions.

Advisory Qwen feedback is NOT an oracle, training admission, or a weight update.
This is a new collector so prior source-bound run-002 evidence remains verifiable.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
from pathlib import Path
import re

_spec = importlib.util.spec_from_file_location(
    "mtc_principal_round1", Path(__file__).with_name("mtc_principal_001.py")
)
principal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(principal)
mtc = principal.mtc

PLAN_SCHEMA = "trinite.mtc-principal-round2.plan.v1"
MANIFEST_SCHEMA = "trinite.mtc-principal-round2.manifest.v2"
EXPECTED_BASELINE_MATCHES = 3
CONTRACT_PATHS = (
    "AGENTS.md", ".github/AGENTS.md", "scripts/AGENTS.md",
    "tests/AGENTS.md", "docs/AGENTS.md",
    "docs/AGENT-IBAE-GOVERNANCE.md", "docs/DATA.md", "docs/INVARIANTS.md",
    "docs/MTC-LFM3-001.md", "docs/MTC-PRINCIPAL-001.md",
    "docs/MTC-PRINCIPAL-R2-001.md",
    "scripts/mtc_lfm3_001.py", "scripts/mtc_principal_001.py",
    "scripts/mtc_principal_round2_001.py",
    "tests/mtc_principal_round2_001/test_round2.py",
    ".github/workflows/mtc-principal-round2-001-contract.yml",
)


def closure():
    root = Path(__file__).resolve().parents[1]
    return {name: mtc.sha(mtc.source_bytes(root / name)) for name in CONTRACT_PATHS}


def training_tasks():
    return [t for t in mtc.tasks() if t["split"] == "train"]


def reject_nested_output(out, teacher_run, previous_run):
    """Keep new plan/capture directories outside immutable parent archives."""
    destination = Path(out).resolve(strict=False)
    for archive in (teacher_run, previous_run):
        parent = Path(archive).resolve(strict=True)
        if destination == parent or parent in destination.parents:
            raise ValueError("Round 2 output must be outside parent evidence archives")


def record_call(body, expected_model):
    """Record one inherited call, distinguishing malformed envelopes from wrong models."""
    call = principal.record_call(body, expected_model)
    if call["error"] == "model_identity_mismatch":
        parsed = mtc.strict_json(base64.b64decode(
            call["response_base64"], validate=True
        ))
        if not isinstance(parsed, dict):
            call["error"] = "invalid_response_shape"
    return call


def reconstruct(call, body, expected_model):
    if isinstance(call, dict) and call.get("error") == "invalid_response_shape":
        if (set(call) != {"request_sha256", "response_base64", "http_status", "error"}
                or call["request_sha256"] != mtc.sha(mtc.canonical(body))
                or call["http_status"] != 200):
            raise ValueError("invalid malformed-response call record")
        raw = base64.b64decode(call["response_base64"], validate=True)
        if len(raw) > mtc.MAX_HTTP or isinstance(mtc.strict_json(raw), dict):
            raise ValueError("malformed-response failure not supported by bytes")
        return None
    return principal.reconstruct(call, body, expected_model)


def feedback_of(response, model):
    try:
        return principal.feedback_of(response, model)
    except UnicodeError:
        return None


def call_event(index, item_id, stage, call):
    return {"index": index, "item_id": item_id, "stage": stage, "call": call}


def check_installed_model(model):
    if mtc.model_sha(mtc.inventory(), model["name"]) != model["digest"]:
        raise ValueError("installed model digest drift after recorded call")


def prior_evidence(teacher_run, previous_run, manifest_identity):
    """Revalidate *both* earlier archives before any new plan or inference."""
    if type(manifest_identity) is not str or re.fullmatch(
            r"sha256:[0-9a-f]{64}", manifest_identity) is None:
        raise ValueError("invalid externally retained Principal manifest identity")
    principal.verify_command(argparse.Namespace(
        teacher_run=teacher_run, run=previous_run,
        manifest_identity=manifest_identity
    ))
    prev_plan = principal.check_plan(mtc.strict_json(
        mtc.checked_file(Path(previous_run) / "plan.json")
    ))
    records = mtc.read_attempts(Path(previous_run) / "attempts.jsonl")
    expected = training_tasks()
    if len(records) != len(expected):
        raise ValueError("previous Principal inventory incomplete")
    result = []
    for i, task in enumerate(expected):
        row = records[i]
        if (row["item_id"] != task["id"] or row["status"] != "completed"
                or type(row["revised_answer"]) is not str
                or row["teacher_call"] is None):
            raise ValueError("previous revised answer unavailable: " + task["id"])
        teacher_response = mtc.strict_json(base64.b64decode(
            row["teacher_call"]["response_base64"], validate=True
        ))
        msg = teacher_response.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        if (type(content) is not str or not content.strip() or
                len(content.encode("utf-8")) > principal.MAX_TEXT or
                mtc.parse_answer(teacher_response) != row["revised_answer"] or
                row["revised_match"] is not (row["revised_answer"] == mtc.oracle(task))):
            raise ValueError("previous teacher revision invalid: " + task["id"])
        result.append(({
            "item_id": task["id"],
            "previous_row_sha256": mtc.sha(mtc.canonical(row)),
            "baseline_answer": row["revised_answer"],
            "baseline_match": row["revised_match"],
        }, content))
    if sum(record["baseline_match"] for record, _ in result) != EXPECTED_BASELINE_MATCHES:
        raise ValueError("previous Principal baseline is not the required 3/4")
    return prev_plan, result


def plan_value(teacher_run, previous_run, manifest_identity):
    previous, items = prior_evidence(teacher_run, previous_run, manifest_identity)
    installed = mtc.inventory()
    for model in (previous["teacher"], previous["principal"]):
        if mtc.model_sha(installed, model["name"]) != model["digest"]:
            raise ValueError("installed model digest drift")
    return {
        "schema": PLAN_SCHEMA, "source_closure": closure(),
        "prior_principal_manifest_identity": manifest_identity,
        "prior_teacher_manifest_identity": previous["prior_manifest_identity"],
        "teacher": previous["teacher"], "principal": previous["principal"],
        "items": [record for record, _ in items],
        "response_policy": mtc.RESPONSE_POLICY,
        "principal_options": principal.PRINCIPAL_OPTIONS,
        "teacher_options": mtc.SETTINGS,
        "planned_calls_max": 2 * len(training_tasks()),
        "rights_status": "QUARANTINED_NOT_ADMITTED",
        "training_admission": False,
    }


def validate_plan(plan):
    if type(plan) is not dict or set(plan) != {
        "schema", "source_closure", "prior_principal_manifest_identity",
        "prior_teacher_manifest_identity", "teacher", "principal", "items",
        "response_policy", "principal_options", "teacher_options",
        "planned_calls_max", "rights_status", "training_admission"
    }:
        raise ValueError("invalid Round 2 plan fields")
    if (plan["schema"] != PLAN_SCHEMA or plan["source_closure"] != closure()
            or plan["response_policy"] != mtc.RESPONSE_POLICY
            or plan["principal_options"] != principal.PRINCIPAL_OPTIONS
            or plan["teacher_options"] != mtc.SETTINGS
            or type(plan["planned_calls_max"]) is not int
            or plan["planned_calls_max"] != 2 * len(training_tasks())
            or plan["rights_status"] != "QUARANTINED_NOT_ADMITTED"
            or plan["training_admission"] is not False):
        raise ValueError("Round 2 source or frozen policy drift")
    for key in ("prior_principal_manifest_identity", "prior_teacher_manifest_identity"):
        if (type(plan[key]) is not str or
                re.fullmatch(r"sha256:[0-9a-f]{64}", plan[key]) is None):
            raise ValueError("invalid parent manifest identity")
    if (type(plan["principal"]) is not dict or
            type(plan["teacher"]) is not dict or
            set(plan["principal"]) != {"name", "digest"} or
            set(plan["teacher"]) != {"name", "digest"} or
            type(plan["principal"]["name"]) is not str or
            principal.PRINCIPAL_PATTERN.fullmatch(plan["principal"]["name"]) is None or
            type(plan["teacher"]["name"]) is not str or
            mtc.validate_models(plan["teacher"]["name"]) != [plan["teacher"]["name"]]):
        raise ValueError("invalid Principal/teacher identity")
    for model in (plan["principal"], plan["teacher"]):
        if (type(model["digest"]) is not str or
                re.fullmatch(r"sha256:[a-f0-9]{64}", model["digest"]) is None):
            raise ValueError("invalid model digest")
    items = plan["items"]
    if type(items) is not list or len(items) != len(training_tasks()):
        raise ValueError("invalid Round 2 task count")
    for record, task in zip(items, training_tasks()):
        if (type(record) is not dict or set(record) != {
                "item_id", "previous_row_sha256", "baseline_answer", "baseline_match"
            } or record["item_id"] != task["id"] or
                type(record["previous_row_sha256"]) is not str or
                re.fullmatch(r"sha256:[a-f0-9]{64}", record["previous_row_sha256"]) is None or
                type(record["baseline_answer"]) is not str or
                len(record["baseline_answer"]) > 32 or
                type(record["baseline_match"]) is not bool or
                record["baseline_match"] is not (record["baseline_answer"] == mtc.oracle(task))):
            raise ValueError("invalid prior row or oracle binding")
    if sum(record["baseline_match"] for record in items) != EXPECTED_BASELINE_MATCHES:
        raise ValueError("Round 2 plan baseline is not the required 3/4")
    return plan


def check_prior(plan, teacher_run, previous_run):
    prev, row_and_content = prior_evidence(
        teacher_run, previous_run, plan["prior_principal_manifest_identity"]
    )
    if (prev["prior_manifest_identity"] != plan["prior_teacher_manifest_identity"]
            or prev["teacher"] != plan["teacher"]
            or prev["principal"] != plan["principal"]
            or [x for x, _ in row_and_content] != plan["items"]):
        raise ValueError("prior evidence or model lineage drift")
    return [content for _, content in row_and_content]


def verdict_alignment(feedback, matched):
    if feedback is None or feedback["verdict"] == "uncertain":
        return None
    return (feedback["verdict"] == "correct") is matched


def validate_records(plan, originals, records):
    tasks = training_tasks()
    if type(records) is not list or len(records) > len(tasks):
        raise ValueError("excess Round 2 attempts")
    for i, row in enumerate(records):
        task, item = tasks[i], plan["items"][i]
        if type(row) is not dict or set(row) != {
            "index", "item_id", "previous_row_sha256", "baseline_match",
            "principal_call", "principal_feedback", "verdict_matches_oracle",
            "teacher_call", "revised_answer", "revised_match", "status"
        }:
            raise ValueError("invalid Round 2 attempt fields")
        if (type(row["index"]) is not int or row["index"] != i or
                row["item_id"] != task["id"] or
                row["previous_row_sha256"] != item["previous_row_sha256"] or
                row["baseline_match"] is not item["baseline_match"]):
            raise ValueError("Round 2 lineage mismatch")
        review_body = principal.review_payload(plan, task, originals[i])
        if row["principal_call"] is None:
            raise ValueError("missing recorded Principal call")
        reviewed = reconstruct(
            row["principal_call"], review_body, plan["principal"]["name"]
        )
        feedback = feedback_of(reviewed, plan["principal"]["name"]) if (
            reviewed is not None) else None
        if (row["principal_feedback"] != feedback or
                row["verdict_matches_oracle"] is not
                verdict_alignment(feedback, item["baseline_match"])):
            raise ValueError("advisory Principal verdict was not reconstructed")
        if feedback is None:
            if (row["status"] != "principal_failed_or_invalid" or
                    row["teacher_call"] is not None or
                    row["revised_answer"] is not None or
                    row["revised_match"] is not False):
                raise ValueError("invalid failed Principal row")
            continue
        revision_body = principal.revision_payload(
            plan, task, originals[i], feedback
        )
        if row["teacher_call"] is None:
            raise ValueError("missing recorded teacher call")
        revised = reconstruct(
            row["teacher_call"], revision_body, plan["teacher"]["name"]
        )
        answer = mtc.parse_answer(revised) if revised is not None else None
        matched = (answer == mtc.oracle(task)) if answer is not None else False
        if (row["revised_answer"] != answer or
                row["revised_match"] is not matched or
                row["status"] != ("completed" if revised is not None
                                  else "revision_transport_failed")):
            raise ValueError("Round 2 revision / oracle mismatch")


def validate_call_journal(records, calls):
    expected = []
    for row in records:
        expected.append(call_event(
            row["index"], row["item_id"], "principal", row["principal_call"]
        ))
        if row["teacher_call"] is not None:
            expected.append(call_event(
                row["index"], row["item_id"], "teacher", row["teacher_call"]
            ))
    if calls != expected:
        raise ValueError("Round 2 call journal mismatch")


def summary(rows):
    # A missing/unparseable revised answer is UNGRADED, not proof that
    # a formerly correct answer regressed. Count transitions only when
    # a final typed answer is actually available.
    scored = [x for x in rows if x["revised_answer"] is not None]
    return {
        "rows": len(rows),
        "revisions_scored": len(scored),
        "ungraded": len(rows) - len(scored),
        "baseline_oracle_matches": sum(x["baseline_match"] for x in rows),
        "revised_oracle_matches": sum(x["revised_match"] for x in rows),
        "principal_feedback_obtained": sum(x["principal_feedback"] is not None for x in rows),
        "principal_verdict_oracle_agreements": sum(
            x["verdict_matches_oracle"] is True for x in rows),
        "principal_verdict_oracle_disagreements": sum(
            x["verdict_matches_oracle"] is False for x in rows),
        "corrected": sum(not x["baseline_match"] and x["revised_match"] for x in scored),
        "retained": sum(x["baseline_match"] and x["revised_match"] for x in scored),
        "regressions": sum(x["baseline_match"] and not x["revised_match"] for x in scored),
        "still_incorrect": sum(not x["baseline_match"] and not x["revised_match"] for x in scored),
        "training_admission": False, "rights_status": "QUARANTINED_NOT_ADMITTED",
        "comparison_class": "exposed_second_round_no_causal_control",
    }


def plan_command(args):
    reject_nested_output(args.out, args.teacher_run, args.previous_run)
    new_plan = validate_plan(plan_value(
        args.teacher_run, args.previous_run, args.previous_manifest_identity
    ))
    out = mtc.create_private(args.out)
    mtc.write_new(out / "plan.json", mtc.canonical(new_plan))
    print("round2_request_identity=" + mtc.sha(mtc.canonical(new_plan)))
    print("baseline=" + str(sum(x["baseline_match"] for x in new_plan["items"])) +
          "/4; principal_calls_max=4; revisions_max=4; no inference")


def capture_command(args):
    if not args.authorize_round2:
        raise ValueError("separate Round 2 owner authorization required")
    reject_nested_output(args.out, args.teacher_run, args.previous_run)
    raw = mtc.checked_file(args.plan)
    if mtc.sha(raw) != args.request_identity:
        raise ValueError("Round 2 request identity mismatch")
    plan = validate_plan(mtc.strict_json(raw))
    originals = check_prior(plan, args.teacher_run, args.previous_run)
    installed = mtc.inventory()
    for model in (plan["principal"], plan["teacher"]):
        if mtc.model_sha(installed, model["name"]) != model["digest"]:
            raise ValueError("installed model digest drift")
    # No silent retry/resume: an interrupted HTTP request may already have executed.
    out = mtc.create_private(args.out)
    mtc.write_new(out / "plan.json", raw)
    mtc.write_new(out / "calls.jsonl", b"")
    mtc.write_new(out / "attempts.jsonl", b"")
    records = []
    for index, task in enumerate(training_tasks()):
        item, original = plan["items"][index], originals[index]
        row = {
            "index": index, "item_id": item["item_id"],
            "previous_row_sha256": item["previous_row_sha256"],
            "baseline_match": item["baseline_match"],
            "principal_call": None, "principal_feedback": None,
            "verdict_matches_oracle": None,
            "teacher_call": None, "revised_answer": None,
            "revised_match": False, "status": "principal_failed_or_invalid",
        }
        review_body = principal.review_payload(plan, task, original)
        row["principal_call"] = record_call({
            **review_body, "_expected_digest": plan["principal"]["digest"]
        }, plan["principal"]["name"])
        mtc.append_line(out / "calls.jsonl", call_event(
            index, item["item_id"], "principal", row["principal_call"]
        ))
        check_installed_model(plan["principal"])
        reviewed = reconstruct(
            row["principal_call"], review_body, plan["principal"]["name"]
        )
        feedback = feedback_of(
            reviewed, plan["principal"]["name"]
        ) if reviewed is not None else None
        row["principal_feedback"] = feedback
        row["verdict_matches_oracle"] = verdict_alignment(
            feedback, item["baseline_match"]
        )
        if feedback is not None:
            revision_body = principal.revision_payload(
                plan, task, original, feedback
            )
            row["teacher_call"] = record_call({
                **revision_body, "_expected_digest": plan["teacher"]["digest"]
            }, plan["teacher"]["name"])
            mtc.append_line(out / "calls.jsonl", call_event(
                index, item["item_id"], "teacher", row["teacher_call"]
            ))
            check_installed_model(plan["teacher"])
            revised = reconstruct(
                row["teacher_call"], revision_body, plan["teacher"]["name"]
            )
            row["revised_answer"] = mtc.parse_answer(revised) if (
                revised is not None) else None
            row["revised_match"] = (row["revised_answer"] == mtc.oracle(task)) if (
                row["revised_answer"] is not None) else False
            row["status"] = "completed" if revised is not None else "revision_transport_failed"
        mtc.append_line(out / "attempts.jsonl", row)
        records.append(row)
        if any(call is not None and call["error"] == "model_identity_mismatch"
               for call in (row["principal_call"], row["teacher_call"])):
            raise ValueError("mismatched model response; partial archive retained")
    validate_records(plan, originals, records)
    validate_call_journal(records, mtc.read_attempts(out / "calls.jsonl"))
    mtc.write_new(out / "summary.json", mtc.canonical(summary(records)))
    manifest = {"schema": MANIFEST_SCHEMA, "members": {
        file: mtc.sha(mtc.checked_file(out / file))
        for file in ("plan.json", "calls.jsonl", "attempts.jsonl", "summary.json")
    }}
    mtc.write_new(out / "manifest.json", mtc.canonical(manifest))
    print("round2_manifest_identity=" + mtc.sha(mtc.canonical(manifest)))
    print("training_admission=0; no weight update; prior archives unchanged")


def verify_command(args):
    out = args.run
    manifest_raw = mtc.checked_file(out / "manifest.json")
    if mtc.sha(manifest_raw) != args.manifest_identity:
        raise ValueError("external Round 2 manifest identity mismatch")
    manifest = mtc.strict_json(manifest_raw)
    if (type(manifest) is not dict or set(manifest) != {"schema", "members"} or
            manifest["schema"] != MANIFEST_SCHEMA or
            type(manifest["members"]) is not dict or
            set(manifest["members"]) != {
                "plan.json", "calls.jsonl", "attempts.jsonl", "summary.json"
            }):
        raise ValueError("invalid Round 2 manifest")
    if {p.name for p in out.iterdir()} != {
        "plan.json", "calls.jsonl", "attempts.jsonl", "summary.json", "manifest.json"
    }:
        raise ValueError("unexpected Round 2 archive membership")
    for name, identity in manifest["members"].items():
        if mtc.sha(mtc.checked_file(out / name)) != identity:
            raise ValueError("Round 2 member hash mismatch")
    plan = validate_plan(mtc.strict_json(mtc.checked_file(out / "plan.json")))
    originals = check_prior(plan, args.teacher_run, args.previous_run)
    records = mtc.read_attempts(out / "attempts.jsonl")
    validate_records(plan, originals, records)
    validate_call_journal(records, mtc.read_attempts(out / "calls.jsonl"))
    if len(records) != len(training_tasks()):
        raise ValueError("incomplete Round 2 records")
    if mtc.checked_file(out / "summary.json") != mtc.canonical(summary(records)):
        raise ValueError("Round 2 summary mismatch")
    print("verified_round2_recorded=" + args.manifest_identity)
    print("training_admission=0; no_student_training_or_weight_update")


def main(argv=None):
    p = argparse.ArgumentParser(description="MTC-PRINCIPAL-R2-001 chained supervision")
    sub = p.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("plan")
    freeze.add_argument("--teacher-run", type=Path, required=True)
    freeze.add_argument("--previous-run", type=Path, required=True)
    freeze.add_argument("--previous-manifest-identity", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--teacher-run", type=Path, required=True)
    capture.add_argument("--previous-run", type=Path, required=True)
    capture.add_argument("--plan", type=Path, required=True)
    capture.add_argument("--request-identity", required=True)
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--authorize-round2", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--teacher-run", type=Path, required=True)
    verify.add_argument("--previous-run", type=Path, required=True)
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--manifest-identity", required=True)
    args = p.parse_args(argv)
    try:
        {"plan": plan_command, "capture": capture_command,
         "verify": verify_command}[args.command](args)
    except (ValueError, TypeError, KeyError, OSError, mtc.TransportError) as exc:
        p.exit(2, "MTC-PRINCIPAL-R2-001: " + type(exc).__name__ + ": " +
               str(exc)[:180] + "\n")


if __name__ == "__main__":
    main()
