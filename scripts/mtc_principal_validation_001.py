#!/usr/bin/env python3
"""MTC-PRINCIPAL-VALIDATION-001: examiner audit + no-feedback teacher control.

No automatic corpus admission, training, cloud inference, retries or promotion.
All Qwen verdicts are advisory and scored against arithmetic independent of Qwen.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import stat

_SPEC = importlib.util.spec_from_file_location(
    "mtc_r2_tagged_for_validation",
    Path(__file__).with_name("mtc_principal_round2_002.py")
)
r2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(r2)
principal, mtc = r2.principal, r2.mtc

PLAN_SCHEMA = "trinite.mtc-principal-validation.plan.v1"
MANIFEST_SCHEMA = "trinite.mtc-principal-validation.manifest.v1"
AUTH_SCHEMA = "trinite.mtc-principal-validation.owner-authorization.v1"
CLAIM_SCHEMA = "trinite.mtc-principal-validation.execution-claim.v1"
OWNER = "Trent Slade"
MAX_EXAMS = 8
CONTROL_COUNT = 4
MAX_CALLS = MAX_EXAMS + CONTROL_COUNT
AUTHORIZATION = (
    "execute MTC-PRINCIPAL-VALIDATION-001 exactly once: eight Qwen examiner "
    "classifications and four LFM3 no-feedback controls; no retry or resume"
)
CONTRACT_PATHS = (
    "AGENTS.md", ".github/AGENTS.md", "scripts/AGENTS.md",
    "tests/AGENTS.md", "docs/AGENTS.md", "docs/AGENT-IBAE-GOVERNANCE.md",
    "docs/DATA.md", "docs/INVARIANTS.md",
    "docs/MTC-LFM3-001.md", "docs/MTC-PRINCIPAL-001.md",
    "docs/MTC-PRINCIPAL-R2-002.md", "docs/MTC-PRINCIPAL-VALIDATION-001.md",
    "scripts/mtc_lfm3_001.py", "scripts/mtc_principal_001.py",
    "scripts/mtc_principal_round2_002.py",
    "scripts/mtc_principal_validation_001.py",
    "tests/mtc_principal_validation_001/test_validation.py",
    ".github/workflows/mtc-principal-validation-001-contract.yml",
)


def closure():
    root = Path(__file__).resolve().parents[1]
    return {p: mtc.sha(mtc.source_bytes(root / p)) for p in CONTRACT_PATHS}


def examiner_cases():
    """Balanced correct/incorrect deterministic formal claims, exposed diagnostic."""
    # Identical tasks are deliberately used in paired correct/incorrect form
    # to isolate a false-positive reviewer verdict; not a held-out benchmark.
    seeds = (
        ("mod11", "addmod", 8, 15, 11, "2", "1"),
        ("mod13", "mulmod", 12, 7, 13, "3", "6"),
        ("mod19", "addmod", 37, 42, 19, "4", "3"),
        ("mod17", "mulmod", 9, 8, 17, "5", "4"),
    )
    result = []
    for ident, op, a, b, modulus, wrong, right in seeds:
        for variant, claim in (("incorrect", wrong), ("correct", right)):
            item = {
                "id": f"{ident}-{variant}", "operation": op, "a": a, "b": b,
                "modulus": modulus, "claimed_answer": claim,
            }
            expected = arithmetic_answer(item)
            if (variant == "correct") is not (claim == expected):
                raise ValueError("invalid static examiner challenge")
            result.append(item)
    return result


def arithmetic_answer(item):
    if item["operation"] == "addmod":
        return str((item["a"] + item["b"]) % item["modulus"])
    if item["operation"] == "mulmod":
        return str((item["a"] * item["b"]) % item["modulus"])
    raise ValueError("unknown arithmetic operation")


def truth(item):
    return "correct" if item["claimed_answer"] == arithmetic_answer(item) else "incorrect"


def control_tasks():
    return [task for task in mtc.tasks() if task["split"] == "train"]


def previous_evidence(teacher_run, baseline_run, baseline_identity):
    if type(baseline_identity) is not str or re.fullmatch(
            r"sha256:[a-f0-9]{64}", baseline_identity) is None:
        raise ValueError("invalid previous Principal manifest identity")
    principal.verify_command(argparse.Namespace(
        teacher_run=teacher_run, run=baseline_run, manifest_identity=baseline_identity
    ))
    prev = principal.check_plan(mtc.strict_json(
        mtc.checked_file(Path(baseline_run) / "plan.json")
    ))
    rows = mtc.read_attempts(Path(baseline_run) / "attempts.jsonl")
    tasks = control_tasks()
    if len(rows) != len(tasks) or sum(x["revised_match"] for x in rows) != 3:
        raise ValueError("expected verified 3/4 prior Principal baseline")
    frozen = []
    for row, task in zip(rows, tasks):
        if (row["item_id"] != task["id"] or row["status"] != "completed"
                or type(row["revised_answer"]) is not str or
                row["revised_match"] is not (row["revised_answer"] == mtc.oracle(task))):
            raise ValueError("invalid prior Principal revised row")
        frozen.append({
            "item_id": task["id"],
            "previous_row_sha256": mtc.sha(mtc.canonical(row)),
            "previous_revised_match": row["revised_match"],
        })
    return prev, frozen


def plan_value(teacher_run, baseline_run, baseline_identity):
    prev, rows = previous_evidence(teacher_run, baseline_run, baseline_identity)
    models = (prev["principal"], prev["teacher"])
    installed = mtc.inventory()
    for model in models:
        if mtc.model_sha(installed, model["name"]) != model["digest"]:
            raise ValueError("installed model digest drift during freeze")
    return {
        "schema": PLAN_SCHEMA, "source_closure": closure(),
        "prior_principal_manifest_identity": baseline_identity,
        "prior_teacher_manifest_identity": prev["prior_manifest_identity"],
        "principal": prev["principal"], "teacher": prev["teacher"],
        "baseline_rows": rows,
        "exams": examiner_cases(),
        "response_policy": mtc.RESPONSE_POLICY,
        "principal_options": principal.PRINCIPAL_OPTIONS,
        "control_options": mtc.SETTINGS,
        "planned_calls": MAX_CALLS,
        "training_admission": False,
        "rights_status": "QUARANTINED_NOT_ADMITTED",
        "comparison_class": "exposed_examiner_diagnostic_plus_no_feedback_repeat",
    }


def validate_plan(plan):
    if type(plan) is not dict or set(plan) != {
        "schema", "source_closure", "prior_principal_manifest_identity",
        "prior_teacher_manifest_identity", "principal", "teacher",
        "baseline_rows", "exams", "response_policy", "principal_options",
        "control_options", "planned_calls", "training_admission",
        "rights_status", "comparison_class",
    }:
        raise ValueError("invalid validation-plan fields")
    if (plan["schema"] != PLAN_SCHEMA or plan["source_closure"] != closure()
            or plan["exams"] != examiner_cases()
            or plan["response_policy"] != mtc.RESPONSE_POLICY
            or plan["principal_options"] != principal.PRINCIPAL_OPTIONS
            or plan["control_options"] != mtc.SETTINGS
            or type(plan["planned_calls"]) is not int
            or plan["planned_calls"] != MAX_CALLS
            or plan["rights_status"] != "QUARANTINED_NOT_ADMITTED"
            or plan["training_admission"] is not False
            or plan["comparison_class"] !=
            "exposed_examiner_diagnostic_plus_no_feedback_repeat"):
        raise ValueError("validation-plan source or policy drift")
    for key in ("prior_principal_manifest_identity", "prior_teacher_manifest_identity"):
        if (type(plan[key]) is not str or re.fullmatch(
                r"sha256:[a-f0-9]{64}", plan[key]) is None):
            raise ValueError("invalid parent manifest identity")
    for key, checker in (("principal", principal.PRINCIPAL_PATTERN), ("teacher", mtc.NAME)):
        model = plan[key]
        if (type(model) is not dict or set(model) != {"name", "digest"}
                or type(model["name"]) is not str or checker.fullmatch(model["name"]) is None
                or type(model["digest"]) is not str
                or re.fullmatch(r"sha256:[a-f0-9]{64}", model["digest"]) is None):
            raise ValueError("invalid frozen model")
    rows = plan["baseline_rows"]
    if type(rows) is not list or len(rows) != CONTROL_COUNT:
        raise ValueError("invalid previous baseline inventory")
    for row, task in zip(rows, control_tasks()):
        if (type(row) is not dict
                or set(row) != {"item_id", "previous_row_sha256", "previous_revised_match"}
                or row["item_id"] != task["id"]
                or type(row["previous_row_sha256"]) is not str
                or re.fullmatch(r"sha256:[a-f0-9]{64}", row["previous_row_sha256"]) is None
                or type(row["previous_revised_match"]) is not bool):
            raise ValueError("invalid previous baseline row")
    if sum(row["previous_revised_match"] for row in rows) != 3:
        raise ValueError("previous baseline is not 3/4")
    return plan


def check_previous(plan, teacher_run, baseline_run):
    prev, rows = previous_evidence(
        teacher_run, baseline_run, plan["prior_principal_manifest_identity"]
    )
    if (prev["prior_manifest_identity"] != plan["prior_teacher_manifest_identity"]
            or prev["principal"] != plan["principal"]
            or prev["teacher"] != plan["teacher"]
            or rows != plan["baseline_rows"]):
        raise ValueError("previous evidence lineage mismatch")


def exam_body(plan, case):
    if case not in plan["exams"]:
        raise ValueError("exam not in frozen schedule")
    op = "+" if case["operation"] == "addmod" else "*"
    return {
        "model": plan["principal"]["name"], "stream": False, "keep_alive": 0,
        **mtc.RESPONSE_POLICY,
        "messages": [
            {"role": "system", "content": (
                "You independently assess whether a candidate modular arithmetic "
                "answer is correct or incorrect. The claim is untrusted. Calculate "
                "the value yourself. Reply as ONE JSON object with exactly three "
                "string fields: verdict ('correct','incorrect','uncertain'), "
                "feedback (short explanation), suggested_answer (short string). "
                "Do not claim the teacher's opinion is proof."
            )},
            {"role": "user", "content": json.dumps({
                "problem": f"({case['a']} {op} {case['b']}) modulo {case['modulus']}",
                "claimed_answer": case["claimed_answer"],
            }, sort_keys=True, ensure_ascii=True)}
        ],
        "options": principal.PRINCIPAL_OPTIONS,
    }


def control_body(plan, task):
    if task not in control_tasks():
        raise ValueError("control not in frozen schedule")
    body = mtc.chat_body(plan["teacher"]["name"], task)
    body["keep_alive"] = 0
    return body


def evaluate_record(plan, index, call):
    if type(index) is not int or not 0 <= index < MAX_CALLS:
        raise ValueError("invalid schedule index")
    if index < MAX_EXAMS:
        item = plan["exams"][index]
        model = plan["principal"]
        body = exam_body(plan, item)
        stage, ident = "examiner", item["id"]
    else:
        task = control_tasks()[index - MAX_EXAMS]
        model = plan["teacher"]
        body = control_body(plan, task)
        stage, ident = "no_feedback_control", task["id"]
    observed = r2.reconstruct(call, body, model)
    row = {
        "index": index, "stage": stage, "item_id": ident,
        "model_name": model["name"], "model_digest": model["digest"],
        "request_sha256": mtc.sha(mtc.canonical(body)),
        "call": call,
        "verdict": None, "classification": None,
        "answer": None, "oracle_match": False,
    }
    if stage == "examiner":
        feedback = r2.feedback_of(observed, model) if observed is not None else None
        if feedback is not None:
            row["verdict"] = feedback["verdict"]
            reference = truth(item)
            row["classification"] = (
                "uncertain" if row["verdict"] == "uncertain"
                else "TP" if reference == "correct" and row["verdict"] == "correct"
                else "FP" if reference == "incorrect" and row["verdict"] == "correct"
                else "FN" if reference == "correct"
                else "TN"
            )
    else:
        row["answer"] = mtc.parse_answer(observed) if observed is not None else None
        row["oracle_match"] = (
            row["answer"] == mtc.oracle(task) if row["answer"] is not None else False
        )
    return row


def report(rows):
    exams = [r for r in rows if r["stage"] == "examiner"]
    controls = [r for r in rows if r["stage"] == "no_feedback_control"]
    count = lambda name: sum(r["classification"] == name for r in exams)
    control_scored = [r for r in controls if r["answer"] is not None]
    return {
        "planned": MAX_CALLS,
        "attempted": len(rows),
        "examiner_cases": len(exams),
        "examiner_valid_verdicts": sum(x["verdict"] is not None for x in exams),
        "true_positive": count("TP"),
        "false_positive": count("FP"),
        "true_negative": count("TN"),
        "false_negative": count("FN"),
        "uncertain": count("uncertain"),
        "invalid_examiner": count(None),
        "no_feedback_controls": len(controls),
        "no_feedback_scored": len(control_scored),
        "no_feedback_matches": sum(x["oracle_match"] for x in control_scored),
        "previous_supervised_baseline_matches": 3,
        "comparison_class": "exposed_examiner_diagnostic_plus_no_feedback_repeat",
        "training_admission": False,
        "rights_status": "QUARANTINED_NOT_ADMITTED",
    }


def reject_nested(path, teacher_run, baseline_run):
    dst = Path(path).resolve(strict=False)
    for parent in (teacher_run, baseline_run):
        root = Path(parent).resolve(strict=True)
        if dst == root or root in dst.parents:
            raise ValueError("new output must stay outside immutable parent archives")


def authorization_value(plan, identity, approval_path, capture_path):
    return {
        "schema": AUTH_SCHEMA,
        "approved_by": OWNER,
        "request_identity": identity,
        "approval_file": str(Path(approval_path).resolve(strict=False)),
        "capture_out": str(Path(capture_path).resolve(strict=False)),
        "prior_principal_manifest_identity": plan["prior_principal_manifest_identity"],
        "prior_teacher_manifest_identity": plan["prior_teacher_manifest_identity"],
        "principal": plan["principal"], "teacher": plan["teacher"],
        "planned_calls": MAX_CALLS,
        "one_use": True, "training_admission": False,
        "authorization": AUTHORIZATION,
    }


def read_authorization(plan, identity, approval_path, capture_path):
    path = Path(approval_path).resolve(strict=True)
    parent = path.parent
    st = parent.stat()
    if not parent.is_dir() or (os.name == "posix" and (
            st.st_uid != os.geteuid() or stat.S_IMODE(st.st_mode) & 0o077)):
        raise ValueError("unsafe authorization directory")
    raw = mtc.checked_file(path)
    value = mtc.strict_json(raw)
    if value != authorization_value(plan, identity, path, capture_path):
        raise ValueError("exact-scope owner authorization required")
    return raw, value


def claim_value(approval_raw, authorization):
    return {
        "schema": CLAIM_SCHEMA,
        "authorization_sha256": mtc.sha(approval_raw),
        "request_identity": authorization["request_identity"],
        "approval_file": authorization["approval_file"],
        "capture_out": authorization["capture_out"],
        "one_use": True,
    }


def claim_path(approval_path):
    path = Path(approval_path).resolve(strict=True)
    return path.with_name(path.name + ".execution-claim.json")


def plan_command(args):
    reject_nested(args.out, args.teacher_run, args.baseline_run)
    value = validate_plan(plan_value(
        args.teacher_run, args.baseline_run, args.baseline_manifest_identity
    ))
    dst = mtc.create_private(args.out)
    mtc.write_new(dst / "plan.json", mtc.canonical(value))
    print("validation_request_identity=" + mtc.sha(mtc.canonical(value)))
    print("examiner_calls=8; no_feedback_calls=4; no inference")


def capture_command(args):
    reject_nested(args.out, args.teacher_run, args.baseline_run)
    reject_nested(args.owner_approval, args.teacher_run, args.baseline_run)
    plan_raw = mtc.checked_file(args.plan)
    if mtc.sha(plan_raw) != args.request_identity:
        raise ValueError("validation request identity mismatch")
    plan = validate_plan(mtc.strict_json(plan_raw))
    auth_raw, authorization = read_authorization(
        plan, args.request_identity, args.owner_approval, args.out
    )
    check_previous(plan, args.teacher_run, args.baseline_run)
    installed = mtc.inventory()
    for model in (plan["principal"], plan["teacher"]):
        if mtc.model_sha(installed, model["name"]) != model["digest"]:
            raise ValueError("installed model digest drift before capture")
    if Path(args.out).exists() or Path(args.out).is_symlink():
        raise FileExistsError("capture output already exists")
    claim_raw = mtc.canonical(claim_value(auth_raw, authorization))
    # Consumable only once: even failure/interruption preserves this claim.
    mtc.write_new(claim_path(args.owner_approval), claim_raw)
    out = mtc.create_private(args.out)
    for name, raw in (
        ("plan.json", plan_raw), ("owner-authorization.json", auth_raw),
        ("execution-claim.json", claim_raw), ("attempts.jsonl", b"")
    ):
        mtc.write_new(out / name, raw)
    records = []
    for i in range(MAX_CALLS):
        if i < MAX_EXAMS:
            body = exam_body(plan, plan["exams"][i])
            model = plan["principal"]
        else:
            body = control_body(plan, control_tasks()[i-MAX_EXAMS])
            model = plan["teacher"]
        call = r2.record_call(body, model)
        row = evaluate_record(plan, i, call)
        mtc.append_line(out / "attempts.jsonl", row)
        records.append(row)
        # Drift is terminal; preserve the last call before stopping.
        if call["error"] in (
                "model_digest_drift", "model_digest_check_failed",
                "model_identity_mismatch"):
            raise ValueError("model identity drift/mismatch; partial archive retained")
    validate_records(plan, records)
    mtc.write_new(out / "summary.json", mtc.canonical(report(records)))
    manifest = {"schema": MANIFEST_SCHEMA, "members": {
        name: mtc.sha(mtc.checked_file(out / name))
        for name in (
            "plan.json", "owner-authorization.json", "execution-claim.json",
            "attempts.jsonl", "summary.json"
        )
    }}
    mtc.write_new(out / "manifest.json", mtc.canonical(manifest))
    print("validation_manifest_identity=" + mtc.sha(mtc.canonical(manifest)))
    print("training_admission=0; source archives unchanged")


def validate_records(plan, rows):
    if type(rows) is not list or len(rows) > MAX_CALLS:
        raise ValueError("invalid validation record count")
    for i, actual in enumerate(rows):
        if type(actual) is not dict or set(actual) != {
                "index", "stage", "item_id", "model_name", "model_digest",
                "request_sha256", "call", "verdict", "classification",
                "answer", "oracle_match",
        }:
            raise ValueError("invalid attempt fields")
        reconstructed = evaluate_record(plan, i, actual["call"])
        if actual != reconstructed:
            raise ValueError("recorded exam/control classification mismatch")


def verify_command(args):
    out = args.run
    raw = mtc.checked_file(out / "manifest.json")
    if mtc.sha(raw) != args.manifest_identity:
        raise ValueError("external examiner manifest identity mismatch")
    manifest = mtc.strict_json(raw)
    if (type(manifest) is not dict or set(manifest) != {"schema", "members"}
            or manifest["schema"] != MANIFEST_SCHEMA
            or type(manifest["members"]) is not dict
            or set(manifest["members"]) != {
                "plan.json", "owner-authorization.json", "execution-claim.json",
                "attempts.jsonl", "summary.json",
            }):
        raise ValueError("invalid examiner manifest")
    if {x.name for x in out.iterdir()} != {
            "plan.json", "owner-authorization.json", "execution-claim.json",
            "attempts.jsonl", "summary.json", "manifest.json",
    }:
        raise ValueError("unexpected examiner archive members")
    for name, digest in manifest["members"].items():
        if mtc.sha(mtc.checked_file(out / name)) != digest:
            raise ValueError("examiner manifest member mismatch")
    plan_raw = mtc.checked_file(out / "plan.json")
    plan = validate_plan(mtc.strict_json(plan_raw))
    auth_raw = mtc.checked_file(out / "owner-authorization.json")
    approval = mtc.strict_json(auth_raw)
    if type(approval) is not dict or type(approval.get("approval_file")) is not str:
        raise ValueError("invalid archived examiner authorization")
    if approval != authorization_value(
            plan, mtc.sha(plan_raw), approval["approval_file"], out):
        raise ValueError("archived examiner authorization scope mismatch")
    claim_raw = mtc.checked_file(out / "execution-claim.json")
    if claim_raw != mtc.canonical(claim_value(auth_raw, approval)):
        raise ValueError("archived one-use claim mismatch")
    check_previous(plan, args.teacher_run, args.baseline_run)
    rows = mtc.read_attempts(out / "attempts.jsonl")
    validate_records(plan, rows)
    if len(rows) != MAX_CALLS:
        raise ValueError("incomplete examiner/control recording")
    if mtc.checked_file(out / "summary.json") != mtc.canonical(report(rows)):
        raise ValueError("examiner/control summary mismatch")
    print("verified_validation_recorded=" + args.manifest_identity)
    print("training_admission=0; no_native_student_training")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MTC-PRINCIPAL-VALIDATION-001")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--teacher-run", type=Path, required=True)
    plan.add_argument("--baseline-run", type=Path, required=True)
    plan.add_argument("--baseline-manifest-identity", required=True)
    plan.add_argument("--out", type=Path, required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--teacher-run", type=Path, required=True)
    capture.add_argument("--baseline-run", type=Path, required=True)
    capture.add_argument("--plan", type=Path, required=True)
    capture.add_argument("--request-identity", required=True)
    capture.add_argument("--owner-approval", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--teacher-run", type=Path, required=True)
    verify.add_argument("--baseline-run", type=Path, required=True)
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--manifest-identity", required=True)
    args = parser.parse_args(argv)
    try:
        {"plan": plan_command, "capture": capture_command,
         "verify": verify_command}[args.command](args)
    except (OSError, ValueError, TypeError, KeyError, mtc.TransportError) as exc:
        parser.exit(2, "MTC-PRINCIPAL-VALIDATION-001: " + type(exc).__name__ +
                    ": " + str(exc)[:150] + "\n")


if __name__ == "__main__":
    main()
