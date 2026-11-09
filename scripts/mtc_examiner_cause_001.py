#!/usr/bin/env python3
"""MTC-EXAMINER-CAUSE-001: isolated solve / review / forced-choice diagnostic.

Frozen, exposed arithmetic probes. Qwen is an observation, never an oracle.
No training/admission, automatic retry, cloud requests or model-weight mutation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess

_SPEC = importlib.util.spec_from_file_location(
    "mtc_validation_for_diagnostic",
    Path(__file__).with_name("mtc_principal_validation_001.py"),
)
v = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v)
r2, principal, mtc = v.r2, v.principal, v.mtc

PLAN_SCHEMA = "trinite.mtc-examiner-cause.plan.v1"
MANIFEST_SCHEMA = "trinite.mtc-examiner-cause.manifest.v1"
AUTH_SCHEMA = "trinite.mtc-examiner-cause.owner-authorization.v1"
CLAIM_SCHEMA = "trinite.mtc-examiner-cause.execution-claim.v1"
VALIDATION_IDENTITY = "sha256:8a8d909c72ed2e47ab0d2903b337939fc7d1e9bc9789ffc0c8e331efd36b65f9"
LOG_RECEIPT_IDENTITY = "sha256:a0d7fd7912c3baa4164d89390e4e7317b18c019a8c64b30dd3d7edde599a29b7"
LOG_SHA = "sha256:852e26b7014dec088eef949f8ee34e79c6988dfe94266339e6c3eab998979cf4"
SOURCE_FREEZE_HEAD = "aa88b4f6963b61de83be2c0eac37aed39384679c"
OWNER = "Trent Slade"
TOTAL = 16
AUTHORIZATION = (
    "execute MTC-EXAMINER-CAUSE-001 exactly once: 4 Qwen solve-only, "
    "8 original claim-review, 4 paired forced-choice; no retries or resume"
)
CLOSURE = (
    "AGENTS.md", ".github/AGENTS.md", "scripts/AGENTS.md", "tests/AGENTS.md",
    "docs/AGENTS.md", "docs/AGENT-IBAE-GOVERNANCE.md",
    "docs/DATA.md", "docs/INVARIANTS.md", "docs/OLLAMA-LOG-001.md",
    "docs/MTC-PRINCIPAL-VALIDATION-001.md",
    "docs/MTC-EXAMINER-CAUSE-001.md",
    "scripts/mtc_lfm3_001.py", "scripts/mtc_principal_001.py",
    "scripts/mtc_principal_round2_002.py", "scripts/mtc_principal_validation_001.py",
    "scripts/mtc_examiner_cause_001.py",
    "tests/mtc_examiner_cause_001/test_examiner_cause.py",
    ".github/workflows/mtc-examiner-cause-001-contract.yml",
)
HEX_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def closure():
    root = Path(__file__).resolve().parents[1]
    return {p: mtc.sha(mtc.source_bytes(root / p)) for p in CLOSURE}


def source_commit():
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        text=True, check=True,
    ).stdout.strip()


def probes():
    # Exposed original four problems: no held-out inference claimed.
    pairs = v.examiner_cases()
    tasks = []
    for i in range(4):
        incorrect, correct = pairs[2 * i:2 * i + 2]
        if (incorrect["id"].split("-")[0] != correct["id"].split("-")[0]
                or v.truth(incorrect) != "incorrect"
                or v.truth(correct) != "correct"):
            raise ValueError("invalid frozen examiner pairs")
        tasks.append({"id": correct["id"].split("-")[0], "operation": correct["operation"],
                      "a": correct["a"], "b": correct["b"], "modulus": correct["modulus"]})
    return tasks


def choice_cases():
    pairs = v.examiner_cases()
    cases = []
    for i, task in enumerate(probes()):
        wrong, right = pairs[2 * i:2 * i + 2]
        if i % 2 == 0:
            a, b = wrong["claimed_answer"], right["claimed_answer"]
        else:
            a, b = right["claimed_answer"], wrong["claimed_answer"]
        cases.append({"id": task["id"], "operation": task["operation"],
                      "a": task["a"], "b": task["b"], "modulus": task["modulus"],
                      "candidate_a": a, "candidate_b": b})
    return cases


def scheduled():
    # Preserve lane + order; no adaptive extra calls and no answer-conditioned prompts.
    return (
        [{"lane": "solve", "case": t} for t in probes()]
        + [{"lane": "review", "case": t} for t in v.examiner_cases()]
        + [{"lane": "choice", "case": t} for t in choice_cases()]
    )


def historical(teacher_run, baseline_run, validation_run):
    """Independent re-verification of immutable historical request/response archives."""
    v.verify_command(argparse.Namespace(
        teacher_run=teacher_run, baseline_run=baseline_run,
        run=validation_run, manifest_identity=VALIDATION_IDENTITY,
    ))
    prior = v.validate_plan(mtc.strict_json(
        mtc.checked_file(Path(validation_run) / "plan.json")))
    summary_raw = mtc.checked_file(Path(validation_run) / "summary.json")
    summary = mtc.strict_json(summary_raw)
    expected = {
        "planned": 12, "attempted": 12, "examiner_cases": 8,
        "examiner_valid_verdicts": 8, "true_positive": 4,
        "false_positive": 4, "true_negative": 0, "false_negative": 0,
        "uncertain": 0, "invalid_examiner": 0,
        "no_feedback_controls": 4, "no_feedback_scored": 4,
        "no_feedback_matches": 2, "previous_supervised_baseline_matches": 3,
        "comparison_class": "exposed_examiner_diagnostic_plus_no_feedback_repeat",
        "training_admission": False, "rights_status": "QUARANTINED_NOT_ADMITTED",
    }
    if summary != expected:
        raise ValueError("historical examiner diagnostic summary differs")
    if prior["principal"]["name"] != "qwen3.5:27b":
        raise ValueError("unexpected historical Principal model")
    return {"principal": prior["principal"],
            "summary_sha256": mtc.sha(summary_raw),
            "prior_teacher_manifest_identity": prior["prior_teacher_manifest_identity"],
            "prior_principal_manifest_identity": prior["prior_principal_manifest_identity"]}


def plan_value(teacher_run, baseline_run, validation_run):
    prev = historical(teacher_run, baseline_run, validation_run)
    if mtc.model_sha(mtc.inventory(), prev["principal"]["name"]) != prev["principal"]["digest"]:
        raise ValueError("Qwen installed digest differs from prior validation")
    commit = source_commit()
    if GIT_SHA.fullmatch(commit) is None:
        raise ValueError("invalid Git HEAD")
    return {"schema": PLAN_SCHEMA, "source_closure": closure(), "source_commit": commit,
            "historical_validation_identity": VALIDATION_IDENTITY,
            "historical_summary_sha256": prev["summary_sha256"],
            "historical_teacher_identity": prev["prior_teacher_manifest_identity"],
            "historical_principal_identity": prev["prior_principal_manifest_identity"],
            "milestone_log_receipt_identity": LOG_RECEIPT_IDENTITY,
            "milestone_log_sha256": LOG_SHA,
            "milestone_code_head": SOURCE_FREEZE_HEAD,
            "principal": prev["principal"],
            "response_policy": mtc.RESPONSE_POLICY,
            "principal_options": principal.PRINCIPAL_OPTIONS,
            "schedule": scheduled(), "planned_calls": TOTAL,
            "comparison_class": "exposed_paired_within_problem_three_lane_diagnostic",
            "rights_status": "QUARANTINED_NOT_ADMITTED", "training_admission": False}


def validate_plan(plan):
    required = {"schema", "source_closure", "source_commit",
                "historical_validation_identity", "historical_summary_sha256",
                "historical_teacher_identity", "historical_principal_identity",
                "milestone_log_receipt_identity", "milestone_log_sha256",
                "milestone_code_head", "principal", "response_policy",
                "principal_options", "schedule", "planned_calls", "comparison_class",
                "rights_status", "training_admission"}
    if type(plan) is not dict or set(plan) != required:
        raise ValueError("invalid diagnostic plan fields")
    model = plan["principal"]
    if (plan["schema"] != PLAN_SCHEMA or plan["source_closure"] != closure()
            or type(plan["source_commit"]) is not str
            or GIT_SHA.fullmatch(plan["source_commit"]) is None
            or plan["historical_validation_identity"] != VALIDATION_IDENTITY
            or plan["milestone_log_receipt_identity"] != LOG_RECEIPT_IDENTITY
            or plan["milestone_log_sha256"] != LOG_SHA
            or plan["milestone_code_head"] != SOURCE_FREEZE_HEAD
            or type(plan["historical_summary_sha256"]) is not str
            or HEX_SHA.fullmatch(plan["historical_summary_sha256"]) is None
            or any(type(plan[k]) is not str or HEX_SHA.fullmatch(plan[k]) is None
                   for k in ("historical_teacher_identity", "historical_principal_identity"))
            or type(model) is not dict or set(model) != {"name", "digest"}
            or model["name"] != "qwen3.5:27b" or type(model["digest"]) is not str
            or HEX_SHA.fullmatch(model["digest"]) is None
            or plan["response_policy"] != mtc.RESPONSE_POLICY
            or plan["principal_options"] != principal.PRINCIPAL_OPTIONS
            or plan["schedule"] != scheduled() or type(plan["planned_calls"]) is not int
            or plan["planned_calls"] != TOTAL
            or plan["comparison_class"] != "exposed_paired_within_problem_three_lane_diagnostic"
            or plan["rights_status"] != "QUARANTINED_NOT_ADMITTED"
            or plan["training_admission"] is not False):
        raise ValueError("diagnostic source, identity, or protocol drift")
    return plan


def check_historical(plan, teacher_run, baseline_run, validation_run):
    prior = historical(teacher_run, baseline_run, validation_run)
    if (prior["principal"] != plan["principal"]
            or prior["summary_sha256"] != plan["historical_summary_sha256"]
            or prior["prior_teacher_manifest_identity"] != plan["historical_teacher_identity"]
            or prior["prior_principal_manifest_identity"] != plan["historical_principal_identity"]):
        raise ValueError("historical lineage mismatch")


def problem(case):
    symbol = "+" if case["operation"] == "addmod" else "*"
    return f"({case['a']} {symbol} {case['b']}) modulo {case['modulus']}"


def chat_body(plan, entry):
    lane, case = entry["lane"], entry["case"]
    model = plan["principal"]["name"]
    if entry not in plan["schedule"]:
        raise ValueError("unknown diagnostic request")
    if lane == "review":
        return v.exam_body(
            {"principal": plan["principal"], "exams": v.examiner_cases()}, case
        )  # Exact prior reviewer prompt for comparison.
    if lane == "solve":
        system = ("Solve modular arithmetic independently with no proposed answer. "
                  "Reply as one JSON object with exactly one string field 'answer' "
                  "containing the decimal integer result. No explanations or tools.")
        user = json.dumps({"problem": problem(case)}, sort_keys=True,
                          ensure_ascii=True)
    elif lane == "choice":
        system = ("Independently calculate this modular arithmetic problem. "
                  "Two candidate answers are presented; exactly one is correct. "
                  "Choose A or B, or 'uncertain'. Reply as one JSON object with "
                  "exactly two string fields: choice ('A','B','uncertain') and "
                  "answer (the calculated decimal integer). Do not accept either "
                  "candidate without checking the arithmetic.")
        user = json.dumps({
            "problem": problem(case), "candidate_A": case["candidate_a"],
            "candidate_B": case["candidate_b"],
        }, sort_keys=True, ensure_ascii=True)
    else:
        raise ValueError("invalid diagnostic lane")
    return {"model": model, "stream": False, "keep_alive": 0,
            **mtc.RESPONSE_POLICY,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": principal.PRINCIPAL_OPTIONS}


def choice_of(response, model):
    if (not isinstance(response, dict) or response.get("model") != model
            or response.get("done") is not True or response.get("done_reason") == "length"):
        return None
    msg = response.get("message")
    if not isinstance(msg, dict) or type(msg.get("content")) is not str:
        return None
    try:
        result = mtc.strict_json(msg["content"])
    except ValueError:
        return None
    if (type(result) is not dict or set(result) != {"choice", "answer"}
            or type(result["choice"]) is not str
            or result["choice"] not in ("A", "B", "uncertain")
            or type(result["answer"]) is not str
            or len(result["answer"]) > 32):
        return None
    return result


def evaluate(plan, index, call):
    if type(index) is not int or not 0 <= index < TOTAL:
        raise ValueError("invalid schedule index")
    entry = plan["schedule"][index]
    lane, case = entry["lane"], entry["case"]
    body, model = chat_body(plan, entry), plan["principal"]
    response = r2.reconstruct(call, body, model)
    oracle = v.arithmetic_answer(case)
    row = {"index": index, "lane": lane, "item_id": case["id"],
           "model_name": model["name"], "model_digest": model["digest"],
           "request_sha256": mtc.sha(mtc.canonical(body)),
           "call": call, "answer": None, "oracle_match": False,
           "feedback": None, "verdict": None, "classification": None,
           "choice": None, "choice_match": False,
           "suggestion_oracle_match": False, "suggestion_claim_match": False}
    if lane == "solve":
        row["answer"] = mtc.parse_answer(response) if response is not None else None
        row["oracle_match"] = row["answer"] == oracle if row["answer"] is not None else False
    elif lane == "review":
        feedback = r2.feedback_of(response, model) if response is not None else None
        if feedback is not None:
            row["feedback"] = feedback
            row["verdict"] = feedback["verdict"]
            actual = v.truth(case)
            row["classification"] = (
                "uncertain" if feedback["verdict"] == "uncertain" else
                "TP" if actual == "correct" and feedback["verdict"] == "correct" else
                "FP" if actual == "incorrect" and feedback["verdict"] == "correct" else
                "FN" if actual == "correct" else "TN")
            row["suggestion_oracle_match"] = feedback["suggested_answer"] == oracle
            row["suggestion_claim_match"] = (
                feedback["suggested_answer"] == case["claimed_answer"])
    else:
        parsed = choice_of(response, model["name"]) if response is not None else None
        if parsed is not None:
            row["choice"], row["answer"] = parsed["choice"], parsed["answer"]
            expected = "A" if case["candidate_a"] == oracle else "B"
            row["choice_match"] = parsed["choice"] == expected
            row["oracle_match"] = parsed["answer"] == oracle
    return row


def report(rows):
    solve = [x for x in rows if x["lane"] == "solve"]
    review = [x for x in rows if x["lane"] == "review"]
    choice = [x for x in rows if x["lane"] == "choice"]
    count = lambda label: sum(x["classification"] == label for x in review)
    return {
        "planned": TOTAL, "attempted": len(rows),
        "solve_cases": len(solve), "solve_valid_answers": sum(x["answer"] is not None for x in solve),
        "solve_matches": sum(x["oracle_match"] for x in solve),
        "review_cases": len(review), "review_valid_verdicts": sum(x["verdict"] is not None for x in review),
        "true_positive": count("TP"), "true_negative": count("TN"),
        "false_positive": count("FP"), "false_negative": count("FN"),
        "review_uncertain": count("uncertain"),
        "review_invalid": count(None),
        "review_suggested_oracle_matches": sum(x["suggestion_oracle_match"] for x in review),
        "review_suggested_claim_matches": sum(x["suggestion_claim_match"] for x in review),
        "choice_cases": len(choice), "choice_valid": sum(x["choice"] is not None for x in choice),
        "choice_correct": sum(x["choice_match"] for x in choice),
        "choice_answer_correct": sum(x["oracle_match"] for x in choice),
        "paired_both_claims_approved": sum(
            review[2*i]["verdict"] == review[2*i+1]["verdict"] == "correct"
            for i in range(len(review)//2)
        ),
        "prior_false_positive": 4, "prior_true_positive": 4,
        "comparison_class": "exposed_paired_within_problem_three_lane_diagnostic",
        "rights_status": "QUARANTINED_NOT_ADMITTED", "training_admission": False,
    }


def reject_nested(path, teacher_run, baseline_run, validation_run):
    dst = Path(path).resolve(strict=False)
    for directory in (teacher_run, baseline_run, validation_run):
        root = Path(directory).resolve(strict=True)
        if dst == root or root in dst.parents:
            raise ValueError("new files cannot be placed in frozen archives")


def approval_value(plan, identity, approval_path, capture_path):
    return {
        "schema": AUTH_SCHEMA, "approved_by": OWNER,
        "request_identity": identity,
        "approval_file": str(Path(approval_path).resolve(strict=False)),
        "capture_out": str(Path(capture_path).resolve(strict=False)),
        "principal": plan["principal"], "planned_calls": TOTAL,
        "historical_validation_identity": VALIDATION_IDENTITY,
        "one_use": True, "training_admission": False,
        "authorization": AUTHORIZATION,
    }


def read_approval(plan, identity, approval_path, capture_path):
    given = Path(approval_path)
    if given.is_symlink():
        raise ValueError("authorization symlink forbidden")
    try:
        path = given.resolve(strict=True)
    except (RuntimeError, FileNotFoundError):
        raise ValueError("exact owner authorization required") from None
    parent = path.parent
    info = parent.stat()
    if not parent.is_dir() or (os.name == "posix" and (
            info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077)):
        raise ValueError("unsafe authorization directory")
    raw = mtc.checked_file(path)
    if mtc.strict_json(raw) != approval_value(plan, identity, path, capture_path):
        raise ValueError("exact-scope owner authorization required")
    return raw


def claim_value(raw, identity, approval_path, capture_path):
    return {"schema": CLAIM_SCHEMA, "authorization_sha256": mtc.sha(raw),
            "request_identity": identity,
            "approval_file": str(Path(approval_path).resolve(strict=False)),
            "capture_out": str(Path(capture_path).resolve(strict=False)),
            "one_use": True}


def plan_command(args):
    reject_nested(args.out, args.teacher_run, args.baseline_run, args.validation_run)
    plan = validate_plan(plan_value(args.teacher_run, args.baseline_run,
                                    args.validation_run))
    dest = mtc.create_private(args.out)
    raw = mtc.canonical(plan)
    mtc.write_new(dest / "plan.json", raw)
    print("examiner_cause_request_identity=" + mtc.sha(raw))
    print("planned_calls=16; solve=4; review=8; choice=4; inference=0")


def capture_command(args):
    for dest in (args.out, args.owner_approval):
        reject_nested(dest, args.teacher_run, args.baseline_run, args.validation_run)
    raw = mtc.checked_file(args.plan)
    if mtc.sha(raw) != args.request_identity:
        raise ValueError("frozen diagnostic request identity mismatch")
    plan = validate_plan(mtc.strict_json(raw))
    approved_raw = read_approval(plan, args.request_identity, args.owner_approval, args.out)
    check_historical(plan, args.teacher_run, args.baseline_run, args.validation_run)
    if mtc.model_sha(mtc.inventory(), plan["principal"]["name"]) != plan["principal"]["digest"]:
        raise ValueError("installed Principal digest drift before capture")
    if Path(args.out).exists() or Path(args.out).is_symlink():
        raise FileExistsError("capture output already exists")
    approval_path = Path(args.owner_approval).resolve(strict=True)
    claim_path = approval_path.with_name(approval_path.name + ".execution-claim.json")
    claim = mtc.canonical(claim_value(approved_raw, args.request_identity, approval_path,
                                     args.out))
    mtc.write_new(claim_path, claim)  # One-use, non-retryable; preserve failed claim.
    out = mtc.create_private(args.out)
    for name, payload in (
        ("plan.json", raw), ("owner-authorization.json", approved_raw),
        ("execution-claim.json", claim), ("attempts.jsonl", b""),
    ):
        mtc.write_new(out / name, payload)
    rows = []
    for i, entry in enumerate(plan["schedule"]):
        call = r2.record_call(chat_body(plan, entry), plan["principal"])
        row = evaluate(plan, i, call)
        mtc.append_line(out / "attempts.jsonl", row)
        rows.append(row)
        if call["error"] in ("model_digest_drift", "model_digest_check_failed",
                             "model_identity_mismatch"):
            raise ValueError("Principal digest mismatch: partial archive retained")
    validate_rows(plan, rows)
    mtc.write_new(out / "summary.json", mtc.canonical(report(rows)))
    members = ("plan.json", "owner-authorization.json",
               "execution-claim.json", "attempts.jsonl", "summary.json")
    manifest = {"schema": MANIFEST_SCHEMA, "members": {
        name: mtc.sha(mtc.checked_file(out / name)) for name in members}}
    mb = mtc.canonical(manifest)
    mtc.write_new(out / "manifest.json", mb)
    print("examiner_cause_manifest_identity=" + mtc.sha(mb))
    print("training_admission=0; original archives unchanged")


def validate_rows(plan, rows):
    if type(rows) is not list or len(rows) > TOTAL:
        raise ValueError("invalid diagnostic row count")
    for i, row in enumerate(rows):
        if type(row) is not dict or row != evaluate(plan, i, row["call"]):
            raise ValueError("recorded examiner diagnosis differs from reconstruction")


def verify_command(args):
    out = Path(args.run)
    raw = mtc.checked_file(out / "manifest.json")
    if mtc.sha(raw) != args.manifest_identity:
        raise ValueError("external diagnostic manifest identity mismatch")
    manifest = mtc.strict_json(raw)
    members = {"plan.json", "owner-authorization.json",
               "execution-claim.json", "attempts.jsonl", "summary.json"}
    if (type(manifest) is not dict or set(manifest) != {"schema", "members"}
            or manifest["schema"] != MANIFEST_SCHEMA
            or type(manifest["members"]) is not dict
            or set(manifest["members"]) != members
            or {x.name for x in out.iterdir()} != members | {"manifest.json"}):
        raise ValueError("invalid diagnostic manifest inventory")
    for name, digest in manifest["members"].items():
        if mtc.sha(mtc.checked_file(out / name)) != digest:
            raise ValueError("diagnostic manifest member digest mismatch")
    plan_raw = mtc.checked_file(out / "plan.json")
    plan = validate_plan(mtc.strict_json(plan_raw))
    archived_approval = mtc.checked_file(out / "owner-authorization.json")
    approval = mtc.strict_json(archived_approval)
    if (type(approval) is not dict or type(approval.get("approval_file")) is not str
            or type(approval.get("capture_out")) is not str
            or approval != approval_value(
                plan, mtc.sha(plan_raw), approval["approval_file"], out)):
        raise ValueError("invalid archived owner authorization")
    expected_claim = mtc.canonical(claim_value(
        archived_approval, mtc.sha(plan_raw), approval["approval_file"], out))
    if mtc.checked_file(out / "execution-claim.json") != expected_claim:
        raise ValueError("invalid archived one-use claim")
    check_historical(plan, args.teacher_run, args.baseline_run, args.validation_run)
    rows = mtc.read_attempts(out / "attempts.jsonl")
    validate_rows(plan, rows)
    if len(rows) != TOTAL or mtc.checked_file(out / "summary.json") != mtc.canonical(report(rows)):
        raise ValueError("incomplete diagnostic archive or summary")
    print("verified_examiner_cause_recorded=" + args.manifest_identity)
    print("claims=exposed_diagnostic_only; training_admission=0; no_inference_on_verify")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MTC-EXAMINER-CAUSE-001")
    sub = parser.add_subparsers(dest="command", required=True)
    def parents(p):
        p.add_argument("--teacher-run", type=Path, required=True)
        p.add_argument("--baseline-run", type=Path, required=True)
        p.add_argument("--validation-run", type=Path, required=True)
    plan = sub.add_parser("plan")
    parents(plan)
    plan.add_argument("--out", type=Path, required=True)
    capture = sub.add_parser("capture")
    parents(capture)
    capture.add_argument("--plan", type=Path, required=True)
    capture.add_argument("--request-identity", required=True)
    capture.add_argument("--owner-approval", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    parents(verify)
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--manifest-identity", required=True)
    args = parser.parse_args(argv)
    try:
        {"plan": plan_command, "capture": capture_command,
         "verify": verify_command}[args.command](args)
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError,
            mtc.TransportError) as exc:
        parser.exit(2, "MTC-EXAMINER-CAUSE-001: " + type(exc).__name__ +
                    ": " + str(exc)[:150] + "\n")


if __name__ == "__main__":
    main()
