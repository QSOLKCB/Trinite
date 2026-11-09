#!/usr/bin/env python3
"""MTC-TEACHER-RECRUITMENT-001: per-candidate advisory arithmetic gate.

Owner-private, versioned, explicit-approval capture. No native learning/admission,
no implicit Ollama pull, no cloud fallback or automatic retries.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import stat

_SPEC = importlib.util.spec_from_file_location(
    "mtc_cause_for_recruitment",
    Path(__file__).with_name("mtc_examiner_cause_001.py"),
)
cause = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cause)
v, r2, mtc = cause.v, cause.r2, cause.mtc

PLAN_SCHEMA = "trinite.mtc-teacher-recruitment.plan.v1"
MANIFEST_SCHEMA = "trinite.mtc-teacher-recruitment.manifest.v1"
AUTH_SCHEMA = "trinite.mtc-teacher-recruitment.owner-authorization.v1"
CLAIM_SCHEMA = "trinite.mtc-teacher-recruitment.execution-claim.v1"
OWNER = "Trent Slade"
CALLS = 16
# Strong role isolation: each tag receives its own plan, authorization, run and
# digest. The 14 GB GLM option is community Q3, not the official Q4 weights.
CANDIDATES = (
    "phi4-reasoning:plus",
    "byczech/glm-4.7-flash-16G-UD-Q3_K_XL:30b",
)
EXPECTED_HISTORY = cause.VALIDATION_IDENTITY
CAUSE_MANIFEST = "sha256:34e39cbc6db430a634c14a5d6ca941e7b50af7bd515684f470ad96908fd9d7cf"
OPTIONS = {"temperature": 0, "seed": 29, "num_predict": 2048, "num_ctx": 4096}
RESPONSE_POLICY = {"format": "json"}  # No think:false: respect model default.
AUTHORIZATION = (
    "execute MTC-TEACHER-RECRUITMENT-001 exactly once for the frozen "
    "candidate and 16 exposed arithmetic tasks; no retry, resume or training"
)
CLOSURE_PATHS = (
    "AGENTS.md", ".github/AGENTS.md", "scripts/AGENTS.md",
    "tests/AGENTS.md", "docs/AGENTS.md", "docs/AGENT-IBAE-GOVERNANCE.md",
    "docs/DATA.md", "docs/INVARIANTS.md",
    "docs/MTC-PRINCIPAL-VALIDATION-001.md", "docs/MTC-EXAMINER-CAUSE-001.md",
    "docs/MTC-TEACHER-RECRUITMENT-001.md",
    "scripts/mtc_lfm3_001.py", "scripts/mtc_principal_001.py",
    "scripts/mtc_principal_round2_002.py",
    "scripts/mtc_principal_validation_001.py",
    "scripts/mtc_examiner_cause_001.py", "scripts/mtc_teacher_recruitment_001.py",
    "tests/mtc_teacher_recruitment_001/test_recruitment.py",
    ".github/workflows/mtc-teacher-recruitment-001-contract.yml",
)
HEX_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
HEX_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def closure():
    root = Path(__file__).resolve().parents[1]
    return {p: mtc.sha(mtc.source_bytes(root / p)) for p in CLOSURE_PATHS}


def schedule():
    return cause.scheduled()


def reject_historical_output(path, teacher_run, baseline_run, validation_run, cause_run):
    cause.reject_nested(path, teacher_run, baseline_run, validation_run)
    existing = Path(cause_run).resolve(strict=True)
    new = Path(path).resolve(strict=False)
    if new == existing or existing in new.parents:
        raise ValueError("new output overlaps immutable examiner-cause archive")


def parent_evidence(teacher_run, baseline_run, validation_run, cause_run):
    hist = cause.historical(teacher_run, baseline_run, validation_run)
    if hist["principal"]["name"] != "qwen3.5:27b":
        raise ValueError("historical examiner evidence mismatch")
    cause.verify_command(argparse.Namespace(
        teacher_run=teacher_run, baseline_run=baseline_run,
        validation_run=validation_run, run=cause_run,
        manifest_identity=CAUSE_MANIFEST,
    ))
    raw = mtc.checked_file(Path(cause_run) / "summary.json")
    summary = mtc.strict_json(raw)
    if (summary.get("planned") != 16 or summary.get("attempted") != 16
            or summary.get("solve_matches") != 0
            or summary.get("false_positive") != 4
            or summary.get("true_negative") != 0
            or summary.get("paired_both_claims_approved") != 4
            or summary.get("choice_correct") != 2
            or summary.get("choice_answer_correct") != 2
            or summary.get("rights_status") != "QUARANTINED_NOT_ADMITTED"
            or summary.get("training_admission") is not False):
        raise ValueError("historical Qwen failure summary mismatch")
    return {**hist, "examiner_cause_summary_sha256": mtc.sha(raw)}


def value_of_plan(candidate, teacher_run, baseline_run, validation_run, cause_run):
    if candidate not in CANDIDATES:
        raise ValueError("candidate must be explicitly allowlisted")
    history = parent_evidence(teacher_run, baseline_run, validation_run, cause_run)
    installed = mtc.inventory()
    digest = mtc.model_sha(installed, candidate)
    commit = cause.source_commit()
    if HEX_COMMIT.fullmatch(commit) is None:
        raise ValueError("invalid source commit")
    return {
        "schema": PLAN_SCHEMA,
        "source_closure": closure(), "source_commit": commit,
        "historical_validation_manifest": EXPECTED_HISTORY,
        "historical_cause_manifest": CAUSE_MANIFEST,
        "historical_cause_summary_sha256": history["examiner_cause_summary_sha256"],
        "historical_summary_sha256": history["summary_sha256"],
        "historical_teacher_manifest": history["prior_teacher_manifest_identity"],
        "historical_principal_manifest": history["prior_principal_manifest_identity"],
        "candidate": {"name": candidate, "digest": digest},
        "candidate_family": "phi4_reasoning_plus" if candidate == CANDIDATES[0]
                            else "glm47_flash_community_q3",
        "response_policy": RESPONSE_POLICY,
        "options": OPTIONS,
        "schedule": schedule(), "planned_calls": CALLS,
        "comparison_class": "exposed_independent_per_candidate_advisory_diagnostic",
        "claim_scope": "arithmetic_supervision_gate_not_general_coding_benchmark",
        "rights_status": "QUARANTINED_NOT_ADMITTED", "training_admission": False,
    }


def validate_plan(plan):
    required = {
        "schema", "source_closure", "source_commit",
        "historical_validation_manifest", "historical_cause_manifest",
        "historical_cause_summary_sha256", "historical_summary_sha256",
        "historical_teacher_manifest", "historical_principal_manifest",
        "candidate", "candidate_family", "response_policy", "options",
        "schedule", "planned_calls", "comparison_class", "claim_scope",
        "rights_status", "training_admission"
    }
    if type(plan) is not dict or set(plan) != required:
        raise ValueError("invalid candidate plan fields")
    candidate = plan["candidate"]
    if type(candidate) is not dict or set(candidate) != {"name", "digest"}:
        raise ValueError("invalid candidate identity")
    if (candidate["name"] not in CANDIDATES
            or type(candidate["digest"]) is not str
            or HEX_SHA.fullmatch(candidate["digest"]) is None
            or plan["candidate_family"] != (
                "phi4_reasoning_plus" if candidate["name"] == CANDIDATES[0]
                else "glm47_flash_community_q3"
            )
            or plan["schema"] != PLAN_SCHEMA or plan["source_closure"] != closure()
            or type(plan["source_commit"]) is not str
            or HEX_COMMIT.fullmatch(plan["source_commit"]) is None
            or plan["historical_validation_manifest"] != EXPECTED_HISTORY
            or plan["historical_cause_manifest"] != CAUSE_MANIFEST
            or any(type(plan[p]) is not str or HEX_SHA.fullmatch(plan[p]) is None
                   for p in ("historical_cause_summary_sha256",
                             "historical_summary_sha256",
                             "historical_teacher_manifest",
                             "historical_principal_manifest"))
            or plan["response_policy"] != RESPONSE_POLICY
            or plan["options"] != OPTIONS or plan["schedule"] != schedule()
            or type(plan["planned_calls"]) is not int or plan["planned_calls"] != CALLS
            or plan["comparison_class"] !=
                "exposed_independent_per_candidate_advisory_diagnostic"
            or plan["claim_scope"] != "arithmetic_supervision_gate_not_general_coding_benchmark"
            or plan["rights_status"] != "QUARANTINED_NOT_ADMITTED"
            or plan["training_admission"] is not False):
        raise ValueError("candidate source or policy drift")
    return plan


def check_parents(plan, teacher_run, baseline_run, validation_run, cause_run):
    history = parent_evidence(teacher_run, baseline_run, validation_run, cause_run)
    if (history["examiner_cause_summary_sha256"] != plan["historical_cause_summary_sha256"]
            or history["summary_sha256"] != plan["historical_summary_sha256"]
            or history["prior_teacher_manifest_identity"] !=
                plan["historical_teacher_manifest"]
            or history["prior_principal_manifest_identity"] !=
                plan["historical_principal_manifest"]):
        raise ValueError("historical evidence lineage drift")


def chat_body(plan, entry):
    if entry not in plan["schedule"]:
        raise ValueError("unknown recruitment schedule entry")
    case, lane = entry["case"], entry["lane"]
    model = plan["candidate"]["name"]
    if lane == "review":
        # Preserve the old examiner's exact system/user prompt text, but use
        # new explicitly frozen model, options and native thinking default.
        old = v.exam_body(
            {"principal": plan["candidate"], "exams": v.examiner_cases()}, case)
        messages = old["messages"]
    else:
        # Reuse prior probes/choice framing without relying on Qwen's verdicts.
        template = cause.chat_body({
            "principal": plan["candidate"], "schedule": cause.scheduled()
        }, entry)
        messages = template["messages"]
    return {
        "model": model, "stream": False, "keep_alive": 0,
        **RESPONSE_POLICY, "messages": messages, "options": OPTIONS,
    }


def evaluate(plan, index, call):
    if type(index) is not int or not 0 <= index < CALLS:
        raise ValueError("invalid candidate index")
    entry = plan["schedule"][index]
    lane, case = entry["lane"], entry["case"]
    model = plan["candidate"]
    body = chat_body(plan, entry)
    observed = r2.reconstruct(call, body, model)
    oracle = v.arithmetic_answer(case)
    row = {
        "index": index, "lane": lane, "item_id": case["id"],
        "candidate": model["name"], "model_digest": model["digest"],
        "request_sha256": mtc.sha(mtc.canonical(body)), "call": call,
        "answer": None, "oracle_match": False,
        "feedback": None, "verdict": None, "classification": None,
        "choice": None, "choice_match": False,
        "suggestion_oracle_match": False,
    }
    if lane == "solve":
        row["answer"] = mtc.parse_answer(observed) if observed is not None else None
        row["oracle_match"] = row["answer"] == oracle if row["answer"] is not None else False
    elif lane == "review":
        feedback = r2.feedback_of(observed, model) if observed is not None else None
        if feedback is not None:
            row["feedback"] = feedback
            row["verdict"] = feedback["verdict"]
            truth = v.truth(case)
            row["classification"] = (
                "uncertain" if row["verdict"] == "uncertain" else
                "TP" if truth == "correct" and row["verdict"] == "correct" else
                "FP" if truth == "incorrect" and row["verdict"] == "correct" else
                "FN" if truth == "correct" else "TN"
            )
            row["suggestion_oracle_match"] = feedback["suggested_answer"] == oracle
    else:
        parsed = cause.choice_of(observed, model["name"]) if observed is not None else None
        if parsed is not None:
            row["choice"] = parsed["choice"]
            row["answer"] = parsed["answer"]
            selected = "A" if case["candidate_a"] == oracle else "B"
            row["choice_match"] = parsed["choice"] == selected
            row["oracle_match"] = parsed["answer"] == oracle
    return row


def report(rows):
    solve = [x for x in rows if x["lane"] == "solve"]
    review = [x for x in rows if x["lane"] == "review"]
    choice = [x for x in rows if x["lane"] == "choice"]
    count = lambda label: sum(x["classification"] == label for x in review)
    return {
        "planned": CALLS, "attempted": len(rows),
        "candidate": rows[0]["candidate"] if rows else None,
        "solve_cases": len(solve), "solve_valid": sum(x["answer"] is not None for x in solve),
        "solve_correct": sum(x["oracle_match"] for x in solve),
        "review_cases": len(review), "review_valid": sum(x["verdict"] is not None for x in review),
        "true_positive": count("TP"), "true_negative": count("TN"),
        "false_positive": count("FP"), "false_negative": count("FN"),
        "review_uncertain": count("uncertain"), "review_invalid": count(None),
        "review_suggestion_correct": sum(x["suggestion_oracle_match"] for x in review),
        "choice_cases": len(choice), "choice_valid": sum(x["choice"] is not None for x in choice),
        "choice_correct": sum(x["choice_match"] for x in choice),
        "choice_answer_correct": sum(x["oracle_match"] for x in choice),
        "pair_both_approved": sum(
            review[2*i]["verdict"] == review[2*i+1]["verdict"] == "correct"
            for i in range(len(review)//2)
        ),
        "comparison_class": "exposed_independent_per_candidate_advisory_diagnostic",
        "training_admission": False, "rights_status": "QUARANTINED_NOT_ADMITTED",
        "automatic_promotion": False,
    }


def authorization_value(plan, identity, approval_path, capture_path):
    return {
        "schema": AUTH_SCHEMA, "approved_by": OWNER,
        "request_identity": identity,
        "approval_file": str(Path(approval_path).resolve(strict=False)),
        "capture_out": str(Path(capture_path).resolve(strict=False)),
        "candidate": plan["candidate"], "planned_calls": CALLS,
        "historical_validation_manifest": EXPECTED_HISTORY,
        "historical_cause_manifest": CAUSE_MANIFEST,
        "one_use": True, "training_admission": False,
        "authorization": AUTHORIZATION,
    }


def read_authorization(plan, identity, approval_path, capture_path):
    p = Path(approval_path)
    if p.is_symlink():
        raise ValueError("symlink authorization rejected")
    try:
        approved = p.resolve(strict=True)
    except (RuntimeError, FileNotFoundError):
        raise ValueError("exact-scope owner approval absent") from None
    info = approved.parent.stat()
    if not approved.parent.is_dir() or (
            os.name == "posix" and (
                info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077)):
        raise ValueError("unsafe owner approval directory")
    raw = mtc.checked_file(approved)
    if mtc.strict_json(raw) != authorization_value(plan, identity, approved, capture_path):
        raise ValueError("exact-scope owner authorization mismatch")
    return raw


def execution_claim(approval_raw, identity, approval_path, capture_path):
    return {
        "schema": CLAIM_SCHEMA, "authorization_sha256": mtc.sha(approval_raw),
        "request_identity": identity, "approval_file": str(Path(approval_path).resolve(strict=False)),
        "capture_out": str(Path(capture_path).resolve(strict=False)),
        "one_use": True,
    }


def plan_command(args):
    reject_historical_output(args.out, args.teacher_run, args.baseline_run,
                             args.validation_run, args.cause_run)
    value = validate_plan(value_of_plan(args.candidate, args.teacher_run,
                                        args.baseline_run, args.validation_run, args.cause_run))
    out = mtc.create_private(args.out)
    raw = mtc.canonical(value)
    mtc.write_new(out / "plan.json", raw)
    print("recruitment_request_identity=" + mtc.sha(raw))
    print("candidate=" + args.candidate)
    print("planned_calls=16; solve=4; review=8; choice=4; inference=0; training_admission=0")


def capture_command(args):
    for path in (args.out, args.owner_approval):
        reject_historical_output(path, args.teacher_run, args.baseline_run,
                                 args.validation_run, args.cause_run)
    raw = mtc.checked_file(args.plan)
    if mtc.sha(raw) != args.request_identity:
        raise ValueError("frozen recruitment request identity mismatch")
    plan = validate_plan(mtc.strict_json(raw))
    auth_raw = read_authorization(
        plan, args.request_identity, args.owner_approval, args.out)
    check_parents(plan, args.teacher_run, args.baseline_run, args.validation_run, args.cause_run)
    if mtc.model_sha(mtc.inventory(), plan["candidate"]["name"]) != plan["candidate"]["digest"]:
        raise ValueError("installed candidate digest drift before capture")
    if Path(args.out).exists() or Path(args.out).is_symlink():
        raise FileExistsError("capture output already exists")
    approval_path = Path(args.owner_approval).resolve(strict=True)
    claim_path = approval_path.with_name(approval_path.name + ".execution-claim.json")
    claim = mtc.canonical(execution_claim(
        auth_raw, args.request_identity, approval_path, args.out))
    mtc.write_new(claim_path, claim)  # Consumption happens BEFORE inference.
    out = mtc.create_private(args.out)
    for name, data in (
        ("plan.json", raw), ("owner-authorization.json", auth_raw),
        ("execution-claim.json", claim), ("attempts.jsonl", b"")
    ):
        mtc.write_new(out / name, data)
    rows = []
    for i, entry in enumerate(plan["schedule"]):
        call = r2.record_call(chat_body(plan, entry), plan["candidate"])
        row = evaluate(plan, i, call)
        mtc.append_line(out / "attempts.jsonl", row)
        rows.append(row)
        if call["error"] in ("model_digest_drift", "model_digest_check_failed",
                             "model_identity_mismatch"):
            raise ValueError("candidate identity drift; preserve partial archive")
    validate_rows(plan, rows)
    mtc.write_new(out / "summary.json", mtc.canonical(report(rows)))
    names = ("plan.json", "owner-authorization.json",
             "execution-claim.json", "attempts.jsonl", "summary.json")
    manifest = {"schema": MANIFEST_SCHEMA, "members": {
        p: mtc.sha(mtc.checked_file(out / p)) for p in names
    }}
    mb = mtc.canonical(manifest)
    mtc.write_new(out / "manifest.json", mb)
    print("recruitment_manifest_identity=" + mtc.sha(mb))
    print("candidate=" + plan["candidate"]["name"])
    print("training_admission=0; no_promotion; previous_archives_unchanged")


def validate_rows(plan, rows):
    if type(rows) is not list or len(rows) > CALLS:
        raise ValueError("invalid recruitment attempt count")
    for i, row in enumerate(rows):
        if type(row) is not dict or row != evaluate(plan, i, row["call"]):
            raise ValueError("recruitment record reconstruction mismatch")


def verify_command(args):
    out = Path(args.run)
    raw = mtc.checked_file(out / "manifest.json")
    if mtc.sha(raw) != args.manifest_identity:
        raise ValueError("external recruitment manifest identity mismatch")
    manifest = mtc.strict_json(raw)
    members = {
        "plan.json", "owner-authorization.json", "execution-claim.json",
        "attempts.jsonl", "summary.json",
    }
    if (type(manifest) is not dict or set(manifest) != {"schema", "members"}
            or manifest["schema"] != MANIFEST_SCHEMA
            or type(manifest["members"]) is not dict
            or set(manifest["members"]) != members
            or {p.name for p in out.iterdir()} != members | {"manifest.json"}):
        raise ValueError("invalid recruitment manifest members")
    for name, expected in manifest["members"].items():
        if mtc.sha(mtc.checked_file(out / name)) != expected:
            raise ValueError("recruitment member digest mismatch")
    pr = mtc.checked_file(out / "plan.json")
    plan = validate_plan(mtc.strict_json(pr))
    auth_raw = mtc.checked_file(out / "owner-authorization.json")
    auth = mtc.strict_json(auth_raw)
    if (type(auth) is not dict or type(auth.get("approval_file")) is not str or
            auth != authorization_value(
                plan, mtc.sha(pr), auth["approval_file"], out)):
        raise ValueError("invalid archived recruitment authorization")
    if mtc.checked_file(out / "execution-claim.json") != mtc.canonical(
        execution_claim(auth_raw, mtc.sha(pr), auth["approval_file"], out)
    ):
        raise ValueError("invalid one-use recruitment claim")
    check_parents(plan, args.teacher_run, args.baseline_run, args.validation_run, args.cause_run)
    rows = mtc.read_attempts(out / "attempts.jsonl")
    validate_rows(plan, rows)
    if len(rows) != CALLS or mtc.checked_file(out / "summary.json") != mtc.canonical(report(rows)):
        raise ValueError("recruitment archive incomplete")
    print("verified_recruitment_recorded=" + args.manifest_identity)
    print("candidate=" + plan["candidate"]["name"])
    print("claims=exposed_advisory_only; no_inference_on_verify; training_admission=0")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MTC-TEACHER-RECRUITMENT-001")
    sub = parser.add_subparsers(dest="command", required=True)
    def ancestors(p):
        p.add_argument("--teacher-run", type=Path, required=True)
        p.add_argument("--baseline-run", type=Path, required=True)
        p.add_argument("--validation-run", type=Path, required=True)
        p.add_argument("--cause-run", type=Path, required=True)
    p = sub.add_parser("plan")
    ancestors(p)
    p.add_argument("--candidate", choices=CANDIDATES, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("capture")
    ancestors(p)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--request-identity", required=True)
    p.add_argument("--owner-approval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("verify")
    ancestors(p)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--manifest-identity", required=True)
    args = parser.parse_args(argv)
    try:
        {"plan": plan_command, "capture": capture_command,
         "verify": verify_command}[args.command](args)
    except (OSError, ValueError, TypeError, KeyError, mtc.TransportError) as err:
        parser.exit(2, "MTC-TEACHER-RECRUITMENT-001: " +
                    type(err).__name__ + ": " + str(err)[:150] + "\n")


if __name__ == "__main__":
    main()
