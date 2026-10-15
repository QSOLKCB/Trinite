"""Source-bound native composition and complete replay for auxiliary conformance.

The original updater/scorer/tensor reader are reused. Only the tiny engineering
profile can create requests or states; the proposed research profile is blocked.
"""
from pathlib import Path
import math
import os
import re
import stat
import resource
import time

from ..comparison_checkpoint import progress, restore_tensors, tensor_payload
from ..contracts import (ContractError, ModelConfig, exact_keys, identity,
                         json_bytes, parse_json, require_identity)
from ..model import Decoder
from ..observation import BundleObserver
from ..retention_v2 import evidence
from ..retention_v2.budget.plan import BudgetRetentionPlan
from ..tokenizer import ByteTokenizer
from ..training import (TrainingState, create_optimizer, evaluate, model_identity,
                        require_environment, run_steps)
from ..procedural_qec_v3 import experiment as v3
from .inputs import revalidate
from .scheduler import coverage, schedule, schedule_receipt
from .spec import (ARMS, SEEDS, assess_training_counts, cell_name, cells,
    diagnostic_profile, diagnostic_rows, metadata_bytes, metadata_limit, profile_settings, report_assessment,
    schema, settings, specification)

OUTPUTS = evidence.OUTPUTS
ROOT_ARTIFACTS = (*evidence.ROOT_ARTIFACTS, "inputs.json")
ROOT = Path(__file__).resolve().parents[3]
FAILURE_SCHEMA = "trinite.procedural-qec-auxiliary-failure.v1"
FAILED_MAX_FILES = 128
FAILED_MAX_BYTES = 128 * 1024 * 1024


def output_path(path, inputs):
    """Reject source destinations before writes, including dot/symlink aliases."""
    path = evidence.safe_path(path).resolve(strict=False)
    for source in (ROOT, inputs.repository, inputs.historical_source):
        source = evidence.safe_path(source).resolve(strict=True)
        if path == source or source in path.parents:
            raise ContractError("auxiliary outputs must be outside producer source checkouts")
    return path


def error_text(exc):
    # JSON can expand each character to six bytes (control escapes); keep four
    # receipts plus cell names below the 64 KiB root error inventory limit.
    name = type(exc).__name__.encode("utf-8", "backslashreplace").decode("utf-8")[:128]
    message = str(exc).encode("utf-8", "backslashreplace").decode("utf-8")
    return name + ": " + message[:1800] + (" [truncated]" if len(message) > 1800 else "")


def failure_record(error):
    if type(error) is not str or not 1 <= len(error) <= 2048:
        raise ContractError("invalid auxiliary failure text")
    return {"schema": FAILURE_SCHEMA, "error": error,
            "learning_adequacy": False, "phase5_ready": False}


def retain_failure(root, error):
    """Retain failures even when admission failed before train_cell made a dir."""
    root = evidence.safe_path(root)
    root.mkdir(parents=True, exist_ok=True)
    expected = failure_record(error)
    path = root / "failure.json"
    if path.exists():
        if evidence.read_bytes(path, 16 * 1024) != json_bytes(expected):
            raise ContractError("auxiliary failure receipt differs from worker error")
    else:
        evidence.write(path, expected)


def worker_errors(root):
    errors = parse_json(evidence.read_bytes(Path(root) / "worker-errors.json", 64 * 1024), canonical=True)
    if type(errors) is not dict or not set(errors).issubset({cell_name(*c) for c in cells()}):
        raise ContractError("invalid auxiliary error inventory")
    for error in errors.values():
        failure_record(error)
    return errors


def failed_commitments(root, error):
    """Bind every bounded partial byte without asserting completed custody."""
    root = evidence.safe_path(root)
    if not root.is_dir():
        raise ContractError("missing auxiliary failed cell directory")
    allowed_dirs = {"provenance", "provenance/artifacts", "provenance/artifacts/sha256",
        "provenance/artifact_records", "provenance/artifact_records/sha256",
        "provenance/events", "provenance/events/sha256"}
    leaf = re.compile(r"provenance/(?:artifacts/sha256/[0-9a-f]{64}|"
                      r"(?:artifact_records|events)/sha256/[0-9a-f]{64}\.json)\Z")
    allowed_files = set(evidence.ARTIFACTS) | {"failure.json"}
    result, total, entries = {}, 0, 0
    def scan_error(error):
        raise ContractError("cannot enumerate auxiliary failed artifacts") from error
    for directory, dirs, names in os.walk(root, followlinks=False, onerror=scan_error):
        entries += len(dirs) + len(names)
        if entries > FAILED_MAX_FILES:
            raise ContractError("auxiliary failed artifact count exceeded")
        for name in dirs:
            path = evidence.safe_path(Path(directory) / name)
            if path.relative_to(root).as_posix() not in allowed_dirs or not path.is_dir():
                raise ContractError("unexpected auxiliary failed artifact directory")
        for name in names:
            path = evidence.safe_path(Path(directory) / name)
            relative = path.relative_to(root).as_posix()
            if (relative not in allowed_files and not leaf.fullmatch(relative)
                    or not stat.S_ISREG(path.stat().st_mode)):
                raise ContractError("unexpected auxiliary failed artifact file")
            raw = evidence.read_bytes(path)
            total += len(raw)
            if total > FAILED_MAX_BYTES:
                raise ContractError("auxiliary failed artifact bytes exceeded")
            result[relative] = identity(raw)
    if evidence.read_bytes(root / "failure.json", 16 * 1024) != json_bytes(failure_record(error)):
        raise ContractError("auxiliary failed receipt differs from worker error")
    return dict(sorted(result.items()))


def sources():
    # Preserve the already-reviewed bounded stable-tree traversal. It includes
    # every new package module and all transitive numerical/admission imports.
    result = v3.sources()
    root = Path(__file__).resolve().parents[3]
    for name in ("scripts/run_procedural_qec_auxiliary.py",
                 "scripts/archive_procedural_qec_auxiliary.py",
                 ".github/workflows/procedural-qec-auxiliary.yml",
                 "docs/evidence/procedural-qec-control-order/audit.py",
                 "docs/evidence/procedural-qec-control-order/comparison.json",
                 "docs/evidence/procedural-qec-control-order/comparison.json.sha256"):
        result[name] = identity(evidence.read_bytes(root / name, 1024 * 1024))
    if len(json_bytes(result)) > 96 * 1024:
        raise ContractError("auxiliary source receipt exceeds request budget")
    return dict(sorted(result.items()))


def candidate_plan_for(seed, run_profile):
    """Build and validate immutable plan data only; no model or native update."""
    cell_name(seed, ARMS[0])
    s = profile_settings(run_profile)
    cfg = ModelConfig(context_length=s["context_length"], blocks=s["blocks"],
        width=s["width"], heads=s["heads"], feed_forward_width=s["feed_forward_width"])
    return BudgetRetentionPlan(model=cfg, seed=seed, steps=s["steps"], batch_size=8,
        validation_every=s["validation_every"], max_seconds=s["max_seconds"],
        max_target_tokens=s["max_target_tokens"], **specification()["optimizer"])


def plan_for(seed, run_profile):
    settings(run_profile)  # Execution admission remains disabled for research.
    return candidate_plan_for(seed, run_profile)


def input_commitments(inputs):
    revalidate(inputs)
    return {"mapping_receipt_identity": identity(json_bytes(inputs.receipt)),
        "historical_dataset_identity": identity(inputs.historical_dataset_raw),
        "historical_manifest_identity": identity(inputs.historical_manifest_raw),
        "dataset_identity": identity(inputs.dataset_raw),
        "manifest_identity": identity(inputs.manifest_raw)}


def encoded_items(inputs):
    revalidate(inputs)
    tokenizer = ByteTokenizer(64)
    encoded = {}
    for e in inputs.items:
        if e["split"] != "train":
            raise ContractError("auxiliary inputs must all be admitted training rows")
        value = tokenizer.encode(e["text"], answer_start=len(e["prompt"].encode()), pad_to=64)
        encoded[e["example_identity"]] = {"identity": e["example_identity"],
            "ids": value.input_ids, "mask": value.attention_mask, "loss": value.loss_mask}
    return encoded


def state_for(seed, arm, run_profile, inputs):
    cell_name(seed, arm)
    plan = plan_for(seed, run_profile)  # Fail closed before any model call.
    require_environment()
    encoded = encoded_items(inputs)
    selected = schedule(inputs, seed, arm, settings(run_profile)["macro_rounds"])
    visits = [encoded[e["example_identity"]] for e in selected]
    if sum(sum(row["loss"]) for row in visits) > plan.max_target_tokens:
        raise ContractError("auxiliary schedule exceeds target budget")
    # Reuse the established unique, train-only engineering diagnostic view.
    selected_unique = [e for mode in ("final", "prefix", "trace")
                       for e in diagnostic_rows(inputs.items, run_profile, mode)]
    unique = [encoded[e["example_identity"]] for e in selected_unique]
    if len({r["identity"] for r in unique}) != len(unique):
        raise ContractError("diagnostic rows must be unique")
    model = Decoder(plan.model, lane="dense", seed=seed)
    return TrainingState(plan, "dense", model, create_optimizer(model, plan),
        {"train": visits, "validation": unique}, identity(inputs.dataset_raw),
        identity(inputs.manifest_raw), model_identity(model), evaluate(model, unique, 8))


def request_core(run_profile, inputs):
    s = settings(run_profile)
    return {"schema": schema("request", run_profile),
        "run_profile": run_profile, "specification": specification(),
        "specification_identity": identity(json_bytes(specification())),
        "source": sources(), "environment": require_environment(),
        "inputs": input_commitments(inputs),
        "schedules": {cell_name(seed, arm): schedule_receipt(
            inputs, seed, arm, s["macro_rounds"]) for seed, arm in cells()}}


def prepare_request(path, run_profile, inputs):
    settings(run_profile)
    path = output_path(path, inputs)
    core = request_core(run_profile, inputs)
    return {"request_identity": evidence.freeze(path, core), "planned_cells": 4,
            "run_profile": run_profile, "scientific_protocol_frozen": False}


def read_request(path, expected_identity, inputs):
    raw = evidence.read_bytes(path, 128 * 1024)
    record = parse_json(raw, canonical=True)
    return evidence.read_frozen(path, expected_identity,
                               request_core(record.get("run_profile"), inputs))


def context(state, seed, arm, run_profile, request_identity, inputs):
    require_identity(request_identity)
    if state.config != plan_for(seed, run_profile) or state.lane != "dense":
        raise ContractError("auxiliary state/plan mismatch")
    expected = encoded_items(inputs)
    selected = schedule(inputs, seed, arm, settings(run_profile)["macro_rounds"])
    if state.data["train"] != [expected[e["example_identity"]] for e in selected]:
        raise ContractError("auxiliary state schedule/encoding mismatch")
    unique = [expected[e["example_identity"]] for mode in ("final", "prefix", "trace")
              for e in diagnostic_rows(inputs.items, run_profile, mode)]
    if state.data["validation"] != unique:
        raise ContractError("auxiliary unique diagnostic view mismatch")
    commitments = input_commitments(inputs)
    if (state.dataset_identity != commitments["dataset_identity"]
            or state.manifest_identity != commitments["manifest_identity"]):
        raise ContractError("auxiliary state admission mismatch")
    return {"schema": schema("checkpoint", run_profile),
        "seed": seed, "arm": arm, "run_profile": run_profile,
        "request_identity": request_identity,
        "specification_identity": identity(json_bytes(specification())),
        "source": sources(), "environment": require_environment(),
        "inputs": commitments, "plan": state.config.to_dict(),
        "initialization_identity": state.initialization_identity,
        "initial_train_loss_hex": state.initial_train_loss,
        "schedule": schedule_receipt(inputs, seed, arm, settings(run_profile)["macro_rounds"])}


def snapshot(state, seed, arm, run_profile, request_identity, inputs):
    core = context(state, seed, arm, run_profile, request_identity, inputs)
    payload = tensor_payload(state)
    metadata = metadata_bytes({**core, "model_identity": model_identity(state.model),
        "step": state.step, "cursor": state.cursor, "target_tokens": state.target_tokens,
        "history": state.history, "validation": state.validation}, run_profile)
    return payload, metadata


def restore(payload, metadata, seed, arm, run_profile, request_identity,
            expected_identities, inputs):
    settings(run_profile)
    if (type(payload) is not bytes or type(metadata) is not bytes
            or len(payload) > 32 * 1024 * 1024 or len(metadata) > metadata_limit(run_profile)
            or (identity(payload), identity(metadata)) != expected_identities):
        raise ContractError("auxiliary checkpoint external identities/type/size mismatch")
    record = parse_json(metadata, canonical=True)
    state = state_for(seed, arm, run_profile, inputs)
    core = context(state, seed, arm, run_profile, request_identity, inputs)
    exact_keys(record, set(core) | {"model_identity", "step", "cursor", "target_tokens",
                                   "history", "validation"}, "auxiliary checkpoint")
    if any(json_bytes(record[k]) != json_bytes(v) for k, v in core.items()):
        raise ContractError("auxiliary checkpoint source/input/arm/plan mismatch")
    state.step, state.cursor, state.target_tokens = record["step"], record["cursor"], record["target_tokens"]
    state.history, state.validation = record["history"], record["validation"]
    progress(state)
    return restore_tensors(state, payload, record["model_identity"])


def report_for(state, inputs, seed, arm, run_profile, request_identity, seconds_hex, rss):
    settings(run_profile)  # Reject research before any scoring/model access.
    items = inputs.items
    for mode in ("final", "prefix", "trace"):
        if diagnostic_rows(items, run_profile, mode) != v3.scoring_view(
                items, diagnostic_profile(run_profile), mode):
            raise ContractError("authoritative scorer diagnostic view differs")
    scores, predictions = v3.scores_for(state.model, items, diagnostic_profile(run_profile))
    losses = v3.diagnostic_losses(state.model, items, diagnostic_profile(run_profile), 8)
    selected = schedule(inputs, seed, arm, settings(run_profile)["macro_rounds"])
    exposed = {e["example_identity"] for e in selected}
    mapping = {e["example_identity"]: e for e in items}
    slices = {}
    for mode in ("prefix", "trace"):
        for row in predictions[mode]:
            item = mapping[row["example_identity"]]
            group = slices.setdefault(item["kind"] + "/" + mode,
                {"visited": {"correct": 0, "examples": 0},
                 "unvisited": {"correct": 0, "examples": 0}})
            target = group["visited" if row["example_identity"] in exposed else "unvisited"]
            target["examples"] += 1
            target["correct"] += int(row["correct"])
    for group in slices.values():
        for target in group.values():
            target["accuracy_defined"] = target["examples"] > 0
    return ({**context(state, seed, arm, run_profile, request_identity, inputs),
        "schema": schema("report", run_profile), "outcome": "completed",
        "model_identity": model_identity(state.model), "steps": state.step,
        "target_tokens": state.target_tokens,
        "mixed_all_modes_loss_hex": evaluate(state.model, state.data["validation"], 8),
        "final_only_loss_hex": losses["final"]["all_loss_hex"],
        "loss_by_mode_and_task": losses, "auxiliary_visited_unvisited": slices,
        "schedule_coverage": coverage(inputs, selected, arm),
        "training_wall_seconds_hex": seconds_hex, "peak_process_rss_kib": rss,
        "scores": scores, "exposure": v3.exposure(state.data["train"], items),
        **report_assessment(run_profile, scores)}, {"predictions": predictions})


def train_cell(root, seed, arm, run_profile, request, request_identity, inputs):
    settings(run_profile)
    root = output_path(root, inputs)
    _, request_raw = read_request(request, request_identity, inputs)
    root.mkdir(parents=True, exist_ok=False)
    try:
        state = state_for(seed, arm, run_profile, inputs)
        started = time.perf_counter()
        run_steps(state)
        seconds = time.perf_counter() - started
        payload, metadata = snapshot(state, seed, arm, run_profile, request_identity, inputs)
        (root / "tensors.safetensors").write_bytes(payload)
        (root / "metadata.json").write_bytes(metadata)
        evidence.write(root / "steps.json", {"steps": state.history, "train_diagnostics": state.validation})
        report, predictions = report_for(state, inputs, seed, arm, run_profile,
            request_identity, seconds.hex(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        evidence.write(root / "report.json", report)
        evidence.write(root / "predictions.json", predictions)
        read_request(request, request_identity, inputs)
        observer = BundleObserver(root / "provenance")
        observer.record("procedural-qec-auxiliary-" + run_profile, inputs={
            "request.json": request_raw, "dataset.json": inputs.dataset_raw,
            "manifest.json": inputs.manifest_raw, "inputs.json": json_bytes(inputs.receipt)},
            outputs={name: evidence.read_bytes(root / name) for name in OUTPUTS})
        evidence.write(root / "verification.json", observer.finalize())
        return report
    except Exception as exc:
        retain_failure(root, error_text(exc))
        raise


def verification_core(root, request_identity, run_profile="engineering"):
    settings(run_profile)
    errors = worker_errors(root)
    records = {}
    for cell in cells():
        name = cell_name(*cell)
        directory = Path(root) / name
        if name in errors:
            records[name] = {"outcome": "failed", "artifacts": failed_commitments(directory, errors[name])}
        else:
            if (directory / "failure.json").exists() or (directory / "failure.json").is_symlink():
                raise ContractError("unexpected failure receipt in completed auxiliary cell")
            records[name] = {"outcome": "completed",
                "artifacts": evidence.commitments(directory, evidence.ARTIFACTS)}
    return {"schema": schema("verification-request", run_profile),
        "request_identity": request_identity,
        "root_artifacts": evidence.commitments(root, ROOT_ARTIFACTS), "cells": records}


def read_verification(path, expected_identity, root, request_identity, run_profile="engineering"):
    settings(run_profile)
    require_identity(expected_identity)
    raw = evidence.read_bytes(path, 128 * 1024)
    if identity(raw) != expected_identity:
        raise ContractError("auxiliary external verification identity mismatch")
    record = parse_json(raw, canonical=True)
    if json_bytes(record) != json_bytes(verification_core(root, request_identity, run_profile)):
        raise ContractError("auxiliary artifacts differ from external commitments")
    return record


def verify_cell(root, seed, arm, run_profile, request, request_identity, inputs, commitments):
    settings(run_profile)
    _, request_raw = read_request(request, request_identity, inputs)
    evidence.check_commitments(root, commitments, evidence.ARTIFACTS)
    bound = evidence.read_bound(root, {"request.json": request_raw,
        "dataset.json": inputs.dataset_raw, "manifest.json": inputs.manifest_raw,
        "inputs.json": json_bytes(inputs.receipt)})
    state = restore(bound["tensors.safetensors"], bound["metadata.json"], seed, arm,
        run_profile, request_identity, (commitments["tensors.safetensors"],
                                       commitments["metadata.json"]), inputs)
    if state.step != state.config.steps:
        raise ContractError("auxiliary final checkpoint was not reached")
    report = parse_json(bound["report.json"], canonical=True)
    wall = float.fromhex(report["training_wall_seconds_hex"])
    if (not math.isfinite(wall) or not 0 < wall <= state.config.max_seconds
            or wall.hex() != report["training_wall_seconds_hex"]):
        raise ContractError("invalid auxiliary resource report")
    rss = report["peak_process_rss_kib"]
    if type(rss) is not int or rss < 1:
        raise ContractError("invalid auxiliary RSS")
    expected_report, predictions = report_for(state, inputs, seed, arm, run_profile,
                                              request_identity, wall.hex(), rss)
    if (bound["report.json"] != json_bytes(expected_report)
            or bound["predictions.json"] != json_bytes(predictions)
            or bound["steps.json"] != json_bytes({"steps": state.history,
                                                  "train_diagnostics": state.validation})):
        raise ContractError("auxiliary fresh predictions/loss/history differ")
    replayed = state_for(seed, arm, run_profile, inputs)
    run_steps(replayed)
    if snapshot(replayed, seed, arm, run_profile, request_identity, inputs) != (
            bound["tensors.safetensors"], bound["metadata.json"]):
        raise ContractError("complete native auxiliary model/AdamW/history replay differs")
    evidence.check_commitments(root, commitments, evidence.ARTIFACTS)
    return {"seed": seed, "arm": arm, "outcome": "completed", "report": report,
        "report_identity": identity(bound["report.json"]), "checks": {
            "closed_custody": True, "request_and_source_binding": True,
            "safe_tensor_structure": True, "scheduled_visits_and_targets": True,
            "final_predictions_and_loss": True, "complete_optimizer_replay": True}}


def summary_for(root, req, request, request_identity, verification, verification_identity, inputs, errors):
    settings(req["run_profile"])
    v = read_verification(verification, verification_identity, root, request_identity, req["run_profile"])
    evidence.check_commitments(root, v["root_artifacts"], ROOT_ARTIFACTS)
    if (evidence.read_bytes(Path(root) / "request.json") != evidence.read_bytes(request)
            or evidence.read_bytes(Path(root) / "inputs.json") != json_bytes(inputs.receipt)):
        raise ContractError("auxiliary retained input/request mismatch")
    if errors != worker_errors(root):
        raise ContractError("auxiliary worker error copy differs from retained errors")
    records = []
    for seed, arm in cells():
        name = cell_name(seed, arm)
        try:
            if name in errors:
                raise ContractError(errors[name])
            records.append(verify_cell(Path(root) / name, seed, arm, req["run_profile"],
                request, request_identity, inputs, v["cells"][name]["artifacts"]))
        except Exception as exc:  # Retain ordinary backend/resource failures, not cancellation.
            records.append({"seed": seed, "arm": arm, "outcome": "failed",
                            "error": error_text(exc)})
    # Recheck all cells, including failed partial outputs, after every replay.
    read_verification(verification, verification_identity, root, request_identity, req["run_profile"])
    return assemble_summary(req["run_profile"], records, errors, request_identity,
                            verification_identity)


def assemble_summary(run_profile, records, errors, request_identity, verification_identity):
    """Shared pure report assembly. It neither runs nor certifies an experiment.

    Native summary_for admits execution and replays every cell first. Synthetic
    tests exercise the dormant candidate branch without making target outcomes.
    """
    profile_settings(run_profile)
    require_identity(request_identity)
    require_identity(verification_identity)
    names = {cell_name(*c) for c in cells()}
    if (type(records) is not list or len(records) != len(names)
            or {cell_name(r["seed"], r["arm"]) for r in records} != names
            or type(errors) is not dict or not set(errors).issubset(names)):
        raise ContractError("auxiliary summary requires exact four-cell inventory")
    complete = not errors and all(r["outcome"] == "completed" for r in records)
    pairing_errors = []
    if complete:
        required_checks = {"closed_custody", "request_and_source_binding", "safe_tensor_structure",
            "scheduled_visits_and_targets", "final_predictions_and_loss", "complete_optimizer_replay"}
        for r in records:
            if (type(r.get("checks")) is not dict or set(r["checks"]) != required_checks
                    or any(value is not True for value in r["checks"].values())):
                raise ContractError("auxiliary completed cell lacks full verification checks")
        for seed in SEEDS:
            a, b = [next(r["report"] for r in records if r["seed"] == seed
                         and r["arm"] == arm) for arm in ARMS]
            for key in ("initialization_identity", "plan", "inputs"):
                if a[key] != b[key]:
                    pairing_errors.append("seed " + str(seed) + " differs in " + key)
            # A matched plan cannot substitute for the exact original primary sequence.
            for key in ("original_row_sequence_identity", "prospective_row_sequence_identity",
                        "normalized_row_sequence_identity"):
                if a["schedule"]["primary"][key] != b["schedule"]["primary"][key]:
                    pairing_errors.append("seed " + str(seed) + " differs in " + key)
    result = {"schema": schema("summary", run_profile),
        "request_identity": request_identity, "verification_request_identity": verification_identity,
        "run_profile": run_profile, "cells": records, "worker_errors": errors,
        "pairing_errors": pairing_errors,
        "outcome": "completed" if complete and not pairing_errors else "failed",
        "training_gate_evaluated": False, "training_gate_pass": False,
        "selected_arm": None, "held_out_scored": False, "learning_adequacy": False,
        "phase5_ready": False, "format_selection_eligible": False, "release_ready": False,
        "evidence_class": "training-only-exploratory-candidate" if run_profile == "research"
                          else "engineering-conformance-only"}
    if run_profile == "research":
        result.update(contrasts={}, positive_fold_contrast_both_seeds=False)
        if complete and not pairing_errors:
            for r in records:
                expected = report_assessment(run_profile, r["report"]["scores"])
                if any(r["report"].get(key) != value for key, value in expected.items()):
                    raise ContractError("candidate report gate/claim differs from diagnostic counts")
            criteria = assess_training_counts({cell_name(r["seed"], r["arm"]): {
                task: {key: row[key] for key in ("correct", "examples")}
                for task, row in r["report"]["scores"]["final"]["tasks"].items()}
                for r in records})
            result.update(training_gate_evaluated=True,
                training_gate_pass=criteria["treatment_training_gate_pass"],
                positive_fold_contrast_both_seeds=criteria["positive_fold_contrast_both_seeds"],
                contrasts=criteria["contrasts"])
    return result


def run_matrix(output, request, request_identity, verification, inputs):
    # Every destination must pass before admission, directory creation or model work.
    output = output_path(output, inputs)
    verification = output_path(verification, inputs)
    req, request_raw = read_request(request, request_identity, inputs)
    if (verification == output or output in verification.parents
            or verification in output.parents or verification.exists()):
        raise ContractError("verification request must be new and outside output tree")
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_bytes(request_raw)
    evidence.write(output / "inputs.json", inputs.receipt)
    errors = {}
    for seed, arm in cells():
        try:
            train_cell(output / cell_name(seed, arm), seed, arm, req["run_profile"],
                       request, request_identity, inputs)
        except Exception as exc:
            error = error_text(exc)
            retain_failure(output / cell_name(seed, arm), error)
            errors[cell_name(seed, arm)] = error
    evidence.write(output / "worker-errors.json", errors)
    core = verification_core(output, request_identity, req["run_profile"])
    evidence.write(verification, core)
    v_identity = identity(json_bytes(core))
    summary = summary_for(output, req, request, request_identity, verification, v_identity, inputs, errors)
    evidence.write(output / "summary.json", summary)
    return {"outcome": summary["outcome"], "summary_identity": identity(json_bytes(summary)),
        "verification_request_identity": v_identity, "cells": 4,
        "training_gate_pass": summary["training_gate_pass"], "phase5_ready": False,
        "evidence_class": summary["evidence_class"]}


def verify_only(output, request, request_identity, verification, verification_identity, inputs):
    req, _ = read_request(request, request_identity, inputs)
    errors = parse_json(evidence.read_bytes(Path(output) / "worker-errors.json"), canonical=True)
    summary = summary_for(output, req, request, request_identity, verification,
                          verification_identity, inputs, errors)
    retained = evidence.read_bytes(Path(output) / "summary.json", 16 * 1024 * 1024)
    if retained != json_bytes(summary):
        raise ContractError("auxiliary retained summary differs from fresh replay")
    return {"outcome": summary["outcome"], "summary_identity": identity(retained),
        "verification_request_identity": verification_identity,
        "replayed_cells": sum(c["outcome"] == "completed" for c in summary["cells"]),
        "learning_adequacy": False, "phase5_ready": False,
        "evidence_class": summary["evidence_class"]}
