"""Source-bound native composition and complete replay for auxiliary conformance.

The original updater/scorer/tensor reader are reused. Only the tiny engineering
profile can create requests or states; the proposed research profile is blocked.
"""
from pathlib import Path
import math
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
from .spec import ARMS, SEEDS, cell_name, cells, settings, specification

OUTPUTS = evidence.OUTPUTS
ROOT_ARTIFACTS = (*evidence.ROOT_ARTIFACTS, "inputs.json")


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


def plan_for(seed, run_profile):
    cell_name(seed, ARMS[0])
    s = settings(run_profile)
    cfg = ModelConfig(context_length=s["context_length"], blocks=s["blocks"],
        width=s["width"], heads=s["heads"], feed_forward_width=s["feed_forward_width"])
    return BudgetRetentionPlan(model=cfg, seed=seed, steps=s["steps"], batch_size=8,
        validation_every=s["validation_every"], max_seconds=s["max_seconds"],
        max_target_tokens=s["max_target_tokens"], **specification()["optimizer"])


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
                       for e in v3.scoring_view(inputs.items, "smoke", mode)]
    unique = [encoded[e["example_identity"]] for e in selected_unique]
    if len({r["identity"] for r in unique}) != len(unique):
        raise ContractError("diagnostic rows must be unique")
    model = Decoder(plan.model, lane="dense", seed=seed)
    return TrainingState(plan, "dense", model, create_optimizer(model, plan),
        {"train": visits, "validation": unique}, identity(inputs.dataset_raw),
        identity(inputs.manifest_raw), model_identity(model), evaluate(model, unique, 8))


def request_core(run_profile, inputs):
    s = settings(run_profile)
    return {"schema": "trinite.procedural-qec-auxiliary-request.v1",
        "run_profile": run_profile, "specification": specification(),
        "specification_identity": identity(json_bytes(specification())),
        "source": sources(), "environment": require_environment(),
        "inputs": input_commitments(inputs),
        "schedules": {cell_name(seed, arm): schedule_receipt(
            inputs, seed, arm, s["macro_rounds"]) for seed, arm in cells()}}


def prepare_request(path, run_profile, inputs):
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
              for e in v3.scoring_view(inputs.items, "smoke", mode)]
    if state.data["validation"] != unique:
        raise ContractError("auxiliary unique diagnostic view mismatch")
    commitments = input_commitments(inputs)
    if (state.dataset_identity != commitments["dataset_identity"]
            or state.manifest_identity != commitments["manifest_identity"]):
        raise ContractError("auxiliary state admission mismatch")
    return {"schema": "trinite.procedural-qec-auxiliary-checkpoint.v1",
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
    metadata = json_bytes({**core, "model_identity": model_identity(state.model),
        "step": state.step, "cursor": state.cursor, "target_tokens": state.target_tokens,
        "history": state.history, "validation": state.validation})
    if len(metadata) > 1024 * 1024:
        raise ContractError("auxiliary checkpoint metadata exceeds byte budget")
    return payload, metadata


def restore(payload, metadata, seed, arm, run_profile, request_identity,
            expected_identities, inputs):
    if (type(payload) is not bytes or type(metadata) is not bytes
            or len(payload) > 32 * 1024 * 1024 or len(metadata) > 1024 * 1024
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
    items = inputs.items
    scores, predictions = v3.scores_for(state.model, items, "smoke")
    losses = v3.diagnostic_losses(state.model, items, "smoke", 8)
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
        "schema": "trinite.procedural-qec-auxiliary-report.v1", "outcome": "completed",
        "model_identity": model_identity(state.model), "steps": state.step,
        "target_tokens": state.target_tokens,
        "mixed_all_modes_loss_hex": evaluate(state.model, state.data["validation"], 8),
        "final_only_loss_hex": losses["final"]["all_loss_hex"],
        "loss_by_mode_and_task": losses, "auxiliary_visited_unvisited": slices,
        "schedule_coverage": coverage(inputs, selected, arm),
        "training_wall_seconds_hex": seconds_hex, "peak_process_rss_kib": rss,
        "scores": scores, "exposure": v3.exposure(state.data["train"], items),
        "held_out_scored": False, "training_gate_evaluated": False,
        "learning_adequacy": False, "phase5_ready": False,
        "format_selection_eligible": False, "release_ready": False,
        "evidence_class": "engineering-conformance-only"}, {"predictions": predictions})


def train_cell(root, seed, arm, run_profile, request, request_identity, inputs):
    _, request_raw = read_request(request, request_identity, inputs)
    root = evidence.safe_path(root)
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
        observer.record("procedural-qec-auxiliary-engineering", inputs={
            "request.json": request_raw, "dataset.json": inputs.dataset_raw,
            "manifest.json": inputs.manifest_raw, "inputs.json": json_bytes(inputs.receipt)},
            outputs={name: evidence.read_bytes(root / name) for name in OUTPUTS})
        evidence.write(root / "verification.json", observer.finalize())
        return report
    except Exception as exc:
        evidence.write(root / "failure.json", {
            "schema": "trinite.procedural-qec-auxiliary-failure.v1",
            "error": type(exc).__name__ + ": " + str(exc),
            "learning_adequacy": False, "phase5_ready": False})
        raise


def verification_core(root, request_identity):
    return {"schema": "trinite.procedural-qec-auxiliary-verification-request.v1",
        "request_identity": request_identity,
        "root_artifacts": evidence.commitments(root, ROOT_ARTIFACTS),
        "cells": {cell_name(*c): evidence.commitments(Path(root) / cell_name(*c),
                                                    evidence.ARTIFACTS) for c in cells()}}


def read_verification(path, expected_identity, root, request_identity):
    require_identity(expected_identity)
    raw = evidence.read_bytes(path, 128 * 1024)
    if identity(raw) != expected_identity:
        raise ContractError("auxiliary external verification identity mismatch")
    record = parse_json(raw, canonical=True)
    if json_bytes(record) != json_bytes(verification_core(root, request_identity)):
        raise ContractError("auxiliary artifacts differ from external commitments")
    return record


def verify_cell(root, seed, arm, run_profile, request, request_identity, inputs, commitments):
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
    v = read_verification(verification, verification_identity, root, request_identity)
    evidence.check_commitments(root, v["root_artifacts"], ROOT_ARTIFACTS)
    if (evidence.read_bytes(Path(root) / "request.json") != evidence.read_bytes(request)
            or evidence.read_bytes(Path(root) / "inputs.json") != json_bytes(inputs.receipt)):
        raise ContractError("auxiliary retained input/request mismatch")
    if type(errors) is not dict or not set(errors).issubset({cell_name(*c) for c in cells()}):
        raise ContractError("invalid auxiliary error inventory")
    records = []
    for seed, arm in cells():
        name = cell_name(seed, arm)
        try:
            if name in errors:
                raise ContractError(errors[name])
            records.append(verify_cell(Path(root) / name, seed, arm, req["run_profile"],
                request, request_identity, inputs, v["cells"][name]))
        except (ContractError, OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            records.append({"seed": seed, "arm": arm, "outcome": "failed",
                            "error": type(exc).__name__ + ": " + str(exc)})
    complete = not errors and all(r["outcome"] == "completed" for r in records)
    pairing_errors = []
    if complete:
        for seed in SEEDS:
            a, b = [next(r["report"] for r in records if r["seed"] == seed
                         and r["arm"] == arm) for arm in ARMS]
            for key in ("initialization_identity", "plan", "inputs"):
                if a[key] != b[key]:
                    pairing_errors.append("seed " + str(seed) + " differs in " + key)
    return {"schema": "trinite.procedural-qec-auxiliary-summary.v1",
        "request_identity": request_identity, "verification_request_identity": verification_identity,
        "run_profile": req["run_profile"], "cells": records, "worker_errors": errors,
        "pairing_errors": pairing_errors,
        "outcome": "completed" if complete and not pairing_errors else "failed",
        "training_gate_evaluated": False, "training_gate_pass": False,
        "selected_arm": None, "held_out_scored": False, "learning_adequacy": False,
        "phase5_ready": False, "format_selection_eligible": False, "release_ready": False,
        "evidence_class": "engineering-conformance-only"}


def run_matrix(output, request, request_identity, verification, inputs):
    req, request_raw = read_request(request, request_identity, inputs)
    # Reject symlinks before resolving dot-segment aliases in the containment test.
    output = evidence.safe_path(output).resolve(strict=False)
    verification = evidence.safe_path(verification).resolve(strict=False)
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
            errors[cell_name(seed, arm)] = type(exc).__name__ + ": " + str(exc)
    evidence.write(output / "worker-errors.json", errors)
    core = verification_core(output, request_identity)
    evidence.write(verification, core)
    v_identity = identity(json_bytes(core))
    summary = summary_for(output, req, request, request_identity, verification, v_identity, inputs, errors)
    evidence.write(output / "summary.json", summary)
    return {"outcome": summary["outcome"], "summary_identity": identity(json_bytes(summary)),
        "verification_request_identity": v_identity, "cells": 4,
        "training_gate_pass": False, "phase5_ready": False,
        "evidence_class": "engineering-conformance-only"}


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
        "evidence_class": "engineering-conformance-only"}
