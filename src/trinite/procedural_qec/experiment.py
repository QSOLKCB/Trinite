"""QEC-guided trace vs final-only CPU learning experiment.

All scored samples are development/training rows. The frozen native AdamW
updater, model, tokenizer and final-answer scorer are reused without patches.
"""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import math
import resource
import time

from ..comparison_checkpoint import progress, restore_tensors, tensor_payload
from ..contracts import (ContractError, ModelConfig, exact_keys, identity, json_bytes,
                         parse_json, require_identity)
from ..foundations import score_examples
from ..model import Decoder
from ..observation import BundleObserver
from ..retention_v2 import evidence
from ..retention_v2.budget.plan import BudgetRetentionPlan
from ..tokenizer import ByteTokenizer
from ..training import (TrainingState, create_optimizer, evaluate, model_identity,
                        require_environment, run_steps)
from . import data

PROTOCOL_IDENTITY = "sha256:0777f713e24a88daf3b83d105cc1a210fcb717f7bbd40375acaed7688c73d8e3"
PROTOCOL_COMMIT = "8a6116251e1d270dc39623dadfa58c18911a0be6"
ARMS = ("final-only", "procedural")
SEEDS = (0, 1)
OUTPUTS = ("report.json", "predictions.json", "steps.json",
           "tensors.safetensors", "metadata.json")


def protocol():
    raw = evidence.read_bytes(Path(__file__).parent / "protocol.json", 16384)
    if identity(raw) != PROTOCOL_IDENTITY:
        raise ContractError("frozen procedural/QEC protocol changed")
    value = parse_json(raw, canonical=True)
    if (value["schema"] != "trinite.procedural-qec-protocol.v1"
            or value["donor"]["pin_identity"] != data.verify_pin()["pin_identity"]):
        raise ContractError("procedural QEC protocol/donor mismatch")
    return value


def sources():
    root = Path(__file__).resolve().parents[3]
    paths = (Path(__file__), root / "src/trinite/procedural_qec/data.py",
             root / "src/trinite/procedural_qec/protocol.json",
             root / "src/trinite/procedural_qec/__init__.py",
             root / "scripts/run_procedural_qec.py",
             root / "src/trinite/model.py", root / "src/trinite/training.py",
             root / "src/trinite/quantizer.py",
             root / "src/trinite/comparison.py",
             root / "src/trinite/comparison_checkpoint.py",
             root / "src/trinite/retention_v2/budget/plan.py",
             root / "src/trinite/retention_v2/evidence.py",
             root / "src/trinite/foundations.py", root / "requirements.lock")
    result = data.source_receipt()
    result.update({str(p.relative_to(root)): identity(evidence.read_bytes(p, 1024 * 1024))
                   for p in paths})
    return dict(sorted(result.items()))


def cells():
    return tuple((seed, arm) for seed in SEEDS
                 for arm in (ARMS if seed == 0 else ARMS[::-1]))


def cell_name(seed, arm):
    if (type(seed) is not int or type(arm) is not str
            or (seed, arm) not in cells()):
        raise ContractError("unknown fixed procedural research cell")
    return "procedural-qec-" + str(seed) + "-" + arm


def settings(run_profile):
    if type(run_profile) is not str or run_profile not in ("smoke", "research"):
        raise ContractError("unsupported procedural run profile")
    return protocol()["profiles"][run_profile]


def plan_for(seed, run_profile):
    if seed not in SEEDS or type(seed) is not int:
        raise ContractError("invalid procedural seed")
    s = settings(run_profile)
    cfg = ModelConfig(context_length=s["context_length"], blocks=s["blocks"],
        width=s["width"], heads=s["heads"], feed_forward_width=s["feed_forward_width"])
    return BudgetRetentionPlan(model=cfg, seed=seed, steps=s["steps"],
        batch_size=s["batch_size"], validation_every=s["validation_every"],
        max_seconds=s["max_seconds"], max_target_tokens=s["max_target_tokens"],
        **protocol()["optimizer"])


def examples():
    raw, manifest = data.dataset()
    data.audit_bytes(raw, manifest)
    return raw, manifest, parse_json(raw, canonical=True)["examples"]


def ranked(items, seed, label):
    if not items:
        raise ContractError("empty required procedural stream: " + label)
    return sorted(items, key=lambda e: identity(json_bytes(
        {"seed": seed, "item": e["example_identity"]})))


def schedule(items, seed, arm, steps):
    """Same eight updates/slots, fixed arithmetic and primary final visits.

    One auxiliary slot changes from extra original-answer examples to exact
    local step or complete trace targets. This is a recipe, not token matching.
    """
    cell_name(seed, arm)
    groups = {(kind, mode): ranked(
        [e for e in items if e["kind"] == kind and e["mode"] == mode],
        seed, kind + "/" + mode)
        for kind, mode in (("arithmetic", "final"), ("fold-sum", "final"),
                           ("gf3", "final"), ("fold-sum", "prefix"),
                           ("fold-sum", "trace"), ("gf3", "prefix"),
                           ("gf3", "trace"))}
    result = []
    for i in range(steps):
        result.extend(groups["arithmetic", "final"][(4 * i + j) % len(groups["arithmetic", "final"])]
                      for j in range(4))
        result.extend(groups["fold-sum", "final"][(2 * i + j) % len(groups["fold-sum", "final"])]
                      for j in range(2))
        result.append(groups["gf3", "final"][i % len(groups["gf3", "final"])])
        kind = "fold-sum" if i % 2 == 0 else "gf3"
        if arm == "final-only":
            mode = "final"
        else:
            mode = "prefix" if (i // 2) % 2 == 0 else "trace"
        selected = groups[kind, mode]
        result.append(selected[(i // 2) % len(selected)])
    if len(result) != steps * 8:
        raise ContractError("procedural schedule must have eight rows per update")
    return result


def exposure(rows, items):
    lookup = {e["example_identity"]: (e["kind"], e["mode"]) for e in items}
    total = {kind: {mode: {"visits": 0, "target_tokens": 0, "input_tokens": 0}
                    for mode in data.MODES} for kind in data.KIND}
    for row in rows:
        k, mode = lookup[row["identity"]]
        target = total[k][mode]
        target["visits"] += 1
        target["target_tokens"] += sum(row["loss"])
        target["input_tokens"] += sum(row["mask"])
    return total


def state_for(seed, arm, run_profile, items=None):
    cell_name(seed, arm)
    require_environment()
    plan = plan_for(seed, run_profile)
    if items is None:
        _, _, items = examples()
    selected = [e for e in items if e["split"] == "train"]
    if len(selected) != len(items):
        raise ContractError("procedural model cannot consume nontraining data")
    tokenizer = ByteTokenizer(plan.model.context_length)
    encoded = {}
    for e in selected:
        value = tokenizer.encode(e["text"], answer_start=len(e["prompt"].encode()))
        encoded[e["example_identity"]] = {"identity": e["example_identity"],
            "ids": value.input_ids, "mask": value.attention_mask,
            "loss": value.loss_mask}
    schedule_items = schedule(selected, seed, arm, plan.steps)
    visits = [encoded[e["example_identity"]] for e in schedule_items]
    n_targets = sum(sum(e["loss"]) for e in visits)
    if n_targets > plan.max_target_tokens:
        raise ContractError("procedural scored targets exceed frozen budget")
    unique = [encoded[e["example_identity"]] for e in selected]
    # Diagnostic evaluations are training-only. No test/validation labels.
    if run_profile == "smoke":
        first = {}
        for e in selected:
            key = (e["kind"], e["mode"], e["task"])
            first.setdefault(key, e["example_identity"])
        unique = [encoded[key] for key in first.values()]
    model = Decoder(plan.model, lane="dense", seed=seed)
    source_data, admission = data.dataset()
    return TrainingState(plan, "dense", model, create_optimizer(model, plan),
        {"train": visits, "validation": unique}, identity(source_data),
        identity(admission), model_identity(model),
        evaluate(model, unique, plan.batch_size))


def request_core(run_profile):
    settings(run_profile)
    raw, manifest, _ = examples()
    return {"schema": "trinite.procedural-qec-request.v1",
        "run_profile": run_profile, "protocol": protocol(),
        "protocol_identity": PROTOCOL_IDENTITY, "protocol_commit": PROTOCOL_COMMIT,
        "source": sources(), "environment": require_environment(),
        "dataset_identity": identity(raw), "manifest_identity": identity(manifest)}


def freeze_request(path, run_profile):
    raw = json_bytes({**request_core(run_profile),
        "created_at": datetime.now(timezone.utc).isoformat()})
    path = evidence.safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"request_identity": identity(raw), "protocol_identity": PROTOCOL_IDENTITY,
            "planned_cells": 4, "run_profile": run_profile}


def read_request(path, expected_identity):
    require_identity(expected_identity)
    raw = evidence.read_bytes(path, 128 * 1024)
    if identity(raw) != expected_identity:
        raise ContractError("procedural request differs from independent commitment")
    record = parse_json(raw, canonical=True)
    expected = request_core(record["run_profile"])
    exact_keys(record, set(expected) | {"created_at"}, "procedural experiment request")
    if any(json_bytes(record[k]) != json_bytes(v) for k, v in expected.items()):
        raise ContractError("procedural request/source/protocol/environment/data mismatch")
    try:
        if datetime.fromisoformat(record["created_at"]).utcoffset() is None:
            raise ValueError("timezone required")
    except (ValueError, TypeError) as exc:
        raise ContractError("invalid procedural request timestamp") from exc
    return record, raw


def context(state, seed, arm, run_profile, request_identity):
    if state.config != plan_for(seed, run_profile) or state.lane != "dense":
        raise ContractError("procedural state/plan mismatch")
    require_identity(request_identity)
    return {"schema": "trinite.procedural-qec-checkpoint.v1",
        "seed": seed, "arm": arm, "run_profile": run_profile,
        "request_identity": request_identity, "protocol_identity": PROTOCOL_IDENTITY,
        "source": sources(), "environment": require_environment(),
        "plan": state.config.to_dict(), "dataset_identity": state.dataset_identity,
        "manifest_identity": state.manifest_identity,
        "initialization_identity": state.initialization_identity,
        "initial_train_loss_hex": state.initial_train_loss,
        "schedule_identity": identity(json_bytes(
            [e["identity"] for e in state.data["train"]]))}


def snapshot(state, seed, arm, run_profile, request_identity):
    raw = tensor_payload(state)
    metadata = json_bytes({**context(state, seed, arm, run_profile, request_identity),
        "model_identity": model_identity(state.model), "step": state.step,
        "cursor": state.cursor, "target_tokens": state.target_tokens,
        "history": state.history, "validation": state.validation})
    return raw, metadata


def restore(payload, metadata, seed, arm, run_profile, request_identity):
    record = parse_json(metadata, canonical=True)
    state = state_for(seed, arm, run_profile)
    core = context(state, seed, arm, run_profile, request_identity)
    exact_keys(record, set(core) | {"model_identity", "step", "cursor",
                                   "target_tokens", "history", "validation"},
               "procedural checkpoint")
    if any(json_bytes(record[k]) != json_bytes(v) for k, v in core.items()):
        raise ContractError("procedural checkpoint context mismatch")
    state.step, state.cursor, state.target_tokens = (
        record["step"], record["cursor"], record["target_tokens"])
    state.history, state.validation = record["history"], record["validation"]
    progress(state)
    return restore_tensors(state, payload, record["model_identity"])


def scoring_view(items, run_profile, mode):
    settings(run_profile)
    subset = [e for e in items if e["mode"] == mode and e["split"] == "train"]
    if run_profile == "research":
        return subset
    selected = {}
    for e in subset:
        key = (e["task"], e["mode"])
        current = selected.get(key)
        if current is None or e["semantic_identity"] < current:
            selected[key] = e["semantic_identity"]
    return [e for e in subset if e["semantic_identity"] == selected[e["task"], e["mode"]]]


def scores_for(model, items, run_profile):
    before = model_identity(model)
    scores, predictions = {}, {}
    for mode in data.MODES:
        scored = scoring_view(items, run_profile, mode)
        cap = settings(run_profile)["generation_cap" if mode != "trace" else "diagnostic_cap"]
        if run_profile == "smoke" and mode == "trace":
            cap = 24  # Integration checks complete traces, not just two tokens.
        if not scored:
            raise ContractError("procedural scoring view empty: " + mode)
        value, rows = score_examples(model, scored, "train", cap)
        scores[mode] = value
        predictions[mode] = rows
    if model_identity(model) != before:
        raise ContractError("procedural scoring mutated model")
    return scores, predictions


def report_for(state, items, seed, arm, run_profile, request_identity, seconds_hex, rss):
    scores, predictions = scores_for(state.model, items, run_profile)
    exp = exposure(state.data["train"], items)
    original = scores["final"]["tasks"]
    failures = sorted(t for t, v in original.items()
                      if v["correct"] * 10 < v["examples"] * 9)
    return ({**context(state, seed, arm, run_profile, request_identity),
        "schema": "trinite.procedural-qec-report.v1", "outcome": "completed",
        "model_identity": model_identity(state.model),
        "steps": state.step, "target_tokens": state.target_tokens,
        "final_train_loss_hex": evaluate(state.model, state.data["validation"],
                                         state.config.batch_size),
        "training_wall_seconds_hex": seconds_hex, "peak_process_rss_kib": rss,
        "scores": scores, "exposure": exp, "training_task_failures": failures,
        "held_out_scored": False, "learning_adequacy": False,
        "phase5_ready": False, "format_selection_eligible": False,
        "scope": "training-only QEC/fold procedural supervision, no fresh reasoning claim"},
        {"predictions": predictions})


def train_cell(root, seed, arm, run_profile, request, request_identity, items, dataset_raw, manifest_raw):
    _, request_raw = read_request(request, request_identity)
    root = evidence.safe_path(root)
    root.mkdir(parents=True, exist_ok=False)
    try:
        state = state_for(seed, arm, run_profile, items)
        t0 = time.perf_counter()
        run_steps(state)
        seconds = time.perf_counter() - t0
        payload, metadata = snapshot(state, seed, arm, run_profile, request_identity)
        (root / "tensors.safetensors").write_bytes(payload)
        (root / "metadata.json").write_bytes(metadata)
        evidence.write(root / "steps.json",
                       {"steps": state.history, "train_diagnostics": state.validation})
        report, predictions = report_for(
            state, items, seed, arm, run_profile, request_identity,
            seconds.hex(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        evidence.write(root / "report.json", report)
        evidence.write(root / "predictions.json", predictions)
        read_request(request, request_identity)
        collector = BundleObserver(root / "provenance")
        collector.record("procedural-qec", inputs={
            "request.json": request_raw, "dataset.json": dataset_raw,
            "manifest.json": manifest_raw},
            outputs={name: evidence.read_bytes(root / name) for name in OUTPUTS})
        receipt = collector.finalize()
        evidence.write(root / "verification.json", receipt)
        return report
    except Exception as exc:
        evidence.write(root / "failure.json", {
            "schema": "trinite.procedural-qec-failure.v1",
            "error": type(exc).__name__ + ": " + str(exc),
            "learning_adequacy": False, "phase5_ready": False})
        raise


def verification_core(root, request_identity):
    return {"schema": "trinite.procedural-qec-verification-request.v1",
        "request_identity": request_identity,
        "root_artifacts": evidence.commitments(root, evidence.ROOT_ARTIFACTS),
        "cells": {cell_name(*c): evidence.commitments(
            Path(root) / cell_name(*c), evidence.ARTIFACTS) for c in cells()}}


def freeze_verification(path, root, request_identity):
    core = verification_core(root, request_identity)
    evidence.write(path, core)
    return identity(json_bytes(core))


def read_verification(path, expected_identity, root, request_identity):
    require_identity(expected_identity)
    raw = evidence.read_bytes(path, 128 * 1024)
    if identity(raw) != expected_identity:
        raise ContractError("procedural external verification identity mismatch")
    record = parse_json(raw, canonical=True)
    if json_bytes(record) != json_bytes(verification_core(root, request_identity)):
        raise ContractError("procedural cell/root artifacts differ from external commitments")
    return record


def verify_cell(root, seed, arm, run_profile, request, request_identity, items,
                dataset_raw, manifest_raw, commitments):
    _, request_raw = read_request(request, request_identity)
    evidence.check_commitments(root, commitments, evidence.ARTIFACTS)
    bound = evidence.read_bound(root, {
        "request.json": request_raw, "dataset.json": dataset_raw,
        "manifest.json": manifest_raw})
    state = restore(bound["tensors.safetensors"], bound["metadata.json"],
                    seed, arm, run_profile, request_identity)
    if state.step != state.config.steps:
        raise ContractError("procedural final checkpoint was not reached")
    record = parse_json(bound["report.json"], canonical=True)
    wall = float.fromhex(record["training_wall_seconds_hex"])
    if (not math.isfinite(wall) or not 0 < wall <= state.config.max_seconds
            or wall.hex() != record["training_wall_seconds_hex"]):
        raise ContractError("invalid procedural training resource report")
    rss = record["peak_process_rss_kib"]
    if type(rss) is not int or rss < 1:
        raise ContractError("invalid procedural RSS")
    expected_report, expected_predictions = report_for(
        state, items, seed, arm, run_profile, request_identity, wall.hex(), rss)
    if (bound["report.json"] != json_bytes(expected_report)
            or bound["predictions.json"] != json_bytes(expected_predictions)
            or bound["steps.json"] != json_bytes({
                "steps": state.history, "train_diagnostics": state.validation})):
        raise ContractError("fresh final scoring/history does not match retained evidence")
    # Complete native optimizer replay; no selected checkpoint or shortcut.
    replayed = state_for(seed, arm, run_profile, items)
    run_steps(replayed)
    if snapshot(replayed, seed, arm, run_profile, request_identity) != (
            bound["tensors.safetensors"], bound["metadata.json"]):
        raise ContractError("full native procedural optimizer/AdamW/history replay differs")
    evidence.check_commitments(root, commitments, evidence.ARTIFACTS)
    return {"seed": seed, "arm": arm, "outcome": "completed",
            "report_identity": identity(bound["report.json"]), "report": record,
            "checks": {"closed_custody": True, "source_request": True,
                       "safe_checkpoint": True, "original_answer_scoring": True,
                       "complete_optimizer_replay": True}}


def summary_for(root, req, request, request_identity, verification, verification_identity,
                *, errors):
    v = read_verification(verification, verification_identity, root, request_identity)
    evidence.check_commitments(root, v["root_artifacts"], evidence.ROOT_ARTIFACTS)
    if evidence.read_bytes(Path(root) / "request.json") != evidence.read_bytes(request):
        raise ContractError("procedural retained root request mismatch")
    raw, manifest, items = examples()
    records = []
    for seed, arm in cells():
        name = cell_name(seed, arm)
        try:
            if name in errors:
                raise ContractError(errors[name])
            records.append(verify_cell(Path(root) / name, seed, arm, req["run_profile"],
                                       request, request_identity, items, raw, manifest,
                                       v["cells"][name]))
        except (ContractError, OSError, ValueError, KeyError, OverflowError) as exc:
            records.append({"seed": seed, "arm": arm, "outcome": "failed",
                            "error": type(exc).__name__ + ": " + str(exc)})
    complete = not errors and all(r["outcome"] == "completed" for r in records)
    comparisons, pairing_errors = {}, []
    if complete:
        for seed in SEEDS:
            a, b = [next(r["report"] for r in records
                         if r["seed"] == seed and r["arm"] == arm) for arm in ARMS]
            for k in ("initialization_identity", "plan", "dataset_identity", "manifest_identity"):
                if a[k] != b[k]:
                    pairing_errors.append("seed " + str(seed) + " differs in " + k)
            if (a["exposure"]["arithmetic"] != b["exposure"]["arithmetic"]
                    or a["scores"]["final"]["examples"] != b["scores"]["final"]["examples"]):
                pairing_errors.append("seed " + str(seed) + " mismatched common learning/scoring")
            comparisons[str(seed)] = {task: {
                "control": a["scores"]["final"]["tasks"][task]["correct"],
                "procedural": b["scores"]["final"]["tasks"][task]["correct"],
                "examples": a["scores"]["final"]["tasks"][task]["examples"]}
                for task in sorted(a["scores"]["final"]["tasks"])}
    # A passing training-only joint result is still NOT fresh held-out adequacy.
    training_pass = (complete and not pairing_errors and req["run_profile"] == "research"
                     and all(not r["report"]["training_task_failures"] for r in records
                             if r["arm"] == "procedural"))
    return {"schema": "trinite.procedural-qec-summary.v1",
        "request_identity": request_identity,
        "verification_request_identity": verification_identity,
        "protocol_identity": PROTOCOL_IDENTITY, "run_profile": req["run_profile"],
        "outcome": "completed" if complete and not pairing_errors else "failed",
        "cells": records, "worker_errors": errors, "pairing_errors": pairing_errors,
        "training_gate_pass": training_pass, "contrasts": comparisons,
        "selected_arm": None, "held_out_scored": False, "learning_adequacy": False,
        "phase5_ready": False, "format_selection_eligible": False,
        "release_ready": False}


def run_matrix(output, request, request_identity, verification):
    req, raw_request = read_request(request, request_identity)
    raw, manifest, items = examples()
    output = evidence.safe_path(output)
    verification = evidence.safe_path(verification)
    if verification == output or output in verification.parents or verification in output.parents:
        raise ContractError("external verification request must be outside output tree")
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_bytes(raw_request)
    errors = {}
    for seed, arm in cells():
        try:
            train_cell(output / cell_name(seed, arm), seed, arm, req["run_profile"],
                       request, request_identity, items, raw, manifest)
        except Exception as exc:
            errors[cell_name(seed, arm)] = type(exc).__name__ + ": " + str(exc)
    evidence.write(output / "worker-errors.json", errors)
    v_identity = freeze_verification(verification, output, request_identity)
    summary = summary_for(output, req, request, request_identity,
                          verification, v_identity, errors=errors)
    evidence.write(output / "summary.json", summary)
    return {"outcome": summary["outcome"], "summary_identity": identity(json_bytes(summary)),
            "verification_request_identity": v_identity, "cells": len(cells()),
            "training_gate_pass": summary["training_gate_pass"],
            "held_out_scored": False, "phase5_ready": False}


def verify_only(output, request, request_identity, verification, verification_identity):
    req, _ = read_request(request, request_identity)
    output = evidence.safe_path(output)
    errors = parse_json(evidence.read_bytes(output / "worker-errors.json"), canonical=True)
    if type(errors) is not dict or not set(errors).issubset({cell_name(*c) for c in cells()}):
        raise ContractError("invalid procedural worker error inventory")
    summary = summary_for(output, req, request, request_identity,
                          verification, verification_identity, errors=errors)
    retained = evidence.read_bytes(output / "summary.json", 16 * 1024 * 1024)
    if retained != json_bytes(summary):
        raise ContractError("procedural retained aggregate differs from fresh replay")
    return {"outcome": summary["outcome"], "summary_identity": identity(retained),
            "verification_request_identity": verification_identity,
            "replayed_cells": sum(c["outcome"] == "completed" for c in summary["cells"]),
            "learning_adequacy": False, "phase5_ready": False}
