"""Reviewable candidate settings and an explicitly bounded engineering profile.

The prospective recipe is data, not authorization or a scientific protocol
freeze. No code path in this increment permits its research execution.
"""
from collections import Counter

from ..contracts import ContractError, integer, json_bytes

ARMS = ("extra-final", "extra-procedural")
SEEDS = (0, 1)
OPTIMIZER = {"learning_rate": "0.001", "beta1": "0.9", "beta2": "0.999",
             "optimizer_epsilon": "0.00000001", "weight_decay": "0.01",
             "gradient_clip": "1"}


def cells():
    return tuple((seed, arm) for seed in SEEDS
                 for arm in (ARMS if seed == 0 else ARMS[::-1]))


def cell_name(seed, arm):
    integer(seed, "auxiliary seed", 0, 1)
    if type(arm) is not str or arm not in ARMS:
        raise ContractError("unknown auxiliary arm")
    return "auxiliary-qec-" + str(seed) + "-" + arm


def specification():
    return {
        "schema": "trinite.procedural-qec-auxiliary-candidate.v1",
        "status": "engineering implementation; scientific protocol NOT frozen",
        "research_execution_enabled": False,
        "arms": list(ARMS), "seeds": list(SEEDS), "geometry_loss": "0",
        "optimizer": dict(OPTIMIZER), "batch_size": 8, "pad_to": 64,
        "checkpoint_metadata_byte_bounds": {"engineering": 1048576, "research": 8388608},
        "primary": "exact mapped historical v2 final-only eight-row batches",
        "extra": "fold-prefix,gf3-prefix,fold-trace,gf3-trace; own eight-row cursors",
        "normalization": "unchanged per-native-update answer-and-EOS target mean",
        "diagnostics": {"research": "every unique admitted training row by mode/task",
            "engineering": "one semantic family per training task and mode",
            "generation_cap": 8, "trace_cap": 24, "checkpoint_selection": False},
        "research_candidate": {"macro_rounds": 4096, "steps": 8192,
            "blocks": 2, "width": 64, "heads": 4, "feed_forward_width": 128,
            "context_length": 64, "validation_every": 2048,
            "max_seconds": 960, "max_target_tokens": 1048576},
        "engineering": {"macro_rounds": 8, "steps": 16,
            "blocks": 1, "width": 16, "heads": 2, "feed_forward_width": 32,
            "context_length": 64, "validation_every": 4,
            "max_seconds": 60, "max_target_tokens": 4096},
        "gates": {"add": [60, 66], "subtract": [60, 66],
            "multiply": [60, 66], "divide": [60, 66],
            "fold-sum": [1035, 1150], "gf3": [49, 54]},
        "contrast": "treatment fold count strictly exceeds paired control in BOTH seeds",
        "remaining_authorization": "reviewed scientific freeze and explicit host/time/storage/resources",
        "claims": "engineering conformance only; no learning or milestone conclusion"}


def profile_settings(run_profile):
    """Pure candidate configuration, never permission to construct a model."""
    if type(run_profile) is not str or run_profile not in ("engineering", "research"):
        raise ContractError("unknown auxiliary candidate profile")
    return specification()["research_candidate" if run_profile == "research" else "engineering"]


def metadata_limit(run_profile):
    profile_settings(run_profile)
    return specification()["checkpoint_metadata_byte_bounds"][run_profile]


def metadata_bytes(record, run_profile):
    raw = json_bytes(record)
    if len(raw) > metadata_limit(run_profile):
        raise ContractError("auxiliary checkpoint metadata exceeds byte budget")
    return raw


def schema(kind, run_profile):
    profile_settings(run_profile)
    if kind not in ("request", "checkpoint", "report", "verification-request", "summary"):
        raise ContractError("unknown auxiliary record kind")
    return "trinite.procedural-qec-auxiliary-" + (
        "research-" if run_profile == "research" else "") + kind + ".v1"


def diagnostic_profile(run_profile):
    profile_settings(run_profile)
    return "research" if run_profile == "research" else "smoke"


def diagnostic_rows(items, run_profile, mode):
    """Data-only unique train view shared by native state and checkpoint checks."""
    profile_settings(run_profile)
    if mode not in ("final", "prefix", "trace"):
        raise ContractError("unknown auxiliary diagnostic mode")
    train = [e for e in items if e["split"] == "train"]
    if len({e["example_identity"] for e in train}) != len(train):
        raise ContractError("diagnostic rows must be unique")
    subset = [e for e in train if e["mode"] == mode]
    if run_profile == "research":
        expected = {task: total for task, (_, total) in specification()["gates"].items()
                    if mode == "final" or task in ("fold-sum", "gf3")}
        if dict(Counter(e["task"] for e in subset)) != expected:
            raise ContractError("candidate diagnostic unique task denominators differ")
        return subset
    selected = {}
    for e in subset:
        task = e["task"]
        selected[task] = min(selected.get(task, e["semantic_identity"]), e["semantic_identity"])
    return [e for e in subset if e["semantic_identity"] == selected[e["task"]]]


def settings(run_profile):
    if type(run_profile) is not str or run_profile != "engineering":
        raise ContractError("research freeze/execution disabled; scientific and resource approval required")
    return profile_settings(run_profile)


def assess_training_counts(counts):
    """Pure candidate-rule arithmetic, never a certificate of research execution.

    Every cell and exact unique diagnostic denominator is mandatory. A caller
    can use this to test acceptance logic before a scientific protocol exists;
    engineering model reports deliberately never invoke it.
    """
    if type(counts) is not dict or set(counts) != {cell_name(*c) for c in cells()}:
        raise ContractError("candidate criteria require every one of four cells")
    gates = specification()["gates"]
    results = {cell_name(*cell): assess_cell_counts(counts[cell_name(*cell)])
               for cell in cells()}
    contrasts = {str(seed): {task:
        counts[cell_name(seed, ARMS[1])][task]["correct"]
        - counts[cell_name(seed, ARMS[0])][task]["correct"]
        for task in gates} for seed in SEEDS}
    return {"schema": "trinite.procedural-qec-auxiliary-candidate-criteria.v1",
        "cells": results, "contrasts": contrasts,
        "treatment_training_gate_pass": all(results[cell_name(seed, ARMS[1])]
            ["training_gate_pass"] for seed in SEEDS),
        "positive_fold_contrast_both_seeds": all(contrasts[str(seed)]["fold-sum"] > 0
                                                for seed in SEEDS),
        "evidence_verified": False, "learning_adequacy": False, "phase5_ready": False}


def assess_cell_counts(tasks):
    """Original exact six-task thresholds, with no claim about evidence custody."""
    gates = specification()["gates"]
    if type(tasks) is not dict or set(tasks) != set(gates):
        raise ContractError("candidate criteria require exact six-task inventory")
    failures = []
    for task, (threshold, total) in gates.items():
        row = tasks[task]
        if type(row) is not dict or set(row) != {"correct", "examples"}:
            raise ContractError("candidate criteria require exact correct/examples fields")
        integer(row["correct"], "correct", 0, total)
        if type(row["examples"]) is not int or row["examples"] != total:
            raise ContractError("candidate criteria require unique original task denominator")
        if row["correct"] < threshold:
            failures.append(task)
    return {"training_task_failures": sorted(failures), "training_gate_pass": not failures}


def report_assessment(run_profile, scores):
    """Pure dormant report plumbing; numerical callers must pass settings first."""
    profile_settings(run_profile)
    research = run_profile == "research"
    result = {"held_out_scored": False, "training_gate_evaluated": research,
        "learning_adequacy": False, "phase5_ready": False,
        "format_selection_eligible": False, "release_ready": False,
        "evidence_class": "training-only-exploratory-candidate" if research
                          else "engineering-conformance-only"}
    if research:
        if type(scores) is not dict or set(scores) != {"final", "prefix", "trace"}:
            raise ContractError("candidate report requires all diagnostic modes")
        for mode, value in scores.items():
            expected = {task: total for task, (_, total) in specification()["gates"].items()
                        if mode == "final" or task in ("fold-sum", "gf3")}
            if (type(value) is not dict or type(value.get("tasks")) is not dict
                    or set(value["tasks"]) != set(expected)
                    or type(value.get("examples")) is not int
                    or value["examples"] != sum(expected.values())):
                raise ContractError("candidate report unique diagnostic inventory differs")
            for task, total in expected.items():
                row = value["tasks"][task]
                if (type(row) is not dict or type(row.get("examples")) is not int
                        or row["examples"] != total):
                    raise ContractError("candidate report repeated/missing task denominator")
                integer(row.get("correct"), "diagnostic correct", 0, total)
            if (type(value.get("correct")) is not int or value["correct"] != sum(
                    row["correct"] for row in value["tasks"].values())):
                raise ContractError("candidate diagnostic aggregate contradicts task counts")
        result.update(assess_cell_counts({task: {key: row[key] for key in ("correct", "examples")}
            for task, row in scores["final"]["tasks"].items()}))
    return result
