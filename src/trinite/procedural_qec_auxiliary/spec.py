"""Reviewable candidate settings and an explicitly bounded engineering profile.

The prospective recipe is data, not authorization or a scientific protocol
freeze. No code path in this increment permits its research execution.
"""
from ..contracts import ContractError, integer

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
        "primary": "exact mapped historical v2 final-only eight-row batches",
        "extra": "fold-prefix,gf3-prefix,fold-trace,gf3-trace; own eight-row cursors",
        "normalization": "unchanged per-native-update answer-and-EOS target mean",
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


def settings(run_profile):
    if type(run_profile) is not str or run_profile != "engineering":
        raise ContractError("research freeze/execution disabled; scientific and resource approval required")
    return specification()["engineering"]


def assess_training_counts(counts):
    """Pure candidate-rule arithmetic, never a certificate of research execution.

    Every cell and exact unique diagnostic denominator is mandatory. A caller
    can use this to test acceptance logic before a scientific protocol exists;
    engineering model reports deliberately never invoke it.
    """
    if type(counts) is not dict or set(counts) != {cell_name(*c) for c in cells()}:
        raise ContractError("candidate criteria require every one of four cells")
    gates = specification()["gates"]
    results = {}
    for seed, arm in cells():
        name = cell_name(seed, arm)
        tasks = counts[name]
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
        results[name] = {"training_task_failures": sorted(failures),
                         "training_gate_pass": not failures}
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
