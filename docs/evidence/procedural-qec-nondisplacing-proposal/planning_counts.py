#!/usr/bin/env python3
"""Read-only historical-source schedule planning; no model or experiment run."""
import argparse
from collections import defaultdict
import hashlib
import types
import json
from pathlib import Path
import subprocess
import sys


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--historical-source", type=Path, required=True)
    parser.add_argument("--proposal", type=Path)
    args = parser.parse_args()
    reference = "d01ed20a11213761836ae9f022c9f1ca38c3780b"
    audit_digest = "sha256:00894610c3ebb2d61590a6320aeedd029fdfaae7b66344c137632d097a1dfa53"
    rel = "docs/evidence/procedural-qec-control-order/audit.py"
    audit_path = args.repository / rel
    expected = subprocess.run(["git", "--no-replace-objects", "-C", str(args.repository), "show",
                               reference + ":" + rel], capture_output=True,
                              check=True, timeout=60).stdout
    if sha(expected) != audit_digest or audit_path.read_bytes() != expected:
        raise RuntimeError("audit differs from independently pinned baseline")
    # Execute the verified bytes directly, never a cached .pyc or import shadow.
    audit = types.ModuleType("historical_audit")
    audit.__file__ = str(audit_path.resolve())
    exec(compile(expected, audit.__file__, "exec"), audit.__dict__)
    retained = audit.read_receipt(audit.HERE / "comparison.json")
    retained_record = audit.parse(retained)
    payload = audit.run_worker(args.historical_source, "historical")
    audit.validate_payload(payload, "historical")
    rows = payload["dataset"]["examples"]
    def count(selected):
        groups = defaultdict(lambda: {"visits": 0, "target_tokens": 0,
                                      "input_tokens": 0})
        for row in selected:
            length = len(row["text"].encode("utf-8")) + 2
            if row["split"] != "train" or length > 64:
                raise RuntimeError("nontraining or overlength planning row")
            group = groups[row["kind"] + "/" + row["mode"]]
            group["visits"] += 1
            group["target_tokens"] += len(row["answer"].encode("utf-8")) + 1
            group["input_tokens"] += length
        totals = {key: sum(value[key] for value in groups.values())
                  for key in ("visits", "target_tokens", "input_tokens")}
        return {"by_kind_and_mode": dict(groups), "totals": totals,
                "original_row_sequence_identity": sha(canonical(
                    [row["example_identity"] for row in selected])),
                "normalized_row_sequence_identity": sha(canonical(
                    [audit.normalized(row) for row in selected]))}
    cells = []
    for seed in (0, 1):
        ranked = {}
        for kind, mode in (("fold-sum", "final"), ("gf3", "final"),
                           ("fold-sum", "prefix"), ("gf3", "prefix"),
                           ("fold-sum", "trace"), ("gf3", "trace")):
            group = [row for row in rows if (row["kind"], row["mode"]) == (kind, mode)]
            ranked[kind, mode] = sorted(group, key=lambda row: sha(canonical(
                {"seed": seed, "item": row["example_identity"]})))
        primary = [rows[index] for index in payload["schedules"]["v2"][seed * 2]]
        if len(primary) != 32768:
            raise RuntimeError("historical primary length mismatch")
        primary_mathematical_identity = sha(canonical(
            [(row["prompt"], row["answer"]) for row in primary]))
        expected_primary = retained_record["cells"][seed * 2]
        if (expected_primary["seed"] != seed
                or expected_primary["arm"] != "final-only"
                or primary_mathematical_identity != expected_primary["historical_sequence_identity"]):
            raise RuntimeError("historical primary sequence commitment mismatch")
        for arm in ("extra-final", "extra-procedural"):
            counters = defaultdict(int)
            extra = []
            interleaved = []
            for macro in range(4096):
                kind = "fold-sum" if macro % 2 == 0 else "gf3"
                mode = "prefix" if (macro // 2) % 2 == 0 else "trace"
                key = (kind, "final") if arm == "extra-final" else (kind, mode)
                group = ranked[key]
                batch = [group[(counters[key] + j) % len(group)] for j in range(8)]
                counters[key] += 8
                extra.extend(batch)
                interleaved.extend(primary[macro * 8:(macro + 1) * 8])
                interleaved.extend(batch)
            recovered = [row for start in range(0, len(interleaved), 16)
                         for row in interleaved[start:start + 8]]
            if recovered != primary:
                raise RuntimeError("primary ordered subsequence changed")
            primary_count, extra_count, all_count = map(count, (primary, extra, interleaved))
            if all_count["totals"]["target_tokens"] > 1048576:
                raise RuntimeError("existing scored-target bound exceeded")
            cells.append({"seed": seed, "arm": arm,
                          "primary": primary_count, "extra": extra_count,
                          "combined": all_count, "native_updates": 8192,
                          "original_audit_primary_sequence_identity": primary_mathematical_identity,
                          "padded_encoded_positions": 8192 * 8 * 64,
                          "model_input_positions": 8192 * 8 * 63,
                          "within_existing_target_ceiling": True})
    audit.verify_source(args.historical_source, "historical")
    if audit_path.read_bytes() != expected:
        raise RuntimeError("audit changed during planning")
    if any(name.split(".")[0] in ("torch", "numpy") for name in sys.modules):
        raise RuntimeError("unexpected numerical import")
    result = {"classification": "pre-outcome mathematical planning; not research outcomes",
              "implementation_status": "proposal only; no protocol or request frozen",
              "planning_script_identity": sha(Path(__file__).read_bytes()),
              "proposal_identity": sha(args.proposal.read_bytes()) if args.proposal else None,
              "audit_reference_commit": reference,
              "audit_script_identity": sha(expected),
              "original_comparison_receipt_identity": sha(retained),
              "historical_producer_commit": audit.PINS["historical"]["commit"],
              "historical_dataset_identity": audit.PINS["historical"]["dataset"],
              "historical_manifest_identity": audit.PINS["historical"]["manifest"],
              "recipe": "4096 macro-rounds; historical eight-row primary then eight-row extra; fold-prefix, gf3-prefix, fold-trace, gf3-trace cadence; per-mode procedural and per-kind final cursors; historical hash ranks; pad_to64",
              "count_semantics": "input_tokens includes BOS and EOS but excludes PAD; target_tokens counts UTF-8 answer bytes plus EOS; model inputs shift off last encoded position",
              "cells": cells,
              "aggregate": {"native_training_updates": 32768,
                            "training_row_visits": 262144,
                            "training_target_tokens": sum(cell["combined"]["totals"]["target_tokens"] for cell in cells)},
              "limitations": ["No prospective-current-source row mapping implemented or certified", "No model calls, training, workflow launch, native replay or new freeze", "No wall-time, RSS, FLOP, energy or learning estimate", "Only historical source reconstruction was executed; complete old/current audit not rerun"]}
    sys.stdout.buffer.write(canonical(result))


if __name__ == "__main__":
    main()
