#!/usr/bin/env python3
"""Reconstruct the retained historical ordering failure, without model imports.

Evidence-only reader. It neither accepts supplied datasets as truth nor writes
to either producer. Run with Python 3.11+ and Git; no installed packages needed.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PINS = {
    "historical": {
        "commit": "93522bf8f18fe38970ceafdd2ace792f8922c382",
        "dataset": "sha256:120b95ac68825b50b92dc0210d1f142a30980bfe5acedcfafe6d8f3cdadf83c4",
        "manifest": "sha256:315168dd6be4909cead81a45a4c3b5b1abb125cc69f3ca773b5145f18b0ac412",
    },
    "current": {
        "commit": "cc996e272800a5c02667001f7c96c8d11cb513cc",
        "dataset": "sha256:9ea3223ebd099933f9c05a3ccb9db43e4a7d19e3c1edf5922c326c2eec16d944",
        "manifest": "sha256:528cb27db98781db8148dd4f029e9a392d0d2a786d393f2ef399dcbdad95ff49",
    },
}
RECEIPT_SHA256 = "7e0160ac9ca5c6241dd8001bbc8775c6dd1c51266755533705debd819d9a087b"
HERE = Path(__file__).resolve().parent
CELLS = tuple((seed, arm) for seed in (0, 1)
              for arm in ("final-only", "procedural"))
STEPS, BATCH, ROWS = 4096, 8, 3876
ROW_KEYS = {
    "answer", "carrier", "example_identity", "family_id", "generator_revision",
    "kind", "mode", "parent_example_identity", "prompt", "semantic_identity",
    "source_ids", "split", "task", "text", "transformation_lineage", "values",
    "verifier_outcome",
}
IGNORED_ROW_FIELDS = frozenset(("generator_revision", "example_identity"))


class AuditError(ValueError):
    """Evidence cannot establish the exact retained audit."""


def require(condition, message):
    # Do not use assert: rejection must survive python -O.
    if not condition:
        raise AuditError(message)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def identity(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def parse(raw):
    value = json.loads(raw, object_pairs_hook=unique_pairs)
    require(canonical(value) == raw, "noncanonical JSON bytes")
    return value


def read_receipt(path):
    raw = path.read_bytes()
    require(identity(raw) == "sha256:" + RECEIPT_SHA256, "retained receipt identity mismatch")
    sidecar = path.with_name(path.name + ".sha256").read_bytes()
    require(sidecar == (RECEIPT_SHA256 + "  comparison.json\n").encode(),
            "retained receipt sidecar mismatch")
    parse(raw)
    return raw


def git(root, *args, input_bytes=None):
    # Ignore ambient Git overrides and do not refresh/write the producer index.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    process = subprocess.run(
        ["git", "--no-replace-objects", "--literal-pathspecs", "-c", "core.fsmonitor=false",
         "-C", str(root), *args], input=input_bytes, capture_output=True, env=env,
        check=False, timeout=60)
    require(process.returncode == 0, "Git source check failed: " +
            process.stderr.decode(errors="replace").strip())
    return process.stdout


def verify_source(root, role):
    """Before imports: fixed commit, clean index/worktree, exact complete src."""
    root = Path(root).resolve(strict=True)
    require(git(root, "rev-parse", "--show-toplevel").decode().strip() == str(root),
            "source must be a Git checkout root")
    require(git(root, "rev-parse", "HEAD").decode().strip() == PINS[role]["commit"],
            role + " source commit mismatch")
    require(not git(root, "status", "--porcelain=v1", "--untracked-files=all"),
            role + " source checkout is dirty")
    entries = git(root, "ls-tree", "-rz", "HEAD", "--", "src").split(b"\0")
    expected = {}
    for entry in filter(None, entries):
        header, name = entry.split(b"\t", 1)
        mode, kind, blob = header.split()
        require(mode in (b"100644", b"100755") and kind == b"blob",
                "unsupported source tree member")
        expected[name.decode()] = blob
    require("src/trinite/procedural_qec/data.py" in expected, "missing producer source")
    require(not (root / "src").is_symlink(), "symlink producer source root")
    actual = set()
    for path in (root / "src").rglob("*"):
        require(not path.is_symlink(), "symlink in producer source")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        else:
            require(path.is_dir(), "nonregular source member")
    # Includes ignored files: bytecode, extensions or untracked import shadows
    # are refused rather than executed or silently discarded.
    require(actual == set(expected), "unexpected or missing source files")
    blobs = git(root, "cat-file", "--batch", input_bytes=b"\n".join(expected.values()) + b"\n")
    offset = 0
    for name, expected_blob in expected.items():
        end = blobs.index(b"\n", offset)
        blob, kind, size = blobs[offset:end].split()
        require(blob == expected_blob and kind == b"blob", "Git blob mismatch")
        size = int(size)
        start = end + 1
        require(blobs[start + size:start + size + 1] == b"\n", "truncated Git blob")
        require((root / name).read_bytes() == blobs[start:start + size],
                "source bytes differ from pinned Git blob: " + name)
        offset = start + size + 1
    require(offset == len(blobs), "extra Git blob output")
    return root


def extract_v2(source, contracts):
    """Compile only four pure functions and their two literal constants.

    The entire containing file was checked against its fixed commit first.
    Its Torch-dependent imports, training and checkpoint code never execute.
    """
    path = source / "src/trinite/procedural_qec/experiment.py"
    tree = ast.parse(path.read_bytes(), filename=str(path))
    wanted = {"cells", "cell_name", "ranked", "schedule"}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in wanted]
    require(len(functions) == len(wanted) and {n.name for n in functions} == wanted,
            "historical scheduler function inventory mismatch")
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ("ARMS", "SEEDS"):
                require(target.id not in constants, "duplicate scheduler constant")
                constants[target.id] = ast.literal_eval(node.value)
    require(constants == {"ARMS": ("final-only", "procedural"), "SEEDS": (0, 1)},
            "historical scheduler constants mismatch")
    namespace = {**constants, "identity": contracts.identity,
                 "json_bytes": contracts.json_bytes, "ContractError": contracts.ContractError}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["schedule"]


def worker(source, role):
    source = verify_source(source, role)
    sys.path.insert(0, str(source / "src"))
    from trinite import contracts
    from trinite.procedural_qec import data
    raw, manifest = data.dataset()
    require(identity(raw) == PINS[role]["dataset"], role + " regenerated dataset mismatch")
    require(identity(manifest) == PINS[role]["manifest"], role + " regenerated manifest mismatch")
    # Exercise original admission instead of trusting a claimed identity field.
    data.audit_bytes(raw, manifest)
    dataset = parse(raw)
    rows = dataset["examples"]
    row_map(dataset)
    indexes = {row["example_identity"]: i for i, row in enumerate(rows)}
    schedulers = {"v2": extract_v2(source, contracts)}
    if role == "current":
        from trinite.procedural_qec_v3.scheduler import schedule
        schedulers["v3"] = schedule
    schedules = {}
    for name, schedule in schedulers.items():
        schedules[name] = []
        for seed, arm in CELLS:
            selected = schedule(rows, seed, arm, STEPS)
            require(len(selected) == STEPS * BATCH, "schedule length mismatch")
            require(all(row == rows[indexes[row["example_identity"]]] for row in selected),
                    "scheduler returned an altered or unadmitted row")
            schedules[name].append([indexes[row["example_identity"]] for row in selected])
    require(not any(n.split(".")[0] in ("torch", "numpy") for n in sys.modules),
            "unexpected numerical dependency imported")
    verify_source(source, role)
    return {"dataset": dataset, "manifest": parse(manifest), "schedules": schedules}


def run_worker(source, role):
    # -I ignores ambient Python paths; -S excludes site packages; -B avoids
    # source writes. verify_source also refuses pre-existing bytecode caches.
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
         "--worker", role, "--source", str(source)],
        capture_output=True, check=False, timeout=300)
    require(process.returncode == 0, role + " reconstruction failed: " +
            process.stderr.decode(errors="replace").strip())
    require(len(process.stdout) <= 16 * 1024 * 1024, "worker evidence exceeds bound")
    return parse(process.stdout)


def row_map(dataset):
    require(type(dataset) is dict and set(dataset) == {"schema", "examples"},
            "dataset schema keys mismatch")
    require(dataset["schema"] == "trinite.procedural-qec-data.v2", "dataset schema mismatch")
    rows = dataset["examples"]
    require(type(rows) is list and len(rows) == ROWS, "dataset row count mismatch")
    result, ids = {}, set()
    for row in rows:
        require(type(row) is dict and set(row) == ROW_KEYS, "row schema keys mismatch")
        require(type(row["prompt"]) is str and type(row["answer"]) is str,
                "row mathematical key must contain strings")
        require(row["text"] == row["prompt"] + row["answer"] and row["split"] == "train",
                "row text or training split mismatch")
        key = (row["prompt"], row["answer"])
        require(key not in result, "non-bijective mathematical row mapping")
        ident = row["example_identity"]
        require(type(ident) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", ident),
                "invalid row identity")
        require(ident not in ids, "duplicate example identity")
        ids.add(ident)
        result[key] = row
    return result


def normalized(row):
    # Keep EVERY other field, including family/source/parent/lineage metadata.
    return {key: value for key, value in row.items() if key not in IGNORED_ROW_FIELDS}


def validate_payload(payload, role):
    require(type(payload) is dict and set(payload) == {"dataset", "manifest", "schedules"},
            "worker schema mismatch")
    require(identity(canonical(payload["dataset"])) == PINS[role]["dataset"],
            role + " dataset identity mismatch")
    require(identity(canonical(payload["manifest"])) == PINS[role]["manifest"],
            role + " manifest identity mismatch")
    mapping = row_map(payload["dataset"])
    expected = {"v2"} if role == "historical" else {"v2", "v3"}
    schedules = payload["schedules"]
    require(type(schedules) is dict and set(schedules) == expected, "scheduler inventory mismatch")
    for cells in schedules.values():
        require(type(cells) is list and len(cells) == len(CELLS), "schedule cell count mismatch")
        for selected in cells:
            require(type(selected) is list and len(selected) == STEPS * BATCH,
                    "schedule visit count mismatch")
            require(all(type(i) is int and 0 <= i < ROWS for i in selected),
                    "invalid scheduled row index")
    return mapping


def comparison(historical, current):
    oldmap = validate_payload(historical, "historical")
    newmap = validate_payload(current, "current")
    require(oldmap.keys() == newmap.keys(), "mathematical rows differ")
    require(all(canonical(normalized(row)) == canonical(normalized(newmap[key]))
                for key, row in oldmap.items()),
            "row content differs beyond the two permitted identity fields")
    changed = sum(row["example_identity"] != newmap[key]["example_identity"]
                  for key, row in oldmap.items())
    require(changed == ROWS, "expected all example identities to change")
    result = {"old_source": PINS["historical"]["commit"], "current_source": PINS["current"]["commit"],
              "old_dataset": PINS["historical"]["dataset"], "current_dataset": PINS["current"]["dataset"],
              "dataset_rows": ROWS, "numerical_rows_equal": True,
              "only_example_and_generator_id_differ": True,
              "changed_example_identity_count": changed, "cells": []}
    def sequence(payload, scheduler, cell):
        rows = payload["dataset"]["examples"]
        return [(rows[i]["prompt"], rows[i]["answer"])
                for i in payload["schedules"][scheduler][cell]]
    for cell, (seed, arm) in enumerate(CELLS):
        old = sequence(historical, "v2", cell)
        new = sequence(current, "v3", cell)
        current_v2 = sequence(current, "v2", cell)
        differences = [i for i in range(len(old)) if old[i] != new[i]]
        require(bool(differences), "expected historical order mismatch is absent")
        first = differences[0]
        oldcounts, newcounts = Counter(old), Counter(new)
        result["cells"].append({
            "seed": seed, "arm": arm, "visits": len(old),
            "numeric_visit_differences": len(differences),
            "first_difference": {"slot": first, "update": first // BATCH,
                                 "historical": list(old[first]), "current": list(new[first])},
            "distinct_visit_frequency_changes": sum(oldcounts[k] != newcounts[k]
                                                       for k in oldcounts.keys() | newcounts.keys()),
            "current_v2_function_matches_current_v3": current_v2 == new,
            "historical_sequence_identity": identity(canonical(old)),
            "current_sequence_identity": identity(canonical(new)),
            "slot_differences": [sum(old[i] != new[i] for i in range(j, len(old), BATCH))
                                 for j in range(BATCH)],
        })
    return canonical(result)


def audit(historical_source, current_source, receipt=HERE / "comparison.json"):
    retained = read_receipt(receipt)
    # Validate BOTH checkouts before allowing ANY producer to execute.
    historical_source = verify_source(historical_source, "historical")
    current_source = verify_source(current_source, "current")
    historical = run_worker(historical_source, "historical")
    current = run_worker(current_source, "current")
    reproduced = comparison(historical, current)
    require(reproduced == retained, "reconstructed comparison differs from retained receipt")
    require(read_receipt(receipt) == retained, "receipt changed during audit")
    verify_source(historical_source, "historical")
    verify_source(current_source, "current")
    return reproduced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-source", type=Path)
    parser.add_argument("--current-source", type=Path)
    parser.add_argument("--receipt", type=Path, default=HERE / "comparison.json")
    parser.add_argument("--worker", choices=tuple(PINS), help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            require(args.source is not None, "worker source required")
            sys.stdout.buffer.write(canonical(worker(args.source, args.worker)))
        else:
            require(args.historical_source is not None and args.current_source is not None,
                    "both --historical-source and --current-source are required")
            result = audit(args.historical_source, args.current_source, args.receipt)
            sys.stdout.buffer.write(result)
            print("Verified exact retained ordering-failure receipt sha256:" + RECEIPT_SHA256,
                  file=sys.stderr)
        return 0
    except (AuditError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print("CONTROL AUDIT REJECTED: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
