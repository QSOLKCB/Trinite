# SPDX-License-Identifier: MPL-2.0
"""TT-003-DIAG-001: zero-update CPU inference on exactly 972 TRAIN rows."""
from __future__ import annotations

import os
from pathlib import Path
import time

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from teacher_training_v1 import numerical
from teacher_diag_v1 import core

OUTPUT_FILES = ("request.json", "predictions.jsonl", "summary.json")


def raw_greedy(model, prompt: str) -> list[int]:
    """Existing uncached greedy semantics, no targets/conditioning or repair."""
    import torch
    from trinite.tokenizer import BOS, EOS
    if type(prompt) is not str or not prompt.isascii():
        raise ContractError("noncanonical inference prompt")
    prefix = [BOS, *prompt.encode("ascii")]
    output = []
    with torch.no_grad():
        for _ in range(5):
            if len(prefix) > model.config.context_length:
                raise ContractError("inference exceeds context; never truncate")
            ids = torch.tensor([prefix], dtype=torch.int64, device="cpu")
            mask = torch.ones_like(ids, dtype=torch.bool)
            token = int(model(ids, mask).logits[0, -1].argmax())
            output.append(token)
            if token == EOS:
                break
            prefix.append(token)
    return output


def write(directory: Path, name: str, data: bytes) -> None:
    if name not in (*OUTPUT_FILES, "manifest.json", "failure.json"):
        raise ContractError("unknown diagnostic output")
    with (directory / name).open("xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def restore_models(dataset: Path, teacher_run: Path):
    from trinite.training import model_identity
    p, request, _, report = core.historical_inputs(dataset, teacher_run)
    numerical.configure()
    verified = numerical.verify_run(teacher_run, p["teacher_run_manifest_identity"],
                                    dataset, p["teacher_request_identity"])
    if verified.get("verified") is not True or verified.get("optimizer_replayed") is not False:
        raise ContractError("historical checkpoint final state verification failed")
    raw_tensor = core.read_member(teacher_run, "tensors.safetensors", core.MAX_TENSOR_BYTES)
    raw_meta = core.read_member(teacher_run, "metadata.json")
    final = numerical.restore(dataset, request, p["teacher_request_identity"],
                              raw_tensor, raw_meta)
    initial = numerical.fresh_state(dataset, request, p["teacher_request_identity"])
    if (model_identity(initial.model) != p["initial_model_identity"]
            or model_identity(final.model) != p["teacher_model_identity"]):
        raise ContractError("initial/final source-bound model identity differs")
    if numerical.predict_train_only(final.model, dataset, 8) != report["train_only_predictions"]:
        raise ContractError("historical eight-prediction diagnostic differs")
    return initial.model, final.model


def compute(dataset: Path, teacher_run: Path, models, *, deadline=None, sink=None):
    from trinite.training import model_identity
    _, request, metadata, _ = core.historical_inputs(dataset, teacher_run)
    rows, visited = core.scheduled_rows(metadata, request, dataset)
    initial, final = models
    before = (model_identity(initial), model_identity(final))
    if before != (core.plan()["initial_model_identity"], core.plan()["teacher_model_identity"]):
        raise ContractError("supplied model state does not match frozen teacher")
    results = []
    for row in rows:
        if deadline is not None and time.monotonic() > deadline:
            raise ContractError("diagnostic CPU forward deadline exceeded")
        task = row["observation"]["task"]
        result = {
            "id": row["id"], "family": row["family"], "task": task,
            "expected": row["answer"], "visited": row["id"] in visited,
            "initial": core.classify(raw_greedy(initial, row["prompt"]), row["answer"], task),
            "final": core.classify(raw_greedy(final, row["prompt"]), row["answer"], task),
        }
        results.append(result)
        if sink is not None:
            sink.write(json_bytes(result))
            sink.flush()
    if len(results) != core.ROW_COUNT or (
            model_identity(initial), model_identity(final)) != before:
        raise ContractError("incomplete inference or mutated checkpoint state")
    return results


def summary_for(rows: list[dict], request: dict) -> dict:
    result = core.analysis_summary(rows)
    result.update({
        "request_identity": request["request_identity"],
        "dataset_manifest_identity": request["dataset_manifest_identity"],
        "teacher_run_manifest_identity": request["teacher_run_manifest_identity"],
        "initial_model_identity": request["plan"]["initial_model_identity"],
        "final_model_identity": request["plan"]["teacher_model_identity"],
        "training_authorized": False, "optimizer_updates": 0,
        "heldout_scored": False, "scientific_confirmation": False,
        "execution_scope": "full-972-exposed-train-only-native-greedy",
    })
    return result


def run(frozen_dir: Path, anchor: str, dataset: Path, teacher_run: Path, output: Path) -> dict:
    request = core.verify_request(frozen_dir, anchor, dataset, teacher_run)
    # Never overwrite an original packet or previous experiment.
    output = Path(output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ContractError("diagnostic output directory must be fresh")
    output.mkdir(mode=0o700)
    write(output, "request.json", core.read_member(frozen_dir, "request.json"))
    start = time.monotonic()
    try:
        models = restore_models(dataset, teacher_run)
        with (output / "predictions.jsonl").open("xb") as stream:
            rows = compute(dataset, teacher_run, models,
                           deadline=start + request["plan"]["max_execution_seconds"],
                           sink=stream)
            stream.flush()
            os.fsync(stream.fileno())
        write(output, "summary.json", json_bytes(summary_for(rows, dict(request, request_identity=anchor))))
        core.verify_request(frozen_dir, anchor, dataset, teacher_run)
        manifest = {
            "schema": core.MANIFEST_SCHEMA,
            "status": "COMPLETED_READ_ONLY_TRAIN_AUDIT",
            "request_identity": anchor,
            "members": [
                {"name": n, "identity": identity(core.read_member(output, n)),
                 "bytes": (output / n).stat().st_size}
                for n in OUTPUT_FILES
            ],
            "train_rows": core.ROW_COUNT,
            "no_optimizer_updates": True, "heldout_scored": False,
        }
        write(output, "manifest.json", json_bytes(manifest))
        return {"manifest_identity": identity(json_bytes(manifest)),
                "status": manifest["status"], "train_rows": core.ROW_COUNT,
                "optimizer_updates": 0, "heldout_scored": False}
    except Exception as err:
        try:
            journal = output / "predictions.jsonl"
            write(output, "failure.json", json_bytes({
                "schema": "trinite.teacher-frozen-diagnostic-failure.v1",
                "successful": False, "type": type(err).__name__,
                "message": str(err)[:512],
                "partial_prediction_bytes": journal.stat().st_size if journal.exists() else 0,
            }))
        except OSError:
            pass
        raise


def verify(frozen_dir: Path, anchor: str, dataset: Path, teacher_run: Path,
           output: Path, external_result_identity: str) -> dict:
    """Byte/custody/tally verification, not a fresh execution of the neural model."""
    from trinite.contracts import require_identity
    require_identity(external_result_identity)
    req = core.verify_request(frozen_dir, anchor, dataset, teacher_run)
    core.check_members(output, core.RESULT_FILES)
    manifest_raw = core.read_member(output, "manifest.json")
    if identity(manifest_raw) != external_result_identity:
        raise ContractError("independent diagnostic result anchor differs")
    manifest = parse_json(manifest_raw, canonical=True)
    if (set(manifest) != {"schema", "status", "request_identity", "members",
                         "train_rows", "no_optimizer_updates", "heldout_scored"}
            or manifest["schema"] != core.MANIFEST_SCHEMA
            or manifest["status"] != "COMPLETED_READ_ONLY_TRAIN_AUDIT"
            or manifest["request_identity"] != anchor
            or manifest["train_rows"] != core.ROW_COUNT
            or manifest["no_optimizer_updates"] is not True
            or manifest["heldout_scored"] is not False):
        raise ContractError("non-successful or malformed result manifest")
    items = [
        {"name": n, "bytes": len(core.read_member(output, n)),
         "identity": identity(core.read_member(output, n))}
        for n in OUTPUT_FILES
    ]
    if manifest["members"] != items:
        raise ContractError("diagnostic artifact identities mismatch")
    if core.read_member(output, "request.json") != core.read_member(frozen_dir, "request.json"):
        raise ContractError("diagnostic request substituted")
    saved = core.read_member(output, "predictions.jsonl")
    if not saved.endswith(b"\n"):
        raise ContractError("partial diagnostic JSONL")
    rows = [parse_json(line + b"\n", canonical=True) for line in saved.splitlines()]
    _, teacher, metadata, _ = core.historical_inputs(dataset, teacher_run)
    original, visited = core.scheduled_rows(metadata, teacher, dataset)
    if len(rows) != len(original):
        raise ContractError("incomplete diagnostic")
    for result, expected in zip(rows, original):
        if (result["id"] != expected["id"]
                or result["family"] != expected["family"]
                or result["task"] != expected["observation"]["task"]
                or result["expected"] != expected["answer"]
                or result["visited"] is not (expected["id"] in visited)):
            raise ContractError("TRAIN row/schedule mismatch")
    calculated = summary_for(rows, dict(req, request_identity=anchor))
    if parse_json(core.read_member(output, "summary.json"), canonical=True) != calculated:
        raise ContractError("summary is not independently reproducible from raw rows")
    return {"verified": True, "scope": "recorded-raw-output-and-custody",
            "native_forward_replayed": False, "optimizer_updates": 0,
            "heldout_scored": False, "train_rows": core.ROW_COUNT}


def replay_forward(frozen_dir: Path, anchor: str, dataset: Path, teacher_run: Path,
                   output: Path, external_result_identity: str) -> dict:
    """Optional full native forward replay; no backward/optimizer update."""
    outcome = verify(frozen_dir, anchor, dataset, teacher_run,
                     output, external_result_identity)
    rows = [parse_json(line + b"\n", canonical=True)
            for line in core.read_member(output, "predictions.jsonl").splitlines()]
    candidate = compute(dataset, teacher_run, restore_models(dataset, teacher_run))
    if candidate != rows:
        raise ContractError("native full-forward replay differs")
    return dict(outcome, native_forward_replayed=True,
                scope="all-972-initial-and-final-native-forward-replay")
