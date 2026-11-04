"""TT-003 opt-in CPU teacher producer using Trinite's unchanged native updater.

Numerical work is strictly CPU and limited to admitted TT-002 TRAIN examples.
No test or development labels enter this module's updates or diagnostics.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import time

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from teacher_training_v1 import spec

REPORT_SCHEMA = "trinite.teacher-train-only-report.v1"
METADATA_SCHEMA = "trinite.teacher-native-checkpoint.v1"
MANIFEST_SCHEMA = "trinite.teacher-run-manifest.v1"
OUTPUTS = ("approval.json", "request.json", "tensors.safetensors",
           "metadata.json", "report.json", "manifest.json")


def configure():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from trinite.training import require_environment
    return require_environment()


def native_plan(request):
    from trinite.training import RunConfig
    from trinite.contracts import ModelConfig
    p = request["plan"]
    return RunConfig(
        model=ModelConfig.from_dict(request["model"]),
        seed=p["seed"], steps=p["steps"], batch_size=p["batch_size"],
        learning_rate=p["learning_rate"], validation_every=p["validation_every"],
        max_seconds=p["max_seconds"], max_target_tokens=p["max_target_tokens"],
    )


def training_rows(packet: Path, request: dict) -> list[dict]:
    """Reads only the TT-002 train file after full packet verification."""
    from trinite.tokenizer import ByteTokenizer
    from trinite.contracts import require_identity
    root = Path(packet)
    raw = (root / "train.jsonl").read_bytes()
    if identity(raw) != request["dataset_train_identity"]:
        raise ContractError("TT-002 training bytes disagree with request")
    model = native_plan(request).model
    tok = ByteTokenizer(model.context_length)
    items = []
    for line in raw.splitlines(keepends=True):
        row = parse_json(line, canonical=True)
        if row["split"] != "train" or row["family"] not in ("S1", "S2", "W"):
            raise ContractError("nontrain or unexpected teacher row")
        enc = tok.encode(row["prompt"] + row["answer"],
                         answer_start=len(row["prompt"].encode("utf-8")))
        if (sum(enc.loss_mask) != row["supervised_tokens"]
                or len(enc.input_ids) != row["complete_context_tokens"]):
            raise ContractError("teacher row numerical encoding mismatch")
        items.append(dict(identity=row["id"], ids=enc.input_ids,
                          mask=enc.attention_mask, loss=enc.loss_mask))
    if len(items) != 972 or len({r["identity"] for r in items}) != 972:
        raise ContractError("teacher training rows missing or repeated")
    items.sort(key=lambda r: identity(json_bytes(
        {"seed": request["plan"]["seed"], "item": r["identity"]})))
    visits = [items[i % len(items)] for i in range(
        request["plan"]["steps"] * request["plan"]["batch_size"])]
    if (identity(json_bytes([r["identity"] for r in visits]))
            != request["train_schedule_identity"] or
            sum(sum(e["loss"]) for e in visits) != request["scheduled_target_tokens"]):
        raise ContractError("TT-003 training schedule mismatch")
    return items


def fresh_state(packet: Path, request: dict, request_anchor: str):
    from trinite.model import Decoder
    from trinite.training import (
        TrainingState, create_optimizer, evaluate, model_identity,
    )
    config = native_plan(request)
    rows = training_rows(packet, request)
    model = Decoder(config.model, lane="dense", seed=config.seed,
                    parameter_budget=2_000_000)
    if sum(p.numel() for p in model.parameters()) != request["model_parameters"]:
        raise ContractError("actual teacher parameter inventory mismatch")
    # Existing RunConfig/run_steps names validation diagnostics; here it is
    # intentionally an already-exposed TRAIN-only diagnostic, not heldout.
    data = {"train": rows, "validation": rows[:8]}
    optim = create_optimizer(model, config)
    return TrainingState(
        config, "dense", model, optim, data,
        request["dataset_train_identity"], request_anchor,
        model_identity(model), evaluate(model, rows[:8], 4),
    )


def predict_train_only(model, packet: Path, count: int = 8) -> list[dict]:
    """Greedy answers for exposed TRAIN examples only; raw invalids retained."""
    import torch
    from trinite.tokenizer import BOS, EOS
    raw = (Path(packet) / "train.jsonl").read_bytes()
    predictions = []
    for line in raw.splitlines(keepends=True)[:count]:
        row = parse_json(line, canonical=True)
        prompt = row["prompt"]
        ids = [BOS, *prompt.encode("utf-8")]
        generated = []
        with torch.no_grad():
            for _ in range(5):  # four answer bytes plus EOS, no retries
                batch = torch.tensor([ids], dtype=torch.int64, device="cpu")
                mask = torch.ones_like(batch, dtype=torch.bool)
                logits = model(batch, mask).logits
                token = int(logits[0, -1].argmax())
                generated.append(token)
                if token == EOS:
                    break
                ids.append(token)
        expected = [*row["answer"].encode("ascii"), EOS]
        predictions.append({"id": row["id"], "expected": expected,
                            "generated": generated, "exact": generated == expected})
    return predictions


def _metadata(state, request: dict, request_anchor: str) -> dict:
    from trinite.training import model_identity, require_environment
    return {
        "schema": METADATA_SCHEMA, "request_identity": request_anchor,
        "dataset_manifest_identity": request["dataset_manifest_identity"],
        "dataset_train_identity": request["dataset_train_identity"],
        "plan": state.config.to_dict(),
        "source_closure": request["source_closure"],
        "environment": require_environment(),
        "initialization_identity": state.initialization_identity,
        "initial_train_loss_hex": state.initial_train_loss,
        "final_model_identity": model_identity(state.model),
        "step": state.step, "cursor": state.cursor,
        "target_tokens": state.target_tokens, "history": state.history,
        "train_only_diagnostics": state.validation,
        "data_scope": "TRAIN_ROWS_ONLY_NOT_HELDOUT",
    }


def checkpoint(state, request: dict, request_anchor: str) -> tuple[bytes, bytes]:
    from trinite.comparison_checkpoint import tensor_payload
    payload = tensor_payload(state)
    meta = json_bytes(_metadata(state, request, request_anchor))
    if len(meta) > 1024 * 1024:
        raise ContractError("teacher checkpoint metadata exceeds bound")
    return payload, meta


def restore(packet: Path, request: dict, anchor: str, raw_tensor: bytes, raw_meta: bytes):
    from trinite.comparison_checkpoint import restore_tensors
    from trinite.training import evaluate, model_identity
    record = parse_json(raw_meta, canonical=True)
    expected_keys = {
        "schema", "request_identity", "dataset_manifest_identity",
        "dataset_train_identity", "plan", "source_closure", "environment",
        "initialization_identity", "initial_train_loss_hex",
        "final_model_identity", "step", "cursor", "target_tokens",
        "history", "train_only_diagnostics", "data_scope",
    }
    if set(record) != expected_keys or record["schema"] != METADATA_SCHEMA:
        raise ContractError("teacher checkpoint metadata schema mismatch")
    state = fresh_state(packet, request, anchor)
    initial = _metadata(state, request, anchor)
    mutable = {"final_model_identity", "step", "cursor", "target_tokens",
               "history", "train_only_diagnostics"}
    if any(record[k] != initial[k] for k in expected_keys - mutable):
        raise ContractError("teacher checkpoint source/plan/environment mismatch")
    for name, value in (("step", record["step"]), ("cursor", record["cursor"]),
                        ("target_tokens", record["target_tokens"]),
                        ("history", record["history"]),
                        ("validation", record["train_only_diagnostics"])):
        setattr(state, name, value)
    state = restore_tensors(state, raw_tensor, record["final_model_identity"])
    if state.step != request["plan"]["steps"] or state.target_tokens != request["scheduled_target_tokens"]:
        raise ContractError("incomplete or over-budget teacher checkpoint")
    if model_identity(state.model) != record["final_model_identity"]:
        raise ContractError("teacher final model mismatch")
    return state


def _write(directory: Path, name: str, raw: bytes):
    with (directory / name).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def run(request_dir: Path, request_anchor: str, dataset: Path,
        approval_path: Path, approval_anchor: str, output: Path) -> dict:
    """Explicit owner-approved CPU execution; never callable from PR CI."""
    request = spec.verify_request(request_dir, request_anchor, dataset)
    # Preserve the exact request identity as a separate value, not self-reported.
    spec.load_approval(approval_path, approval_anchor, request, request_anchor)
    output = Path(output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ContractError("teacher output must be new under an existing parent")
    output.mkdir(mode=0o700)
    # Preserve the run's intended inputs before importing/loading model tensors.
    _write(output, "request.json", spec._member(request_dir, "request.json"))
    _write(output, "approval.json", Path(approval_path).read_bytes())
    state = None
    start = time.monotonic()
    try:
        configure()
        from trinite.training import run_steps, evaluate, model_identity
        state = fresh_state(dataset, request, request_anchor)
        run_steps(state)
        if state.step != request["plan"]["steps"] or state.target_tokens != request["scheduled_target_tokens"]:
            raise ContractError("teacher numerical run did not complete its frozen schedule")
        tensors, metadata = checkpoint(state, request, request_anchor)
        _write(output, "tensors.safetensors", tensors)
        _write(output, "metadata.json", metadata)
        before = model_identity(state.model)
        predictions = predict_train_only(state.model, dataset, 8)
        if model_identity(state.model) != before:
            raise ContractError("training-only inference mutated teacher state")
        final_loss = evaluate(state.model, state.data["validation"], 4)
        # Source-bound independent safe-restore from the produced bytes.
        del state
        state = restore(dataset, request, request_anchor, tensors, metadata)
        if model_identity(state.model) != before:
            raise ContractError("safe-restored teacher differs from completed run")
        report = {
            "schema": REPORT_SCHEMA, "status": "COMPLETED_TRAIN_ONLY",
            "request_identity": request_anchor,
            "approval_identity": approval_anchor,
            "dataset_manifest_identity": request["dataset_manifest_identity"],
            "optimizer_updates": state.step,
            "target_tokens": state.target_tokens,
            "initial_train_loss_hex": state.initial_train_loss,
            "final_train_loss_hex": final_loss,
            "model_identity": before,
            "train_only_predictions": predictions,
            "heldout_scored": False, "learning_claim": False,
            "gpu_used": False, "weights_license": "UNDECIDED",
        }
        _write(output, "report.json", json_bytes(report))
        observed = [(name, (output / name).read_bytes()) for name in OUTPUTS[:-1]]
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "COMPLETED_TRAIN_ONLY",
            "request_identity": request_anchor, "approval_identity": approval_anchor,
            "members": [{"name": name, "identity": identity(raw), "bytes": len(raw)}
                        for name, raw in observed],
            "optimizer_updates": state.step, "heldout_scored": False,
            "scientific_confirmation": False,
        }
        _write(output, "manifest.json", json_bytes(manifest))
        return {"run_manifest_identity": identity(json_bytes(manifest)),
                "status": "COMPLETED_TRAIN_ONLY",
                "optimizer_updates": state.step, "heldout_scored": False}
    except Exception as exc:
        # Preserve failure evidence; incomplete directories are never valid runs.
        try:
            _write(output, "failure.json", json_bytes({
                "schema": "trinite.teacher-run-failure.v1", "successful": False,
                "error_type": type(exc).__name__,
                "message": str(exc)[:512], "elapsed_hex": float(time.monotonic() - start).hex(),
                "optimizer_updates_completed": 0 if state is None else state.step,
            }))
        except Exception:
            pass
        raise


def verify_run(output: Path, external_manifest: str, dataset: Path,
               request_anchor: str) -> dict:
    """Read-only native safe restore and train-only diagnosis, not optimizer replay."""
    from trinite.contracts import require_identity
    require_identity(external_manifest)
    require_identity(request_anchor)
    # The saved request is anchored, but the run directory contains more members
    # than the single-file frozen request; use strict identity + regeneration.
    raw_request = spec._member(output, "request.json")
    if identity(raw_request) != request_anchor:
        raise ContractError("teacher run request identity mismatch")
    request = parse_json(raw_request, canonical=True)
    if request != spec.request_for(dataset, request["dataset_manifest_identity"]):
        raise ContractError("teacher run bound sources or data changed")
    expected_names = set(OUTPUTS)
    names = set()
    with os.scandir(output) as entries:
        for item in entries:
            if item.name not in expected_names or item.name in names:
                raise ContractError("unknown teacher run artifact")
            names.add(item.name)
    if names != expected_names:
        raise ContractError("incomplete teacher run artifact inventory")
    manifest_raw = spec._member(output, "manifest.json")
    if identity(manifest_raw) != external_manifest:
        raise ContractError("externally anchored teacher run manifest mismatch")
    manifest = parse_json(manifest_raw, canonical=True)
    if (manifest.get("schema") != MANIFEST_SCHEMA
            or manifest.get("request_identity") != request_anchor
            or manifest.get("status") != "COMPLETED_TRAIN_ONLY"):
        raise ContractError("teacher run manifest schema/status mismatch")
    observed = []
    for name in OUTPUTS[:-1]:
        raw = spec._member(output, name, 32 * 1024 * 1024 if name == "tensors.safetensors" else spec.MAX_FILE)
        observed.append({"name": name, "identity": identity(raw), "bytes": len(raw)})
    if observed != manifest["members"]:
        raise ContractError("teacher run file identities differ")
    spec.load_approval(output / "approval.json", manifest["approval_identity"],
                       request, request_anchor)
    configure()
    raw_tensor = spec._member(output, "tensors.safetensors", 32 * 1024 * 1024)
    raw_meta = spec._member(output, "metadata.json")
    state = restore(dataset, request, request_anchor, raw_tensor, raw_meta)
    report = parse_json(spec._member(output, "report.json"), canonical=True)
    if (report.get("schema") != REPORT_SCHEMA
            or report.get("request_identity") != request_anchor
            or report.get("approval_identity") != manifest["approval_identity"]
            or report.get("optimizer_updates") != state.step
            or report.get("target_tokens") != state.target_tokens
            or report.get("status") != "COMPLETED_TRAIN_ONLY"
            or report.get("model_identity") != _metadata(state, request, request_anchor)["final_model_identity"]
            or report.get("initial_train_loss_hex") != state.initial_train_loss
            or report.get("weights_license") != "UNDECIDED"
            or report.get("target_tokens") != request["scheduled_target_tokens"]
            or report.get("train_only_predictions") != predict_train_only(state.model, dataset, 8)
            or report.get("final_train_loss_hex") != __import__("trinite.training", fromlist=["evaluate"]).evaluate(
                state.model, state.data["validation"], 4)
            or report.get("optimizer_updates") != request["plan"]["steps"]
            or report.get("heldout_scored") is not False
            or report.get("gpu_used") is not False
            or report.get("learning_claim") is not False):
        raise ContractError("teacher final train-only report not independently reproduced")
    return {"verified": True, "scope": "source-and-safe-final-state-with-train-only-diagnosis",
            "optimizer_updates_verified": state.step, "optimizer_replayed": False,
            "heldout_scored": False, "teacher_generalization_claim": False}
