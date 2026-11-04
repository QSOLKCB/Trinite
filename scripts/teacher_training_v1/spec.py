"""TT-003 standard-library-only planning; validates before importing Torch."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import tempfile

from trinite.contracts import (
    ContractError, ModelConfig, identity, json_bytes, parse_json, require_identity,
)
from teacher_lab_v1 import core as tt001
from teacher_dataset_v1 import core as tt002

PLAN = "configs/teacher/gf3-teacher-training-v1.json"
REQUEST_SCHEMA = "trinite.teacher-training-request.v1"
APPROVAL_SCHEMA = "trinite.teacher-training-local-approval.v1"
MAX_FILE = 4 * 1024 * 1024
# Source closure for the *new* teacher producer. Existing TT-001/TT-002 receipts
# remain unchanged and use their original source maps.
CLOSURE = (
    "scripts/teacher_training.py", "scripts/teacher_training_v1/__init__.py",
    "scripts/teacher_training_v1/spec.py", "scripts/teacher_training_v1/numerical.py",
    "scripts/teacher_training_v1/cuda_preflight.py",
    "configs/teacher/gf3-teacher-training-v1.json",
    "src/trinite/__init__.py", "src/trinite/contracts.py",
    "src/trinite/tokenizer.py", "src/trinite/model.py", "src/trinite/quantizer.py",
    "src/trinite/inspection.py", "src/trinite/training.py",
    "src/trinite/comparison_checkpoint.py",
    "src/trinite/comparison_training.py",
    "src/trinite/comparison_data.py", "src/trinite/qec_source.py",
    "src/trinite/qec-pin.json", "src/trinite/comparison-protocol.json",
    "src/trinite/data.py", "requirements.lock",
    "docs/TEACHER-LAB-TT-003.md", "docs/INVARIANTS.md",
    "docs/DATA.md", "docs/PROVENANCE.md", "AGENTS.md", "LICENSE",
)


def settings() -> tuple[dict, ModelConfig]:
    p = tt001.source_json(PLAN)
    expected = {
        "schema": "trinite.teacher-training-plan.v1",
        "dataset_schema": tt002.MANIFEST_SCHEMA,
        "model_config": "configs/teacher/gf3-teacher-90k-v1.json",
        "role": "teacher", "lane": "dense",
        "backend": "pinned-cpu-float32", "seed": 17, "steps": 32,
        "batch_size": 4, "max_target_tokens": 640, "max_seconds": 600,
        "validation_every": 32, "learning_rate": "0.001",
        "training_authorized": False,
        "evaluation": "train-only-diagnostic-no-heldout",
    }
    if p != expected or any(type(p[k]) is not type(v) for k, v in expected.items()):
        raise ContractError("TT-003 plan changed or needs a new version")
    config = ModelConfig.from_dict(tt001.source_json(p["model_config"]))
    constructor, run_limit = tt001.native_limits()
    if config.parameter_count > min(constructor, run_limit):
        raise ContractError("teacher model outside the current CPU native budget")
    # Explicit token upper bound (4 answer bytes plus EOS per scheduled visit).
    if p["steps"] * p["batch_size"] * 5 > p["max_target_tokens"]:
        raise ContractError("teacher plan has insufficient supervised-token budget")
    return p, config


def _member(folder: Path, name: str, size: int = MAX_FILE) -> bytes:
    if type(name) is not str or name not in ("request.json", "approval.json",
                                             "manifest.json", "metadata.json",
                                             "report.json", "tensors.safetensors"):
        raise ContractError("unknown bounded teacher artifact")
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ContractError("invalid teacher artifact directory")
    path = folder / name
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > size:
        raise ContractError("invalid teacher artifact member")
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    if len(raw) > size:
        raise ContractError("oversize teacher artifact")
    return raw


def load_approval(path: Path, anchor: str, request: dict,
                  request_anchor: str) -> dict:
    require_identity(anchor)
    require_identity(request_anchor)
    path = Path(path)
    meta = path.lstat()
    if not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1 or meta.st_size > 4096:
        raise ContractError("invalid approval file")
    raw = path.read_bytes()
    if identity(raw) != anchor:
        raise ContractError("external approval anchor mismatch")
    value = parse_json(raw, canonical=True)
    expected = {
        "schema": APPROVAL_SCHEMA,
        "request_identity": request_anchor,
        "dataset_manifest_identity": request["dataset_manifest_identity"],
        "owner": "QSOLKCB",
        "scope": "one-explicit-owner-local-cpu-teacher-experiment",
        "approved_updates": request["plan"]["steps"],
        "training_authorized": True,
    }
    if value != expected or any(type(value[k]) is not type(v)
                                for k, v in expected.items()):
        raise ContractError("approval does not match frozen teacher run")
    # A local declaration is a boundary, not cryptographic owner authentication.
    return value


def request_for(dataset: Path, external_dataset_anchor: str) -> dict:
    require_identity(external_dataset_anchor)
    p, model = settings()
    verified = tt002.verify(Path(dataset), external_dataset_anchor)
    if verified["training_authorized"] is not False:
        raise ContractError("TT-002 data packet changed training authority")
    # Read training rows only. TT-002 separately verifies the privileged archive;
    # TT-003 does not use those withheld labels as model updates/diagnostics.
    from trinite.tokenizer import ByteTokenizer
    tok = ByteTokenizer(model.context_length)
    train = []
    for line in (Path(dataset) / "train.jsonl").read_bytes().splitlines(keepends=True):
        item = parse_json(line, canonical=True)
        if item["split"] != "train":
            raise ContractError("nontrain label in teacher training rows")
        encoded = tok.encode(item["prompt"] + item["answer"],
                             answer_start=len(item["prompt"].encode("utf-8")))
        if (sum(encoded.loss_mask) != item["supervised_tokens"] or
                len(encoded.input_ids) != item["complete_context_tokens"]):
            raise ContractError("train-only tokenizer/label mismatch")
        train.append({"identity": item["id"], "ids": encoded.input_ids,
                      "mask": encoded.attention_mask, "loss": encoded.loss_mask})
    if len(train) != 972:
        raise ContractError("teacher requires the verified complete TT-002 train partition")
    train.sort(key=lambda e: identity(json_bytes(
        {"seed": p["seed"], "item": e["identity"]})))
    visits = [train[i % len(train)] for i in range(p["steps"] * p["batch_size"])]
    total = sum(sum(row["loss"]) for row in visits)
    if total > p["max_target_tokens"]:
        raise ContractError("scheduled teacher targets exceed frozen plan")
    paths = sorted(set(CLOSURE) | set(tt002.SOURCES) |
                   {p["model_config"], tt002.PLAN,
                    "configs/teacher/source-candidates-v1.json"})
    sources = [{"path": name, "identity": identity(tt001.source_bytes(name))}
               for name in paths]
    return {
        "schema": REQUEST_SCHEMA,
        "status": "FROZEN_NOT_AUTHORIZED",
        "role": "teacher", "backend": p["backend"], "lane": "dense",
        "model": model.to_dict(), "model_parameters": model.parameter_count,
        "plan": p, "dataset_manifest_identity": external_dataset_anchor,
        "dataset_rights_identity": identity(json_bytes(tt002.rights()[0])),
        "dataset_train_identity": identity((Path(dataset) / "train.jsonl").read_bytes()),
        "scheduled_visits": len(visits), "scheduled_target_tokens": total,
        "train_schedule_identity": identity(json_bytes([r["identity"] for r in visits])),
        "source_closure": sources,
        "training_authorized": False, "heldout_scored": False,
        "cuda_gpu_training_available": False,
        "weights_license": "UNDECIDED",
    }


def freeze(folder: Path, dataset: Path, dataset_anchor: str) -> str:
    request = request_for(dataset, dataset_anchor)
    # No circular hash inside the record: caller anchors the file bytes.
    folder = Path(folder)
    if folder.exists() or folder.is_symlink() or not folder.parent.is_dir():
        raise ContractError("new request destination required")
    stage = Path(tempfile.mkdtemp(prefix=".tt003-freeze-", dir=folder.parent))
    try:
        with (stage / "request.json").open("xb") as out:
            out.write(json_bytes(request)); out.flush(); os.fsync(out.fileno())
        if folder.exists() or folder.is_symlink():
            raise ContractError("request destination already exists")
        os.rename(stage, folder)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return identity(json_bytes(request))


def verify_request(folder: Path, external: str, dataset: Path) -> dict:
    require_identity(external)
    raw = _member(folder, "request.json")
    if identity(raw) != external:
        raise ContractError("externally anchored TT-003 request mismatch")
    observed = parse_json(raw, canonical=True)
    expected = request_for(dataset, observed["dataset_manifest_identity"])
    if observed != expected:
        raise ContractError("TT-003 request source/dataset/schedule mismatch")
    names = set()
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.name != "request.json" or entry.name in names:
                raise ContractError("request directory has unexpected contents")
            names.add(entry.name)
    if names != {"request.json"}:
        raise ContractError("request directory has unexpected contents")
    return observed
