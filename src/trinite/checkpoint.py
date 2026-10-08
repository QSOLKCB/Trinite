"""Versioned, bounded safetensors + JSON checkpoints; no pickle or observers."""
from pathlib import Path
import random
import struct

import numpy as np
import torch
from safetensors.torch import load, save

from .contracts import ContractError, exact_keys, identity, integer, json_bytes, parse_json, require_identity
from .inspection import tensor_identity
from .training import (RunConfig, TrainingState, create_state, environment, model_identity,
                       require_environment, require_optimizer, source_receipt)

SCHEMA = "trinite.training-checkpoint.v1"
MAX_TENSORS = 32 * 1024 * 1024
MAX_METADATA = 1024 * 1024


def hex_value(value, label):
    if type(value) is not str:
        raise ContractError(f"{label} requires an exact float hex string")
    try:
        number = float.fromhex(value)
    except ValueError as error:
        raise ContractError(f"invalid {label}") from error
    if not np.isfinite(number) or number.hex() != value:
        raise ContractError(f"{label} requires finite canonical float hex")
    return number


def rng_snapshot() -> tuple[dict, dict]:
    python = random.getstate()
    numpy = np.random.get_state()
    return ({"python_version": python[0], "python_state": list(python[1]),
             "python_gauss_hex": None if python[2] is None else python[2].hex(),
             "numpy_algorithm": numpy[0], "numpy_position": int(numpy[2]),
             "numpy_has_gauss": int(numpy[3]), "numpy_gauss_hex": float(numpy[4]).hex()},
            {"rng.torch_cpu": torch.get_rng_state().clone(),
             "rng.numpy_keys": torch.tensor([int(n) for n in numpy[1]], dtype=torch.int64, device="cpu")})


def _validate_rng(metadata, tensors):
    exact_keys(metadata, {"python_version", "python_state", "python_gauss_hex", "numpy_algorithm",
                          "numpy_position", "numpy_has_gauss", "numpy_gauss_hex"}, "RNG metadata")
    if metadata["python_version"] != 3 or type(metadata["python_version"]) is not int:
        raise ContractError("unsupported Python RNG state")
    state = metadata["python_state"]
    if type(state) is not list or len(state) != 625:
        raise ContractError("invalid Python RNG state shape")
    for value in state[:-1]: integer(value, "Python RNG word", 0, 2**32-1)
    integer(state[-1], "Python RNG position", 0, 624)
    gauss = metadata["python_gauss_hex"]
    python_state = (3, tuple(state), None if gauss is None else hex_value(gauss, "Python Gaussian cache"))
    random.Random(0).setstate(python_state)
    if metadata["numpy_algorithm"] != "MT19937":
        raise ContractError("unsupported NumPy RNG algorithm")
    integer(metadata["numpy_position"], "NumPy RNG position", 0, 624)
    integer(metadata["numpy_has_gauss"], "NumPy Gaussian flag", 0, 1)
    numpy_keys = tensors["rng.numpy_keys"]
    if (numpy_keys.dtype != torch.int64 or numpy_keys.shape != (624,)
            or bool((numpy_keys < 0).any()) or bool((numpy_keys >= 2**32).any())):
        raise ContractError("invalid NumPy RNG key tensor")
    numpy_state = ("MT19937", np.array(numpy_keys.tolist(), dtype=np.uint32),
                   metadata["numpy_position"], metadata["numpy_has_gauss"],
                   hex_value(metadata["numpy_gauss_hex"], "NumPy Gaussian cache"))
    np.random.RandomState(0).set_state(numpy_state)
    cpu = tensors["rng.torch_cpu"]
    if cpu.dtype != torch.uint8 or cpu.shape != torch.get_rng_state().shape:
        raise ContractError("invalid Torch CPU RNG tensor")
    try:
        torch.Generator(device="cpu").set_state(cpu)
    except RuntimeError as error:
        raise ContractError("invalid Torch CPU RNG state") from error
    return python_state, numpy_state, cpu


def validate_progress(state: TrainingState) -> None:
    if state.model.config != state.config.model or state.model.lane != state.lane:
        raise ContractError("training state model/plan/lane mismatch")
    if type(state.history) is not list or type(state.validation) is not list:
        raise ContractError("checkpoint histories require lists")
    integer(state.step, "checkpoint step", 0, state.config.steps)
    if (type(state.cursor) is not int or state.cursor != state.step*state.config.batch_size
            or type(state.target_tokens) is not int or state.target_tokens != state.cursor*2
            or len(state.history) != state.step):
        raise ContractError("checkpoint step, data cursor, tokens, and history disagree")
    order = [e["identity"] for e in state.data["train"]]
    for i, entry in enumerate(state.history):
        exact_keys(entry, {"step", "loss_hex", "gradient_norm_hex", "target_tokens", "examples"}, "step history")
        expected = [order[(i*state.config.batch_size+j) % 36] for j in range(state.config.batch_size)]
        if (type(entry["step"]) is not int or entry["step"] != i+1
                or type(entry["target_tokens"]) is not int or entry["target_tokens"] != state.config.batch_size*2
                or entry["examples"] != expected):
            raise ContractError("checkpoint history differs from the frozen data schedule")
        if hex_value(entry["loss_hex"], "loss") < 0 or hex_value(entry["gradient_norm_hex"], "gradient norm") < 0:
            raise ContractError("negative loss/gradient norm")
    expected_validation = [i for i in range(1, state.step+1)
                           if i % state.config.validation_every == 0 or i == state.config.steps]
    if len(state.validation) != len(expected_validation):
        raise ContractError("checkpoint validation schedule differs from plan")
    for i, entry in zip(expected_validation, state.validation):
        exact_keys(entry, {"step", "loss_hex"}, "validation history")
        if type(entry["step"]) is not int or entry["step"] != i or hex_value(entry["loss_hex"], "validation loss") < 0:
            raise ContractError("invalid validation history")
    if hex_value(state.initial_train_loss, "initial training loss") < 0:
        raise ContractError("negative initial loss")
    require_identity(state.initialization_identity)
    state.model.validate_state()
    require_optimizer(state)


def checkpoint_bytes(state: TrainingState) -> tuple[bytes, bytes]:
    require_environment()
    validate_progress(state)
    tensors = {}
    for name, parameter in state.model.named_parameters():
        if parameter.grad is not None:
            raise ContractError("checkpoint requires a completed step with cleared gradients")
        tensors["model."+name] = parameter.detach().clone().contiguous()
        slots = state.optimizer.state.get(parameter, {})
        if state.step:
            exact_keys(slots, {"step", "exp_avg", "exp_avg_sq"}, "AdamW state")
            for slot, value in slots.items():
                if (value.dtype != torch.float32 or value.device.type != "cpu"
                        or not bool(torch.isfinite(value).all())
                        or value.shape != (() if slot == "step" else parameter.shape)
                        or (slot == "step" and float(value) != state.step)
                        or (slot == "exp_avg_sq" and bool((value < 0).any()))):
                    raise ContractError("invalid named AdamW tensor")
                tensors["optimizer."+name+"."+slot] = value.detach().clone().contiguous()
        elif slots:
            raise ContractError("zero-step checkpoint cannot have optimizer moments")
    rng, rng_tensors = rng_snapshot()
    _validate_rng(rng, rng_tensors)
    tensors.update(rng_tensors)
    payload = save(tensors)
    if len(payload) > MAX_TENSORS:
        raise ContractError("checkpoint tensor payload exceeds 32 MiB")
    metadata = {"schema": SCHEMA, "plan": state.config.to_dict(), "plan_identity": state.config.content_identity,
                "lane": state.lane, "dataset_identity": state.dataset_identity, "manifest_identity": state.manifest_identity,
                "initialization_identity": state.initialization_identity, "initial_train_loss": state.initial_train_loss,
                "step": state.step, "cursor": state.cursor, "target_tokens": state.target_tokens,
                "history": state.history, "validation": state.validation, "environment": environment(),
                "source": source_receipt(), "rng": rng, "tensor_file_identity": identity(payload),
                "model_identity": model_identity(state.model),
                "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype), "identity": tensor_identity(t)}
                            for n, t in tensors.items()}}
    meta = json_bytes(metadata)
    if len(meta) > MAX_METADATA:
        raise ContractError("checkpoint metadata exceeds 1 MiB")
    return payload, meta


def load_checkpoint_bytes(payload: bytes, meta: bytes, dataset: bytes, manifest: bytes,
                          config: RunConfig, lane: str) -> TrainingState:
    require_environment()
    if type(payload) is not bytes or not 8 <= len(payload) <= MAX_TENSORS or type(meta) is not bytes or len(meta) > MAX_METADATA:
        raise ContractError("checkpoint exceeds byte bounds")
    m = exact_keys(parse_json(meta, canonical=True), {"schema", "plan", "plan_identity", "lane", "dataset_identity",
        "manifest_identity", "initialization_identity", "initial_train_loss", "step", "cursor", "target_tokens",
        "history", "validation", "environment", "source", "rng", "tensor_file_identity", "model_identity", "tensors"}, "checkpoint")
    if (m["schema"] != SCHEMA or json_bytes(m["plan"]) != json_bytes(config.to_dict()) or m["plan_identity"] != config.content_identity
            or m["lane"] != lane or m["dataset_identity"] != identity(dataset) or m["manifest_identity"] != identity(manifest)
            or json_bytes(m["source"]) != json_bytes(source_receipt()) or json_bytes(m["environment"]) != json_bytes(environment())
            or m["tensor_file_identity"] != identity(payload)):
        raise ContractError("checkpoint schema/plan/lane/input/source/environment/bytes mismatch")
    header_length = struct.unpack("<Q", payload[:8])[0]
    if not 1 <= header_length <= MAX_METADATA or 8+header_length > len(payload):
        raise ContractError("invalid or excessive safetensors header")
    parse_json(payload[8:8+header_length])  # Reject duplicate keys and unsupported metadata before allocation.
    try:
        tensors = load(payload)
    except Exception as error:
        raise ContractError("invalid safetensors payload") from error
    if type(m["tensors"]) is not dict or set(tensors) != set(m["tensors"]):
        raise ContractError("checkpoint tensor inventory mismatch")
    state = create_state(config, lane, dataset, manifest)
    if state.initialization_identity != m["initialization_identity"] or state.initial_train_loss != m["initial_train_loss"]:
        raise ContractError("checkpoint initialization receipt does not replay")
    for key in ("step", "cursor", "target_tokens", "history", "validation"):
        setattr(state, key, m[key])
    validate_progress(state)
    expected = {"rng.torch_cpu", "rng.numpy_keys"}
    for name, p in state.model.named_parameters():
        expected.add("model."+name)
        if state.step: expected.update("optimizer."+name+"."+s for s in ("step", "exp_avg", "exp_avg_sq"))
    if set(tensors) != expected:
        raise ContractError("checkpoint contains missing or unexpected tensor names")
    for name, tensor in tensors.items():
        info = exact_keys(m["tensors"][name], {"shape", "dtype", "identity"}, "tensor receipt")
        if json_bytes(info) != json_bytes({"shape": list(tensor.shape), "dtype": str(tensor.dtype), "identity": tensor_identity(tensor)}):
            raise ContractError("checkpoint tensor receipt mismatch")
    python_rng, numpy_rng, cpu_rng = _validate_rng(m["rng"], tensors)
    # All candidate model/optimizer tensors are validated before applying them.
    for name, p in state.model.named_parameters():
        for key, shape in [("model."+name, p.shape)] + (
                [("optimizer."+name+"."+s, () if s == "step" else p.shape)
                 for s in ("step", "exp_avg", "exp_avg_sq")] if state.step else []):
            t = tensors[key]
            if t.dtype != torch.float32 or t.shape != shape or not bool(torch.isfinite(t).all()):
                raise ContractError("checkpoint tensor shape/dtype/finite contract failed")
        if state.step and (float(tensors["optimizer."+name+".step"]) != state.step
                          or bool((tensors["optimizer."+name+".exp_avg_sq"] < 0).any())):
            raise ContractError("checkpoint AdamW step or variance is invalid")
    with torch.no_grad():
        for name, p in state.model.named_parameters():
            p.copy_(tensors["model."+name])
            if state.step:
                state.optimizer.state[p] = {s: tensors["optimizer."+name+"."+s].clone()
                                            for s in ("step", "exp_avg", "exp_avg_sq")}
    if model_identity(state.model) != m["model_identity"]:
        raise ContractError("checkpoint model identity mismatch")
    validate_progress(state)
    # Restore caller RNGs only after every checkpoint check has succeeded.
    random.setstate(python_rng); np.random.set_state(numpy_rng); torch.set_rng_state(cpu_rng)
    return state


def save_checkpoint(path: Path, state: TrainingState) -> dict:
    payload, meta = checkpoint_bytes(state)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    (path/"tensors.safetensors").write_bytes(payload)
    (path/"metadata.json").write_bytes(meta)
    return {"tensors_identity": identity(payload), "metadata_identity": identity(meta)}


def read_checkpoint(path: Path) -> tuple[bytes, bytes]:
    path = Path(path)
    if path.is_symlink() or not path.is_dir() or {p.name for p in path.iterdir()} != {"tensors.safetensors", "metadata.json"}:
        raise ContractError("checkpoint requires exactly its two regular members")
    output = []
    for name, limit in (("tensors.safetensors", MAX_TENSORS), ("metadata.json", MAX_METADATA)):
        member = path/name
        if member.is_symlink() or not member.is_file():
            raise ContractError("checkpoint symlinks/special files are forbidden")
        with member.open("rb") as stream: data = stream.read(limit+1)
        if len(data) > limit: raise ContractError("checkpoint member exceeds byte bound")
        output.append(data)
    return tuple(output)
