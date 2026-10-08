"""Bounded detached tensor inventories and selected quantizer snapshots."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import platform
import sys

import torch

from .contracts import ContractError, identity, integer, json_bytes
from .model import Decoder, Linear, MODEL_ID
from .quantizer import quantize, require_float_tensor

MAX_INSPECTION_BYTES = 32 * 1024 * 1024


def tensor_identity(tensor: torch.Tensor) -> str:
    """Hash contiguous native little-endian CPU bytes in bounded chunks."""
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
            or tensor.layout != torch.strided or sys.byteorder != "little"
            or tensor.numel() * tensor.element_size() > MAX_INSPECTION_BYTES):
        raise ContractError("tensor identity requires bounded little-endian CPU storage")
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for offset in range(0, raw.numel(), 65536):
        digest.update(bytes(raw[offset:offset+65536].tolist()))
    return "sha256:" + digest.hexdigest()


def _budget(model: Decoder, max_bytes: int) -> None:
    if not isinstance(model, Decoder):
        raise ContractError("inspection requires a Decoder")
    integer(max_bytes, "inspection byte budget", 1, MAX_INSPECTION_BYTES)
    if sum(p.numel() * p.element_size() for p in model.parameters()) > max_bytes:
        raise ContractError("model tensors exceed inspection byte budget")
    model.validate_state()


def inventory(model: Decoder, *, max_bytes: int = 8 * 1024 * 1024) -> dict:
    """Inspect actual named tensors, aliases, modules and effective-code scope.

    max_bytes bounds the unique parameter payload inspected, not process RSS.
    No model forward, RNG call, hooks, gradients or file writes occur here.
    """
    _budget(model, max_bytes)
    aliases = {}
    parameters = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        require_float_tensor(parameter, name)
        if id(parameter) in aliases:
            aliases[id(parameter)].append(name)
            continue
        names = [name]
        aliases[id(parameter)] = names
        parameters.append({"names": names, "shape": list(parameter.shape),
                           "dtype": "float32", "device": "cpu", "elements": parameter.numel(),
                           "trainable": parameter.requires_grad,
                           "content_identity": tensor_identity(parameter)})
    linears = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                entry = {"module": name, "weight": name+".weight", "lane": module.lane,
                         "elements": module.weight.numel()}
                if module.lane == "ternary":
                    value = quantize(module.weight)
                    entry.update(scale_hex=float(value.scale).hex(),
                                 code_identity=tensor_identity(value.codes),
                                 codes={str(i): int((value.codes == i).sum()) for i in (-1, 0, 1)})
                linears.append(entry)
    unique = sum(p.numel() for p in model.parameters())
    ternary = sum(e["elements"] for e in linears if e["lane"] == "ternary")
    package = Path(__file__).resolve().parent
    source = {name: identity((package/name).read_bytes())
              for name in ("contracts.py", "quantizer.py", "model.py", "inspection.py")}
    core = {"schema": "trinite.model-inventory.v1", "model": MODEL_ID,
            "config": model.config.to_dict(), "config_identity": model.config.content_identity,
            "lane": model.lane, "initialization_seed": model.seed,
            "source": source, "parameters": parameters, "linear_layers": linears,
            "modules": [{"name": name, "type": type(module).__name__}
                        for name, module in model.named_modules()],
            "counts": {"unique_parameters": unique, "ternary_forward_elements": ternary,
                       "floating_forward_elements": unique-ternary,
                       "latent_float32_bytes": unique*4},
            "forward": {"dtype": "float32", "device": "cpu", "attention": "explicit-matmul-softmax",
                        "padding": "nonempty-right-padded-prefix; zero masked states/logits",
                        "gelu": "exact", "rms_epsilon": model.config.rms_epsilon,
                        "quantizer_absmean_reduction": "float64-then-float32-scale"}}
    return {**core, "inventory_identity": identity(json_bytes(core)),
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "platform": platform.platform(), "byte_order": sys.byteorder,
                            "threads": torch.get_num_threads(),
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}


@dataclass(frozen=True)
class QuantizerSnapshot:
    codes: torch.Tensor
    scale: torch.Tensor


def quantizer_snapshots(model: Decoder, names: tuple[str, ...], *,
                        max_bytes: int = 64 * 1024) -> dict[str, QuantizerSnapshot]:
    """Copy only explicitly selected effective codes/scales, after preflight."""
    if not isinstance(model, Decoder):
        raise ContractError("quantizer inspection requires a Decoder")
    model.validate_state()
    modules = {name: m for name, m in model.named_modules()
               if isinstance(m, Linear) and m.lane == "ternary"}
    if (type(names) is not tuple or not names
            or any(type(n) is not str or n not in modules for n in names)
            or len(set(names)) != len(names)):
        raise ContractError("select distinct ternary linear module names")
    integer(max_bytes, "quantizer snapshot budget", 1, MAX_INSPECTION_BYTES)
    if sum(modules[n].weight.numel() + 4 for n in names) > max_bytes:
        raise ContractError("selected codes/scales exceed inspection byte budget")
    snapshots = {}
    with torch.no_grad():
        for name in names:
            value = quantize(modules[name].weight)
            snapshots[name] = QuantizerSnapshot(value.codes.clone(), value.scale.clone())
    return snapshots


def capture_inventory(captures: dict[str, torch.Tensor]) -> dict:
    """Summarize detached captures; numeric arrays stay in the explicit API."""
    if type(captures) is not dict or not all(type(n) is str for n in captures):
        raise ContractError("captures require a named mapping")
    if sum(t.numel()*t.element_size() for t in captures.values()
           if isinstance(t, torch.Tensor)) > MAX_INSPECTION_BYTES:
        raise ContractError("capture payload exceeds inspection budget")
    result = {}
    for name, tensor in captures.items():
        require_float_tensor(tensor, name)
        if tensor.requires_grad or tensor.grad_fn is not None:
            raise ContractError("inspection captures must be detached")
        result[name] = {"shape": list(tensor.shape), "dtype": "float32", "device": "cpu",
                        "content_identity": tensor_identity(tensor)}
    return result
