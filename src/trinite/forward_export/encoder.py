"""Pure bounded FINAL-FORWARD encoding using the one native quantizer.

There is deliberately no tensor reader, checkpoint loader, file writer, CLI,
training dependency or inference backend here. The caller owns an idle model
and explicitly captures source bytes before calling this function.
"""
import math
import platform
import sys

import numpy as np
import safetensors
from safetensors.torch import save
import torch
from torch import nn

from ..contracts import (ContractError, ModelConfig, TOKENIZER_ID, identity,
                         integer, json_bytes, require_identity)
from ..inspection import tensor_identity
from ..model import Block, Decoder, Linear, MAX_PARAMETERS, MODEL_ID, RMSNorm
from ..packed_ternary import CODEC_ID, MAX_CODE_COUNT, pack_codes
from ..quantizer import quantize
from ..tokenizer import BOS, EOS, PAD, VOCABULARY_SIZE
from . import FORMAT_ID, ForwardExport, MAX_MANIFEST_BYTES, MAX_PAYLOAD_BYTES
from .source import SourceSnapshot


def _layout(config):
    """Expected registered structure, derived from a revalidated config."""
    d, f = config.width, config.feed_forward_width
    modules = [("", Decoder, ("token_embedding", "position_embedding", "output_weight"),
                ("blocks", "final_norm")),
               ("blocks", nn.ModuleList, (), tuple(str(i) for i in range(config.blocks)))]
    shapes = {"token_embedding": (VOCABULARY_SIZE, d),
              "position_embedding": (config.context_length, d)}
    linears = set()
    for i in range(config.blocks):
        prefix = f"blocks.{i}"
        children = ("attention_norm", "q", "k", "v", "o", "feed_forward_norm", "up", "down")
        modules.append((prefix, Block, (), children))
        for name in children:
            path = prefix + "." + name
            norm = name in ("attention_norm", "feed_forward_norm")
            modules.append((path, RMSNorm if norm else Linear, ("weight",), ()))
            weight = path + ".weight"
            shapes[weight] = ((d,) if norm else (f, d) if name == "up" else
                              (d, f) if name == "down" else (d, d))
            if not norm:
                linears.add(weight)
    modules.append(("final_norm", RMSNorm, ("weight",), ()))
    shapes["final_norm.weight"] = (d,)
    return modules, shapes, linears


def _preflight(model, max_payload_bytes, max_manifest_bytes):
    integer(max_payload_bytes, "export payload budget", 1, MAX_PAYLOAD_BYTES)
    integer(max_manifest_bytes, "export manifest budget", 1, MAX_MANIFEST_BYTES)
    if type(model) is not Decoder or type(model.config) is not ModelConfig:
        raise ContractError("export requires the exact native Decoder and ModelConfig")
    # Avoid dataclasses.asdict/deepcopy on a maliciously altered frozen config.
    config = ModelConfig.from_dict(vars(model.config).copy())
    if (model.config != model._construction_config or type(model.lane) is not str
            or model.lane != "ternary"):
        raise ContractError("export requires an unchanged config and the ternary lane")
    integer(model.seed, "initialization seed label", 0, 2**32-1)
    integer(config.parameter_count, "unique parameter count", 1, MAX_PARAMETERS)
    if sys.byteorder != "little" or torch.is_autocast_enabled("cpu"):
        raise ContractError("export requires little-endian CPU float32 without autocast")
    specifications, shapes, linears = _layout(config)
    # Bound each matrix before finite scans, tensor copies, quantization or packing.
    for name in linears:
        if math.prod(shapes[name]) > MAX_CODE_COUNT:
            raise ContractError("individual matrix exceeds the packed codec code ceiling")
    packed_bytes = sum((math.prod(shapes[n])+3)//4 for n in linears)
    float_elements = sum(math.prod(s) for n, s in shapes.items() if n not in linears)
    float_bytes = 4 * (float_elements + len(linears))
    # Conservative, public preflight allowances, not claimed RSS limits. Actual
    # serialized bytes are checked again and reported separately in the manifest.
    float_members = len(shapes)
    payload_ceiling = packed_bytes + float_bytes + 8 + 1024 * float_members
    metadata_ceiling = 16384 + 2048 * len(shapes) + 512 * len(specifications)
    if payload_ceiling > max_payload_bytes or metadata_ceiling > max_manifest_bytes:
        raise ContractError("export exceeds conservative payload/metadata preflight budget")
    if (type(model._parameter_shapes) is not dict
            or any(type(shape) is not tuple or any(type(d) is not int for d in shape)
                   for shape in model._parameter_shapes.values())
            or model._parameter_shapes != shapes):
        raise ContractError("constructed tensor inventory differs from the canonical config")

    # Native state validation alone does not reject hooks, extra buffers, aliased
    # linears, subclasses or replaced block implementations. Close that boundary.
    for key, value in vars(torch.nn.modules.module).items():
        if key.startswith("_global_") and "hook" in key and value:
            raise ContractError("global module hooks are outside the export profile")
    modules = {}
    parameters = {}
    seen_storage = []
    for name, kind, parameter_names, child_names in specifications:
        if not name:
            module = model
        else:
            parent, _, leaf = name.rpartition(".")
            module = modules[parent]._modules[leaf]
        if (type(module) is not kind or type(module._parameters) is not dict
                or tuple(module._parameters) != parameter_names
                or type(module._modules) is not dict or tuple(module._modules) != child_names
                or module._buffers or module._non_persistent_buffers_set):
            raise ContractError("export requires the canonical native module/tensor inventory")
        # nn.Module resolves ordinary instance attributes before registered
        # parameters/children. A __dict__ shadow could otherwise make forward
        # use different weights or operations from this registered inventory.
        for leaf in parameter_names:
            if leaf in vars(module) or getattr(module, leaf) is not module._parameters[leaf]:
                raise ContractError("registered parameter is shadowed by attribute resolution")
        for leaf in child_names:
            if leaf in vars(module) or getattr(module, leaf) is not module._modules[leaf]:
                raise ContractError("registered child module is shadowed by attribute resolution")
        for key, value in vars(module).items():
            if (("hook" in key and value) or callable(value) or key in {
                    "forward", "validate_state", "_validate_inputs", "_call_impl",
                    "_wrapped_call_impl", "named_parameters", "named_modules", "modules",
                    "parameters", "__call__"}
                    or (key == "_compiled_call_impl" and value is not None)):
                raise ContractError("module hooks and overridden execution are not exportable")
        if kind is Linear and (type(module.lane) is not str or module.lane != "ternary"):
            raise ContractError("linear lane must be exact ternary")
        if kind is Block and (type(module.heads) is not int or type(module.head_width) is not int):
            raise ContractError("attention head dimensions must be exact integers")
        modules[name] = module
        for leaf in parameter_names:
            tensor = module._parameters[leaf]
            path = name + "." + leaf if name else leaf
            if path == "output_weight":
                if tensor is not model.token_embedding:
                    raise ContractError("output weight must be the token embedding alias")
                continue
            if (type(tensor) is not nn.Parameter or tuple(tensor.shape) != shapes[path]
                    or tensor.device.type != "cpu" or tensor.dtype != torch.float32
                    or tensor.layout != torch.strided or not tensor.is_contiguous()
                    or tensor.storage_offset() != 0
                    or tensor.untyped_storage().nbytes() != tensor.numel()*4):
                raise ContractError("export requires canonical contiguous CPU float32 parameters")
            start = tensor.untyped_storage().data_ptr()
            end = start + tensor.untyped_storage().nbytes()
            # Separately wrapped external buffers can have distinct storage
            # objects and starting pointers while still sharing physical bytes.
            # Half-open intervals reject every intersection but allow adjacency.
            if any(start < previous_end and previous_start < end
                   for previous_start, previous_end in seen_storage):
                raise ContractError("unexpected overlapping parameter storage")
            seen_storage.append((start, end))
            parameters[path] = tensor
    # Call the authoritative validation operation without trusting an instance override.
    Decoder.validate_state(model)
    return config, modules, parameters, linears, packed_bytes, float_bytes


def export_forward(model: Decoder, *, source: SourceSnapshot,
                   expected_source_identity: str,
                   max_payload_bytes: int = MAX_PAYLOAD_BYTES,
                   max_manifest_bytes: int = MAX_MANIFEST_BYTES) -> ForwardExport:
    """Encode the actual current ternary forward weights as immutable bytes.

    ``source`` must be explicitly captured with ``read_source_snapshot``; retain
    its expected identity independently. Source freshness and a trusted unchanged
    interpreter are caller responsibilities. This API neither loads nor proves
    origin from a checkpoint, and accepts no arbitrary checkpoint-origin hash.
    The model must be idle (no concurrent updates). No model forward, RNG, gradient
    update, file/network operation, backend substitution or checkpoint/tensor deserialization occurs.
    """
    require_identity(expected_source_identity)
    if type(source) is not SourceSnapshot:
        raise ContractError("export requires an immutable explicit SourceSnapshot")
    source_receipt = source.receipt()
    if source.content_identity != expected_source_identity:
        raise ContractError("source snapshot differs from expected source identity")
    config, modules, parameters, linears, packed_size, float_size = _preflight(
        model, max_payload_bytes, max_manifest_bytes)
    versions = {name: tensor._version for name, tensor in parameters.items()}
    latent = []
    ternary = []
    floating = []
    tensors = {}
    chunks = []
    offset = 0
    with torch.no_grad():
        for name, parameter in parameters.items():
            latent.append({"name": name,
                           "aliases": ["output_weight"] if name == "token_embedding" else [],
                           "shape": list(parameter.shape), "dtype": "float32",
                           "elements": parameter.numel(), "bytes": parameter.numel()*4,
                           "identity": tensor_identity(parameter)})
            if name in linears:
                value = quantize(parameter)
                if (type(value.codes) is not torch.Tensor or type(value.scale) is not torch.Tensor
                        or value.codes.device.type != "cpu" or value.codes.layout != torch.strided
                        or value.codes.dtype != torch.int8 or value.codes.shape != parameter.shape
                        or value.scale.shape != () or value.scale.dtype != torch.float32
                        or value.scale.device.type != "cpu" or not bool(torch.isfinite(value.scale))
                        or not float(value.scale) > 0):
                    raise ContractError("native quantizer returned invalid codes/float32 scale")
                codes = tuple(value.codes.reshape(-1).tolist())
                packed = pack_codes(codes)
                scale_name = "scale." + name
                tensors[scale_name] = value.scale.detach().clone().contiguous()
                ternary.append({"name": name, "shape": list(parameter.shape),
                                "code_count": parameter.numel(), "code_dtype": "int8",
                                "code_identity": tensor_identity(value.codes),
                                "occupancy": {str(i): codes.count(i) for i in (-1, 0, 1)},
                                "offset": offset, "bytes": len(packed), "identity": identity(packed),
                                "scale_tensor": scale_name, "scale_hex": float(value.scale).hex(),
                                "scale_identity": tensor_identity(value.scale)})
                chunks.append(packed)
                offset += len(packed)
            else:
                key = "float." + name
                tensors[key] = parameter.detach().clone().contiguous()
                floating.append({"name": name, "tensor": key, "shape": list(parameter.shape),
                                 "dtype": "float32", "elements": parameter.numel(),
                                 "bytes": parameter.numel()*4,
                                 "identity": tensor_identity(tensors[key])})
        packed_payload = b"".join(chunks)
        float_payload = save(tensors)
        if (len(packed_payload) != packed_size
                or len(packed_payload)+len(float_payload) > max_payload_bytes):
            raise ContractError("serialized export exceeds payload budget or predicted code count")
        # Do not publish a torn snapshot when ordinary in-place updates interleave.
        for entry in latent:
            name = entry["name"]
            if (parameters[name]._version != versions[name]
                    or tensor_identity(parameters[name]) != entry["identity"]):
                raise ContractError("model changed during final-forward export")
        if model.config != config:
            raise ContractError("model config changed during final-forward export")
        final = _preflight(model, max_payload_bytes, max_manifest_bytes)
        if (any(final[1][n] is not module for n, module in modules.items())
                or any(final[2][n] is not tensor for n, tensor in parameters.items())):
            raise ContractError("model inventory changed during final-forward export")
    state = {"schema": "trinite.final-forward-model-state.v1", "model": MODEL_ID,
             "config_identity": config.content_identity, "lane": "ternary", "tensors": latent}
    float_inventory = {name: {"shape": list(tensor.shape), "dtype": "float32",
                              "bytes": tensor.numel()*4, "identity": tensor_identity(tensor)}
                       for name, tensor in sorted(tensors.items())}
    tokenizer = {"id": TOKENIZER_ID, "byte_tokens": [0, 255], "bos": BOS,
                 "eos": EOS, "pad": PAD, "vocabulary_size": VOCABULARY_SIZE,
                 "source_identity": next(e["identity"] for e in source_receipt["members"]
                                         if e["name"] == "tokenizer.py")}
    manifest = {
        "schema": FORMAT_ID, "claim": "engineering-candidate; final-forward encode only",
        "model": MODEL_ID, "lane": "ternary", "quantizer": config.quantizer,
        "config": config.to_dict(), "config_identity": config.content_identity,
        "tokenizer": tokenizer, "tokenizer_identity": identity(json_bytes(tokenizer)),
        "source": source_receipt, "source_identity": expected_source_identity,
        "source_claim": "caller-supplied captured input bytes; no executed-code attestation",
        "state": state, "state_identity": identity(json_bytes(state)),
        "initialization_seed_label": model.seed,
        "lineage": "current in-memory state only; no checkpoint or training-origin proof",
        "modules": [{"name": name, "type": type(module).__name__}
                    for name, module in modules.items()],
        "codec": CODEC_ID, "byte_order": "little", "matrix_order": "row-major",
        "ternary_tensors": ternary, "floating_tensors": floating,
        "float_tensor_inventory": float_inventory,
        "members": {
            "packed_codes": {"bytes": len(packed_payload), "identity": identity(packed_payload)},
            "float_tensors": {"bytes": len(float_payload), "identity": identity(float_payload),
                              "format": "safetensors"}},
        "counts": {"unique_parameters": config.parameter_count, "matrices": len(linears),
                   "ternary_codes": sum(parameters[n].numel() for n in linears),
                   "floating_forward_elements": sum(e["elements"] for e in floating),
                   "scale_elements": len(linears), "latent_float32_bytes": config.parameter_count*4,
                   "packed_code_bytes": len(packed_payload), "float_raw_bytes": float_size,
                   "safetensors_framing_bytes": len(float_payload)-float_size,
                   "serialized_payload_bytes": len(packed_payload)+len(float_payload)},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": str(np.__version__), "safetensors": str(safetensors.__version__),
                        "machine": platform.machine(), "byte_order": sys.byteorder,
                        "threads": torch.get_num_threads(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
    }
    raw = json_bytes(manifest)
    if len(raw) > max_manifest_bytes:
        raise ContractError("serialized export exceeds manifest byte budget")
    return ForwardExport(raw, packed_payload, float_payload)
