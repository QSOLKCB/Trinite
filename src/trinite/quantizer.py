"""CPU float32 absmean ternary forward and an explicit surrogate gradient."""
from dataclasses import dataclass

import torch

from .contracts import ContractError

MAX_MATRIX_ELEMENTS = 2_000_000


def require_float_tensor(value: object, label: str) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != torch.float32 or value.layout != torch.strided):
        raise ContractError(f"{label} requires a strided CPU float32 tensor")
    if not bool(torch.isfinite(value).all()):
        raise ContractError(f"{label} must contain only finite values")
    return value


@dataclass(frozen=True)
class QuantizedWeight:
    codes: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor


class _Surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, latent, effective, inside):
        ctx.save_for_backward(inside)
        return effective

    @staticmethod
    def backward(ctx, gradient):
        (inside,) = ctx.saved_tensors
        return gradient * inside, None, None


def quantize(weight: torch.Tensor) -> QuantizedWeight:
    """Return int8 codes, detached scalar scale, and float32 STE weight.

    Absmean reduces detached float32 inputs in float64 to avoid sum overflow,
    then rounds the scale to float32. No subtract/add STE cancellation occurs.
    The chosen backward is not the derivative of the quantized forward.
    """
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
            or not 1 <= weight.numel() <= MAX_MATRIX_ELEMENTS):
        raise ContractError("quantizer requires a nonempty bounded weight matrix")
    require_float_tensor(weight, "quantizer weight")
    with torch.no_grad():
        scale = weight.detach().double().abs().mean().clamp_min(1e-8).float()
        unit = weight.detach() / scale
        codes = unit.round().clamp(-1, 1).to(torch.int8)
        effective = codes.float() * scale
        inside = unit.abs() < 1
    return QuantizedWeight(codes, scale, _Surrogate.apply(weight, effective, inside))


def quantize_four(weight: torch.Tensor) -> QuantizedWeight:
    """Classical odd four-level codebook; frozen comparison protocol v1.

    Zero maps to +1, +/-2 unit ties map inward, +/-3 STE boundaries stop.
    Scale is detached. Reject effective overflow rather than emitting infinity.
    """
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
            or not 1 <= weight.numel() <= MAX_MATRIX_ELEMENTS):
        raise ContractError("four-state quantizer requires a bounded weight matrix")
    require_float_tensor(weight, "four-state weight")
    with torch.no_grad():
        scale = (weight.detach().double().abs().mean()/2).clamp_min(1e-8).float()
        unit = weight.detach()/scale
        magnitude = torch.where(unit.abs() > 2, 3, 1)
        codes = (magnitude*torch.where(unit < 0, -1, 1)).to(torch.int8)
        effective = codes.float()*scale
        require_float_tensor(effective, "four-state effective weight")
        inside = unit.abs() < 3
    return QuantizedWeight(codes, scale, _Surrogate.apply(weight, effective, inside))
