"""Independent symmetric ternary adaptation of LSQ's scale surrogate.

This uses projected direct scales, not a log/softplus parameterization. Endpoint
surrogates are explicitly frozen; they are not finite-difference derivatives.
"""
import math
import torch
from trinite.contracts import ContractError
from trinite.quantizer import QuantizedWeight, require_float_tensor

QUANTIZER_ID = 'trinite.learned-positive-ternary-ste.v1'
FLOOR = 1e-8
MAX_ELEMENTS = 65536


def require_scale(scale):
    require_float_tensor(scale, 'learned ternary scale')
    if scale.ndim != 0 or bool(scale < FLOOR):
        raise ContractError('scale must be scalar and at least float32(1e-8)')
    return scale


def initial_scale(weight):
    require_matrix(weight)
    return weight.detach().double().abs().mean().clamp_min(FLOOR).float()


def require_matrix(weight):
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
            or not 1 <= weight.numel() <= MAX_ELEMENTS):
        raise ContractError('learned ternary requires a bounded nonempty matrix')
    require_float_tensor(weight, 'learned ternary weight')


class _Surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale, effective, inside, h):
        ctx.save_for_backward(inside, h)
        return effective

    @staticmethod
    def backward(ctx, gradient):
        require_float_tensor(gradient, 'incoming learned-scale gradient')
        inside, h = ctx.saved_tensors
        dw = torch.where(inside, gradient, 0.)
        ds = ((gradient.double()*h.double()).sum()/math.sqrt(gradient.numel())).float()
        require_float_tensor(ds, 'learned-scale surrogate gradient')
        return dw, ds, None, None, None


def quantize(weight, scale):
    require_matrix(weight)
    require_scale(scale)
    if torch.is_autocast_enabled('cpu'):
        raise ContractError('autocast is outside learned-scale float32 contract')
    with torch.no_grad():
        unit = weight.detach()/scale.detach()
        codes = unit.round().clamp(-1, 1).to(torch.int8)
        effective = codes.float()*scale.detach()
        inside = unit.abs() < 1
        # Finite weights and positive scales can still yield +/-inf unit values.
        # Clamp before subtraction, never compute an inf residual or 0*inf.
        residual = codes.float()-unit.clamp(-1, 1)
        h = torch.where(unit <= -1, -1., torch.where(unit >= 1, 1., residual))
        require_float_tensor(effective, 'learned ternary effective weight')
    return QuantizedWeight(codes, scale.detach().clone(),
                           _Surrogate.apply(weight, scale, effective, inside, h))


def project_scale_(scale):
    """Post-AdamW positive projection; moments are deliberately untouched."""
    require_float_tensor(scale, 'scale before projection')
    if scale.ndim != 0:
        raise ContractError('projection requires scalar scale')
    with torch.no_grad():
        scale.clamp_(min=FLOOR)
    require_scale(scale)
