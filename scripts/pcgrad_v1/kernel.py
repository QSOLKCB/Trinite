"""Independent two-objective PCGrad algebra; no model, optimizer or RNG mutation.

Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020.
Implemented from the mathematical method; no upstream code copied.
"""
from dataclasses import dataclass
import torch
from trinite.contracts import ContractError

MAX_ELEMENTS = 65536
ARMS = ('sum', 'pcgrad')


@dataclass(frozen=True)
class Combination:
    gradient: torch.Tensor
    projected: tuple[torch.Tensor, torch.Tensor]
    diagnostics: dict


def vector(value):
    if (type(value) is not torch.Tensor or value.device.type != 'cpu'
            or value.dtype != torch.float32 or value.layout != torch.strided
            or value.ndim != 1 or not 1 <= value.numel() <= MAX_ELEMENTS
            or not bool(torch.isfinite(value).all())):
        raise ContractError('gradient requires bounded finite CPU float32 flat vector')
    return value.detach().clone()


def combine(a, b, *, arm, order=(0, 1)):
    """Project each original against the other original, then SUM (not mean).

    float64 intermediates avoid float32 squared-norm overflow/underflow. The
    actual float32 cast and sum need not have exact real-arithmetic orthogonality.
    The two-objective order is recorded, but each has only one projection target.
    """
    if type(arm) is not str or arm not in ARMS:
        raise ContractError('unsupported gradient combination arm')
    if (type(order) is not tuple or len(order) != 2
            or any(type(i) is not int for i in order) or set(order) != {0, 1}):
        raise ContractError('projection order must be a permutation of (0, 1)')
    originals = (vector(a), vector(b))
    if originals[0].shape != originals[1].shape:
        raise ContractError('gradient vector shapes differ')
    wide = tuple(g.to(torch.float64) for g in originals)
    dot = torch.dot(*wide).item()
    norms = [torch.dot(g, g).item() for g in wide]
    projected = [g.clone() for g in originals]
    applied = [False, False]
    for i in order:
        j = 1-i
        if arm == 'pcgrad' and dot < 0 and norms[j] > 0:
            projected[i] = (wide[i] - (dot/norms[j])*wide[j]).to(torch.float32)
            applied[i] = True
    result = projected[0] + projected[1]
    if any(not bool(torch.isfinite(g).all()) for g in (*projected, result)):
        raise ContractError('nonfinite rounded projection or sum')
    cosine = None if 0 in norms else dot/(norms[0]**.5 * norms[1]**.5)
    return Combination(result, tuple(projected), {
        'dot_hex': dot.hex(), 'squared_norms_hex': [n.hex() for n in norms],
        'cosine_hex': None if cosine is None else cosine.hex(),
        'order': list(order), 'applied': applied,
        'rounded_residual_dots_hex': [torch.dot(projected[i].double(), wide[1-i]).item().hex()
                                      for i in (0, 1)]})
