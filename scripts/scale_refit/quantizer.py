"""Independent TWN-inspired fixed-code least-squares scale refit.

Only the scale changes: codes and the historical surrogate mask stay fixed.
No paper implementation is copied. See docs/SCALE-REFIT.md for attribution.
"""
from dataclasses import dataclass

import torch

from trinite.contracts import ContractError
from trinite.quantizer import QuantizedWeight, _Surrogate, quantize, require_float_tensor

QUANTIZER_ID = 'trinite.ternary-fixed-code-scale-refit.v1'
MAX_ELEMENTS = 65_536
SSE_DENOMINATOR_EXPONENT = 298


def _integers(value: torch.Tensor) -> list[int]:
    # Every finite binary32 value is exactly an integer times 2**-149.
    # Python float represents every binary32 exactly; no rounded subtraction.
    result = []
    for item in value.detach().reshape(-1).tolist():
        numerator, denominator = item.as_integer_ratio()
        result.append(numerator * ((1 << 149) // denominator))
    return result


def exact_sse_numerator(reference: torch.Tensor, actual: torch.Tensor) -> int:
    """Exact squared error = returned nonnegative integer / 2**298.

    Bounded finite CPU float32 tensors only; compares actual rounded outputs.
    Integer arithmetic avoids overflow, underflow, and cancellation even when
    float64 SSE summation would lose a small improvement beside a huge error.
    """
    require_float_tensor(reference, 'SSE reference')
    require_float_tensor(actual, 'SSE actual')
    if reference.shape != actual.shape or not 1 <= reference.numel() <= MAX_ELEMENTS:
        raise ContractError('SSE requires matching nonempty bounded tensors')
    return sum((a-b)**2 for a, b in zip(_integers(reference), _integers(actual)))


@dataclass(frozen=True)
class RefitResult:
    quantized: QuantizedWeight
    baseline_scale: torch.Tensor
    baseline_sse_numerator: int
    selected_sse_numerator: int
    decision: str
    quantizer_id: str = QUANTIZER_ID


def _candidate_scale(weight: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    support = codes != 0
    return (weight.detach().double()[support] * codes[support].double()).mean().float()


def refit(weight: torch.Tensor) -> RefitResult:
    """Generate a rounded least-squares candidate and certify actual SSE.

    The real fixed-code optimum is sum(w*q)/sum(q*q). Nonzero baseline codes
    have the sign of w. Reduce in float64 and cast the resulting mean to
    float32. Accept only a finite candidate with strictly lower exact SSE;
    otherwise return the original reconstruction and scale. This does not
    assert a floating-point global optimum or lower linear-output error.
    """
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
            or not 1 <= weight.numel() <= MAX_ELEMENTS):
        raise ContractError('scale refit requires a bounded nonempty matrix')
    baseline = quantize(weight)
    with torch.no_grad():
        baseline_sse = exact_sse_numerator(weight, baseline.weight)
        support = baseline.codes != 0
        scale, effective = baseline.scale, baseline.weight.detach()
        selected_sse = baseline_sse
        decision = 'zero_support'
        if bool(support.any()):
            candidate_scale = _candidate_scale(weight, baseline.codes)
            candidate = baseline.codes.float() * candidate_scale
            decision = 'nonfinite_candidate'
            if bool(torch.isfinite(candidate_scale)) and bool(torch.isfinite(candidate).all()):
                candidate_sse = exact_sse_numerator(weight, candidate)
                decision = 'no_strict_improvement'
                if candidate_sse < baseline_sse:
                    scale, effective = candidate_scale, candidate
                    selected_sse = candidate_sse
                    decision = 'improved'
        # Deliberately use the historical absmean scale, not the refitted scale.
        inside = (weight.detach() / baseline.scale).abs() < 1
    selected = QuantizedWeight(baseline.codes, scale,
                               _Surrogate.apply(weight, effective, inside))
    return RefitResult(selected, baseline.scale, baseline_sse, selected_sse, decision)
