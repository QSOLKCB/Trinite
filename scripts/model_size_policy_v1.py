"""Prospective checkout-only model-size policy; not a historical runtime retrofit."""

from trinite.contracts import ContractError, ModelConfig

POLICY_ID = "trinite.model-size-policy.v1"
MAX_MODEL_PARAMETERS = 8_000_000_000


def preflight_parameter_count(count: int) -> int:
    """Validate a complete unique-parameter count before model allocation.

    This validates the supplied count, not its completeness or provenance.
    Architecture-specific callers must derive it from validated config shapes.
    Python integers are exact; no float conversion or fixed-width arithmetic.
    """
    if type(count) is not int or not 1 <= count <= MAX_MODEL_PARAMETERS:
        raise ContractError("model requires 1..8,000,000,000 unique parameters")
    return count


def require_matching_count(expected: int, actual: int) -> int:
    """Reject invalid, over-ceiling or mismatched config/inventory counts."""
    preflight_parameter_count(expected)
    preflight_parameter_count(actual)
    if actual != expected:
        raise ContractError("actual parameter inventory disagrees with config")
    return actual


def build_native_decoder(config: ModelConfig, *, lane: str = "ternary",
                         seed: int = 0, parameter_budget: int = 2_000_000):
    """Opt-in native construction with preflight and authoritative native checks.

    The current 5M implementation ceiling and smaller per-run budget still apply.
    This is not a generic model loader, arbitrary factory or scaling permission.
    """
    if type(config) is not ModelConfig:
        raise ContractError("policy adapter requires an exact native ModelConfig")
    # Revalidate rather than trusting even a forcibly mutated frozen dataclass.
    validated = ModelConfig.from_dict(config.to_dict())
    expected = preflight_parameter_count(validated.parameter_count)
    # Import only after config preflight; no backend needed by scalar policy.
    from trinite.model import Decoder
    model = Decoder(validated, lane=lane, seed=seed,
                    parameter_budget=parameter_budget)
    require_matching_count(expected, sum(p.numel() for p in model.parameters()))
    return model
