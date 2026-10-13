"""Bounded in-memory ternary code packing; not a tensor or model importer.

The caller supplies already-quantized codes in logical row-major order. This
primitive stores no shape, scale, tensor name, hash, or model metadata and does
not select a model weight format. See docs/INFERENCE.md for remaining gates.
"""

from ..contracts import ContractError, integer

CODEC_ID = "trinite.ternary-2bit.v1"
MAX_CODE_COUNT = 1_048_576
MAX_PAYLOAD_BYTES = (MAX_CODE_COUNT + 3) // 4
_PAIR_TO_CODE = (0, 1, -1)


def pack_codes(codes: tuple[int, ...]) -> bytes:
    """Pack an exact tuple of built-in -1/0/+1 integers into immutable bytes.

    Four codes occupy each byte, least-significant pair first, with 00=0,
    01=+1, 10=-1 and zero padding. Empty input is allowed. All input is checked
    before allocating the bounded output. Arbitrary iterables, booleans, floats
    and subclasses are rejected rather than coerced or consumed.
    """
    if type(codes) is not tuple:
        raise ContractError("codes must be an exact tuple")
    count = integer(len(codes), "code count", 0, MAX_CODE_COUNT)
    if any(type(code) is not int or code not in (-1, 0, 1) for code in codes):
        raise ContractError("codes must contain only built-in integers -1, 0, +1")

    packed = bytearray((count + 3) // 4)
    for index, code in enumerate(codes):
        pair = 2 if code == -1 else code
        packed[index // 4] |= pair << (2 * (index % 4))
    return bytes(packed)


def unpack_codes(payload: bytes, code_count: int) -> tuple[int, ...]:
    """Validate a complete payload and return its exact ordered integer codes.

    The externally supplied count is required: trailing zero codes and padding
    have identical bits. Reject reserved 11 pairs, nonzero padding, any missing
    or trailing bytes, non-built-in input types and counts outside the fixed
    bound. Validate all bytes before allocating the decoded tuple.
    """
    count = integer(code_count, "code count", 0, MAX_CODE_COUNT)
    if type(payload) is not bytes:
        raise ContractError("payload must be exact bytes")
    if len(payload) != (count + 3) // 4:
        raise ContractError("payload byte count does not match code count")

    remainder = count % 4
    if remainder and payload[-1] >> (2 * remainder):
        raise ContractError("nonzero ternary padding pairs")
    # A reserved 11 pair has both bits set at an even-positioned low bit.
    if any(byte & (byte >> 1) & 0x55 for byte in payload):
        raise ContractError("reserved ternary code pair 11")

    return tuple(
        _PAIR_TO_CODE[(payload[index // 4] >> (2 * (index % 4))) & 3]
        for index in range(count)
    )
