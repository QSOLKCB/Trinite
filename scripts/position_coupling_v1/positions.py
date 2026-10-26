"""Input-only arithmetic syntax and slot positions; never compute an answer."""
import re

from trinite.contracts import ContractError, integer

POSITION_SCHEMA = 'trinite.arithmetic-position-slots.v1'
MODES = ('sequence', 'coupled')
MAX_OPERAND_WIDTH = 4
_PATTERN = re.compile(r'([0-9]{1,4})\+([0-9]{1,4})=\Z', re.ASCII)


def operand_width(prompt: str) -> int:
    """Require already padded equal-width unsigned decimal operands."""
    if type(prompt) is not str:
        raise ContractError('arithmetic prompt must be a string')
    match = _PATTERN.fullmatch(prompt)
    if match is None or len(match[1]) != len(match[2]):
        raise ContractError('prompt requires equal-width 1..4 digit operands: digits+digits=')
    return len(match[1])


def prefix_positions(prompt: str, emitted_count: int, *, mode: str,
                     context_length: int = 64, offset: int = 0) -> tuple[int, ...]:
    """Positions for BOS + complete prompt + emitted slots, not prediction targets.

    The final (D+2-th) emitted slot is reserved for EOS and has position zero.
    No emitted token value or gold label is an input. Malformed generation has
    the same slots as correct generation; this function never repairs output.
    """
    width = operand_width(prompt)
    integer(context_length, 'context length', 2, 256)
    integer(emitted_count, 'emitted count', 0, width + 2)
    integer(offset, 'position offset', 0, context_length - 1)
    if type(mode) is not str or mode not in MODES:
        raise ContractError('position mode must be sequence or coupled')
    length = 1 + len(prompt) + emitted_count
    if length > context_length:
        raise ContractError('prefix exceeds declared context')
    if mode == 'sequence':
        result = tuple(offset + index for index in range(length))
    else:
        digits = tuple(offset + index for index in range(width, 0, -1))
        answer = tuple(offset + index + 1 if index < width + 1 else 0
                       for index in range(emitted_count))
        result = (0, *digits, 0, *digits, 0, *answer)
    if max(result) >= context_length:
        raise ContractError('position address exceeds embedding inventory')
    return result


def decode_answer(prompt: str, emitted: tuple[int, ...]) -> str | None:
    """Decode exact fixed-width reversed bytes + EOS; invalid output is None.

    This only converts a representation. It neither computes nor checks a sum.
    """
    width = operand_width(prompt)
    if (type(emitted) is not tuple or len(emitted) != width + 2
            or any(type(token) is not int for token in emitted)
            or emitted[-1] != 257 or any(not 48 <= token <= 57 for token in emitted[:-1])):
        return None
    text = ''.join(chr(token) for token in reversed(emitted[:-1]))
    return text.lstrip('0') or '0'
