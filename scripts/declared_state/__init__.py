"""Opt-in declared labels and raw-text checks; never a model-state measurement.

No training, inference, repair, dataset admission or truth scoring lives here.
The five signed labels are ordered K, C, R, A, S. See docs/DECLARED-STATE.md.
"""

from dataclasses import dataclass
import re

from trinite.contracts import ContractError, integer

CODEC_ID = "trinite.declared-five-trit.v1"
OBSERVATION_ID = "trinite.declared-output-observation.v1"
REGISTERS = ("K", "C", "R", "A", "S")
MAX_OUTPUT_CHARS = 65_536
_HEX = re.compile(r"0x[0-9A-F]{2}\Z")


def encode_state(state: tuple[int, ...]) -> str:
    """Encode exactly five built-in signed trits, K most significant."""
    if type(state) is not tuple or len(state) != 5:
        raise ContractError("state must be an exact five-item tuple")
    value = 0
    for trit in state:
        integer(trit, "declared trit", -1, 1)
        value = 3 * value + trit + 1
    return f"0x{value:02X}"


def decode_state(code: str) -> tuple[int, ...]:
    """Accept only canonical 0x00..0xF2; reject aliases and reserved bytes."""
    if type(code) is not str or len(code) != 4 or _HEX.fullmatch(code) is None:
        raise ContractError("state code must be canonical 0xHH with uppercase digits")
    value = integer(int(code[2:], 16), "declared state byte", 0, 242)
    reverse = []
    for _ in REGISTERS:
        value, digit = divmod(value, 3)
        reverse.append(digit - 1)
    return tuple(reversed(reverse))


@dataclass(frozen=True)
class OutputObservation:
    """Raw text plus objective checks; issues=() makes no correctness claim."""

    raw_output: str
    code: str | None
    state: tuple[int, ...] | None
    body: str | None
    word_count: int | None
    issues: tuple[str, ...]
    schema: str = OBSERVATION_ID


def observe_output(
    raw_output: str, *, exact_words: int | None = None,
    banned_chars: tuple[str, ...] = (),
) -> OutputObservation:
    """Inspect canonical ``0xHH SPACE body`` without trimming or repairing it.

    The separator is one ASCII space; subsequent characters belong to the body
    verbatim. Word count uses Python str.split() (Unicode whitespace). Bans are
    case-sensitive Unicode code points, without normalization. The 65,536 limit
    counts Python string characters, not encoded bytes or tokenizer tokens.
    Malformed envelopes retain raw_output and have no interpreted body/state.
    Empty/whitespace-only bodies fail even when no other constraint is supplied.
    """
    if type(raw_output) is not str or len(raw_output) > MAX_OUTPUT_CHARS:
        raise ContractError("raw_output must be an exact bounded string")
    if exact_words is not None:
        integer(exact_words, "exact word count", 0, MAX_OUTPUT_CHARS)
    if (type(banned_chars) is not tuple or len(banned_chars) > 256
            or any(type(char) is not str or len(char) != 1 for char in banned_chars)
            or len(set(banned_chars)) != len(banned_chars)):
        raise ContractError("banned_chars must be an exact tuple of <=256 unique characters")
    try:
        code = raw_output[:4]
        state = decode_state(code)
    except ContractError:
        return OutputObservation(raw_output, None, None, None, None, ("invalid_state_code",))
    if raw_output[4:5] != " ":
        return OutputObservation(raw_output, None, None, None, None, ("missing_separator",))
    body = raw_output[5:]
    count = len(body.split())
    issues = []
    if not body.strip():
        issues.append("empty_body")
    if exact_words is not None and count != exact_words:
        issues.append("exact_word_count_mismatch")
    if any(char in body for char in banned_chars):
        issues.append("banned_character_present")
    return OutputObservation(raw_output, code, state, body, count, tuple(issues))
