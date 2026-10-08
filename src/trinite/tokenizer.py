"""UTF-8 byte tokens, source byte spans, and explicit padding/loss masks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import ContractError, TOKENIZER_ID, integer

BOS, EOS, PAD = 256, 257, 258
VOCABULARY_SIZE = 259


@dataclass(frozen=True)
class Encoding:
    input_ids: tuple[int, ...]
    byte_spans: tuple[tuple[int, int] | None, ...]
    attention_mask: tuple[int, ...]
    loss_mask: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "tokenizer_id": TOKENIZER_ID,
            "input_ids": list(self.input_ids),
            "byte_spans": [list(s) if s is not None else None for s in self.byte_spans],
            "attention_mask": list(self.attention_mask),
            "loss_mask": list(self.loss_mask),
        }


class ByteTokenizer:
    def __init__(self, context_length: int = 256):
        self.context_length = integer(context_length, "context_length", 2, 256)

    def encode(self, text: str, *, answer_start: int | None = None,
               pad_to: int | None = None) -> Encoding:
        """BOS + UTF-8 bytes + EOS. Offsets/loss masks refer to token positions.

        answer_start is a UTF-8 *byte* boundary. loss_mask marks tokens whose
        prediction is scored (the trainer later shifts it with the targets).
        BOS/PAD never score; EOS does. None means all byte tokens score.
        """
        if type(text) is not str:
            raise ContractError("text must be a string")
        try:
            raw = text.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ContractError("text contains invalid Unicode") from exc
        if len(raw) + 2 > self.context_length:
            raise ContractError("overlength input; truncation is not supported")
        start = 0 if answer_start is None else integer(answer_start, "answer_start", 0, len(raw))
        try:
            raw[:start].decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ContractError("answer_start splits a UTF-8 character") from exc
        length = len(raw) + 2
        target = length if pad_to is None else integer(pad_to, "pad_to", length, self.context_length)
        extra = target - length
        return Encoding(
            (BOS, *raw, EOS, *([PAD]*extra)),
            (None, *((i, i+1) for i in range(len(raw))), None, *([None]*extra)),
            (*([1]*length), *([0]*extra)),
            (0, *(int(i >= start) for i in range(len(raw))), 1, *([0]*extra)),
        )

    def decode_bytes(self, tokens: Iterable[int]) -> bytes:
        """Discard well-formed boundary specials; retain byte values exactly.

        Optional BOS must be first, optional EOS terminates content, and PAD
        may only be a suffix. No silently ignored internal special tokens.
        """
        values = tuple(tokens)
        output = bytearray()
        ended = False
        padded = False
        eos_seen = False
        for index, token in enumerate(values):
            integer(token, "token", 0, PAD)
            if token == BOS:
                if index != 0:
                    raise ContractError("BOS is only allowed at the beginning")
            elif token == EOS:
                if eos_seen or padded:
                    raise ContractError("unexpected EOS")
                eos_seen = ended = True
            elif token == PAD:
                padded = ended = True
            elif ended:
                raise ContractError("byte token follows EOS/PAD")
            else:
                output.append(token)
        return bytes(output)

    def decode(self, tokens: Iterable[int], *, errors: str = "strict") -> str:
        if errors not in ("strict", "replace", "backslashreplace"):
            raise ContractError("unsupported UTF-8 display policy")
        return self.decode_bytes(tokens).decode("utf-8", errors=errors)
