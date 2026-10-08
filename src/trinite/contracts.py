"""Versioned configuration and exact-byte identities for the Phase 1 artifacts.

This is Trinite's own JSON artifact format, not a PROVENANCE serializer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

CONFIG_SCHEMA = "trinite.model-config.v1"
TOKENIZER_ID = "trinite.utf8-byte.v1"
MAX_SAFE_INTEGER = 9_007_199_254_740_991
_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ContractError(ValueError):
    """An input violates a versioned Trinite contract."""


def integer(value: object, label: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ContractError(f"{label} must be an integer in [{low}, {high}]")
    return value


def exact_keys(value: object, keys: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise ContractError(f"{label} requires exactly: {', '.join(sorted(keys))}")
    return value


def _json_value(value: Any) -> None:
    if value is None or type(value) in (bool, str):
        return
    if type(value) is int:
        integer(value, "JSON integer", -MAX_SAFE_INTEGER, MAX_SAFE_INTEGER)
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():
            _json_value(item)
        return
    raise ContractError("JSON v1 accepts only strings, safe integers, bool, null, lists, objects")


def json_bytes(value: Any) -> bytes:
    _json_value(value)
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise ContractError("invalid JSON v1 value") from exc


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data: bytes, *, canonical: bool = False) -> Any:
    if not isinstance(data, bytes) or data.startswith(b"\xef\xbb\xbf"):
        raise ContractError("JSON requires UTF-8 bytes without BOM")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs)
        encoded = json_bytes(value)
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise ContractError(f"invalid JSON v1: {exc}") from exc
    if canonical and data != encoded:
        raise ContractError("artifact bytes are not canonical Trinite JSON v1")
    return value


def identity(data: bytes) -> str:
    if type(data) is not bytes:
        raise ContractError("content identity requires exact bytes")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def require_identity(value: object) -> str:
    if type(value) is not str or _IDENTITY.fullmatch(value) is None:
        raise ContractError("identity requires sha256:<64 lowercase hex>")
    return value


@dataclass(frozen=True)
class ModelConfig:
    schema: str = CONFIG_SCHEMA
    tokenizer_id: str = TOKENIZER_ID
    vocabulary_size: int = 259
    context_length: int = 256
    blocks: int = 6
    width: int = 128
    heads: int = 4
    feed_forward_width: int = 512
    rms_epsilon: str = "0.00001"
    positional_encoding: str = "learned"
    activation: str = "gelu-exact"
    linear_bias: bool = False
    dropout: str = "0"
    tied_output: bool = True
    dtype: str = "float32"
    quantizer: str = "trinite.absmean-ste.v0"
    geometry_loss: str = "0"

    def __post_init__(self) -> None:
        for key, expected in {
            "schema": CONFIG_SCHEMA, "tokenizer_id": TOKENIZER_ID,
            "vocabulary_size": 259, "rms_epsilon": "0.00001",
            "positional_encoding": "learned", "activation": "gelu-exact",
            "linear_bias": False, "dropout": "0", "tied_output": True,
            "dtype": "float32", "quantizer": "trinite.absmean-ste.v0",
            "geometry_loss": "0",
        }.items():
            value = getattr(self, key)
            if type(value) is not type(expected) or value != expected:
                raise ContractError(f"unsupported {key}; expected {expected!r}")
        integer(self.context_length, "context_length", 2, 256)
        integer(self.blocks, "blocks", 1, 24)
        integer(self.width, "width", 1, 1024)
        integer(self.heads, "heads", 1, self.width)
        integer(self.feed_forward_width, "feed_forward_width", 1, 4096)
        if self.width % self.heads:
            raise ContractError("width must be divisible by heads")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> ModelConfig:
        return cls(**exact_keys(value, set(cls.__dataclass_fields__), "model config"))

    @classmethod
    def load(cls, path: Path) -> ModelConfig:
        return cls.from_dict(parse_json(Path(path).read_bytes()))

    @property
    def content_identity(self) -> str:
        return identity(json_bytes(self.to_dict()))

    @property
    def parameter_count(self) -> int:
        d = self.width
        return (self.blocks * (4*d*d + 2*d*self.feed_forward_width + 2*d)
                + self.vocabulary_size*d + self.context_length*d + d)
