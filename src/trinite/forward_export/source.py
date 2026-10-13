"""Explicit, bounded installed-source capture, separate from the pure encoder."""
from dataclasses import dataclass
from pathlib import Path

from ..contracts import ContractError, identity, json_bytes, require_identity

SOURCE_SCHEMA = "trinite.final-forward-source.v1"
SOURCE_FILES = (
    "__init__.py", "contracts.py", "cpu-dependencies.lock",
    "forward_export/__init__.py", "forward_export/encoder.py",
    "forward_export/source.py", "inspection.py", "model.py",
    "packed_ternary/__init__.py", "quantizer.py", "tokenizer.py",
)
MAX_SOURCE_MEMBER_BYTES = 128 * 1024
MAX_SOURCE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SourceSnapshot:
    """Immutable captured source bytes, not an attestation of process integrity."""
    members: tuple[tuple[str, bytes], ...]

    def receipt(self) -> dict:
        if type(self.members) is not tuple or len(self.members) != len(SOURCE_FILES):
            raise ContractError("source requires the complete fixed runtime closure")
        total = 0
        entries = []
        for item, expected in zip(self.members, SOURCE_FILES):
            if (type(item) is not tuple or len(item) != 2 or type(item[0]) is not str
                    or item[0] != expected or type(item[1]) is not bytes
                    or not 1 <= len(item[1]) <= MAX_SOURCE_MEMBER_BYTES):
                raise ContractError("invalid, missing, reordered or excessive source member")
            total += len(item[1])
            if total > MAX_SOURCE_BYTES:
                raise ContractError("source snapshot exceeds aggregate byte budget")
            entries.append({"name": item[0], "bytes": len(item[1]),
                            "identity": identity(item[1])})
        return {"schema": SOURCE_SCHEMA, "members": entries, "bytes": total}

    @property
    def content_identity(self) -> str:
        return identity(json_bytes(self.receipt()))


def read_source_snapshot(*, expected_identity: str | None = None) -> SourceSnapshot:
    """Explicitly read the fixed installed runtime closure, never caller paths.

    This is the only filesystem operation in this package. Capture immediately
    before encoding in a trusted, unchanged process. An externally retained
    expected identity detects changed installed bytes. It does not establish
    checkpoint origin, a Git commit, training history or executed-code attestation.
    """
    if expected_identity is not None:
        require_identity(expected_identity)
    package = Path(__file__).resolve().parents[1]
    members = []
    total = 0
    for name in SOURCE_FILES:
        path = package / name
        if path.is_symlink() or not path.is_file():
            raise ContractError("source member must be a regular installed file")
        with path.open("rb") as stream:
            raw = stream.read(min(MAX_SOURCE_MEMBER_BYTES, MAX_SOURCE_BYTES-total)+1)
        if not 1 <= len(raw) <= MAX_SOURCE_MEMBER_BYTES:
            raise ContractError("source member exceeds byte budget or is empty")
        total += len(raw)
        if total > MAX_SOURCE_BYTES:
            raise ContractError("source snapshot exceeds aggregate byte budget")
        members.append((name, raw))
    result = SourceSnapshot(tuple(members))
    if expected_identity is not None and result.content_identity != expected_identity:
        raise ContractError("installed source differs from the expected source identity")
    return result
