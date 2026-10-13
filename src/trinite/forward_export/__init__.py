"""Encode-only final-forward artifacts; no importer or generation backend.

Import ``encoder.export_forward`` explicitly in the hash-locked CPU environment.
The package and explicit source-snapshot API remain standard-library only.
"""
from dataclasses import dataclass

from ..contracts import ContractError

FORMAT_ID = "trinite.final-forward-ternary.candidate.v1"
MAX_PAYLOAD_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 512 * 1024


@dataclass(frozen=True)
class ForwardExport:
    """Three immutable byte members, with no model/tensor references or I/O.

    Construction alone is not validation. Only ``export_forward`` produces a
    validated export; there is deliberately no artifact acceptance/import API.
    """
    manifest: bytes
    packed_codes: bytes
    float_tensors: bytes

    def __post_init__(self) -> None:
        if any(type(value) is not bytes for value in
               (self.manifest, self.packed_codes, self.float_tensors)):
            raise ContractError("forward export members must be exact immutable bytes")
