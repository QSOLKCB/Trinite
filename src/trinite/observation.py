"""Pinned PROVENANCE composition adapter; stdlib only, no model/RNG imports."""
import hashlib
import importlib.util
from pathlib import Path

from .contracts import ContractError, identity, json_bytes, parse_json

UPSTREAM_COMMIT = "41451bf690597a75f461299dfcbbee559c83e924"
PIN_IDENTITY = "sha256:8133d0e03bce1f34937c0e722a01e15f56700c3a4e4fbf04c635025411043f79"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024


def verify_pin(project_root: Path | None = None) -> dict:
    package = Path(__file__).resolve().parent
    pin_bytes = (package/"provenance-pin.json").read_bytes()
    if identity(pin_bytes) != PIN_IDENTITY:
        raise ContractError("unsupported PROVENANCE source pin receipt")
    pin = parse_json(pin_bytes, canonical=True)
    if pin["commit"] != UPSTREAM_COMMIT:
        raise ContractError("unsupported PROVENANCE revision")
    checked = []
    for local, entry in pin["files"].items():
        if project_root is None and not local.startswith("src/"):
            continue
        path = Path(project_root)/local if project_root is not None else package.parent/local[4:]
        if path.is_symlink() or not path.is_file():
            raise ContractError("missing or unsafe pinned PROVENANCE source: "+local)
        with path.open("rb") as stream: content = stream.read(1024*1024+1)
        blob = hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
        if identity(content) != entry["content_identity"] or blob != entry["git_blob"]:
            raise ContractError("PROVENANCE source differs from pinned revision: "+local)
        checked.append(local)
    # Do not accept another installed package with the same top-level names.
    for name in ("provenance_core", "provenance_verify"):
        spec = importlib.util.find_spec(name)
        expected = package.parent/name/"__init__.py"
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
            raise ContractError("PROVENANCE namespace does not resolve to the frozen source")
    return {"commit": UPSTREAM_COMMIT, "pin_identity": PIN_IDENTITY,
            "runtime_dependencies": "Python standard library", "checked_files": checked}


class BundleObserver:
    def __init__(self, path: Path, *, actor: str = "trinite.cpu-runner"):
        self.pin = verify_pin()
        from provenance_core import (ArtifactRecord, EventCore, EventEnvelope,
                                     EvidenceClass, ManifestCore, ManifestEnvelope, RetentionState,
                                     canonical_json_bytes)
        from provenance_verify import verify_bundle
        self.ArtifactRecord, self.EventCore, self.EventEnvelope = ArtifactRecord, EventCore, EventEnvelope
        self.EvidenceClass, self.ManifestCore, self.ManifestEnvelope = EvidenceClass, ManifestCore, ManifestEnvelope
        self.RetentionState, self.serialize, self.verify = RetentionState, canonical_json_bytes, verify_bundle
        self.path, self.actor = Path(path), actor
        if type(actor) is not str or not actor or len(actor) > 256:
            raise ContractError("observer actor requires a bounded nonempty string")
        self.path.mkdir(parents=True, exist_ok=False)
        self.artifacts, self.events, self.names = {}, {}, {}
        self.total_bytes = 0
        self.closed = False

    def _json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.serialize(value))

    def retain(self, name: str, content: bytes) -> str:
        if self.closed or type(name) is not str or not name or len(name) > 256:
            raise ContractError("invalid observer artifact name or closed bundle")
        if type(content) is not bytes or len(content) > MAX_ARTIFACT_BYTES:
            raise ContractError("observer artifact exceeds byte limit")
        record = self.ArtifactRecord.from_bytes(content, retention=self.RetentionState.CONTENT_RETAINED,
                                                media_type="application/octet-stream")
        key = record.content_identity
        if name in self.names and self.names[name] != key:
            raise ContractError("observer artifact name already binds different bytes")
        if key not in self.artifacts:
            if len(self.artifacts) >= 64 or self.total_bytes+len(content) > MAX_BUNDLE_BYTES:
                raise ContractError("observer bundle exceeds artifact/count budget")
            path = self.path/"artifacts/sha256"/key.split(":")[1]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            self._json(self.path/"artifact_records/sha256"/(record.record_identity.split(":")[1]+".json"),
                       record.to_dict())
            self.artifacts[key] = record
            self.total_bytes += len(content)
        self.names[name] = key
        return key

    def record(self, operation: str, *, inputs: dict[str, bytes], outputs: dict[str, bytes],
               evidence_class: str = "DERIVED") -> str:
        if self.closed or type(operation) is not str or not operation or len(operation) > 256:
            raise ContractError("invalid observer operation or closed bundle")
        if type(inputs) is not dict or type(outputs) is not dict or len(self.events) >= 32:
            raise ContractError("invalid observer references or event budget")
        try:
            core = self.EventCore(self.EvidenceClass(evidence_class), self.actor, operation,
                                  inputs=tuple(self.retain(n, b) for n, b in inputs.items()),
                                  outputs=tuple(self.retain(n, b) for n, b in outputs.items()))
            envelope = self.EventEnvelope.seal(core)
        except (ValueError, TypeError) as error:
            raise ContractError("observer record violates the pinned event contract") from error
        self.events[envelope.event_identity] = envelope
        self._json(self.path/"events/sha256"/(envelope.event_identity.split(":")[1]+".json"), envelope.to_dict())
        return envelope.event_identity

    def finalize(self) -> dict:
        if self.closed:
            raise ContractError("observer bundle is already frozen")
        self.retain("artifact-name-index.json", self.serialize({"schema": "trinite.observer-index.v1",
                                                               "artifacts": dict(self.names), "pin": self.pin}))
        manifest = self.ManifestEnvelope.seal(self.ManifestCore.build(
            artifacts=list(self.artifacts.values()), events=list(self.events), scope="closed"))
        self._json(self.path/"manifest.json", manifest.to_dict())
        self.closed = True
        report = self.verify(self.path).to_dict()
        # The report stays outside the frozen bundle to preserve exact membership.
        return report


def verify_observation(path: Path) -> dict:
    verify_pin()
    from provenance_verify import verify_bundle
    report = verify_bundle(Path(path)).to_dict()
    if (not report["integrity_verified"] or report["manifest_scope"] != "closed"
            or report["known_missing_artifacts"] or report["errors"]):
        raise ContractError("observer bundle did not verify as closed retained evidence: "+str(report["errors"]))
    return report
