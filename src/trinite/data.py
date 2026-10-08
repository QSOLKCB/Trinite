"""Tiny symbolic arithmetic fixtures and a replayable family split audit.

All content is generated locally from numeric facts; no external corpus or
model is accessed. This is software-conformance data, not a capability test.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from .contracts import (ContractError, ModelConfig, exact_keys, identity, integer,
                        json_bytes, parse_json)
from .tokenizer import ByteTokenizer

SOURCE_ID = "trinite.formal.modular-add.v1"
DATASET_SCHEMA = "trinite.formal-dataset.v1"
MANIFEST_SCHEMA = "trinite.dataset-manifest.v1"
SPLIT_POLICY = "trinite.family-rank-6-1-1.v1"
SEED_MAX = 2**32 - 1
_FILES = ("dataset.json", "manifest.json")
_MAX_BYTES = {"dataset.json": 256 * 1024, "manifest.json": 16 * 1024}
_INFIX = re.compile(r"\((\d+)\+(\d+)\)%(\d+)=\Z")
_FIELDS = re.compile(r"m:(\d+);a:(\d+);b:(\d+);r=\Z")


def generator_identity() -> str:
    """Bind generator/validator, JSON/config contracts, and tokenization source."""
    return identity(json_bytes(implementation_receipt()))


def implementation_receipt() -> dict:
    package = Path(__file__).resolve().parent
    return {name: identity((package/name).read_bytes())
            for name in ("contracts.py", "data.py", "tokenizer.py")}


def _spec(modulus: int, left: int, right: int) -> dict:
    integer(modulus, "modulus", 2, 9)
    integer(left, "left", 0, modulus-1)
    integer(right, "right", 0, modulus-1)
    if left > right:
        raise ContractError("operands must be canonical (left <= right)")
    return {"modulus": modulus, "left": left, "right": right}


def solve(spec: dict) -> str:
    """Generator solver: walk a cyclic counter, without division/remainder."""
    exact_keys(spec, {"modulus", "left", "right"}, "formal specification")
    _spec(**spec)
    state = spec["left"]
    for _ in range(spec["right"]):
        state += 1
        if state == spec["modulus"]:
            state = 0
    return str(state)


def verify_answer(spec: dict, answer: str) -> None:
    """Independent arithmetic check, not the generator solver or renderer."""
    exact_keys(spec, {"modulus", "left", "right"}, "formal specification")
    _spec(**spec)
    expected = (spec["left"] + spec["right"]) % spec["modulus"]
    if type(answer) is not str or answer != str(expected):
        raise ContractError("formal answer failed independent verification")


def parse_prompt(prompt: str, carrier: str) -> dict:
    if type(prompt) is not str:
        raise ContractError("prompt must be a string")
    if carrier == "infix":
        match = _INFIX.fullmatch(prompt)
        if match:
            left, right, modulus = map(int, match.groups())
            return _spec(modulus, left, right)
    elif carrier == "fields":
        match = _FIELDS.fullmatch(prompt)
        if match:
            modulus, left, right = map(int, match.groups())
            return _spec(modulus, left, right)
    raise ContractError("invalid symbolic carrier or prompt")


def _render(spec: dict, carrier: str) -> str:
    m, a, b = spec["modulus"], spec["left"], spec["right"]
    if carrier == "infix":
        return f"({a}+{b})%{m}="
    if carrier == "fields":
        return f"m:{m};a:{a};b:{b};r="
    raise ContractError("unknown carrier")


def _family(modulus: int) -> str:
    return identity(json_bytes({"task": SOURCE_ID, "modulus": modulus}))


def split_families(families: list[str], seed: int) -> dict[str, str]:
    integer(seed, "seed", 0, SEED_MAX)
    if (type(families) is not list or len(families) != 8
            or any(type(f) is not str for f in families)
            or set(families) != {_family(m) for m in range(2, 10)}):
        raise ContractError("split policy requires the eight complete modulus families")
    def rank(family: str) -> tuple[bytes, str]:
        key = json_bytes({"policy": SPLIT_POLICY, "seed": seed, "family": family})
        return hashlib.sha256(key).digest(), family
    ordered = sorted(families, key=rank)
    return {f: ("train" if i < 6 else "validation" if i == 6 else "test")
            for i, f in enumerate(ordered)}


def _formal_source() -> list[dict]:
    # Three unique boundary/lower cases per modulus, including zero and wrap.
    return [_spec(m, a, b) for m in range(2, 10)
            for a, b in ((0, 0), (0, m-1), (1, 1))]


def admission() -> dict:
    formal = _formal_source()
    rendered = [_render(spec, carrier) + solve(spec) for spec in formal
                for carrier in ("infix", "fields")]
    return {
        "source_id": SOURCE_ID,
        "source_content_identity": identity(json_bytes(formal)),
        "origin": "local numeric formal generator; no external text source",
        "author": "Trinite deterministic formal generator; implementation created by "
                  "OpenAI Codex at Trent Slade / QSOL-IMC's request",
        "generation_procedure": {
            "formal_source": "Enumerate moduli 2..9 with operand pairs (0,0), (0,m-1), (1,1)",
            "answer_generation": "Cyclic-counter addition with wrap at the modulus",
            "answer_verification": "Independent integer addition and remainder; parse each prompt",
            "rendering": "Two minimal symbolic carriers: infix and labelled fields",
            "acquisition": "Generated locally; no downloaded or inherited corpus",
        },
        "rights_basis": "numeric mathematical facts rendered with minimal symbols",
        "supporting_reference": {
            "path": "src/trinite/data.py",
            "content_identity": implementation_receipt()["data.py"],
            "symbols": ["_formal_source", "_render", "solve", "verify_answer", "parse_prompt"],
        },
        "rights_evidence": {
            "kind": "inspectable-numeric-source-and-symbolic-payload",
            "formal_source": formal,
            "rendered_payload": rendered,
            "payload_identity": identity(json_bytes(rendered)),
            "review_basis": "Source and all 48 exact texts are retained for scope inspection; "
                            "only numeric facts and the declared symbolic forms are admitted",
        },
        "scope": "only the 24 formal triples and the two specified symbolic carriers",
        "reviewer": "trinite.phase1-symbolic-admission.v1 (automated policy audit)",
        "reviewed_on": "2026-10-09",
        "outcome": "admitted_by_policy",
        "limitations": "not a blanket rights assessment for synthetic data or future prose",
    }


def validate_admission(record: object) -> None:
    """Check required evidence against source/content, not a bare rights label.

    These checks verify the documented admission evidence and narrow scope;
    they do not automate a legal opinion or certify future corpora.
    """
    required = {"source_id", "source_content_identity", "origin", "author",
                "generation_procedure", "rights_basis", "supporting_reference",
                "rights_evidence", "scope", "reviewer", "reviewed_on", "outcome",
                "limitations"}
    record = exact_keys(record, required, "source admission")
    for name in ("origin", "author", "rights_basis", "scope", "reviewer",
                 "reviewed_on", "outcome", "limitations"):
        if type(record[name]) is not str or not record[name].strip():
            raise ContractError(f"source admission requires non-empty {name}")
    procedure = exact_keys(record["generation_procedure"],
                           {"formal_source", "answer_generation", "answer_verification",
                            "rendering", "acquisition"}, "generation procedure")
    if any(type(v) is not str or not v.strip() for v in procedure.values()):
        raise ContractError("generation procedure must identify every step")
    reference = exact_keys(record["supporting_reference"],
                           {"path", "content_identity", "symbols"}, "supporting reference")
    if (reference["path"] != "src/trinite/data.py"
            or reference["content_identity"] != implementation_receipt()["data.py"]
            or reference["symbols"] != ["_formal_source", "_render", "solve",
                                        "verify_answer", "parse_prompt"]):
        raise ContractError("source admission reference does not bind the current source")
    evidence = exact_keys(record["rights_evidence"],
                          {"kind", "formal_source", "rendered_payload", "payload_identity",
                           "review_basis"}, "rights evidence")
    formal = _formal_source()
    payload = [_render(spec, carrier) + solve(spec) for spec in formal
               for carrier in ("infix", "fields")]
    if (record["source_id"] != SOURCE_ID
            or record["source_content_identity"] != identity(json_bytes(formal))
            or evidence["kind"] != "inspectable-numeric-source-and-symbolic-payload"
            or json_bytes(evidence["formal_source"]) != json_bytes(formal)
            or evidence["rendered_payload"] != payload
            or evidence["payload_identity"] != identity(json_bytes(payload))
            or type(evidence["review_basis"]) is not str or not evidence["review_basis"].strip()):
        raise ContractError("source admission evidence does not match the numeric source/payload")


def build_dataset(config: ModelConfig = ModelConfig(), *, seed: int = 0) -> tuple[dict, dict]:
    if not isinstance(config, ModelConfig):
        raise ContractError("config must be a validated ModelConfig")
    integer(seed, "seed", 0, SEED_MAX)
    source_admission = admission()
    validate_admission(source_admission)
    tokenizer = ByteTokenizer(config.context_length)
    assignment = split_families([_family(m) for m in range(2, 10)], seed)
    generator = generator_identity()
    examples = []
    normalized = set()
    text_seen = set()
    for spec in _formal_source():
        semantic = identity(json_bytes({"task": SOURCE_ID, "formal": spec}))
        if semantic in normalized:
            raise ContractError("duplicate normalized formal problem")
        normalized.add(semantic)
        answer = solve(spec)
        verify_answer(spec, answer)
        for carrier in ("infix", "fields"):
            prompt = _render(spec, carrier)
            if parse_prompt(prompt, carrier) != spec:
                raise ContractError("rendered prompt does not describe the formal problem")
            text = prompt + answer
            if text in text_seen:
                raise ContractError("duplicate rendered example")
            text_seen.add(text)
            encoding = tokenizer.encode(text, answer_start=len(prompt.encode("utf-8")))
            core = {
                "source_id": SOURCE_ID, "generator_identity": generator, "seed": seed,
                "formal": spec, "semantic_identity": semantic, "carrier": carrier,
                "family_id": _family(spec["modulus"]),
                "split": assignment[_family(spec["modulus"])],
                "prompt": prompt, "answer": answer, "text": text,
                "verification": "passed-independent-arithmetic-and-prompt-parse",
                "transformations": ["canonical-operands.v1", "symbolic-"+carrier+".v1",
                                    "utf8-byte.v1"],
                "encoding": encoding.to_dict(),
            }
            examples.append({"example_identity": identity(json_bytes(core)), **core})
    dataset = {"schema": DATASET_SCHEMA, "config": config.to_dict(),
               "seed": seed, "examples": examples}
    manifest = {
        "schema": MANIFEST_SCHEMA, "dataset_identity": identity(json_bytes(dataset)),
        "config_identity": config.content_identity, "generator_identity": generator,
        "implementation": implementation_receipt(), "admission": source_admission,
        "seed": seed, "split_policy": SPLIT_POLICY,
        "family_splits": assignment,
        "counts": {s: sum(e["split"] == s for e in examples)
                   for s in ("train", "validation", "test")},
        "curation": {"formal_candidates": 24, "unique_formal_problems": 24,
                     "rendered_candidates": 48, "retained": 48, "excluded": 0,
                     "duplicate_policy": "dedup formal problems; retain labelled carrier variants"},
        "evidence_scope": "software conformance only; not a reasoning benchmark",
    }
    return dataset, manifest


def audit_bytes(dataset_bytes: bytes, manifest_bytes: bytes) -> dict:
    """Verify exact content plus regenerate source, split, and every encoding.

    Producer-supplied digests/rights/verifier labels are never sufficient: the
    entire expected fixture is recomputed under the installed source contract.
    """
    for name, data in zip(_FILES, (dataset_bytes, manifest_bytes)):
        if type(data) is not bytes or len(data) > _MAX_BYTES[name]:
            raise ContractError(f"{name} exceeds the Phase 1 byte limit or is not bytes")
    dataset = exact_keys(parse_json(dataset_bytes, canonical=True),
                         {"schema", "config", "seed", "examples"}, "dataset")
    manifest = parse_json(manifest_bytes, canonical=True)
    if type(manifest) is not dict or "admission" not in manifest:
        raise ContractError("manifest requires source admission evidence")
    validate_admission(manifest["admission"])
    if dataset["schema"] != DATASET_SCHEMA:
        raise ContractError("unsupported dataset schema")
    config = ModelConfig.from_dict(dataset["config"])
    expected_data, expected_manifest = build_dataset(config, seed=dataset["seed"])
    if dataset_bytes != json_bytes(expected_data) or manifest_bytes != json_bytes(expected_manifest):
        raise ContractError("fixture differs from frozen source/admission/split/encoding contract")
    # Independent parse/solve is also part of regeneration, before success.
    return {"schema": "trinite.dataset-audit.v1", "verified": True,
            "dataset_identity": identity(dataset_bytes),
            "manifest_identity": identity(manifest_bytes),
            "counts": manifest["counts"], "families": 8,
            "generator_identity": expected_manifest["generator_identity"],
            "evidence_scope": "software conformance only"}


def audit_directory(path: Path) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ContractError("dataset path must be a real directory")
    if {p.name for p in path.iterdir()} != set(_FILES):
        raise ContractError("dataset directory must contain exactly dataset.json and manifest.json")
    for name in _FILES:
        if (path/name).is_symlink() or not (path/name).is_file():
            raise ContractError("dataset members must be regular files without symlinks")
    contents = []
    for name in _FILES:
        with (path/name).open("rb") as stream:
            contents.append(stream.read(_MAX_BYTES[name] + 1))
    return audit_bytes(*contents)


def write_dataset(path: Path, config: ModelConfig = ModelConfig(), *, seed: int = 0) -> dict:
    # Validate everything before publishing to a new directory. Never overwrite
    # an existing output (including a symlink); incomplete IO remains auditable.
    dataset, manifest = build_dataset(config, seed=seed)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    (path/"dataset.json").write_bytes(json_bytes(dataset))
    (path/"manifest.json").write_bytes(json_bytes(manifest))
    return audit_directory(path)
