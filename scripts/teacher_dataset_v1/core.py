"""TT-002: source-specific, reproducible GF(3) formal data admission.

All data and labels are original finite arithmetic facts. No teacher inference,
training, downloads, native Rust execution or model/optimizer construction.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import tempfile

from trinite.contracts import ContractError, identity, json_bytes, parse_json, require_identity
from teacher_lab_v1 import core as tt001
import qutrit_conformance_v1 as q

PLAN = "configs/teacher/gf3-dataset-v1.json"
MANIFEST_SCHEMA = "trinite.teacher-gf3-dataset.v1"
ROW_SCHEMA = "trinite.teacher-gf3-row.v1"
SPLITS = {"S1": "train", "S2": "train", "W": "train",
          "S1W": "validation", "WS2": "validation",
          "S2W": "test", "WS1": "test"}
MEMBERS = ("plan.json", "dataset-rights.json", "weights-rights.json",
           "train.jsonl", "validation-public.jsonl", "test-public.jsonl",
           "privileged-targets.jsonl", "manifest.json")
MAX_BYTES = 4 * 1024 * 1024
SOURCES = (
    "scripts/teacher_dataset.py", "scripts/teacher_dataset_v1/__init__.py",
    "scripts/teacher_dataset_v1/core.py", "scripts/qutrit_conformance_v1.py",
    "scripts/queue_evaluation_v1.py", "scripts/teacher_lab_v1/__init__.py",
    "scripts/teacher_lab_v1/core.py", "scripts/model_size_policy_v1.py",
    "src/trinite/__init__.py", "src/trinite/contracts.py", "src/trinite/tokenizer.py",
    "docs/QUTRIT-CONFORMANCE-V1.md", "docs/TEACHER-LAB-TT-002.md",
    "docs/DATA.md", "docs/TEACHER-LAB-V1.md", "AGENTS.md", "LICENSE",
)


def plan() -> dict:
    found = tt001.source_json(PLAN)
    expected = {
        "schema": "trinite.teacher-gf3-plan.v1",
        "qec_revision": q.QEC_SHA,
        "donor_revision": q.DONOR_SHA,
        "source_catalog": "configs/teacher/source-candidates-v1.json",
        "split_policy": "primitives-before-compositions.v1",
        "training_authorized": False,
        "data_scope": "original-commissioned-gf3-only",
    }
    if found != expected or any(type(v) is not type(expected[k]) for k, v in found.items()):
        raise ContractError("TT-002 plan or source pin changed")
    catalog = tt001.validate_catalog(tt001.source_json(found["source_catalog"]))
    entries = [e for e in catalog["sources"] if e["id"] == "qec-v1772"]
    if (len(entries) != 1 or entries[0]["revision"] != q.QEC_SHA or
            entries[0]["rights_status"] != "REVIEW_REQUIRED"):
        raise ContractError("QEC candidate provenance mismatch")
    if set(q.CIRCUITS) != set(SPLITS):
        raise ContractError("QEC circuit inventory mismatch")
    return found


def _gates(name: str) -> tuple[str, ...]:
    if name not in SPLITS:
        raise ContractError("invalid GF(3) circuit")
    if name == "W":
        return ("W",)
    if name.startswith("W"):
        return ("W", name[1:])
    if name.endswith("W"):
        return (name[:2], "W")
    return (name,)


def algebraic(word: str, circuit: str, inverse: bool = False) -> str:
    """Original direct Pauli exponent algebra; independent from ket oracle."""
    if type(word) is not str or len(word) != 4 or any(c not in "012" for c in word):
        raise ContractError("invalid GF(3) word")
    a, b, c, d = (int(v) for v in word)
    steps = list(_gates(circuit))
    if inverse:
        steps.reverse()
    for gate in steps:
        if gate == "W":
            a, b, c, d = b, a, d, c
        else:
            if gate not in ("S1", "S2"):
                raise ContractError("invalid SUM gate")
            weight = int(gate[1]) * (-1 if inverse else 1)
            b = (b + weight * a) % 3
            c = (c - weight * d) % 3
    return "".join(str(v) for v in (a, b, c, d))


def rights() -> tuple[dict, dict]:
    dataset = {
        "schema": "trinite.teacher-gf3-dataset-rights.v1",
        "admission": "ADMITTED_FORMAL_ONLY",
        "training_authorized": False,
        "author": "QSOLKCB original formal generator commissioned under owner direction",
        "origin": "finite exact GF(3) arithmetic, not copied prose or upstream datasets",
        "owner_scope": "2026-11-04 user-approved research direction for first-party formal facts",
        "reviewer": "QSOLKCB owner scope; not independent legal certification",
        "review_date": "2026-11-04",
        "rights_basis": "numeric mathematical facts and original minimal ASCII carrier",
        "source": "QSOLKCB/QEC",
        "source_revision": q.QEC_SHA,
        "external_reference": "hmx21/css-decoupling MIT, attributed mathematics only",
        "external_revision": q.DONOR_SHA,
        "code_license": "MPL-2.0",
        "dataset_license": "UNDECIDED_SEPARATE_FROM_CODE",
        "generation": "81 four-trit words x seven circuits x four task views",
        "independent_oracle": "direct GF(3) exponent transform vs nine-ket permutation",
        "family_split": "train primitives; validation/test distinct composite circuits",
        "exposure": "ALL circuit families already exposed in public engineering tests",
        "limits": "not fresh/secret heldout evidence; check questions may be trivial",
        "excluded": "textbooks, copyrighted prose, third-party corpora, pretrained outputs, private files",
    }
    weights = {"schema": "trinite.teacher-gf3-weights-rights.v1",
               "status": "NOT_CREATED", "code_license": "MPL-2.0",
               "weights_license": "UNDECIDED",
               "checkpoint_identity": "NOT_CREATED",
               "dataset_lineage": "TT-002 source-only; no weights generated"}
    return dataset, weights


def jsonl(rows: list[dict]) -> bytes:
    return b"".join(json_bytes(r) for r in rows)


def make_packet() -> tuple[dict[str, bytes], dict]:
    cfg = plan()
    train, val, test, privileged = [], [], [], []
    prompts: set[str] = set()
    ids: set[str] = set()
    family_counts: dict[str, int] = {}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    oracle_checks = 0
    for circuit in q.CIRCUITS:
        split = SPLITS[circuit]
        family_counts[circuit] = 0
        for number in range(81):
            word = "".join(str(number // 3 ** n % 3) for n in (3, 2, 1, 0))
            forward, inverse = algebraic(word, circuit), algebraic(word, circuit, True)
            if algebraic(forward, circuit, True) != word or algebraic(inverse, circuit) != word:
                raise ContractError("GF(3) inverse roundtrip failed")
            wrong = str((int(forward[0]) + 1) % 3) + forward[1:]
            views = (("apply", "", forward), ("inverse", "", inverse),
                     ("check", forward, "1"), ("check", wrong, "0"))
            for task, candidate, answer in views:
                observed = q.public(word, circuit, task, candidate)
                if q.expected(observed) != answer:
                    raise ContractError("independent nine-ket oracle disagrees with GF(3) algebra")
                oracle_checks += 1
                prompt = q.prompt(observed)
                if q.decode_prompt(prompt) != observed or prompt in prompts:
                    raise ContractError("public carrier mismatch or duplicated prompt")
                prompts.add(prompt)
                ledger = q.ledger(observed)
                if (ledger["complete_context_tokens"] > 256 or
                        ledger["supervised_tokens"] != len(answer) + 1 or
                        ledger["truncated"] is not False or ledger["optimizer_updates"] != 0):
                    raise ContractError("native tokenizer supervision bound failed")
                rid = identity(json_bytes({"schema": ROW_SCHEMA, "observation": observed}))
                if rid in ids:
                    raise ContractError("duplicate structural item identity")
                ids.add(rid)
                public = {"schema": ROW_SCHEMA, "id": rid, "prompt": prompt,
                          "observation": observed}
                if split == "train":
                    train.append(dict(public, family=circuit, split=split, answer=answer,
                                      supervised_tokens=len(answer) + 1,
                                      complete_context_tokens=ledger["complete_context_tokens"]))
                elif split == "validation":
                    val.append(public)
                    privileged.append({"id": rid, "family": circuit, "split": split, "answer": answer})
                else:
                    test.append(public)
                    privileged.append({"id": rid, "family": circuit, "split": split, "answer": answer})
                split_counts[split] += 1
                family_counts[circuit] += 1
    if split_counts != {"train": 972, "validation": 648, "test": 648}:
        raise ContractError("GF(3) finite inventory incomplete")
    dataset_rights, weight_rights = rights()
    packet = {
        "plan.json": json_bytes(cfg),
        "dataset-rights.json": json_bytes(dataset_rights),
        "weights-rights.json": json_bytes(weight_rights),
        "train.jsonl": jsonl(train),
        "validation-public.jsonl": jsonl(val),
        "test-public.jsonl": jsonl(test),
        "privileged-targets.jsonl": jsonl(privileged),
    }
    closure = sorted(set(SOURCES) | {PLAN, cfg["source_catalog"]})
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "ADMITTED_FORMAL_ONLY_NO_TRAINING",
        "training_authorized": False,
        "heldout_scored": False,
        "optimizer_updates": 0,
        "scientific_confirmation": False,
        "qec_revision": q.QEC_SHA, "donor_revision": q.DONOR_SHA,
        "split_policy": cfg["split_policy"],
        "split_counts": split_counts, "family_counts": family_counts,
        "independent_oracle_checks": oracle_checks,
        "source_closure": [{"path": p, "identity": identity(tt001.source_bytes(p))}
                           for p in closure],
        "files": [{"name": name, "identity": identity(data), "bytes": len(data)}
                  for name, data in sorted(packet.items())],
    }
    packet["manifest.json"] = json_bytes(manifest)
    if any(len(content) > MAX_BYTES for content in packet.values()):
        raise ContractError("dataset artifact exceeds byte budget")
    return packet, manifest


def prepare(path: Path) -> str:
    packet, _ = make_packet()
    path = Path(path)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise ContractError("destination must be new under an existing directory")
    stage = Path(tempfile.mkdtemp(prefix=f".{path.name}.tt002-", dir=path.parent))
    try:
        for name in MEMBERS:
            with (stage / name).open("xb") as fh:
                fh.write(packet[name])
                fh.flush()
                os.fsync(fh.fileno())
        if path.exists() or path.is_symlink():
            raise ContractError("destination already exists")
        os.rename(stage, path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return identity(packet["manifest.json"])


def verify(path: Path, external_anchor: str) -> dict:
    require_identity(external_anchor)
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ContractError("invalid packet directory")
    names: set[str] = set()
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.name not in MEMBERS or entry.name in names:
                raise ContractError("unexpected packet member")
            names.add(entry.name)
    if names != set(MEMBERS):
        raise ContractError("missing packet member")
    actual: dict[str, bytes] = {}
    for name in MEMBERS:
        member = path / name
        meta = member.lstat()
        if not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1 or meta.st_size > MAX_BYTES:
            raise ContractError("unsafe or oversized packet member")
        with member.open("rb") as stream:
            actual[name] = stream.read(MAX_BYTES + 1)
        if len(actual[name]) > MAX_BYTES:
            raise ContractError("oversized packet payload")
    if identity(actual["manifest.json"]) != external_anchor:
        raise ContractError("externally retained manifest identity mismatch")
    expected, manifest = make_packet()
    if actual != expected:
        raise ContractError("dataset/source/rights/oracle replay mismatch")
    if parse_json(actual["manifest.json"], canonical=True) != manifest:
        raise ContractError("manifest structure mismatch")
    return {"verified": True, "scope": "formal-data-and-independent-oracle-replay",
            "split_counts": manifest["split_counts"], "training_authorized": False,
            "heldout_scored": False, "optimizer_updates": 0}
