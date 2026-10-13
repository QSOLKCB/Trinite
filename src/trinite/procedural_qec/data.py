"""Exact QEC/GF(3) and fold-sum procedural supervision; no model or teacher outputs.

The only permitted learning rows come from existing admitted train families or
new, bounded GF(3) numeric facts. This does not confer fresh held-out status.
"""
from collections import Counter
from itertools import product
from pathlib import Path

from .._qec.qutrit import gf3
from ..contracts import ContractError, identity, json_bytes, parse_json
from ..curriculum_data import audit_bytes as scalar_audit, dataset as scalar_dataset
from ..curriculum_oracle import solve as scalar_solve
from ..qec_source import verify_pin
from ..tokenizer import ByteTokenizer

SCHEMA = "trinite.procedural-qec-data.v1"
KIND = ("arithmetic", "fold-sum", "gf3")
MODES = ("final", "prefix", "trace")
MAX_ROWS = 12000


def source_receipt():
    package = Path(__file__).resolve().parent
    root = package.parent
    names = (package / "data.py", package / "protocol.json",
             root / "curriculum_data.py", root / "curriculum_oracle.py",
             root / "learning_data.py", root / "qec_source.py",
             root / "_qec/qutrit/gf3.py", root / "qec-pin.json",
             root / "tokenizer.py", root / "contracts.py")
    return {str(p.relative_to(root.parent)): identity(p.read_bytes()) for p in names}


def states(kind, values):
    """Donor GF(3) transitions, independently checked by a scalar integer oracle."""
    if kind not in ("fold-sum", "gf3") or type(values) is not list:
        raise ContractError("invalid procedural kind/operands")
    if not (2 <= len(values) <= 4 if kind == "fold-sum" else len(values) == 3):
        raise ContractError("invalid procedural arity")
    if any(type(v) is not int or
           (not -2 <= v <= 2 if kind == "fold-sum" else not 0 <= v <= 2)
           for v in values):
        raise ContractError("out-of-domain procedural operand")
    result = [0]
    for index, value in enumerate(values, 1):
        next_value = (result[-1] + value if kind == "fold-sum"
                      else gf3.add((result[-1],), (value,))[0])
        independently = sum(values[:index])
        if kind == "gf3":
            independently %= 3
        if next_value != independently:
            raise ContractError("pinned QEC / independent scalar oracle disagreement")
        result.append(next_value)
    return result


def make_prompt(kind, mode, values, carrier, index=None):
    if kind not in ("fold-sum", "gf3") or mode not in MODES:
        raise ContractError("invalid procedural rendering")
    tag = {"fold-sum": {"final": "S", "prefix": "P", "trace": "T"},
           "gf3": {"final": "G", "prefix": "H", "trace": "J"}}[kind][mode]
    arg = ",".join(str(v) for v in values)
    suffix = "@" + str(index) if mode == "prefix" else ""
    if carrier == "infix":
        return tag + ":" + arg + suffix + "="
    if carrier == "fields":
        return tag + "(" + arg + ")" + suffix + ":"
    raise ContractError("unrecognized procedural carrier")


def derive(parent, kind, mode, *, index=None, revision):
    values = parent["values"]
    acc = states(kind, values)
    if mode == "prefix":
        if type(index) is not int or not 1 <= index <= len(values):
            raise ContractError("invalid prefix boundary")
        answer = str(acc[index])
    elif mode == "trace":
        if index is not None:
            raise ContractError("unexpected trace index")
        answer = ",".join(map(str, acc))
    elif mode == "final":
        if index is not None:
            raise ContractError("unexpected final index")
        answer = str(acc[-1])
    else:
        raise ContractError("invalid procedural mode")
    prompt = make_prompt(kind, mode, values, parent["carrier"], index)
    core = {"kind": kind, "task": kind, "mode": mode, "values": list(values),
            "carrier": parent["carrier"], "prompt": prompt, "answer": answer,
            "text": prompt + answer, "split": parent["split"],
            "family_id": parent["family_id"],
            "semantic_identity": parent["semantic_identity"],
            "parent_example_identity": parent["parent_example_identity"],
            "transformation_lineage": {"parent_example_identity": parent["parent_example_identity"],
                                       "operation": "running-accumulator." + mode + ".v1",
                                       "prefix_index": index},
            "generator_revision": revision,
            "source_ids": parent["source_ids"] + [SCHEMA],
            "verifier_outcome": "verified"}
    ByteTokenizer(64).encode(core["text"], answer_start=len(prompt.encode("utf-8")))
    return {**core, "example_identity": identity(json_bytes(core))}


def dataset():
    verify_pin()
    parents = {}
    parent_bindings = {}
    for workload in ("arithmetic", "fold"):
        raw, manifest = scalar_dataset(workload)
        scalar_audit(raw, manifest)
        parents[workload] = parse_json(raw, canonical=True)["examples"]
        parent_bindings[workload] = {"dataset": identity(raw), "manifest": identity(manifest)}
    source = source_receipt()
    revision = identity(json_bytes(source))
    rows = []
    # Arithmetic rehearsal is exactly the existing source and targets, not a rewrite.
    for item in parents["arithmetic"]:
        if item["split"] != "train":
            continue
        core = {"kind": "arithmetic", "task": item["task"], "mode": "final",
                "values": [item["formal"]["a"], item["formal"]["b"]],
                "carrier": item["carrier"], "prompt": item["prompt"],
                "answer": item["answer"], "text": item["text"],
                "split": item["split"], "family_id": item["family_id"],
                "semantic_identity": item["semantic_identity"],
                "parent_example_identity": item["example_identity"],
                "transformation_lineage": {"parent_example_identity": item["example_identity"],
                                           "operation": "unchanged-arithmetic.v1",
                                           "prefix_index": None},
                "generator_revision": revision,
                "source_ids": item["source_ids"] + [SCHEMA],
                "verifier_outcome": "verified"}
        ByteTokenizer(64).encode(core["text"], answer_start=len(core["prompt"].encode()))
        rows.append({**core, "example_identity": identity(json_bytes(core))})
    # Import only existing admitted fold-sum training families; no test derivations.
    for item in parents["fold"]:
        if item["split"] != "train" or item["task"] != "sum":
            continue
        if scalar_solve("fold", item["formal"]) != item["answer"]:
            raise ContractError("independent parent fold label disagreement")
        base = {"values": item["formal"]["values"], "carrier": item["carrier"],
                "split": item["split"], "family_id": item["family_id"],
                "semantic_identity": item["semantic_identity"],
                "parent_example_identity": item["example_identity"],
                "source_ids": item["source_ids"]}
        derived = derive(base, "fold-sum", "final", revision=revision)
        # Final-answer prompt and answer must stay byte-for-byte unchanged.
        if derived["text"] != item["text"] or derived["prompt"] != item["prompt"]:
            raise ContractError("parent fold final-answer changed")
        rows.append(derived)
        rows.append(derive(base, "fold-sum", "prefix",
                           index=max(1, len(base["values"]) // 2), revision=revision))
        rows.append(derive(base, "fold-sum", "trace", revision=revision))
    # True direct QEC transfer: the already-vendored GF(3) addition oracle.
    # All tiny GF(3) triples are known training/development facts, not held out.
    for values, carrier in product(product(range(3), repeat=3), ("infix", "fields")):
        values = list(values)
        formal = {"operation": "GF3-running-sum", "values": values}
        semantic = identity(json_bytes(formal))
        base = {"values": values, "carrier": carrier, "split": "train",
                "family_id": "gf3:" + semantic, "semantic_identity": semantic,
                "parent_example_identity": identity(json_bytes({"formal": formal, "carrier": carrier})),
                "source_ids": ["QSOLKCB/QEC:" + verify_pin()["revision"]]}
        rows.append(derive(base, "gf3", "final", revision=revision))
        rows.append(derive(base, "gf3", "prefix", index=2, revision=revision))
        rows.append(derive(base, "gf3", "trace", revision=revision))
    if not rows or len(rows) > MAX_ROWS:
        raise ContractError("procedural curriculum exceeds bounded row count")
    if len({r["example_identity"] for r in rows}) != len(rows) or len({r["text"] for r in rows}) != len(rows):
        raise ContractError("procedural task collision or duplicate text")
    if any(r["split"] != "train" for r in rows):
        raise ContractError("nontraining family entered procedural training material")
    counts = {kind: {mode: sum(r["kind"] == kind and r["mode"] == mode for r in rows)
                     for mode in MODES} for kind in KIND}
    raw = json_bytes({"schema": SCHEMA, "examples": sorted(rows, key=lambda e: e["example_identity"])})
    manifest = json_bytes({"schema": "trinite.procedural-qec-admission.v1",
        "dataset_identity": identity(raw), "parent_identities": parent_bindings,
        "qec_dependency": verify_pin(), "generator_source": source,
        "family_splits": {e["family_id"]: "train" for e in rows},
        "counts": counts,
        "admission": {"source_id": SCHEMA, "outcome": "admitted",
            "origin": "existing family-preserved fold/arithmetic facts plus exhaustive local GF(3) triples",
            "author": "QSOL-IMC commissioned exact procedural generator",
            "rights_basis": "independently generated integer/finite-field facts, minimal symbolic carriers, no donor prose",
            "generation": "pinned QEC GF(3) add and independent sum-modulo oracle; independent fold-sum oracle; source-bound parent admission",
            "scope": "this bounded train-only experiment, not new external/generalization data",
            "reviewer": "automated exact-oracle, family, source, rights and tokenizer audit",
            "reviewed_on": "2026-10-13",
            "limitations": "formal-fact admission, not legal certification, model learning, held-out validation or physics"}})
    return raw, manifest


def audit_bytes(raw, manifest):
    if (type(raw) is not bytes or type(manifest) is not bytes
            or len(raw) > 16 * 1024 * 1024 or len(manifest) > 2 * 1024 * 1024):
        raise ContractError("bounded procedural data and manifest bytes required")
    if (raw, manifest) != dataset():
        raise ContractError("procedural data/admission differs from exact regeneration")
    return parse_json(manifest, canonical=True)
