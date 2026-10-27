"""Bounded original formal music corpus; manifests never become model inputs."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from .core import Event, Meter, Score, to_dict, from_dict, labels
from .oracle import annotate

SCHEMA = "trinite.symbolic-music.v1"
SEED = 0  # Fixed software-conformance split, not a confirmatory test set.
GROUPINGS = (
    (3, 4, (1, 1, 1)),
    (4, 4, (1, 1, 1, 1)), (4, 4, (2, 2)),
    (5, 8, (2, 3)), (5, 8, (3, 2)),
    (7, 8, (2, 2, 3)), (7, 8, (3, 2, 2)), (7, 8, (2, 3, 2)),
    (8, 16, (2, 2, 2, 2)), (8, 16, (3, 3, 2)),
)
SOURCES = (
    "scripts/symbolic_music_v1/__init__.py",
    "scripts/symbolic_music_v1/core.py", "scripts/symbolic_music_v1/corpus.py",
    "scripts/symbolic_music_v1/oracle.py",
    "scripts/run_symbolic_music.py", "docs/SYMBOLIC-MUSIC.md",
)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def family_split():
    """All notation, meter, grouping, transposition and tempo variants stay together."""
    ordered = sorted(range(8), key=lambda n: digest(f"{SCHEMA}:{SEED}:{n}".encode()))
    return {n: ("train" if i < 4 else "validation" if i < 6 else "test")
            for i, n in enumerate(ordered)}


def _score(mask, numerator, denominator, groups, variant):
    unit = Fraction(1, 8 if denominator == 4 else denominator)
    count = int(Fraction(numerator, denominator) / unit)
    pitch = 60 if variant == 0 else 67
    events = [Event(unit, None if 1 <= i <= 3 and mask & (1 << (i - 1)) else pitch)
              for i in range(count)]
    if variant:
        # Coalesce equal adjacent symbols: quarter/dotted values, not float labels.
        merged = []
        for event in events:
            if merged and event.pitch == merged[-1].pitch:
                merged[-1] = replace(event, duration=merged[-1].duration + unit)
            else:
                merged.append(event)
        events = merged
        # Explicit within-bar ties preserve sounding duration, not attack count.
        first = events.pop(0)
        events[:0] = [replace(first, duration=unit / 2, tie_out=True),
                      replace(first, duration=first.duration - unit / 2, tie_in=True)]
    bars = [list(events), list(events)]
    if variant:
        # Continuation across an explicit bar boundary. Adjacent ties can chain.
        bars[0][-1] = replace(bars[0][-1], tie_out=True)
        bars[1][0] = replace(bars[1][0], tie_in=True)
    return Score(Meter(numerator, denominator, groups),
                 tuple(tuple(bar) for bar in bars), 90 if variant == 0 else 120)


def model_input(score):
    """Allowlisted symbolic view. Never accept a row/manifest or append labels/IDs."""
    return {"task": "annotate-rhythm-v1", "score": to_dict(score)}


def generate():
    rows = []
    splits = family_split()
    for mask in range(8):
        for numerator, denominator, groups in GROUPINGS:
            for variant in range(2):
                score = _score(mask, numerator, denominator, groups, variant)
                prompt = model_input(score)
                rows.append({"id": digest(canonical(prompt)),
                             "family": f"rest-mask-{mask}", "split": splits[mask],
                             "input": prompt, "target": labels(score),
                             "lineage": {"rest_mask": mask, "variant": variant,
                                         "seed": SEED, "source": SCHEMA}})
    validate_rows(rows)
    return rows


def validate_rows(rows):
    """Strict regeneration rejects relabeling, split edits and metadata leaks."""
    if type(rows) is not list or len(rows) != 160:
        raise ValueError("expected complete 160-row conformance corpus")
    seen, family_splits, structures = set(), {}, {}
    for row in rows:
        if type(row) is not dict or set(row) != {"id", "family", "split", "input", "target", "lineage"}:
            raise ValueError("invalid row fields")
        lineage = row["lineage"]
        if type(lineage) is not dict or set(lineage) != {"rest_mask", "variant", "seed", "source"}:
            raise ValueError("invalid lineage")
        mask, variant = lineage["rest_mask"], lineage["variant"]
        if type(mask) is not int or mask not in range(8) or type(variant) is not int or variant not in (0, 1):
            raise ValueError("invalid family or variant")
        if type(lineage["seed"]) is not int or lineage["seed"] != SEED or lineage["source"] != SCHEMA:
            raise ValueError("invalid source")
        if row["family"] != f"rest-mask-{mask}" or row["split"] != family_split()[mask]:
            raise ValueError("family split mismatch")
        prompt = row["input"]
        if type(prompt) is not dict or set(prompt) != {"task", "score"} or prompt["task"] != "annotate-rhythm-v1":
            raise ValueError("invalid model input fields")
        score = from_dict(prompt["score"])
        m = score.meter
        if (m.numerator, m.denominator, m.groups) not in GROUPINGS:
            raise ValueError("unsupported corpus grouping")
        expected = _score(mask, m.numerator, m.denominator, m.groups, variant)
        if (canonical(prompt) != canonical(model_input(expected))
                or canonical(row["target"]) != canonical(labels(expected))
                or canonical(row["target"]) != canonical(annotate(prompt["score"]))):
            raise ValueError("formal score/target mismatch")
        identity = digest(canonical(prompt))
        if identity != row["id"] or identity in seen:
            raise ValueError("duplicate or incorrect input identity")
        seen.add(identity)
        family_splits.setdefault(row["family"], set()).add(row["split"])
        # Underlying rest mask is deliberately coarser than meter/notation/pitch.
        structures.setdefault(mask, set()).add(row["split"])
    if any(len(s) != 1 for s in [*family_splits.values(), *structures.values()]):
        raise ValueError("family leakage")


def source_hashes(root):
    if Path(root).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("source root must be the executing checkout")
    return {name: digest((Path(root) / name).read_bytes()) for name in SOURCES}


def admission(root):
    return {"schema": SCHEMA, "outcome": "engineering-conformance-only",
            "author": "Deterministic original generator commissioned by Trent Slade / QSOL-IMC",
            "rights_basis": "Bounded newly generated formal timing facts; no existing composition, score, recording, donor text, model output or third-party asset is consumed.",
            "scope": "Only the 160 formal symbolic rows generated by this exact source closure; not blanket synthetic-data admission or authorization for native training.",
            "reviewer": "Codex source-policy audit for the owner-requested music core",
            "review_date": "2026-10-27", "seed": SEED,
            "procedure": "Enumerate eight rest-mask families, ten meter/groupings and two notation/pitch/tempo variants; validate exact notation, labels, identities and whole-family split.",
            "sources_sha256": source_hashes(root),
            "limitations": "Fully visible exploratory software fixture; no learning, arithmetic transfer or legal certification. Fresh held-out families and reviewed native admission required before a learning study."}


def artifacts(root):
    rows = generate()
    files = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        # Explicitly separate input-only bytes from targets and audit metadata.
        files[f"{split}.inputs.jsonl"] = b"".join(canonical(row["input"]) for row in selected)
        files[f"{split}.audit.jsonl"] = b"".join(canonical(row) for row in selected)
    files["admission.json"] = canonical(admission(root))
    manifest = {"schema": SCHEMA, "count": len(rows), "family_count": 8,
                "files_sha256": {name: digest(data) for name, data in sorted(files.items())},
                "sources_sha256": source_hashes(root),
                "counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}}
    files["manifest.json"] = canonical(manifest)
    return files


def write_bundle(root, destination):
    files = artifacts(root)
    destination = Path(destination)
    destination.mkdir(parents=False, exist_ok=False)
    for name, data in files.items():
        with (destination / name).open("xb") as stream:
            stream.write(data)
    return digest(files["manifest.json"])


def verify_bundle(root, destination, expected_manifest_sha256):
    """Read-only byte replay plus independently retained manifest commitment."""
    if type(expected_manifest_sha256) is not str or len(expected_manifest_sha256) != 64:
        raise ValueError("independent manifest SHA256 required")
    destination = Path(destination)
    expected = artifacts(root)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("bundle must be a real directory")
    if {p.name for p in destination.iterdir()} != set(expected):
        raise ValueError("bundle inventory mismatch")
    for name, data in expected.items():
        path = destination / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != len(data):
            raise ValueError("bundle member type/size mismatch")
        if path.read_bytes() != data:
            raise ValueError(f"source-bound replay mismatch: {name}")
    actual = digest(expected["manifest.json"])
    if actual != expected_manifest_sha256:
        raise ValueError("external manifest commitment mismatch")
    return {"schema": SCHEMA, "verified": True, "manifest_sha256": actual,
            "scope": "source-bound exact corpus replay, not learning or closed custody"}
