# SPDX-License-Identifier: MPL-2.0
"""TT-003-FIT-001 TRAIN-only panel selection; no model/optimizer execution.

This is a deterministic, data-only planning helper. Its caller must first
independently verify the entire TT-002 source-bound dataset and rights packet.
"""
from __future__ import annotations

import hashlib
from itertools import product
import re

FAMILIES = ("S1", "S2", "W")
WORDS = tuple("".join(chars) for chars in product("012", repeat=4))
SEEDS = (17, 29, 43)
WORD_DOMAIN = b"trinite.teacher-fit-001.word.v1\x00"
ROW_DOMAIN = b"trinite.teacher-fit-001.row.v1\x00"
ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
EXPECTED_ROWS = 972
PANEL_WORDS = 6
PANEL_ROWS = 72
UPDATES = 256
BATCH = 4
TARGET_LIMIT = 4096


def _rank(prefix: bytes, text: str) -> tuple[bytes, str]:
    raw = text.encode("ascii", errors="strict")
    return hashlib.sha256(prefix + raw).digest(), text


def select_panel(train_rows: list[dict]) -> dict:
    """Validate TRAIN structural inventory, bind 72 IDs and 1024 visits.

    Does not verify the TT-002 oracle, manifest, rights or source closure.
    The owner-local runner MUST validate those independently before calling.
    """
    if type(train_rows) is not list or len(train_rows) != EXPECTED_ROWS:
        raise ValueError("requires exact original 972-row TRAIN inventory")
    keys: dict[tuple[str, str, str, str], dict] = {}
    ids: set[str] = set()
    for row in train_rows:
        if type(row) is not dict:
            raise ValueError("invalid TRAIN row")
        try:
            rid, family, split = row["id"], row["family"], row["split"]
            obs, answer = row["observation"], row["answer"]
            tokens = row["supervised_tokens"]
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete TRAIN row") from exc
        if (type(rid) is not str or ID_RE.fullmatch(rid) is None
                or rid in ids or type(obs) is not dict
                or split != "train" or family not in FAMILIES
                or obs.get("circuit") != family
                or obs.get("word") not in WORDS
                or obs.get("task") not in ("apply", "inverse", "check")):
            raise ValueError("foreign/duplicate/malformed TRAIN row")
        task, word = obs["task"], obs["word"]
        if task == "check":
            if answer not in ("0", "1") or type(tokens) is not int or tokens != 2:
                raise ValueError("invalid check target")
            view = answer
        else:
            if (type(answer) is not str or len(answer) != 4
                    or set(answer) - set("012")
                    or type(tokens) is not int or tokens != 5):
                raise ValueError("invalid vector target")
            view = ""
        key = (family, word, task, view)
        if key in keys:
            raise ValueError("duplicate structural TRAIN task")
        ids.add(rid)
        keys[key] = row
    expected_keys = {
        (family, word, task, view)
        for family in FAMILIES for word in WORDS
        for task, view in (("apply", ""), ("inverse", ""),
                           ("check", "0"), ("check", "1"))
    }
    if set(keys) != expected_keys:
        raise ValueError("TRAIN task/word/circuit completeness mismatch")
    selected_words = sorted(WORDS, key=lambda w: _rank(WORD_DOMAIN, w))[:PANEL_WORDS]
    panel = [
        keys[(family, word, task, view)]
        for family in FAMILIES for word in selected_words
        for task, view in (("apply", ""), ("inverse", ""),
                           ("check", "0"), ("check", "1"))
    ]
    if len(panel) != PANEL_ROWS:
        raise ValueError("fit panel length differs")
    ordered = sorted(panel, key=lambda row: _rank(ROW_DOMAIN, row["id"]))
    scheduled = [ordered[i % PANEL_ROWS] for i in range(UPDATES * BATCH)]
    targets = sum(row["supervised_tokens"] for row in scheduled)
    if targets > TARGET_LIMIT:
        raise ValueError("native scheduled target budget exceeded")
    return {
        "schema": "trinite.teacher-fit-001-panel.v1",
        "selected_words": selected_words,
        "panel_ids": [row["id"] for row in ordered],
        "visit_ids": [row["id"] for row in scheduled],
        "scheduled_target_tokens": targets,
        "seeds": list(SEEDS),
        "optimizer_updates_per_seed": UPDATES,
        "batch_size": BATCH,
        "heldout_scored": False,
        "training_authorized": False,
    }
