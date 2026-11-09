#!/usr/bin/env python3
"""Optional Heretic 2.x scorer adapter. Importing this file needs Heretic installed.

Do NOT run with Heretic's default configuration (it optimizes refusal removal).
No network calls, dataset downloads, model loading or execution on import.
"""
from pathlib import Path
import sys

# Explicit sibling import. The actual Heretic executable imports this file by
# path only after the owner independently authorizes a locally installed model.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtc_heretic_formal_001 import SYSTEM, cases, summarize  # noqa: E402
from heretic.scorer import Score, Scorer  # noqa: E402
from heretic.utils import Prompt  # noqa: E402


class FormalAuthorityScorer(Scorer):
    """Exposed training-objective score; not a locked evaluator or safety gate."""

    @property
    def reproducible(self) -> bool:
        return True  # Deterministic, static prompt bank; host inference may vary.

    @property
    def score_name(self) -> str:
        return "Formal authority error loss (EXPOSED)"

    def init(self, ctx) -> None:
        self._rows = cases()
        self._prompts = [Prompt(system=SYSTEM, user=row["prompt"])
                         for row in self._rows]

    def get_score(self, ctx) -> Score:
        summary = summarize(self._rows, ctx.get_responses(self._prompts))
        label = (
            f"false approvals {summary['false_approvals']}, "
            f"true rejections {summary['true_rejections']}, "
            f"invalid {summary['invalid']}, "
            f"wrong numeric {summary['numeric_wrong']}"
        )
        return Score(
            value=summary["loss"],
            rich_display=label,
            md_display=label,
        )
