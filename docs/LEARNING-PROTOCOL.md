# Matched learning/evaluation protocol

Frozen before new target outcomes. The canonical [machine protocol](../src/trinite/learning-protocol.json)
chooses constant 0.001 and 4,096 updates after the retained longer-budget
experiment passed all nine dense per-task training checks. This is a manual,
explicit budget decision, not an automatic selection or a geometry intervention.

Train 27 fresh CPU float32 cells: arithmetic/fold/lattice, seeds 0/1/2 and
dense/ternary/four-state. Keep model, initialization, numeric examples, textual
carriers, family splits, exposure order, AdamW, clipping and scored-target budgets
matched. Geometry loss stays zero. Every diagnostic model call during updates
uses training rows only. Select the final update; no early stopping or test tuning.

The new admission attaches each item's generator revision, split seed and
transformation lineage. `ordering_identity` is its parent example identity;
ranking that key preserves the prior numerical schedule despite new metadata
identities. Retain the parent source/artifact references and do not rewrite old
fixtures, requests or evidence. Game theory, books and other curriculum additions
are outside this experiment.

All training workers terminate before held-out scoring. A fresh evaluation
worker verifies every closed training bundle, checkpoint and regenerated train
prediction/loss, checks all pairings, then publishes a source-bound decision.
Any missing/failed/unmatched cell blocks the entire evaluation. Every dense
task must reach 90% exact answer plus EOS in all seeds. One matrix evaluation
process performs this authorization before any held-out model call; it does not
cache verified status for a later invocation.

The existing foundations gates are unchanged: each dense held-out task must
reach 50% exact answer/EOS and beat the per-task training-majority answer by ten
percentage points in every seed. Preserve quantized failures and per-seed quality,
wall-time and latent-float32-byte margins. Equal float32 storage is not packed
savings. Raw counts, mean/min/max and paired differences are descriptive with
three seeds; do not claim significance or superiority.

Each train worker allows 600 seconds, with the plan bounded to 480 seconds and
524,288 target tokens. Evaluation and summary each have a separate 1,200-second
worker bound. The conservative worker envelope is 310 minutes; the manual
workflow allows 360 minutes, including 50 minutes for setup and retention.
This respects the [hosted job limit](https://docs.github.com/en/actions/reference/limits).
Retain complete, blocked, failed, timed-out and partial outputs. A summary-worker
failure retains its original output and a conservative failed inventory.
Runner loss/cancellation can still interrupt retention.

These are visible software/development fixtures with family separation, not a
fresh confirmatory benchmark. Passing the dense gate permits the next separately
frozen Phase 5 scope; it does not prove general reasoning, quantum protection,
geometry benefit, independent replication or release readiness. Preserve all
earlier null/blocked results and disclose any source change after measurement.
