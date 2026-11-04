# TT-003-FIT-001 — scoped Codex instructions (QSOL-IBAE-inspired)

Read root `AGENTS.md` and `docs/TEACHER-LAB-FIT-001.md` before
implementing or reviewing any file under this directory. This file is
*procedural* governance guidance, **not** an assertion that the
QSOL-IBAE native runtime has executed or attested this experiment.

## Authority and evidence states

- Maintain separate statuses: `PROPOSED`, `FROZEN`,
  `OWNER_AUTHORIZED`, `EXECUTED`, `RECORDED_VERIFIED`,
  `NATIVE_REPLAYED`, `FAILED`/`BLOCKED`. A PR, approved code
  review, plan, or model-written approval text is not training authority.
- Governance chooses what may execute; orchestration prepares bounded
  work; the existing native Trinite optimizer executes; independent
  verification establishes what happened. No layer can promote itself.
- Never claim `GENERALIZED` from any TT-002 TRAIN observation,
  unchanged model size, a seed comparison or a signed/hashed receipt.
- Freeze source, dataset, rights, 72-row selection, entire 1,024-visit
  order, seeds 17/29/43, complete configuration, budgets and evidence
  schema before any target update. Source drift means a **new**
  frozen request, not a resealed historical one.
- A complete, separate owner approval is required for **each** seed.
  No new approval may be derived from an existing approval or from
  successful tests. The executor must recheck identities at dispatch.
- Reject missing/extra examples, unsupported rights, mismatched
  source, privileged targets, mutable approval, altered numerical
  environment, unexpected backend and budget or checkpoint changes.
- Every operation is CPU-only with 256 updates, batch four, 4,096
  supervised targets maximum and separate 600s native / 900s outer
  bounds. Record real consumed resources. A watchdog expiry does not
  prove termination, success or optimizer safety.
- Preserve all three attempts, failures, partials and unmodified
  output bytes. No automatic retry, seed replacement, budget extension,
  dynamic sampling, successful-subset report or accidental upload.
- Prefer existing `Decoder`, `RunConfig`, `TrainingState`,
  `run_steps`, `target_loss`, `evaluate` and safe checkpoint
  reader. Do not add a new optimizer, loss, model implementation or
  unnecessary dependencies.
- Software doubles and CI one-step canaries establish engineering
  contracts only. Real fit requires owner-local evidence and honest
  exact-answer performance against precomputed constant-zero/one
  controls. Distinguish recorded verification, forward replay and
  optimizer replay.
- Recovery state must be one of `CONTINUE_EXISTING`,
  `RECONSTRUCT_FRESH`, `REQUEST_APPROVAL` or
  `STOP_UNVERIFIABLE`. Never fabricate live authority from a context
  summary, serialized grant or source hash. A failed attempt is not
  silently resumed under its old approval.

## PR contract

For each implementation PR state affected TRI-Ixx requirements, exact
source/head SHA, changed correctness identities, tests run versus
untested, authority boundaries, source of real evidence and remaining
negative findings. No direct-main edit, automatic merge, release,
checkpoint publication or TT-004 promotion.

Principle: **Correctness outranks speed. Evidence outranks confidence.
Governance outranks autonomy.**
