# Trinite scoped agent instructions — QSOL-IBAE procedural adaptation

These instructions **supplement** the unchanged root `AGENTS.md` and any
more-specific agent files. Read [the versioned governance contract](../../docs/AGENT-IBAE-GOVERNANCE.md)
before changing files in this directory.

**Applies to:** Native model, numerical operations, checkpoint readers, and training implementation.

- Preserve `governance != orchestration != execution != evidence`. A model
  proposal, PR description, passing CI or source checksum is **not** an
  owner-issued authority token.
- Classify information as `OBSERVED`, `DERIVED`, `PROPOSED` or
  `UNKNOWN`. Distinguish frozen intent, approved action, real execution,
  recorded verification, native replay, and independent scientific evidence.
- For mutation or resource-bearing work (training, paid/GPU execution,
  publication, external upload, release or merge), require the existing
  exact-scope, owner-authorised path. An instruction file grants no new
  authority. Read-only analysis and software-double conformance must never
  masquerade as a target run.
- Freeze versioned source/import closures, numerical environment, rights,
  inputs, schedules, budgets and failure semantics *before* a target run.
  Source drift means a new request identity, not an updated historical hash.
- Preserve failed, partial, timed-out and negative results. Never silently
  retry, adjust budgets, replace a seed, suppress an inconvenient output or
  convert unknown state into success.
- On interruption explicitly choose `CONTINUE_EXISTING`,
  `RECONSTRUCT_FRESH`, `REQUEST_APPROVAL` or
  `STOP_UNVERIFIABLE`, supported by observed authority and state. A
  context summary or receipt cannot recreate lost execution authority.
- Do not reseal frozen source evidence. Historical source-bound verification
  must use the exact original revision and identities, including the old
  root `AGENTS.md` bytes. New experiments bind this scoped instruction
  file when it affects their source/decision closure.
- Report exact reviewed Git SHA, actual tests and skipped tests, affected
  `TRI-Ixx` invariants, identity-bearing changes and unresolved blockers.
  Do not assert that merely writing these rules runs QSOL-IBAE's Rust
  authority kernel, creates non-constructible native seals, or proves
  machine attestation.

**Correctness outranks speed. Evidence outranks confidence.
Governance outranks autonomy.**
