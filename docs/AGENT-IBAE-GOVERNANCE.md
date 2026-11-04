# QSOL-IBAE procedural governance for Trinite agent work — v1


> **v2 compatibility note:** This guidance is activated only through new
> directory-scoped `AGENTS.md` files under `scripts/`, `src/trinite/`,
> `tests/`, `docs/`, and `.github/`. It does not change the root
> `AGENTS.md`, so the byte identities of historical source closures remain
> unchanged. New agent instructions are prospective and procedural only;
> historical fixtures are not regenerated. In an actual run, new source inputs
> must be frozen and reviewed on their exact commit.

**Status:** machine-facing procedural instructions and review criteria.
These rules adapt selected authority, evidence and recovery distinctions
from QSOLKCB/QSOL-IBAE, **Invariant-Bounded Agent Execution**.
They do not import or execute its Python/Rust runtime, issue native
governance seals, prove enforcement in Trinite, or replace owner review.

Read root `AGENTS.md`, `docs/INVARIANTS.md`, the affected experiment
contract, and the current exact source/head before any implementation.

## Four non-interchangeable planes

| Plane | Permitted duty | Forbidden promotion |
| --- | --- | --- |
| Governance | Determine allowed operations from human approvals, contracts and limits | Model proposals, PR text or tests do not mint approval |
| Orchestration | Build a frozen plan, order permitted work, record blockers and invoke admitted tools | Cannot enlarge execution budgets, reinterpret permissions or bypass denial |
| Execution | Perform only the declared, authorised bounded native operation | Cannot issue approvals, change expected results or classify its own outputs as verified |
| Evidence and analysis | Record actual receipts, independently verify and classify claims | A hash, passing CI or impressive metric is not producer authentication, causal proof or generalisation |

A model is an information source and proposal generator, not a substitute
for owner authority, an independently verified experiment, or an immutable
receipt. No statement of self-awareness, simulated confidence or apparent
understanding has operational authority.

## Minimum agent state, shown explicitly

For every substantial repository mutation or controlled experiment,
maintain explicit classifications:

- `OBSERVED`: directly inspected source, real tool result or recorded
  native output, with exact provenance where available.
- `DERIVED`: computation, static deduction or comparison from named
  observations; include assumptions and independently recomputable math.
- `PROPOSED`: model-recommended code, run, experiment or conclusion,
  not yet accepted or executed.
- `UNKNOWN`: missing, stale, contradictory or unreproducible state.

For an individual action, distinguish:

1. `PROPOSED`: named operation and reason.
2. `FROZEN`: committed exact source, inputs, protocol, expected scope,
   identity, resources, failure policy and tests.
3. `OWNER_AUTHORIZED`: explicit scope-matched human grant; no implied
   permission from a plan, previous approval, CI success or user silence.
4. `EXECUTED` / `FAILED` / `BLOCKED`: actual action and raw exit
   code, partials and resource record; no status invention.
5. `VERIFIED_RECORDED`: independent byte, source, inventory and derived
   metric checks, not proof of fresh native execution.
6. `VERIFIED_NATIVE_REPLAY`: distinctly stated actual model/tool replay
   matching the declared equality/tolerance contract.
7. `REVIEWED` or `PUBLISHED`: only after the relevant independent
   review and human authority; not automatic promotion from any prior step.

These labels are **documentation conventions** in Trinite, not the
QSOL-IBAE native Rust runtime's accepted sealed states.

## Exact authority and boundedness

For executable experiments, a frozen request must include:
source Git head, complete import/source closure, immutable input and
rights identities, numerical environment and backend, exact planned
operation, target schedule, per-run resource caps, expected evidence,
failure/retry policy and *separate* owner approval. The approval must bind
the exact request, particular action and limits; if any of these drift,
reject and require a new version or authority decision.

For source edits, Codex may work in a branch and open a PR within the
owner's specified scope. Running CI is evidence of that particular
source revision, not authorization to merge, release, publish, train,
upload private artifacts or activate any service. The source of approval
must be unambiguous and be checked again at the action boundary.
Subordinate workers, local models, harnesses, CI and benchmark machinery
never become supervisors and cannot self-extend execution authority.

No optimizer or model evaluation may silently change datasets, weight
state, seed, temperature, EOS constraints, sampling, learning schedule,
budget or backend to force success. Do not silently retry a failed,
timed-out or unknown-outcome action. A timeout signals a state that must
be investigated; it does not prove that a process stopped or that
side effects did not occur. No dependent step may proceed while such
an outcome remains unresolved.

## Immutable historical evidence

Root `AGENTS.md` is included in several source-bound Trinite
closures, including TT-002/TT-003. Therefore **this policy edit
changes future source identity**. It does not retroactively update,
repair or invalidate evidence when verified under its original exact
historical checkout, and it must never be used to reseal old artifacts
as though their source had been unchanged.

When a new experiment requires the adapted guidance, freeze a
**new request version on the exact reviewed commit**, regenerate
admitted source-derived data/rights receipts where that version
requires them, and preserve the old data/checkpoint/manifest separately.
Never run old freeze or verifier source from extracted evidence archives.
Verify each historical result using its own source tree/commit and
retained anchors. Do not make change to root instructions inside an
already frozen target execution and claim equivalent authority.

## Bounded recovery after interruptions

Use one of the following decisions and preserve the evidence that
supports it:

- `CONTINUE_EXISTING`: only when live same-task authority, exact
  source and state, remaining budget and continuous provenance are
  demonstrably available.
- `RECONSTRUCT_FRESH`: start a **new** bounded task/request with fresh
  owner authority, using old records only as non-authoritative context.
- `REQUEST_APPROVAL`: when the intended mutation or continuation
  exceeds or ambiguously intersects granted scope.
- `STOP_UNVERIFIABLE`: when observed state cannot safely establish
  an admissible continuation or recovery.

The invariant adapted here is `IBAE-REC-001`: interruption must
never increase execution authority or invalidate previously verified
work. A saved JSON context, model summary, SHA-256 digest, Git commit
or sidecar may establish identity but cannot reconstruct a lost native
execution capability or issue a continuation lease.

## Mandatory PR/change record

The implementing agent must disclose affected `TRI-Ixx` invariants,
actual tests run (and missing checks), precise Git head, changed
correctness-bearing identities, any benchmark-only data, source/rights
authority, expected/historical compatibility, failures and negative
evidence, unresolved blockers and whether review/merge/run/launch is
authorised. Explicitly distinguish observed success from simulated
software-double conformance and from merely prospective architecture.

No claim of live QSOL-IBAE enforcement, formal proof, remote
attestation, independent-host reproducibility or model competence is
licensed by adopting this document.

**Correctness outranks speed. Evidence outranks confidence.
Governance outranks autonomy.**

**Research principle:** the gates may decide what can be claimed;
they must not be mistaken for evidence about what happened.
