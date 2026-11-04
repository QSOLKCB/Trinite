# TT-003-FIT-001 — three-seed native teacher fit-feasibility experiment

**Status: DESIGN / IMPLEMENTATION REQUIRED / NOT AUTHORISED TO EXECUTE.**
This is a prospective, separately frozen owner-local experiment. This PR publishes
the protocol, panel selection, budgets, acceptance criteria and agent boundary;
it does **not** implement a trainer, perform an optimiser update, create results,
approve training, launch Codex on an owner machine, or permit publication.

## Why this experiment

TT-003-EVIDENCE-001 trained the original random-initialised 90,624-parameter
dense teacher for 32 updates. The independently verified TT-003-DIAG-001 and
TT-003-LOGIT-001 v2 showed no correct complete vector answers, premature EOS,
and zero as best residue at all 1,944 final vector gold-prefix residue positions.
A lower token NLL is not proof of the intended GF(3) operation. In particular,
excluding EOS left only 648/1,944 correct-best-residue outcomes, explained
by the frequency of expected zero. The new hypothesis is narrower: **can the
unchanged native teacher learn to fit a fixed, small, diverse TRAIN-only panel
without relying on constant-zero predictions or early termination?**

References are observational anchors, not imports of diagnostic model data:
original teacher source `6f2ec05b407a3bafe7a5ed947fd9caadc4be65f7`,
behaviour audit source `b3dd408a51e1cbc9b3280a47352ddb09a71b434a`,
logit diagnostic source `a46dd431c0cc4b59277afbf5d11775033cc2dba6`
(unmerged PR #133 at design time), frozen logit request
`8585174ce406a7c57ffd1b3f69fb2996a7488064a3cd54789147ad00a36e52b9`,
and original checkpoint `ec0e42658818fb9fb1712f11117eebc3c5adf101aa668e335ab788b5e2f6a551`.
None is a new training input. Do not rewrite, reseal or replay those artifacts
under an evolved source tree.

## Fixed treatment: three distinct initialization seeds

Run **three independent starts** with seeds `17`, `29`, `43`.
These are **seed-sensitivity replications**, NOT three same-seed determinism
replays. All three use precisely the same panel, row order, 256-update schedule,
optimizer configuration, training token budget, diagnostic positions and scoring.
Each seed starts from a fresh random-initialised model and fresh AdamW state.
The final checkpoint of one run must never initialise another.

- Existing `Decoder` with `lane="dense"`, 90,624 parameters, context 128,
  exact tied-output/GELU/RMSNorm semantics, native CPU FP32 only.
- The existing `RunConfig`, `TrainingState`, `create_optimizer`,
  `run_steps`, `target_loss`, `evaluate` and safe checkpoint/reader;
  no new optimizer, loss, EOS weights, length penalty, gradient logic,
  dropout, sampling policy, architecture, temperature or decoding constraints.
- Per seed: 256 updates, batch size 4, constant learning rate `0.001`,
  existing AdamW defaults/gradient clipping, at most 4,096 supervised
  answer/EOS tokens, at most 600 seconds of native between-update budget
  plus an independently enforced 900-second whole-process supervisor.
  These limits are **not** performance estimates and neither guarantees
  progress; fail on either exhaustion and retain the failure.
- Observe checkpoints at 0, 64, 128, 192 and 256 completed updates.
  Diagnostic observation must not alter training state, update count,
  RNG/order, or optimizer tensors.
- Fixed predeclared policy: no retries, restarts, resuming failed seeds,
  adaptive extension, smaller samples, skipped seed, early stopping,
  'best checkpoint' selection or conditionally changed hyperparameters.
  Failures and partial progress are outcomes, not reasons to silently
  substitute another experiment.

## Dataset: fixed 72-row fit panel from TT-002 TRAIN

Cold-regenerate and independently verify the original TT-002 packet under the
**exact reviewed source closure** used for this experiment. Require separately
retained canonical manifest SHA-256. No privileged validation/test targets
enter model construction, training, diagnostics or selection.

Select exactly six distinct `observation.word` values from the 81 possible
four-trit words. Sort the entire validated 81-word TRAIN inventory by the
ascending SHA-256 digest of the ASCII bytes:

`b"trinite.teacher-fit-001.word.v1\0" + word.encode("ascii")`

Break any digest ties lexicographically by the word itself, then choose the
first six. This is **data-only precommitment**: no model scores, losses,
feedback or held-out results influence the selection.

For each selected word and each TRAIN circuit `S1`, `S2`, `W`, include
exactly four original TT-002 rows: one `apply`, one `inverse`,
one positive `check` (answer `1`) and one negative `check`
(answer `0`). The resulting panel has 72 unique rows:
18 `apply`, 18 `inverse`, 18 positive and 18 negative `check`.
Require duplicate/missing/foreign-task rejection, original TT-002 row
identity, independently reproduced answer/oracle, context and loss-mask
validation. Preserve actual zero/one/two distributions at every residue
position, rather than assuming uniformity.

Use **one identical, seed-independent training order** for all runs:
sort the 72 admitted row IDs by SHA-256 of

`b"trinite.teacher-fit-001.row.v1\0" + row_id.encode("ascii")`

and break collisions by row ID. Visit the list cyclically in batches of four.
All three runs execute the same first 1,024 scheduled visits; freeze a full
visit-ID ledger and summed supervised-token count **before training**,
requiring <= 4,096 actual masked targets. Bind the panel, row order,
complete scheduled visit list and their identities to the request. Reject
unexpected target or ordering changes. These 72 panel rows are not an
independent scientific holdout; fitting them establishes feasibility only.

## Mandatory controls and metrics

Before any run, calculate and commit both fixed predictions for every panel
row, with EOS in the expected position:

- Constant zero: `0<EOS>` for scalar checks, `0000<EOS>` for vectors.
- Constant one: `1<EOS>` for scalar checks, `1111<EOS>` for vectors.

Controls are analytical predictions, never trained or used as decoding
constraints. Record full-answer exactness, each family/task/circuit,
positive-vs-negative scalar, target residue counts by position, EOS timing,
gold-prefix target NLL, and top-ranked residue (especially zero).
Raw greedy decoding must use the unmodified argmax over all 259 tokens,
at most five generated tokens, stop only at EOS, retain invalid output.

At each fixed checkpoint, record and preserve:
(a) complete exact panel answers and per-task/circuit/label counts,
(b) constant-zero and constant-one baselines on the same rows,
(c) vector per-position residue accuracy including the best residue when EOS
is analytically excluded, (d) gold-prefix NLL split by target class and
premature versus terminal EOS, (e) wrong-length and invalid-token outcomes,
(f) model/optimizer/schedule/resource identities, and (g) original errors.
Never equate low NLL, correct EOS, high token rank, or a plausible output
with learned GF(3) operations. Do not tune a threshold on outcomes.

**Primary endpoint:** exact complete answers on all 72 fitted TRAIN rows
at the fixed final 256-update checkpoint, reported separately for each seed.
Fitting **passes** per seed only at 72/72, including 18/18 apply,
18/18 inverse, 18/18 positive and 18/18 negative checks.
Anything else is a failed or partial fit, even if loss improves.
Compare each final seed to both constant-predictor controls, using actual
panel prevalence and no assumed chance level.

**Secondary descriptive outcomes:** exact progress curves at the frozen
checkpoints, first emergence of nonzero input-sensitive outputs,
task-specific losses, residue ranks, prefix-conditioned premature EOS,
visited panel counts, variance across the three seeds, and final output
diversity. All are exposed TRAIN metrics. Three seeds do not establish a
population-level significance result or generalisation.

## Implementation and freeze requirements for Codex

The design PR is deliberately not an executable producer. Add a separately
reviewed, additive implementation under `scripts/teacher_fit_v1/`, a
dedicated CLI `scripts/run_teacher_fit.py`, targeted contract tests, and
a dedicated CI workflow using **software doubles** and bounded synthetic
one-step canaries only. No target full training in CI.

Before a target run, freeze an immutable canonical request with:
exact Git commit; complete transitive source/import closure including
`AGENTS.md`, the scoped `AGENTS.md`, plan, this protocol and tests;
actual TT-002 manifest+rights; full selection and visit inventories;
model/optimizer/environment pins; three seeds and predeclared outcomes;
per-run time/memory/bytes limits; separate evidence destination
identities; and the explicit failure policy. Reject a dirty checkout,
changed source or missing/extra/duplicate panel records.

The owner must provide a **separate canonical, source/request-bound
authorization** for each of the three seed-specific native update
operations. Publication of this plan or opening/merging a PR is not
that authorization. Validate each approval before importing a model,
and recheck at the actual execution boundary. A genuine owner statement
is required; the agent must not manufacture approval from a run plan.
One approval must not be reusable for a different seed, request, source,
schedule, environment or retry.

On each run, produce an independent private directory with request,
approval, exact checkpoint and AdamW tensors, weights-rights record,
initial/final state identities, per-checkpoint train-only predictions
and probabilities/metrics, native resource/error receipts, run manifest
and independent sidecar SHA-256. Never overwrite an existing path.
Distinguish read-only archive/metric verification, restored native
forward replay and **full optimizer replay**; only claim which actually ran.

After all three attempts, make one **separate, read-only**
`TT-003-FIT-001-TRIPLE-SYNTHESIS.md` comparing all seeds, including failed
attempts; accept no post-hoc preferred seed or missing cell. Require
source-bound manifests and independent recomputation of every aggregate,
plus explicit negative and inconclusive outcomes.

All model weights, optimizer moments, data packet, request approvals,
private evidence and private output remain owner-local; none goes into
Git, GitHub Actions artifacts, an external API or a publication. Dataset
rights and weights licences remain independent and undecided.
No GPU fallback; no TT-004, distillation, model scaling, EOS intervention,
or early promotion to a trained teacher.

## QSOL-IBAE-inspired task-level agent discipline

This is **procedural adoption** of the QSOL-IBAE separation of governance,
orchestration, execution and evidence. It does not import, run or claim
assurance by the QSOL-IBAE native Rust runtime. In this protocol:
`PROPOSED` is not `AUTHORIZED`; `EXECUTED` is not
`VERIFIED`; `VERIFIED` TRAIN fit is not `GENERALIZED`.
The model/agent cannot issue its own approvals, increase budgets, declare
final success from prose, convert a timeout into a successful stop, or
turn a failed run into a fresh attempt without a newly versioned request
and human authority.

If interrupted, classify `CONTINUE_EXISTING` only where live, exact,
externally verified authorization and state permit it; otherwise choose
`RECONSTRUCT_FRESH` with **new** authority, `REQUEST_APPROVAL`,
or `STOP_UNVERIFIABLE`. A context summary, receipt or Git commit is
evidence, not a reconstructed live execution capability.
Do not infer success from a checksum alone or from a synthetic test.

Correctness outranks speed. Evidence outranks confidence.
Governance outranks autonomy. The gates govern claims, not inquiry.

## Acceptance / handoff checklist

- [ ] A source-clean, exact-head reviewed implementation PR with green
      contract/CI checks and tested negative paths.
- [ ] Valid TT-002 manifest, source and oracle; frozen 72-row panel and
      1,024-visit schedule identical across seeds.
- [ ] Explicit per-seed owner authorizations with 256-update/4,096-target
      ceilings; no inference from a PR or plan.
- [ ] Three **independent** native attempts in fresh locations, preserving
      failure outputs and original return codes.
- [ ] Independent recorded verification and separately stated forward/
      optimizer replay evidence.
- [ ] Honest comparison to constant-zero/one controls, full 72-row fit,
      residue-specific behaviour and EOS timings.
- [ ] Private three-result synthesis, all hashes; no publication or TT-004.

Affected Trinite boundaries: TRI-I01/I02/I03/I05/I07/I08/I10/I11/I12/I13/I14.
No existing historical source, observer, model forward, tokenizer, original
teacher producer, frozen request, checkpoint or diagnostic is changed here.
