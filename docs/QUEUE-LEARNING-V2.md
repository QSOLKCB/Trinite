# Queue-learning v2: proposed bounded fit-first pilot

**Implementation for review, not execution authorization.** The
[diagnosis](QUEUE-EOS-DIAGNOSIS-V2.md) found no demonstrated numerical defect.
PR #114 retained the proposal, diagnostics and exact data recipe; those evidence
files remain unchanged. The separately source-bound v2 controller now implements
this contract through `scripts/run_queue_learning_v2.py`. Its engineering tests
use mocked numerical updates and harmless subprocesses, not a new scientific
optimizer experiment. The v1 controller/approval cannot execute this study.
After a reviewed green merge of this implementation, obtain fresh supervised
feasibility, freeze and explicit owner approval of that exact freeze. Any source,
data or runtime change requires fresh receipts and approval. CI cannot launch
this study; original v1 producers, fixtures and negative results stay unchanged.

## Rationale and exact changes

V1 failed to generate digits on training rows as well as held-out rows. First
test basic fit on eight distinct training prefixes, then, only if it passes,
test one fourfold horizon increase from 64 to **256 updates** on the same 256
v1 train episodes. Keep seeds 17/29, model, precision, LR, batch, data order,
loss, optimizer, input carrier, parsing and unrestricted greedy decoding fixed.
There is no search, EOS treatment, label relabeling or larger-model treatment.

V1 validation/test targets and results are exposed exploratory evidence. They
are excluded from v2 training, fit checks, validation, test and checkpoint
selection. No claim will call them a fresh confirmatory holdout. V2 uses new
families below. This is a bounded development study with public construction,
not proof of operator blindness, a secret benchmark or significance study.

## Frozen data recipe and overlap rules

[Exact requests](evidence/queue-eos-diagnostics-v2/requests.json),
[proposal](evidence/queue-eos-diagnostics-v2/proposal.json) and
[observation-only overlap audit](evidence/queue-eos-diagnostics-v2/overlap.json)
are preregistered before any fresh target call. `protocol.episodes()` regenerates
384 requests: the unchanged 256 v1 train episodes, 64 new validation and 64 new
test episodes. Evaluation has 16 new families, four per kind, eight variants
each; j=12..15 and capacity=4+j. Each family keeps both initial states, both
masks and both hidden suffixes together. Observation window and forecast horizon
remain eight ticks. Initial state, future +3-arrival alternative and masks match
v1. Steady and ramp formulas extend v1 to these j values. Periodic amplitude is
three rather than two to avoid repeating its 12-parameter schedule cycle; outage
base arrivals use 2+j//3 rather than 2+j//4 to avoid repeating old base schedules.
These are evaluation-family construction changes, not tuned training features.

Per kind, rank j by SHA-256 of ASCII `queue-learning-v2/fresh/{kind}/{j}`;
first two families are validation, last two test. No labels or model predictions
participate in selection. The exact request inventory SHA-256 is
`33d54a234d3320acdee9fccbe461af696003bf3d3b2871d43f9233d25d28c076`.
There are 189 distinct public prefixes across the 384 requests. All aliases stay
within one split; the new held-out observation prefixes have zero overlap with
**all 384 exposed v1 episodes**, not just their training subset.

The conservative related-family key is the complete observed-prefix driving
arrival/service/outage arrays, ignoring kind, capacity, initial state, mask and
the entire hidden future. Equal keys must stay together; fresh evaluation keys
must not equal any exposed v1 key. Exact request, exact canonical observation,
related-family and declared family grouping must be rechecked before freeze
and again before execution, without labels. Reject on overlap; retain the
failure and stop. Do not redraw families, tune the hash salt or discard cases.
Broader algorithmic similarity among these four kinds remains; this does not
test transfer to a wholly new workload kind.
New capacities and some driving schedules extend beyond the v1 train range;
this is a declared extrapolation setting, not an identical-distribution retest.

Admission must independently regenerate these exact commissioned formal facts,
record rights basis and source scope and delegate observations/targets to the
unchanged Rust gate. V1's 384-request admission does not cover the new inventory.
Only train requests may materialize labels in numerical state. Freeze and
overlap checks use observations only; no fresh held-out targets were computed
in this preparation.

## Train-only fit sanity check

Use fresh random initialization with **seed 17**, independent of the main pilot.
Rank the existing train rows by SHA-256 of ASCII episode ID and take the first
eight distinct canonical observation prefixes, one existing label per prefix.
Selection ignores label values, errors and difficulty. `protocol.fit_report`
prepares the forward-only measurement and retains all eight raw outputs. The exact
[selection](evidence/queue-eos-diagnostics-v2/fit-selection.json) has 22 answer/EOS
targets per batch, hence **2,816 actual targets** in 128 updates. All eight are
train-exposed; this measures tiny-pool optimization adequacy only, not prediction
of a hidden future, full-corpus fit or generalization.

Train batch eight for exactly **128 updates**, no interim checkpoint selection.
Collect answer-byte, first-answer and EOS counts/summed NLL/correct counts and
unmodified raw generations plus strict parse status at step 0 and 128. At the
final checkpoint require **8/8 exact canonical decimal+EOS outputs**. Keep every
case in the denominator. Failure, timeout, nonfinite state, budget failure or
anything below 8/8 stops the entire study, with no retry or main pilot. Do not
test alternative subsets, horizons or hyperparameters. Discard fit weights and
optimizer state; main cells start independently from their declared seeds.

## Main pilot and hard limits

| Setting | Exact proposal |
| --- | --- |
| Model | Dense native decoder, 24,768 unique parameters; context 256; byte vocabulary 259; one block, width 32, four heads, FF64; tied input/output; RMS epsilon 1e-5; no dropout/bias |
| Runtime | Linux x86_64 CPU float32 eager; unchanged requirements.lock, one Torch compute thread, strict deterministic algorithms; no autocast/backend substitution |
| Seeds and attempts | Fit seed 17 once; main seeds 17 and 29 once each; three attempts total, sequential, no retries/resume/extra seeds |
| Optimizer | Native single-tensor AdamW, LR .001, betas .9/.999, epsilon 1e-8, decay .01, clip norm 1; foreach/fused/AMSGrad/capturable/maximize/differentiable off |
| Loss | Unchanged mean cross-entropy over shifted decimal answer bytes and EOS only; geometry loss zero |
| Batch/order | Eight; same SHA-256 episode-ID rank as v1, cycling without reshuffle; eight main cycles |
| Update budget | Fit 128; main 256 per seed; **640 total** if all cells execute |
| Visits/targets | Fit 1,024 visits, 2,816 actual targets, hard cap 4,096; main 2,048 visits and 5,600 actual targets per seed, hard cap 8,192 each; global hard target cap 20,480 |
| Wall limits | 120 seconds native updates and 900 seconds whole attempt including imports/admission/diagnostics/checkpoint/inference; 2,700 seconds entire study; stop at the first reached limit |
| Memory | 1 GiB process RSS, 2 GiB aggregate process-tree RSS, 4 GiB per-process address-space limit; parent must monitor/kill/wait and retain observed peaks/failures; these are not whole-machine limits |
| Selection | Main final step 256 only; fit final 128 only; no validation/test election or early fit-success election |
| Generation | Unchanged unrestricted greedy argmax, five emitted tokens, stop on EOS; no retries, repair, numeric constraints or answer-conditioned cap |

No automatic hyperparameter search, larger model, GPU, paid compute or geometry
intervention. Total fit+main actual targets would be 14,016. Failed/unattempted
cells remain in the planned inventory. No claim that fourfold updates will fix
the collapse. Feasibility must check worst-context memory and three engineering
forward/backward/clip timings with zero optimizer steps, conservatively project
the **256-update** budget with fourfold margin, and include diagnostic/IO time.
Failure blocks freeze; no automatic budget enlargement.

Collect detached full-unique-train diagnostics at initialization and final step
only, separately from scheduled visits and per-batch optimizer history. Keep
answer/EOS target counts and summed NLL distinct. Retain all raw generated token
lists, strict parsing status, prefix-conflict audit, checkpoint identities and
losses. Reconcile diagnostic sums against native target_loss; never replace its
objective with a diagnostic implementation.

## Baselines, validation permission and test isolation

Fit lower median and mode constants on **all 256 train episodes only**, including
their variants, with v1 tie rules. Freeze constants before held-out inference.
Keep the fixed v1 causal continuation baseline; missing last public queue,
arrival or service produces an explicit abstention. No baseline fitting,
calibration or choice uses fresh validation/test targets.

Validation is reporting-only after both final checkpoints are anchored. It may
inform a separately preregistered later development study, never alter this
run's horizon, LR, decoder, checkpoint, baselines or data. No validation loss
enters updates, fit diagnostics or selection. Test labels stay isolated from
all development, including validation interpretation. Commit **every planned
validation and test output for both seeds and all three baselines before
scoring any held-out target**; verify all public anchors before private reads.
Report all planned cells together; no adaptive second test access.

Use the existing two-anchor prediction gate, preserving raw tokens and failures
beside gate-interface abstentions. Malformed empty answers stay malformed.
Primary all-case numeric MAE is defined only with full planned numeric coverage;
otherwise null. Also report conditional numeric MAE with its own denominator,
exact/all planned, numeric exact/numeric, numeric coverage, valid-output rate,
and counts for prediction/abstain/malformed/failure/timeout/not_attempted.
Preserve rational sums/counts and null for zero-denominator metrics. Main
denominators are 64 per method/split, regardless of failure or fit-gate stop.
Compare each seed and split with all three fixed baselines; a descriptive gain
requires full coverage and lower MAE than every baseline in each seed. Two seeds
do not establish significance. Constant numeric outputs are not input-sensitive
forecasting. Fresh-family and exposed-v1 metrics are not interchangeable.

Keep partial artifacts, logs, bounded child outputs, resource receipts and
failed comparisons. Never silently skip cases, relabel malformed outputs,
restart, extend limits or publish only successful cells. Missing/invalid
checkpoints cannot enter inference. Safe loading remains closed safetensors,
32 MiB tensors/1 MiB metadata, externally anchored identities, finite/shape/dtype
and named model/AdamW inventory checks. No foreign-environment forensic state
may become training state.

## Review and future approval gate

This PR is preparation only and must not merge automatically. After owner
review, merge and green post-merge CI of the actual implementation, obtain a
**fresh v2 feasibility receipt** and freeze binding that merged SHA, complete
runtime source closure including this proposal/diagnostics/launcher, exact
requests/observations/splits/admission, CPU dependency hashes, Rust locks/build
identities, interpreter/CPU/thread/precision/determinism, budgets and selection.
Retain independent freeze and preparation anchors outside the checkout. Ask the
owner to approve **that exact freeze** before any fit or main optimizer step.
The old v1 approval provides no authorization. Changed source/data/dependencies/
runtime requires a new receipt, freeze and explicit approval, not edited pins.

TOMBSTONE-derived learning needs its own task definition, admission, baselines,
family split, evaluation protocol and authorization in a separate follow-on.


## Implemented controller and operator boundary

Affected requirements: TRI-I02/I03/I07/I08/I11/I12/I13/I14. This is a new v2
execution schema, not a migration of v1 authorization or evidence. The preserved
`protocol.PROTOCOL` remains the non-executable proposal; `spec.PROTOCOL` is the
source-bound executable contract whose admission still requires exact approval.
The complete native updater, decoder and closed safe tensor reader are reused;
no alternative numerical implementation or dependency change is introduced.

The fixed launcher applies the 4 GiB address-space rlimit before native imports.
An outer Linux supervisor covers the whole study, including admission, baseline
construction, commitment and scoring. Inner supervisors cover each sequential
worker and checkpoint validation, charging a main cell's wait for the other cell
against its 900-second lifecycle. An atomic update marker and per-step journal
allow the parent to terminate an update that hangs past 120 seconds. Fit/main
updates and targets are statically reserved; failed cells never replenish them.
An exclusive run claim and per-attempt claims beside the independently retained
preparation anchors prevent repeating a consumed freeze in a new output folder.
Do not remove or copy these claims to manufacture another attempt.

The supervisor monitors transitive descendants, including new sessions, adopts
orphans, kills outstanding processes and reaps them before sealing the final
inventory. Receipts retain bounded stdout/stderr, sampled per-process/tree RSS,
kernel child high-water RSS and cumulative reaped CPU usage. **RSS is sampled**,
not a kernel-enforced allocation ceiling or proof that no transient aggregate
peak occurred. AS is a hard inherited per-process limit. `/proc` inspection
failure is terminal. The fixed workers are cooperative trusted code; the parent
marker prevents accidental unsupervised internal CLI use, not hostile same-user
execution or kernel-level attestation. These are process-tree, not host limits.

Final fit acceptance independently safe-restores the bound step-128 checkpoint
and regenerates all eight canonical decimal+EOS outputs, rather than trusting a
boolean. Each main worker revalidates the fit certificate and starts a fresh
native state. Any failed worker, invalid checkpoint, fit miss or reached budget
stops subsequent work. Stops before the scoring phase do not access fresh
held-out targets; failures during scoring retain partial scores and never
resume. All 640 planned method/episode cells remain in `predictions.json`, including
partial raw outputs, timeout/failure and not-attempted statuses. Unscored exact
counts are explicitly unknown, never invented zero errors. Successful scoring
still uses the unchanged complete two-anchor public barrier before private reads.

Successful scoring retains the unchanged v1 gate's raw `scores/summary.json`
for audit and writes a separately versioned `scores/summary-v2.json`, also used
in `run.json.summary`. The v2 reporting adapter maps every rational metric with
a zero denominator to `null` (including numeric exact-match when no numeric
outputs exist), without changing counts, coverage, nonzero-denominator rational
values or historical v1 results. Verification independently recomputes the raw
v1 aggregation and then checks the normalized v2 summary and run record.

### Post-merge commands (not permission to execute)

Use the hash-locked CPU environment and build both pinned Rust crates offline.
Use a clean checkout of the actual reviewed green main SHA, and fresh disjoint
artifact directories outside it. The feasibility command performs three
engineering forward/backward/clip measurements, **zero optimizer steps**, a
worst-context diagnostic/inference envelope and IO timing:

```sh
python -I -B scripts/run_queue_learning_v2.py feasibility --out /absolute/new/feasibility
python -I -B scripts/run_queue_learning_v2.py freeze \
  --feasibility /absolute/new/feasibility/supervised-feasibility.json \
  --feasibility-sha "$FEASIBILITY_SHA" --out /absolute/new/frozen \
  --anchors /absolute/separate/preparation-anchors.json
```

Only `supervised-feasibility.json`, sealed after successful resource supervision
and reaping, is usable; the inner `feasibility.json` is not an admission receipt.
Retain the printed SHA independently. Retain freeze and preparation SHA anchors
outside the checkout and **stop for the owner's approval of that exact freeze**.
Local JSON cannot prove human authorship, review, green remote CI or chronology.
The canonical declaration (sorted keys, compact separators, final newline) is:

```json
{"approved_by":"Trent Slade","authorization":"execute queue-learning-v2 fit seed 17 once; only after final 8/8 at step 128, main seeds 17 and 29 once at step 256","freeze_sha256":"<FREEZE_SHA>","implementation_commit":"<reviewed merged green main SHA>","reviewed_merged_green":true,"schema":"trinite.queue-learning-owner-approval.v2"}
```

Only after that explicit approval:

```sh
python -I -B scripts/run_queue_learning_v2.py run \
  --frozen /absolute/new/frozen --freeze-sha "$FREEZE_SHA" \
  --anchors /absolute/separate/preparation-anchors.json --anchors-sha "$PREPARATION_SHA" \
  --approval /absolute/separate/owner-approval.json \
  --out /absolute/new/run --run-anchors /absolute/separate/run-anchors
python -I -B scripts/run_queue_learning_v2.py verify \
  --frozen /absolute/new/frozen --freeze-sha "$FREEZE_SHA" \
  --anchors /absolute/separate/preparation-anchors.json --anchors-sha "$PREPARATION_SHA" \
  --run /absolute/new/run --run-anchors /absolute/separate/run-anchors --run-sha "$RUN_SHA"
```

The externally retained final `RUN_SHA` binds `final-inventory.json`, including
the outer supervision receipt, every retained partial artifact and inner run
record. `verify` is read-only custody, source/checkpoint structure and gate
verification, **not optimizer replay**. It does not turn a stopped/incomplete
study into successful execution or establish learning adequacy. Retain the
complete source checkout, dependency locks, Rust builds, feasibility, frozen
inputs, owner approval, claims, both anchor sets and all failures. Do not reseal
historical evidence or reuse a foreign-environment checkpoint as training state.

Engineering conformance:

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/queue_learning_v2 -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/queue_learning_v2 -v
```

The mocked 128/256 counter transitions in these tests are not native optimizer
steps, learning measurements or feasibility approval. Real subprocess tests
exercise bounded output, deadlines, RSS rejection, closed pipes, orphan cleanup
and reaping. Existing diagnostic/native/evaluator suites remain required too.
