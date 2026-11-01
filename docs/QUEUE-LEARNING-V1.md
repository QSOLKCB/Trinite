# Bounded supervised queue-learning pilot v1

Status: prospective implementation for review, 1 November 2026. **No pilot
training is authorized or reported.** Review, merge with green CI, a fresh
source/environment-bound freeze, and Trent Slade's explicit approval of that
frozen run must precede training. The implementation stops at engineering
checks in CI. Preserve every failed attempt and negative outcome. Do not merge
this PR as part of the implementation task.

Affected requirements: [TRI-I01–I03/I05/I07/I08/I11–I14](INVARIANTS.md).
This checkout-only queue lane adds no installed API and changes no historical
producer, admission, numerical updater, fixture or negative result. It reuses
the [authoritative queue](QUEUE-SIM-V1.md), [two-anchor gate](QUEUE-EVALUATION-V1.md),
native `training.run_steps`, `comparison.greedy`, and
`comparison_checkpoint.restore_tensors`. Music admission is never imported.
CPU `requirements.lock` is unchanged. Geometry weight remains zero.

## Frozen prospective distribution and task

The sole target is future `forecast.dropped`, the number of rejected arrivals
in the eight unobserved ticks. Eight completed ticks are public. The producer
contains exactly **384 distinct requests**, with no refill, curation by model
success or extra target episodes. There are 48 structural groups: four workload
kinds times twelve parameter settings. Each group includes two initial queue
states, two hidden-future alternatives, and two measurement masks (eight requests).
The initial queue is either zero or half the buffer capacity, rounded down.
Capacity is 4+j, with j=0..11. At tick t=0..15, before the hidden alternative:

| Kind | Arrivals | Nominal service | Outage |
| --- | --- | --- | --- |
| steady | 2 + j mod 6 | 1 + floor(j/4) | false |
| periodic | 1 + j mod 4 + 2*((t+j) mod 3) | 2 + j mod 3 | false |
| ramp | 1 + floor(j/3) + floor(t/2) | 2 + j mod 4 | false |
| outage | 2 + floor(j/4) | 4 + j mod 3 | t mod 4 = 3 |

The two alternatives add either zero or three offered arrivals to **each
future tick only**. They are balanced and undeclared to inference. Recurrence,
load, capacity, slope and visible outages therefore carry predictive information;
the suffix is not an independent arbitrary schedule. Nevertheless the hidden
alternative cannot be determined from a prefix. Identical observations can have
different targets, so perfect per-episode prediction is generally impossible.
For an equally weighted pair with labels a and b, the minimum average absolute
error is |a-b|/2; when a!=b, maximum exact-match rate for that pair is 1/2.
Masking and initial-state convergence can introduce further aliases. No learned
uncertainty or real-world queue calibration is claimed.

Whole groups are ranked within each kind by SHA-256 of
`queue-learning-v1/split/{kind}/{j}`. The first eight groups train, the next two
validate, and the final two test. This yields **256/64/64 episodes** and
32/8/8 groups. The split table is generated without labels, written before
preparation, and bound by the freeze. Every mask, initial-state and hidden-suffix
variant stays with its group. An exact public-prefix alias across groups is a
hard rejection, even within one split. No item-level shuffle or test tuning.
Public conformance fixtures and test cases are exposed engineering evidence;
they are not held-out model results. The producer grammar is public, so even a
later test split is a finite synthetic family holdout, not a secret benchmark
or fresh confirmation after inspecting/tuning against it.

## Public carrier, supervision and admission

`codec.prompt` accepts only the gate's exact public observation schema and the
fixed declared task. Full requests, future schedules, labels, family/episode IDs,
seeds, hashes, filenames and provenance records are not prompt fields. Unknown
fields fail. `numerical.infer(model, observation_bytes)` calls the native greedy
decoder with that public carrier alone. Checkpoint restoration is privileged
preparation; its train-only state never becomes held-out inference context.
The infer worker has no private request or held-out-label argument. The public
inventory routes files but is not passed to `infer` or tokenized. This is a
program interface boundary, **not OS-enforced isolation** from an operator who
owns all files and code.

The lossless ASCII grammar is `Q1;capacity;observe_ticks;horizon;rows`, followed
by `?forecast.dropped=`. Each row is fourteen characters: one outage bit, one
base64url mask character (bit i is measurement i), and six two-character
base64url unsigned integers (high digit first). The alphabet is
`0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_`. Tick numbers
are the row indices. The independent decoder reconstructs every public field,
then revalidates the queue schema. Missing values keep their required zero
placeholders **and false masks**. Observed zero keeps its true mask. No values
are imputed or merged. This is a lossless input encoding, not weight quantization.

The unchanged native byte tokenizer inserts BOS/EOS. Only decimal answer bytes
and EOS have a loss mask. Prompt, BOS and padding never contribute. Generic
overlength observations fail: neither prompt truncation nor a larger model is
permitted. Check both `BOS + prompt + answer + EOS <= 256` and
`BOS + prompt + 5 generation tokens <= 256`. The fixed generation cap is five
(four digits for the general queue maximum 8160, then EOS), independent of gold
labels. The actual eight-tick pilot answer is at most 256 (three digits).

Admission is a **new queue-specific policy** for these exact generated formal
counts and minimal notation. It records commissioned authorship, exact source
inventory, formal request/split identities, rights basis, reviewer/date, scope,
exclusions and limits. It independently regenerates the policy record and
requests; a self-reported `admitted` flag is insufficient. “Synthetic” and the
software's MPL-2.0 license are not blanket rights exemptions. No external corpus,
music, teacher, pretrained model or expressive prose is admitted. Labels are
obtained only through the authoritative Rust target operation, after split
freeze. Only train rows enter numerical state; validation/test labels are never
materialized by the training adapter.

## Budget, feasibility and checkpoint election

| Setting | Fixed proposal per seed |
| --- | --- |
| Backend | Canonical Linux x86_64 CPU float32 dense, hash-locked CPU dependencies, one Torch thread, strict determinism |
| Seeds | 17 and 29, exactly one attempt each; no automatic retry |
| Model | One block, width 32, four heads, FF64, context 256, tied byte vocabulary 259 |
| Training | Native single-tensor AdamW, LR 0.001, batch 8, 64 updates |
| Other optimizer settings | Native defaults: betas .9/.999, epsilon 1e-8, decay .01, clip 1, no schedule/accumulation |
| Order | Train episode IDs ranked by SHA-256, cycle twice; same order for both seeds |
| Targets | Hard maximum 32,768 supervised answer/EOS tokens; static worst case 2,048 |
| Time | 120 seconds for updates; 900 seconds elapsed whole-worker wall per seed |
| Selection | Final step 64 only, frozen before any held-out inference |

`feasibility` first validates all 384 **observations only**, masks, alias grouping,
context and schedule upper bounds. It then measures three worst-context native
forward/backward/gradient-clip calls on a public engineering batch (arbitrary
bytes, not pilot outcomes). **No optimizer step occurs**; model identities must
remain equal. A fourfold margin times 64 must fit 120 seconds before `freeze`
accepts this source/environment-bound receipt. This is a feasibility estimate,
not measured AdamW/update compliance, a guarantee against host contention, or
training evidence. Actual runs enforce deadlines and retain timeouts. A changed
source, binary, environment or negative feasibility result needs a new retained
receipt; never edit an old fingerprint to pass.
Engineering feasibility may record an absent `origin/main` in a detached shallow
PR checkout. Freeze and training reject that state; neither substitutes `HEAD`
for a missing reviewed-main identity.

The parent imposes a stricter 120-second **entire train child** deadline,
including imports, admission, initialization, updates, diagnostics and IO.
Inference receives the remainder of the seed's 900-second elapsed worker
allowance, including time waiting for the other seed and checkpoint barrier.
Both children have bounded stdout/stderr and process-group kill/wait
through the existing queue gate. Native update time is also measured separately.
The updater's field named `validation` deliberately contains a **train-only
loss diagnostic** at step 64, never held-out data or checkpoint selection.
Unique train counts, scheduled repeated visits and actual targets are distinct.

Checkpoint payloads use the existing closed named model/AdamW safetensors reader,
32 MiB tensor and 1 MiB metadata bounds, exact externally anchored byte identities,
progress/shape/dtype/finite checks and source/environment/plan binding. No pickle,
unsafe fallback, resume/retry or pretrained initialization. Both final checkpoints
are independently anchored before held-out inference starts. Invalid or partial
checkpoints cannot enter inference. Best-effort partial snapshots and failures
are retained; a hard kill may leave incomplete bytes that the reader must reject.
Verification checks safe structure, progress and model identity; it **does not
replay optimizer updates or certify learning adequacy**.

## Baselines, barrier and metrics

The train-only constants are the lower median and the most frequent target
(ties choose the smallest integer), weighted by train episodes, including mask
variants. No validation or test target enters their fit. The fixed causal
baseline repeats the last public arrival and effective service for eight ticks,
starting from the last public end queue. It delegates transitions to the same
Rust queue; it does not receive the actual future request. If any of these three
last measurements is missing, it abstains. Both pilot masks expose these inputs;
all other missing measurements stay missing. No success-based tuning occurs.

For both seeds and all three baselines, retain every raw output and commit
**all validation and test predictions before scoring even one held-out target**.
The separate barrier file binds the complete planned inventory, raw outputs and
each public commitment anchor. Validate every public commitment before opening
any private preparation. Then actually call `score` and `verify` for every case
using independently retained preparation and commitment anchors. Retain per-case
and aggregate verification receipts. Do not recover expected anchors from
mutable manifests. A freeze or checkpoint anchor is not a prediction anchor.
Preparation and commitment anchors are also appended immediately to separate
external journals, so an interrupted construction retains its partial anchors.

Canonical nonnegative decimal bytes followed by EOS are numeric predictions.
`?` followed by EOS is explicit abstention. Invalid bytes, missing EOS, excessive
tokens, leading zeroes, special tokens, out-of-range numbers, failures and
timeouts are retained separately. Invalid/non-numeric outputs become an explicit
abstention **at the gate interface only**, with their original raw failure status
bound in the barrier; they never become numeric predictions. No abstention is a
successful exact match. Planned but unattempted rows remain in denominators.

Primary metric is numeric **MAE with full planned numeric coverage**. Otherwise
all-case MAE is undefined; separately show conditional numeric MAE and its smaller
denominator. Also report exact matches/all planned, numeric exact matches/numeric,
valid-output rate (numeric or explicit abstention)/all planned, numeric coverage,
and counts of every status. Preserve rational numerator/denominator pairs, even
zero denominators; undefined conditional MAE is null. Compare per seed and split
against each baseline on equal denominators. A descriptive improvement requires
full coverage and lower MAE than every baseline in each seed. Do not select a
seed/checkpoint after test scoring or claim significance from two seeds. Report
negative results, incomplete coverage and failures plainly.

Hashes establish byte binding under retained anchors, not that an operator never
saw answers, chronology, remote CI, human review or source-to-binary attestation.
The approval JSON below is an operator declaration, not cryptographic proof of
Trent's authorship. Enforce access and approval outside this same-user process.

## Reproduction and accounting

Use Python 3.11–3.13 with the unchanged CPU lock (see
[acquisition](GETTING_STARTED.md#cpu-reference-model)). Install it in a dedicated
virtual environment with `python -m pip install --require-hashes --no-deps -r
requirements.lock`, then `python -m pip check`. Build pinned Rust 1.90.0:

```sh
(cd external/queue-evaluator-v1 && cargo build --offline --locked --release)
(cd external/queue-sim-v1 && cargo build --offline --locked --release)
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/queue_learning_v1 -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/queue_learning_v1 -v
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/queue_learning_native -v
python -I -B scripts/run_queue_learning_v1.py feasibility --out /absolute/new/feasibility
python -I -B scripts/run_queue_learning_v1.py engineering \
  --out /absolute/new/engineering --anchors /absolute/separate/engineering-anchors
```

These commands run engineering checks only. The actual gate integration tests
execute prepare/commit/score/verify on exposed engineering requests, retain both
anchors separately and exercise invalid/abstained/wrong outputs. They do not
score the proposed pilot's hidden outcomes. The native tests must not perform
an optimizer update. CI never invokes `run`.

After implementation is reviewed, merged and green, use a clean checkout of that
exact main commit. Re-run feasibility there, independently retain its printed
digest as `FEASIBILITY_SHA`, and choose fresh, disjoint destinations outside the
checkout whose parents already exist:

```sh
python -I -B scripts/run_queue_learning_v1.py freeze \
  --feasibility /absolute/new/feasibility/feasibility.json --feasibility-sha "$FEASIBILITY_SHA" \
  --out /absolute/new/frozen --anchors /absolute/separate/preparation-anchors.json
```

Independently retain both printed digests (`FREEZE_SHA`, `PREPARATION_SHA`).
Review the frozen protocol, sources, environment, split/episode inventory and
feasibility receipt. **STOP here for the owner's explicit frozen-run approval.**
Only after that approval, write its declaration to a fresh external JSON file:

```json
{"approved_by":"Trent Slade","authorization":"execute queue-learning-v1 seeds 17 and 29 once","freeze_sha256":"<FREEZE_SHA>","implementation_commit":"<reviewed merged green main SHA>","reviewed_merged_green":true,"schema":"trinite.queue-learning-owner-approval.v1"}
```

Use canonical JSON (sorted keys, compact separators and final newline), then:

```sh
python -I -B scripts/run_queue_learning_v1.py run \
  --frozen /absolute/new/frozen --freeze-sha "$FREEZE_SHA" \
  --anchors /absolute/separate/preparation-anchors.json --anchors-sha "$PREPARATION_SHA" \
  --approval /absolute/separate/owner-approval.json \
  --out /absolute/new/run --run-anchors /absolute/separate/run-anchors
```

Retain printed `CHECKPOINT_SHA`, `BARRIER_SHA` and `RUN_SHA` independently. The
last binds the complete retained run inventory, including resource/failure files.
Read-only gate
and safe-checkpoint verification uses these expected anchors explicitly:

```sh
python -I -B scripts/run_queue_learning_v1.py verify \
  --frozen /absolute/new/frozen --freeze-sha "$FREEZE_SHA" \
  --anchors /absolute/separate/preparation-anchors.json --anchors-sha "$PREPARATION_SHA" \
  --run /absolute/new/run --run-anchors /absolute/separate/run-anchors \
  --checkpoint-sha "$CHECKPOINT_SHA" --barrier-sha "$BARRIER_SHA" --run-sha "$RUN_SHA"
```

Retain the entire freeze, all attempt directories, checkpoints, raw journals,
bounded process outputs, prediction commitments, result/verification receipts,
owner approval and both independently stored anchor sets. Never overwrite,
reseal or discard partial failures. Archive these together with the exact source
checkout, Cargo locks/toolchain and executable, CPU lock and feasibility receipt.
The source closure deliberately overbinds every Python/JSON/lock/text source in
`src/` and the new package plus explicit gate/Rust inputs. Git SHA, source hashes,
binary SHA and measured environment are distinct identities. Source hashes are
not build attestation. Record interpreter, package versions, CPU/features,
thread/dtype/determinism, wall/user/system seconds and process peak RSS; child
deadlines and parent process evidence survive failures. CPU/parent RSS is not
whole-machine memory. Equal tokens are not equal compute. No GPU substitution,
paid resources, hyperparameter search, automatic retries, RL, cross-domain
training, transfer claim or release gate is included.
