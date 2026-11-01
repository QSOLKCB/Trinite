# Context-symbol learning v1

Status: complete execution-path implementation for owner review. **Unrun.** No
scientific optimizer steps belong to implementation or CI. After review, owner
merge and green main CI, prepare fresh feasibility and freeze on that exact
producer SHA, then stop for explicit approval of its full SHA, freeze hash and
schedule. This document and earlier queue approvals do not grant that approval.

Affected requirements: TRI-I02/I03/I05/I07/I08/I10–I14. The native numerical
implementation, context 256, geometry loss zero, queue-v1 negative result,
queue-v2 unrun status and historical fixtures/producers remain unchanged.
This checkout-only lane does not enable racing, TOMBSTONE, ternary evidence
training, a larger model, later research phases or deployment.

## Task and admission

The reference is the unchanged [context-symbol module](CONTEXT-SYMBOL-RELATION-V1.md).
Compatible worlds are the Cartesian product of nonempty context/symbol masks.
Return `1` if all worlds support the public query, `-1` if all contradict, `0`
if mixed. Missing context alone implies none of these labels. Inconsistent masks
are invalid. Empty/malformed output, missing EOS and timeouts are prediction
failures, never logical zero or an uncertainty answer.

Source-specific admission independently regenerates original commissioned
finite tables/masks, native `observe` for all inputs and native
`evaluate-privileged` plus the independent Python finite-world oracle for
**training only**. It records authorship, procedure, source supporting reference,
numeric-fact rights basis, review date and exact inventory scope. Rights attach
to these formal facts/minimal notation; software licensing does not admit a
corpus. Only original/public authorized material is published. No credentials,
private chat metadata, book text, OCR, images or teacher outputs enter the data,
source archive or evidence. Jung remains bibliographic inspiration only.

Use the reviewed procedural-admission `C1` lossless carrier and native tokenizer.
Only dimensions, full total rules, possibility masks and query enter a prompt,
followed by `\n=`. Labels, family IDs, seeds, file paths and provenance stay
outside. Answer bytes and EOS alone receive loss, shifted by the authoritative
native `target_loss`. BOS, delimiter and prompt bytes never receive loss. Native
`ByteTokenizer(256)` measures the entire sequence, including special tokens;
overflow fails rather than truncates. Reserve three generation tokens for all
inputs (two answer bytes plus EOS). Generation is unrestricted greedy argmax,
three emitted tokens maximum, stop on EOS. Accept exactly `-1`+EOS, `0`+EOS or
`1`+EOS. No constrained decoding, stripping, repair, fallback or retry.

## Fresh family protocol

Version `context-symbol-scientific-family-v1` is distinct from #119's 252 public engineering
fixtures. Generate candidates from SHA-256 ASCII
`context-symbol-scientific-family-v1/{index}` for indices 0..4095. The first twelve digest
bytes modulo three form a 4×3 table with three meanings. A declared mutation
increments cell (0,0) modulo three. Both parents must use all three meanings;
their full-table orbits must differ and must not overlap earlier families.
Take the first nine accepted families, without label/model/difficulty selection.
The fixed generator retains each accepted index; rejected candidates are fully
reconstructible from the bounded stream. No redrawing or salt adjustment after
this version is published/frozen.

An early implementation-only aggregation test derived labels on the development
inventory using namespace `context-symbol-family-v1`, inventory SHA-256
`a73e56c3f3fb7a2dd89cd14b0fa0bb909e1c885808208247756951a8b82ec43e`.
Treat all nine development families as exposed. The scientific stream excludes
**every orbit of both parents of those families** before accepting candidates.
The exclusion depends only on public tables, not label values/model outcomes.
Tests now use synthetic heldout outcomes and call the real native privileged
oracle only on training rows or separate pre-exposed conformance examples.
No scientific optimizer steps occurred on either inventory. Retain this
development exposure record; do not present it as an unseen scientific run.

Rank base tables by SHA-256 canonical JSON bytes: first five families train,
next two validation, final two test. Every parent/mutation, all three queries,
context masks {row 0, row 1, all rows}, symbol masks {column 0, all columns},
and identity/reversed-row-and-column/cyclic-meaning-renamed variants inherit
that family and split. There are **648 rows: 360 train / 144 validation / 144
test**, nine families, eighteen semantic table orbits. Exact public duplicates
are forbidden. The declared semantic overlap key ignores masks and query and
canonicalizes independent context, symbol and meaning bijections by exhaustive
column/meaning permutations and row sorting. Parent lineage groups the declared
mutation with its base even though their table orbits differ. The declaration
covers these included mutations, not arbitrary edit-distance equivalence or all
mathematically related finite tables.

Inventory SHA-256 (canonical JSON with final newline):
`21b0311fc629bd95f52a01c0e57c756383599ea4b286a4d01b8c58d799c130d5`.
Maximum reserved prompt-plus-answer sequence: **98 native tokens**. Historical
#119 families have square dimensions n/n/n, n=2/3/4 for every seed; fresh 4/3/3
tables cannot overlap exactly or under the declared dimension-preserving
bijections. Dimension distinguishes historical exposure; all new splits share
4/3/3. This tests transfer across the frozen table families at fixed size and
algorithm. It cannot establish unseen algorithms, arbitrary mutations, larger
sizes, other domains, secret-benchmark performance, psychological truth or
operator blindness. Public construction and software tests expose the method.

Validation is reporting-only after both final checkpoints and every prediction
are committed. No validation/test target enters admission exports, updates,
training diagnostics, hyperparameters or checkpoint selection. Test outcomes
cannot select a retry or a follow-up within this protocol. Any later study
requires its own preregistration and exposure account.

## Exact bounded schedule

| Setting | Frozen value |
| --- | --- |
| Model | Existing native dense decoder, byte vocabulary 259, context 256, block 1, width 32, heads 4, FF64, tied embeddings/output; **24,768 actual unique parameters**, checked against constructed tensors |
| Runtime | Linux x86_64 CPU float32 eager, hash-locked `requirements.lock`, one Torch compute thread, strict deterministic algorithms; no autocast or alternate backend |
| AdamW | LR .001; betas .9/.999; epsilon 1e-8; decay .01; clip norm 1; foreach/fused/AMSGrad/capturable/maximize/differentiable off; unchanged native updater |
| Loss | Native mean cross entropy over shifted answer bytes and EOS; geometry loss zero |
| Batch/order | Eight; train IDs ranked by SHA-256 ASCII ID; cycle without shuffle; same order across seeds |
| Fit | Fresh seed 17; first three negative, two mixed and three positive rows in fixed training order; eight unique consistent prefixes; exactly 128 updates |
| Fit gate | Every canonical answer **and EOS correct at final step 128**, all 8/8; independently reload the actual final checkpoint and regenerate; early success does not count |
| Main | Only after fit passes: independently initialized seeds 17 and 29, exactly 256 updates each, once each; fit weights/optimizer never initialize main |
| Updates | Fit 128 + main 256 + main 256 = **640 maximum**, reservations never replenished after failures |
| Supervised tokens | Fit **2,432**; each main **4,900**; exact enforced ceiling **12,232 overall**, including EOS; visits 1,024 + 2,048 + 2,048 |
| Selection | Final step only; train-only mean loss at initialization/final and fit generations at initialization/final; native `validation` field contains training rows only |
| Time | 120 s cumulative native updates per attempt, 900 s full attempt including import/checkpoint reload/inference, 2,700 s entire study including admission/baselines/commitment/scoring |
| Memory | 1 GiB process RSS, 2 GiB aggregate observed process-tree RSS, 4 GiB per-process address space; core dumps disabled |

No retries, resume, extra seeds, search, GPU, paid compute or automatic budget
expansion. First failure stops dependent stages and retains evidence. A failed
attempt consumes the exact freeze through an exclusive execution claim beside
the independently retained preparation anchor. New output paths do not authorize
another attempt. Source/settings cannot change during execution. Any mismatch,
infrastructure failure or proposed rerun needs a new owner decision and freeze.

The fixed launcher sets address-space limits before native imports. Reused
reviewed Linux supervision monitors transitive descendants, new sessions and
adopted orphans, kills/reaps them, bounds output, records wait4 CPU/high-water
usage, sampled process/tree RSS and wall timing. It also watches an atomic
update marker and monotone update/token counts. Linux `/proc`, subreaper and
rlimit support are required; inspection failure stops execution. RSS is sampled
at approximately 10 ms and cannot rule out transient aggregate peaks. Address
space is a hard inherited per-process limit. These are not host-wide limits,
hostile-code isolation or hardware attestation. Native workers are cooperative;
prompt separation and process arguments do not prevent a malicious same-user
operator from opening private files. The model itself receives no tool access.

Fresh feasibility uses three worst-context forward/backward/clip measurements,
**zero optimizer steps**, unchanged weight identity, worst-context train/inference
and IO envelope, plus native public/train admission timings. Require a fourfold
projected margin within the unchanged 120/900-second limits and supervised
memory bounds. This is engineering projection, not measured AdamW throughput
or a guarantee of scientific completion. Incompatibility stops freeze; propose
the smallest change for owner review rather than silently substituting settings.

## Baselines, commitment and metrics

Fit the episode-weighted majority class on **all 360 training rows only**; ties
choose the smallest numeric class. The fixed context-ignoring shortcut always
uses rule row zero, ignoring context possibilities, while respecting symbol mask
and query; mixed truths within that row give zero. It is a deliberately limited
shortcut, not a trained second neural arm. Freeze both before heldout inference.
The exact native oracle is a correctness reference, not a learned competitor.
Its score is replayed from independently checked targets after the commitment
barrier; it is not an extra pre-barrier model prediction.

Run main-17's final inference, then main-29's final inference without exposing
any heldout scores. Each worker receives train rows and public observations;
its prompt receives only a whitelisted observation. Commit **every raw output
for both seeds and both baselines, validation and test**, before any native
heldout privileged evaluation. The external prediction anchor binds raw tokens,
statuses and all final checkpoint identities. No partial barrier may score.
Partial outputs remain in their journals if a worker fails; later planned cells
stay visible as unattempted. Oracle failures never become semantic labels.

Report full-denominator exact accuracy (144 per method/split), each class's
recall with support, exact rational macro-F1, valid-output coverage, malformed,
failure, timeout and unattempted counts, family accuracy and context-switch
consistency. The latter is **both answers correct** across all paired context
mask interventions sharing every other public fact, plus the changed-label
subset. Missing/malformed responses fail both-answer pairs. Unscored metrics
are null, never invented zeros. Zero-support recall is null; a zero-denominator
class F1 contributes zero under the frozen convention. Macro-F1 averages all
three classes; exact Fraction arithmetic preserves counts.

A descriptive gain requires full valid-output coverage and strictly greater
accuracy than both majority and shortcut for **both seeds** within the reported
split. Report validation and test separately. Do not claim improvement beyond
what that comparison supports. Two seeds and correlated variants establish no
statistical significance. Learned fitting uses far more compute than constant
or finite-rule baselines; the oracle has exact task rules and no learned budget.
Resource receipts measure learner workers and controller/oracle work; per-case
native transport receipts measure oracle wall time/outputs. Equal compute,
context offload, compression and capacity gains are not tested.

## Operator commands and approval

Use an already acquired exact hash-locked CPU environment and Rust 1.90.0. No
implicit network, model downloads or dependency substitution occurs. Commands
below are instructions for later stages, not execution authorization. The owner
must first review/merge the implementation; verify the actual main SHA and green
CI from GitHub, then use a clean checkout of that commit. Retain source checkout,
runtime/locks, build binary and independently recorded artifact hashes.

```sh
python -I -B scripts/run_context_symbol_learning_v1.py build
python -I -B scripts/run_context_symbol_learning_v1.py feasibility --out /absolute/new/feasibility
python -I -B scripts/run_context_symbol_learning_v1.py freeze \
  --feasibility /absolute/new/feasibility/supervised-feasibility.json \
  --feasibility-sha "$FEASIBILITY_SHA" --out /absolute/new/frozen \
  --anchors /absolute/separate/preparation.json
```

Only the outer supervised feasibility receipt is accepted. Freeze binds exact
producer SHA, complete runtime source map/archive, configuration, generated
inventory, split/admission/training rows, dependency lock, interpreter/native
library/oracle executable hashes and runtime/hardware/thread identities. System
libraries/kernel/compiler custody are operator trust boundaries. Retain the
printed freeze/preparation hashes independently. **Stop and ask the owner to
approve that full producer SHA, freeze hash and bounded schedule.** Human
identity, actual owner review, remote main CI and chronology cannot be proved by
a local JSON file. Do not manufacture the declaration without explicit approval.
After approval, its canonical declaration is:

```json
{"approved_by":"Trent Slade","authorization":"execute context-symbol-learning-v1 fit seed 17 once at step 128; only after final 8/8, main seeds 17 and 29 once at step 256","freeze_sha256":"<FULL_FREEZE_SHA256>","optimizer_updates_ceiling":640,"producer_sha":"<FULL_REVIEWED_MERGED_GREEN_SHA>","reviewed_merged_green":true,"schema":"trinite.context-symbol-owner-approval.v1","supervised_token_ceiling":12232}
```

Use a fresh destination, the exact clean producer checkout and unchanged runtime:

```sh
python -I -B scripts/run_context_symbol_learning_v1.py run \
  --frozen /absolute/new/frozen --freeze-sha "$FREEZE_SHA" \
  --anchors /absolute/separate/preparation.json --anchors-sha "$PREPARATION_SHA" \
  --approval /absolute/separate/owner-approval.json --out /absolute/new/run \
  --run-anchor /absolute/separate/run-anchor.json
python -I -B scripts/run_context_symbol_learning_v1.py verify \
  --frozen /absolute/new/frozen --freeze-sha "$FREEZE_SHA" \
  --anchors /absolute/separate/preparation.json --anchors-sha "$PREPARATION_SHA" \
  --run /absolute/new/run --run-anchor /absolute/separate/run-anchor.json --run-sha "$RUN_SHA"
```

Verification needs the producer sources/runtime and native oracle, but performs
no training. It independently regenerates data, safe-restores completed final
checkpoints, checks fit answers and prediction anchors, recomputes native targets
and all reported scores, and leaves run bytes unchanged. Outer interruptions
are retained as incomplete custody, never successful results. Custody and score
replay are **not independent optimizer replay or proof of operator blindness**.

For independent-host, standard-library-only score review, append `--scores-only`
to the same verify command. This still checks external anchors, source archive,
exact data/splits/admission, training tokens, predictions/baselines and recomputes
labels with the independent finite-world oracle and all scores. It does not
load tensors, regenerate fit outputs, check current native binaries or assert a
matching runtime. Its receipt explicitly reports that reduced scope. Use default
verification for same-environment native/checkpoint checks; neither mode trains.

## Evidence deposit and implementation checks

Use a separate evidence branch/PR after the approved execution. Keep the
**trained producer SHA distinct from the evidence commit SHA**. Deposit under
`docs/evidence/context-symbol-learning-v1/`: frozen protocol/data/admission,
source closure ZIP and its hash, feasibility, runtime/lock/executable identities,
owner declaration, preparation/run/prediction anchors and execution claim,
every fit/main checkpoint and metadata, raw predictions/journals, fit results,
baseline constants, score/target files, every partial/failed attempt, bounded
logs and timing/resource receipts. Include a result note with exact commands,
hashes, complete planned/completed denominators, stopped/unrun cells and replay
instructions. Inventory preserves every file; do not select only successes.
Preserve the frozen producer verifier as the read-only entrypoint.

An interrupted/negative run is a valid report. Open the evidence PR for
independent review; verify its exact published commit and CI. Do not merge or
self-approve either PR. Local receipt claims remain operator declarations.

Implementation checks, with optimizer-facing transitions doubled:

```sh
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/context_symbol_learning_v1 -v
PYTHONPATH=src:scripts python -O -B -m unittest discover -s tests/context_symbol_learning_v1 -v
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/queue_learning_v2 -v
PYTHONPATH=src:scripts python -B -m unittest discover -s tests/queue_learning_native -v
(cd external/context-symbol-relation-v1 && cargo test --offline --locked && cargo test --release --offline --locked)
python external/context-symbol-relation-v1/tests/oracle.py external/context-symbol-relation-v1/target/release/trinite-context-symbol-relation-v1
```

CI builds only the context oracle and runs engineering tests; it never runs
this scientific experiment. Native tests use random engineering weights,
forward/gradient checks and safe zero-step tensors; optimizer steps are forbidden.
