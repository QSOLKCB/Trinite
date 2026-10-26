# Perturb → Expose → Correct → Verify: position study v1

Status: prospective development protocol, frozen before target outcomes. No
target training or target inference is authorized by this implementation PR.
Tiny injected tests establish engineering conformance only. The separate
preservation work for the balanced-addition negative/mixed result is not part
of this change. Its original bound artifacts remain the evidence authority.

The one corrective hypothesis is place-value position coupling. Both fresh
dense arms receive identical examples, padded operands, reversed fixed-width
answers, schedule, effective initialization, AdamW and budgets. Only position
addressing differs: control `sequence`, candidate `coupled`, using the unchanged
[PR75 adapter and grammar](POSITION-COUPLING.md). Comparison with the historical
[balanced-addition recipe](BALANCED-ADDITION.md) changes format and data too;
it cannot isolate coupling. This is a controlled prospective study, with no
adaptive answer repair, self-tuning loop or guarantee of improved arithmetic.

## Frozen inventory and exposure

`scripts/position_study_v1/protocol.py` is the machine-readable specification.
`data.py` enumerates the complete corpus and independently checks each label
using decimal column arithmetic, integer addition and a strict prompt parser.
The source-specific admission covers these locally rendered mathematical facts
and minimal symbols, with source identities, commissioned authorship, rights
basis, reviewer/date, formal inventory and transformation lineage. No paper
code, data, prose, pretrained weights or teacher outputs are imported.

An unordered operand pair is a semantic family. Both operand orders and every
admitted padded rendering stay together. Equal sums with different operands are
distinct families. The only carrier is equal-width unsigned `digits+digits=`.
For maximum operand below10 admit widths1 and2; otherwise admit only the
maximum operand's decimal width. Other renderings are outside this finite
admission, never alternate heldouts. Response width is always D+1, from the
prompt's operand width D, never the gold sum. Reverse the zero-padded sum.

Training has these 16 one-digit families:

```
(0,2) (0,4) (0,6) (0,8) (1,2) (1,4) (1,6) (2,3)
(2,5) (2,7) (3,4) (3,6) (4,5) (4,7) (5,6) (6,7)
```

The remaining48 training families are the Cartesian product
`a ∈ {10,11,19,22,37,48}`, `b ∈ {50,55,66,79,88,99}`, plus
`(0,b)` and `(b,b)` for each `b ∈ {10,11,19,22,37,48}`.

The24 in-range unseen families are:

```
(0,0) (0,1) (0,9) (1,1) (1,8) (2,2) (2,8) (3,3)
(0,98) (0,99) (1,98) (1,99) (10,89) (11,89) (12,87) (12,88)
(22,77) (22,78) (33,66) (33,67) (44,55) (44,56) (49,50) (49,51)
```

The12 three-digit shift families are:

```
(0,999) (1,999) (2,998) (2,999) (99,900) (99,901)
(111,222) (111,223) (444,555) (444,556) (555,555) (555,556)
```

The12 four-digit shift families are:

```
(0,9999) (1,9999) (2,9998) (2,9999) (999,9000) (999,9001)
(1111,2222) (1111,2223) (4444,5555) (4444,5556) (5555,5555) (5555,5556)
```

| Partition | Semantic families | Unique rows |
| --- | ---: | ---: |
| Train/seen anchors | 64 | 154 |
| In-range unseen combinations | 24 | 56 |
| Three-digit range/length shift | 12 | 23 |
| Four-digit range/length shift | 12 | 23 |

These inventories include reversals, equal-sum alternatives, operand increments,
zeroes, repeated digits, carry/no-carry and long carry chains. Range and length
shift together within the last two strata; do not infer separate range and
length causal effects. All are development; historical0–12 facts are exposed.
If probes guide a later change they remain development evidence. Fresh
confirmation needs a later independently gated experiment.

Schedule: iterate training families in the listed order, each order `(a,b)` then
`(b,a)`, each admitted width in ascending order. Diagonals deliberately occupy
both order slots; diagnostics contain unique rows only. This yields160 visits:
32 width1 and128 width2. Repeat200 cycles, batch8, 4,000 updates: **32,000
visits, 307,200 live shifted model-input tokens, 121,600 scored answer+EOS
targets per cell**. Per-example predictor input/target lengths are7/3 for
width1 and10/4 for width2, giving300,800 predictor input tokens. Native loss
shifts the padded batch: mixed-width batches retain short-row EOS as a live
unscored input, adding6,400 lookups. Padding is excluded. The request separates
both totals; neither is inherited from the historical format. This static batch
accounting correction was made during engineering validation before any target
outcomes; dataset/order/budget/criteria were not selected from outcomes.

## Numerical and positional contract

Six cells: two arms × paired seeds0/1/2. Context64, two blocks, width64,
four heads, FF128, 86,528 parameters, native random initialization. Native
authoritative `TrainingState/run_steps/target_loss/evaluate`, single-tensor
AdamW, LR0.001, betas0.9/0.999, epsilon1e-8, decay0.01, clip1, constant LR,
accumulation1. Geometry0, dense float32, pinned Linux x86_64 CPU dependencies,
one thread and deterministic algorithms. No change to historical operations.
Numerical model size and state-key inventory stay fixed.

Offset policy `fixed-zero.v1` is stateless, input-independent and shared by both
arms. It consumes no RNG and depends on neither answers nor generated values.
It **does not reproduce per-visit randomized offsets in Position Coupling**.
This intentionally narrow test leaves high-significance rows untrained:
control lookup rows0–9; candidate rows0–3. Candidate rows4/5 used at longer
significances are untrained; control later sequence rows are also untrained.
The request records exact row lookup counts from actual shifted inputs.
Lookup exposure alone cannot establish transfer to a new digit significance.
All possible prompt/cap addresses fit context64; maximum full prefix length17.

A stateless live-prefix adapter derives BOS, complete prompt through the first
`=`, and emitted count from actual right-padded IDs/masks on every call. It
ignores emitted values when choosing positions, including malformed bytes,
extra delimiters and misplaced special tokens. No sidecar/global position state.
Prompt grammar and bounds are strict. Sequence-order causal masking always
applies even when address IDs repeat or descend. Padding positions are0.
For shifted training, the adapter receives actual `ids[:-1]`/`mask[:-1]` and
derives the corresponding actual positions, never label positions `[1:]`.

Generation uses unchanged native greedy operations with an explicit per-prompt
cap **D+2**, derived before generation. Stop only on actual EOS; never force
EOS, repair digits or report unfinished output as success. This bounds malformed
short-width output without passing a global long-width cap into the mapper.
New scoring semantics explicitly bind this cap and representation. Preserve
raw token IDs; compute references only in the evaluator. No oracle answer,
carry state or solver result enters generation or positional addressing.

The scorer uses the authoritative native scorer one selected item at a time
with a complete training-only majority pool. Training items use an internal
test-selection view to select one item, then retain their original train split;
this neither introduces probes nor changes the loss/update split. Training
majorities are width-local; probe majority is over all training widths. Raw
predictions are journalled and fsynced after each completed item. Each declared
score has an exclusive bounded JSONL journal, retained on failure and hash-bound
on completion. A killed process can leave an unfinished last write; such an
attempt cannot certify completion. The new classifier admits leading zeroes in
reversed answers and the D+2 cap; historical decimal/cap5 classification is
incompatible. Independent malformed, early-EOS and missing-EOS flags supplement
the disjoint primary failure category.

Same-shape ordinary offset0 parity: exact logits, captures, canonical masked
loss and all parameter gradients. Cross-shape comparisons: absolute2e-6,
relative1e-5. Same-source/environment resume: exact model/AdamW/history.
Request/checkpoint/scoring identity binds mode, schema, offset and cap policy;
matching parameter hashes alone never certifies matching behaviour.

## Measurement, criteria and limits

Only fixed checkpoints0/1000/2000/3000/4000. Training-only unique diagnostics at
each; probes only at final4000 after all updates. No probe-based checkpoint,
budget, seed, threshold selection, adaptive rerun or early stopping.

Retain exact answer+EOS, decoded numeric correctness, wrong value, malformed,
missing/early EOS, every raw prediction, whole-family all-rendering correctness,
carry-count/chain and length strata. Pair reporting distinguishes consistency
from both-correct: reversal and equal-sum outputs should agree, incremented
outputs should differ with the correct numeric delta. Two identical wrong
answers cannot pass. Include labelled non-neural constant/majority, lookup and
exact-oracle controls; none repair neural predictions.

Descriptive support requires all six cells structurally verified, candidate
seen-family accuracy≥95%, seen regression≤2 percentage points versus paired
control, and **each seed** gaining≥10 points in unseen-family accuracy.
Each length stratum also requires each candidate seed≥25% and gain≥10 points.
Use integer family counts/denominators; report every condition separately.
Failure of length criteria with unseen improvement is mixed support, not a
length-transfer success. Failure to meet criteria leaves the hypothesis
unsupported in this scope, without proving universal incapacity or a different
cause. No significance claim, automatic learning/release gate or reasoning claim.

Bounds: worker1,800 seconds/cell, cumulative update960 seconds sampled at
milestones, native480-second per-invocation ceiling; six workers≤10,800 seconds
plus setup/audits. Per-artifact16MiB, retained computational payload32MiB/cell,
worker log1MiB. Custody copies are extra; no hard RSS or whole-directory cap.
Record actual timing/high-water RSS. Retain every failed, partial and timed-out
attempt without overwriting; original identities must be independently saved.

## Later owner-local workflow and gates

Implementation uses a separately versioned preflight/freeze/run/audit/summarize
lane. Review the exact published commit and green CI before separately
authorizing target execution. Keep source quiescent, use the hash-locked CPU
environment, fresh paths outside the checkout, and independently capture request
and matrix identities before/after execution. Never invoke internal workers.

```sh
PYTHONPATH=src:scripts python -B scripts/run_position_study.py preflight
PYTHONPATH=src:scripts python -B scripts/run_position_study.py freeze --request /ABS/NEW/request.json
# Independently retain the printed sha256 request identity before outcomes.
PYTHONPATH=src:scripts python -B scripts/run_position_study.py run \
  --request /ABS/NEW/request.json --request-identity sha256:REQUEST_ID \
  --output /ABS/NEW/attempt --authorize-run
# Independently retain matrix_identity, including on incomplete matrices.
PYTHONPATH=src:scripts python -B scripts/run_position_study.py audit \
  --request /ABS/NEW/request.json --request-identity sha256:REQUEST_ID \
  --output /ABS/NEW/attempt --expected-identity sha256:MATRIX_ID
PYTHONPATH=src:scripts python -B scripts/run_position_study.py summarize \
  --request /ABS/NEW/request.json --request-identity sha256:REQUEST_ID \
  --output /ABS/NEW/attempt --expected-identity sha256:MATRIX_ID
```

Read-only audit verifies actual admission, source/environment/request binding,
safe tensors, complete evidence inventory, original upstream custody, retained
history consistency and recalculated metrics. It does not rerun inference or
optimizer updates. Deterministic resume is a tested engineering API, not an
automatic target retry workflow. Full numerical replay and independent-host
replication remain separate later operations, never inferred from structural
audit. Retain full source/request/checkpoints/raw outputs/logs/failures and the
original independent identities; do not redact or reseal original packets.

Affected TRI-I01–I03/I05–I08/I10–I14. Historical closures, models, fixtures,
protocols, importers and negative evidence remain unchanged. No LUT sharing,
ternary, larger model, geometry, scratchpad, teacher, external data, KV or GPU.

Motivation: [Position Coupling2405.20671v2](https://arxiv.org/abs/2405.20671v2)
supplies the significance-coupling idea; its larger scale and randomized-offset
training differ materially. [Teaching Arithmetic2307.03381v1](https://arxiv.org/abs/2307.03381v1)
motivates controlling format/sampling; no scratchpad supervision is used here.
This small finite Trinite adaptation is not a paper reproduction.
