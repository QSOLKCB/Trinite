# Finite maths and exact linear algebra

Authorised by Trent Slade on 2026-10-09 after PR #18: prepare an admitted maths
corpus and train a comparison. This separate exploratory pilot permits its own
held-out diagnostics; it changes none of the historical failed gates, selects
no format and does not unlock Phase 5. A corpus can teach narrow operations;
correctness and freedom from drift require measurements, not a corpus guarantee.

## Admission and coverage

`maths_data.py` generates 3,154 examples (1,577 formal items, two carriers).
The corrected v2 orbit split contains 1,998 train, 502 validation and 654 test examples.
Validation remains unused. Every task occurs in every split. These are bounded
numeric/formal facts with newly rendered minimal symbols, under the existing
strict [DATA](DATA.md) policy. No textbooks, websites, explanations, pretrained
weights or teacher outputs enter training. The generator software is MPL-2.0;
the separate data admission records commissioned authorship, exact formal
items, rights basis, source identities, procedures, scope and limitations.

| Tags | Meaning and domain |
| --- | --- |
| UC, IC | Union/intersection cardinality of two 3-bit sets (universe {0,1,2}) |
| BC | C(n,k), n=0..8; k=-1..n+1; out-of-range k has answer 0 |
| MA | (a+b) mod m, m=2..6, operands 0..m-1 |
| VD | Dot product of two length-two vectors, components -1..1 |
| VA0, VA1 | Component 0/1 of vector addition, components -1..1 |
| DT | Determinant of a row-major 2x2 matrix, entries -1..1 |
| MV0, MV1 | Component 0/1 of A v; matrix entries -1..1; v in {(-1,0),(0,1),(1,1)} |
| LS0, LS1 | Component 0/1 of the unique solution A x=b, same matrix/RHS domain; ERR for every singular A |

Answers are integers or reduced rationals (for example 1/2); ERR denotes no
unique solution, encompassing inconsistent and underdetermined systems. Tags
are unique; prompts include all operands and the component tag. Carriers are
`DT:1,0,0,1=` and `DT(1,0,0,1):`, both with answer `1`.

Procedural generator labels use bit counts, Pascal rows, repeated subtraction,
repeated addition and adjugate numerators. The independent oracle uses set
operations, binomial coefficients, modulo, permutation determinant expansion
and rational Gaussian elimination. A strict separate parser checks rendered
operands. Exhaustive replay rejects changed labels, prompts, split assignments,
admissions and source records; bools, noncanonical integers, unknown tags and
out-of-domain operands fail. All examples fit the unchanged byte-tokenizer
context; no silent truncation.

Set relabelling, exchange and simultaneous complement stay together. Binomial
families hold n fixed; modular families hold m fixed. Matrix row/column
permutations, transposition and global sign define conservative algebra orbits.
All tasks, carriers, RHS/vector variants and scalar projections for an orbit
share one split, including vector-pair tasks using the same four operands.
V2 stratifies algebra by singular/nonsingular matrices, with both classes in
every split. The sole absolute-determinant-2 orbit remains training-only to
teach rational answers; there is no held-out rational-solving evidence. Shared arithmetic primitives still appear across families and in the
legacy pack: it is not unseen-algorithm or primitive-isolation evidence.

## Frozen pilot

The machine protocol is `src/trinite/maths-protocol.json`, hash checked by
`maths_learning.protocol()`, with its source-bound request written before runs.
Each protocol producer is published before its own full outcomes. V2 corrects a
coverage error found in V1; both complete experiments are retained. Two seeds each train two
fresh dense models from paired random initialization, in alternating profile
order: control uses unchanged arithmetic train examples; expanded adds the
maths train split. All four cells use the unchanged 86,528-parameter two-block,
width-64 model, context 64, CPU float32, one thread, deterministic kernels,
AdamW at 0.001, batch 8 and 4,096 updates. Geometry loss stays zero. No optimizer,
architecture, quantizer or tokenizer semantics change. No inherited checkpoint
or pretrained weight is used.

Both profiles receive 32,768 example visits; corpus lengths, per-example visits
and scored target counts differ. Therefore this compares these two complete
training recipes, rather than isolating a corpus effect at equal old-task
exposure or equal token compute. A single seeded hash-rank cycle fixes order.
Final step only; intermediate diagnostics use **train** rows; no early stopping,
validation/test tuning or outcome-driven budget changes. The otherwise unused
validation and test rows never enter numerical updates or selection.

The final models and reconstructed initial random models are scored prompt-only
on fixed maths and legacy held-out splits, after training. Final training
accuracy is also retained. The scorer greedily generates unrestricted bytes
with a fixed eight-token cap; complete answer **plus EOS** is primary. No solver,
constrained decoder, teacher forcing or answer hints are used at evaluation.
Each task includes a train-derived majority baseline. Missing EOS, wrong answers
with EOS, carrier disagreements and both-carrier exact counts are diagnostics,
not replacements for primary accuracy. Legacy test is reused and maths test is
visible generated data, so results are exploratory. An improvement cannot
establish general-purpose mathematical reasoning or drift prevention.

Every cell retains full predictions, update history, model and AdamW safetensors,
metadata, identities, resources and a closed PROVENANCE bundle. Fresh review
restores safe tensors, checks exact schedule progress and admission/source/env
context, regenerates every prediction and loss, and compares root copies and
upstream custody receipts. Full optimizer-update replay and independent-host
replication are separate checks. Negative and failed cells remain retained.

## Run and review

Inside the [locked CPU environment](GETTING_STARTED.md):

```sh
PYTHONPATH=src python scripts/run_maths.py --freeze-request /tmp/maths-request.json
# Use the printed request identity verbatim:
PYTHONPATH=src python scripts/run_maths.py --request /tmp/maths-request.json \
  --request-identity sha256:PRINTED_DIGEST --output /tmp/maths-run
PYTHONPATH=src python scripts/run_maths.py --request /tmp/maths-request.json \
  --request-identity sha256:PRINTED_DIGEST --output /tmp/maths-run --verify-only
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/check_research_cpu.py
```

Freeze, workers, summary and verification reject incompatible selectors and
symlink paths. Fixed subprocess argv uses the same interpreter and repository
script, with bounded deadlines and no arbitrary shell/executable. Output and
request writes are exclusive; review is read-only. CPU CI includes the maths
numeric tests alongside every previously required suite. A full training run is
explicit, separate from routine CI.

Requirements affected: TRI-I02/03/05/06/07/08/11/12/13/14; CPU/data/evidence
boundaries are extended. Historical fixtures and numerical modules are unchanged.
The new packaged protocol and corpus identities are separate from earlier runs.

## Retained v1 result and coverage correction

The original protocol was published at
[fbc5c243](https://github.com/QSOLKCB/Trinite/commit/fbc5c243c21ef08417ca77e8795dd0c66bea72fd)
before its four cells. All completed, with every prediction/loss regenerated and
closed evidence verified by the complete summary. Seed 0 control/expanded maths
test counts were 127/351 out of 602; seed 1 counts were 134/332. All four legacy
test counts were 0/64. This is a retained positive aggregate diagnostic with a
critical coverage limitation: **all solve test matrices were singular**. Its
solve scores cannot support a unique-system-solving claim; both expanded models
also underperformed the train-derived ERR majority on those solve tests.

V1's lossless review packet is in `fixtures/maths-v1/`: 361 restored files,
82,323,009 bytes, archive identity
`sha256:294df98f6ae72128ef69b1cff71ddaa392765c57df74e608d0133b8c55b083a2`.
All 232 producer files match published GitHub blobs. This is an explicit review
exception to the small-fixture convention, preserving complete completed runs
and exact producer source, rather than an evolving model-release directory.

V2 changes the data policy/schema and orbit stratification, after the coverage
error and **before its own outcomes**, without changing labels, optimizer,
architecture, update budget or split seed. It neither searches seeds nor chooses
a split using predictions. Source admission requires unique and singular systems
in train/validation/test and rational targets in training. Test has 108 singular
and 96 unique-system scalar/carrier examples across the two solve tasks; there
are no fractional test targets. This correction is exploratory development after
exposure to V1, not a fresh confirmatory benchmark. V1 is not deleted or relabelled.

V2 was published at [d5d50fd0](https://github.com/QSOLKCB/Trinite/commit/d5d50fd006a5232136982d98a19edae150903821) before its outcomes. All four cells completed, with no worker or pairing errors, and their full predictions/losses freshly regenerated from safe checkpoints.
Historical fold-sum/composed adequacy, format selection, scaling and Phase 5
remain blocked regardless of these exploratory diagnostics.

To restore V1, join the ordered parts from its inventory and use the existing
bounded lossless restoration tool:

```sh
cat fixtures/maths-v1/Trinite-maths-v1.zip.part* > /tmp/Trinite-maths-v1.zip
python scripts/restore_foundations_archive.py /tmp/Trinite-maths-v1.zip \
  /tmp/maths-review-v1 --archive-identity sha256:294df98f6ae72128ef69b1cff71ddaa392765c57df74e608d0133b8c55b083a2
cd /tmp/maths-review-v1/producer
cp ../run/request.json /tmp/maths-review-v1-request.json
PYTHONPATH=src python scripts/run_maths.py --request /tmp/maths-review-v1-request.json \
  --request-identity sha256:64e53e5d0063393092792713979384ac8b103d3510080322466b29732d2cfc9a \
  --output ../run --verify-only
```

Use the matching locked environment; the request deliberately binds Python,
packages, device scope, numerical settings and actual source bytes. A different
environment must freeze a new request. The producer packet supports pilot replay;
full historical conformance needs the complete published checkout and fixtures.


## Complete v2 result

Exact complete answer plus EOS, final step; each test has 654 examples:

| Seed | Arithmetic control | Expanded maths | Train-derived majority |
| --- | --- | --- | --- |
| 0 | 159/654 (24.3%) | 274/654 (41.9%) | 288/654 (44.0%) |
| 1 | 157/654 (24.0%) | 324/654 (49.5%) | 288/654 (44.0%) |

The expanded recipe improves over the arithmetic control in both seeds, but only
seed 1 beats the majority baseline overall. This is a partial, task-dependent
gain, not mathematical learning adequacy. No candidate or new training protocol
is selected from these outcomes.

| Task | Test examples | Control seed 0 / 1 correct | Expanded seed 0 / 1 correct | Majority correct |
| --- | --- | --- | --- | --- |
| choose | 22 | 2 / 2 | 8 / 4 | 8 |
| determinant | 34 | 14 / 14 | 24 / 22 | 18 |
| dot | 34 | 10 / 8 | 18 / 19 | 10 |
| intersection-count | 28 | 0 / 8 | 11 / 14 | 12 |
| matvec-0 | 102 | 42 / 42 | 45 / 53 | 42 |
| matvec-1 | 102 | 42 / 38 | 41 / 60 | 42 |
| mod-add | 32 | 4 / 2 | 12 / 15 | 8 |
| solve-0 | 102 | 8 / 11 | 37 / 40 | 54 |
| solve-1 | 102 | 11 / 13 | 38 / 51 | 54 |
| union-count | 28 | 0 / 4 | 14 / 12 | 12 |
| vector-add-0 | 34 | 14 / 10 | 15 / 19 | 14 |
| vector-add-1 | 34 | 12 / 5 | 11 / 15 | 14 |

Determinant and dot-product accuracy exceed both the control and majority in both
seeds. Matrix-vector results vary by seed/component; combinations and unique
linear-system solving show no consistent benefit. Across both solve projections,
unique-system counts are control/expanded **19/18** for seed 0 and **24/44** for
seed 1, each out of 96. Expanded singular-system counts are 57/108 and 47/108;
always returning ERR solves all singular examples. Aggregate solve accuracy stays
below the task-majority baseline in both seeds. These class counts are derived
from retained predictions in `fixtures/maths-v2/analysis.json`, without new model
calls or answer filtering.

Retention is a material negative result: control arithmetic train accuracy is
264/264 in both seeds, while expanded is **153/264 (58.0%)** and **157/264 (59.5%)**.
All four reused arithmetic test results remain **0/64**. Reduced old-task exposure
is part of the expanded recipe; neither a forgetting mechanism nor a corpus-only
causal effect is established. Maths training accuracy is 1,152/1,998 and
1,263/1,998 in the expanded seeds. Learning adequacy remains unresolved.

Expanded maths test outputs have zero missing-EOS cases but 380/330 wrong answers
with EOS. Carrier disagreement is 46/327 and 57/327 paired problems; both carriers
are exactly correct in 126/327 and 138/327 pairs. These diagnostics do not establish
freedom from drift. All individual predictions, families and tasks remain visible.

Both freshly trained control model/AdamW safetensor payloads and complete update
histories match V1 byte-for-byte on this host. Expanded full-update replay and
independent-host replication remain pending. Resource observations describe
numerical training wall time and process peak RSS sampled before output scoring;
they are not matched-token compute or speed comparisons.

The complete V2 review packet is in `fixtures/maths-v2/`: 361 restored files /
81,326,806 bytes. All 232 exact producer files match published GitHub blobs;
all 78 request source artifacts are present. Fresh read-only verification
regenerates the summary and leaves every byte of all 128 run files unchanged.
Restoration checks every original member byte. Both review packets retain
all safe model/optimizer states, histories, predictions, corpora, custody and
exact producer; raw model weights are review evidence, not a release.

```sh
cat fixtures/maths-v2/Trinite-maths-v2.zip.part* > /tmp/Trinite-maths-v2.zip
python scripts/restore_foundations_archive.py /tmp/Trinite-maths-v2.zip \
  /tmp/maths-review-v2 --archive-identity sha256:b410e17388cd9f5774dd16f89c4899f5e730b5a66d13316d6b85c74b343f61e8
cd /tmp/maths-review-v2/producer
cp ../run/request.json /tmp/maths-review-v2-request.json
PYTHONPATH=src python scripts/run_maths.py --request /tmp/maths-review-v2-request.json \
  --request-identity sha256:43b1846e05ce78403daa35c8877bc42a31033455610cad66e462c621d05b444d \
  --output ../run --verify-only
```

Validation: initial frozen producer passed 114 standard-library and 167 required
CPU research tests. After the explicit data-only coverage correction, 115
standard-library tests and all four maths CPU/replay tests passed. The frozen
upstream suite ran 51 cases with its existing root-permission skip; this sandbox
cannot switch to an unprivileged account. Dependency consistency, changed-doc
relative links, source/blob parity, complete-cell custody/scoring regeneration,
read-only byte parity and both lossless restorations passed. Hosted checks on
the final head are reported separately; no unrun full expanded replay or external
replication is claimed.
