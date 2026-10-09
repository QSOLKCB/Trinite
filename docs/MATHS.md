# Finite maths and exact linear algebra

Authorised by Trent Slade on 2026-10-09 after PR #18: prepare an admitted maths
corpus and train a comparison. This separate exploratory pilot permits its own
held-out diagnostics; it changes none of the historical failed gates, selects
no format and does not unlock Phase 5. A corpus can teach narrow operations;
correctness and freedom from drift require measurements, not a corpus guarantee.

## Admission and coverage

`maths_data.py` generates 3,154 examples (1,577 formal items, two carriers).
The orbit split contains 2,452 train, 100 validation and 602 test examples.
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
This deliberately uneven orbit split has very small validation counts for some
tasks. Shared arithmetic primitives still appear across families and in the
legacy pack: it is not unseen-algorithm or primitive-isolation evidence.

## Frozen pilot

The machine protocol is `src/trinite/maths-protocol.json`, hash checked by
`maths_learning.protocol()`, with its source-bound request written before runs.
The exact producer is published before full outcomes. Two seeds each train two
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

## Empirical status

Implementation and software checks are being completed before the frozen pilot.
The final report will retain every cell and all per-task/regression counts.
Historical fold-sum/composed adequacy, format selection, scaling and Phase 5
remain blocked regardless of this exploratory diagnostic.
