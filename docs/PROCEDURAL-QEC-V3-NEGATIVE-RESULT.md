# QEC procedural v3 — retained negative training result

**Observed 13 October 2026. All four research cells completed; the training
gate failed.** This is the completed local execution of the frozen v3 producer
at `cc996e272800a5c02667001f7c96c8d11cb513cc`, with 4,096 native dense CPU updates
per cell, two arms and seeds 0/1. Neither procedural seed passes every original
final-answer task. The historical v2 final-only ordering was also **not
preserved**, so this record does not establish an isolated cursor-correction
effect relative to v2.

All results below are training/development diagnostics. No held-out model
scoring was performed, no arm/checkpoint was selected, and learning adequacy,
format selection, Phase 5, scaling and release remain blocked. The
[original v2 negative record](PROCEDURAL-QEC-NEGATIVE-RESULT.md), its source,
requests, checkpoints and checksums are unchanged.

## Complete final answer plus EOS: primary result

The scorer requires greedy complete canonical final answer **and EOS**. Counts
are correct unique training examples, not repeated update visits. Each
arithmetic task has 66 examples, fold-sum 1,150, and GF(3) 54.

| Task | Seed 0 final-only | Seed 0 procedural | Seed 1 final-only | Seed 1 procedural |
| --- | ---: | ---: | ---: | ---: |
| Add | 65/66 | 61/66 | 66/66 | 65/66 |
| Subtract | 66/66 | 64/66 | 63/66 | 63/66 |
| Multiply | 66/66 | 66/66 | 66/66 | 65/66 |
| Divide | 66/66 | 64/66 | 62/66 | 58/66 |
| Fold-sum | 808/1150 | 614/1150 | 778/1150 | 762/1150 |
| GF(3) | 54/54 | 45/54 | 52/54 | 50/54 |

The prospectively frozen 90% criterion requires at least 60/66 for each
arithmetic task, 1,035/1,150 for fold-sum, and 49/54 for GF(3), in **both
procedural seeds**. Procedural seed 0 fails fold-sum and GF(3); seed 1 fails
fold-sum and division. Both final-only controls fail fold-sum. Within this
current-producer run, procedural final-answer counts are no greater than
the paired control on any task. The observed fold differences are −194 and
−16 correct examples for seeds 0 and 1; GF(3) differences are −9 and −2.
These are descriptive outcomes of this recipe and budget, not evidence that
procedural supervision or reasoning in general cannot work.

## Frozen scope and actual environment

The [v3 protocol](../src/trinite/procedural_qec_v3/protocol.json) was frozen at
`66a8d927ee3026e47268f6fdfae5bad3c00565df`, before this producer and these outcomes.
The request was created at `2026-10-13T14:02:55.007550+00:00`. The 86,528-parameter model uses
random initialization, float32 CPU dense execution, two blocks, width 64,
four heads, context 64, batch eight, native single-tensor AdamW, constant
learning rate 0.001, 4,096 updates and geometry loss zero. This is the research
profile, not the small-model conformance smoke.

The actual host was Linux 6.18.44 x86_64/glibc 2.41, AMD EPYC 9V74 80-Core
Processor, AVX512; CPython 3.12.14 and hash-locked Torch 2.8.0+cpu, deterministic
CPU execution, **one intra-op and nine inter-op threads**. The complete package
and CPU fingerprint is in the unchanged
[request](../fixtures/procedural-qec-v3-negative-20261013/request.json).
The historical v2 run used a different host and Python 3.12.15. This is not
same-host reproduction of historical v2 or independent-host numerical
replication of this v3 run.

Every current-v3 paired seed has equal initialization, model/optimizer plan,
dataset/manifest and arithmetic exposure. The summary has no pairing errors
or worker errors. That within-run pairing does not establish cross-version
historical control parity.

## Teacher-forced diagnostics, separated by objective

All displayed losses are rounded to six decimal places; the exact original
hexadecimal values remain in the unchanged
[full summary](../fixtures/procedural-qec-v3-negative-20261013/summary.json).
Final-only loss covers the final-answer diagnostic rows. Mixed loss covers
all final/prefix/trace diagnostic rows. Neither is free-running accuracy;
the mixed objective includes auxiliary formats absent from control training.

| Arm | Seed | Final-only loss | Mixed all-modes loss | Prefix loss | Trace loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| final-only | 0 | 0.242570 | 5.752373 | 2.327975 | 7.984599 |
| procedural | 0 | 0.388471 | 0.577141 | 0.649505 | 0.612807 |
| final-only | 1 | 0.283334 | 5.945004 | 2.254362 | 8.275706 |
| procedural | 1 | 0.298573 | 0.559522 | 0.680557 | 0.604349 |

The much smaller procedural mixed loss does not imply better original
final-answer learning: both procedural final-only losses are higher than their
paired controls, and their primary counts above fail the gate.

Per-task teacher-forced losses:

| Mode / task | Seed 0 final-only | Seed 0 procedural | Seed 1 final-only | Seed 1 procedural |
| --- | ---: | ---: | ---: | ---: |
| final / Add | 0.022629 | 0.076206 | 0.019967 | 0.026648 |
| final / Subtract | 0.015387 | 0.068615 | 0.076995 | 0.078027 |
| final / Multiply | 0.002435 | 0.007528 | 0.015911 | 0.021351 |
| final / Divide | 0.007536 | 0.028027 | 0.044428 | 0.134037 |
| final / Fold-sum | 0.307222 | 0.474921 | 0.350884 | 0.361528 |
| final / GF(3) | 0.010304 | 0.302834 | 0.054514 | 0.090678 |
| prefix / Fold-sum | 2.296325 | 0.658987 | 2.207693 | 0.686961 |
| prefix / GF(3) | 3.132704 | 0.408415 | 3.440945 | 0.517725 |
| trace / Fold-sum | 7.946991 | 0.624196 | 8.217301 | 0.616872 |
| trace / GF(3) | 9.102548 | 0.274258 | 10.011920 | 0.232077 |

## Actual auxiliary coverage and free-running correctness

Each procedural auxiliary stream receives 1,024 scheduled visits. Its
independent cursor reaches 1,024 distinct of 1,150 fold rows in each mode,
leaving 126 unvisited, and all 54 GF(3) rows in each mode. By comparison, the
historical v2 defect reached only 575 fold and 27 GF(3) distinct rows per mode.
This coverage correction is verified software behavior, not proof of an
isolated causal learning benefit.

| Procedural auxiliary | Seed 0 visited | Seed 0 unvisited | Seed 1 visited | Seed 1 unvisited |
| --- | ---: | ---: | ---: | ---: |
| fold-sum/prefix | 353/1024 | 52/126 | 332/1024 | 46/126 |
| fold-sum/trace | 14/1024 | 3/126 | 14/1024 | 1/126 |
| gf3/prefix | 40/54 | none (0 examples) | 27/54 | none (0 examples) |
| gf3/trace | 13/54 | none (0 examples) | 19/54 | none (0 examples) |

These are unique-row correct/example counts. An empty unvisited set has no
accuracy estimate. The control has zero visited auxiliary rows: its unvisited
fold-prefix counts are 151/1,150 and 166/1,150 (seeds 0/1), GF(3)-prefix
15/54 and 9/54, and both trace tasks are zero correct in both seeds.
Unvisited derived rows still belong to already exposed development families;
they are not held-out families. In particular, only 14/1,024 visited fold
traces are correct in either procedural seed, despite the corrected coverage.

## Exposure: the objectives are not token-matched

| Stream | Final-only visits (each seed) | Procedural visits (each seed) | Final-only target tokens (each seed) | Procedural target tokens seed 0 / 1 |
| --- | ---: | ---: | ---: | ---: |
| arithmetic/final | 16,384 | 16,384 | 42,202 | 42,202 / 42,202 |
| fold-sum/final | 10,240 | 8,192 | 25,068 | 20,061 / 20,061 |
| gf3/final | 6,144 | 4,096 | 12,288 | 8,192 / 8,192 |
| fold-sum/prefix | 0 | 1,024 | 0 | 2,445 / 2,452 |
| fold-sum/trace | 0 | 1,024 | 0 | 11,440 / 11,405 |
| gf3/prefix | 0 | 1,024 | 0 | 2,048 / 2,048 |
| gf3/trace | 0 | 1,024 | 0 | 8,192 / 8,192 |

All cells have 32,768 scheduled row visits. Controls score 79,558 target tokens
per seed; procedural seeds score 94,580 and 94,552. Procedural auxiliary rows
displace 2,048 fold-final and 2,048 GF(3)-final visits per seed while keeping
arithmetic exposure equal. All original final rows are visited in both arms.
Full input-token counts and visited/unvisited identity commitments are in the
original summary. Corrected coverage does not remove the displacement or
target-token differences, and this experiment does not isolate a reasoning
mechanism.

## Historical v2 comparison and failed preservation requirement

The following explicitly compares the immutable
[v2 report](PROCEDURAL-QEC-NEGATIVE-RESULT.md#original-final-answer--eos-training-accuracy)
with this v3 execution. Each cell is **v2 correct → v3 correct**; denominators
are unchanged (66 arithmetic, 1,150 fold, 54 GF(3)). No v2 value is replaced.

| Task | Seed 0 final-only | Seed 0 procedural | Seed 1 final-only | Seed 1 procedural |
| --- | ---: | ---: | ---: | ---: |
| Add | 63 → 65 | 66 → 61 | 64 → 66 | 64 → 65 |
| Subtract | 64 → 66 | 62 → 64 | 66 → 63 | 64 → 63 |
| Multiply | 64 → 66 | 64 → 66 | 66 → 66 | 61 → 65 |
| Divide | 64 → 66 | 65 → 64 | 66 → 62 | 55 → 58 |
| Fold-sum | 786 → 808 | 702 → 614 | 930 → 778 | 607 → 762 |
| GF(3) | 54 → 54 | 47 → 45 | 50 → 52 | 45 → 50 |

These numerical differences are **not causal estimates of the cursor fix**.
The read-only historical audit reconstructed original producer
`93522bf8f18fe38970ceafdd2ace792f8922c382` and its original dataset commitment.
All 3,876 mathematical rows agree after removing only `generator_revision`
and `example_identity`, but all 3,876 example identities changed. Generator
source revision contributes to example identity, which contributes to seeded
ranking. Consequently, the final-only mathematical sequence differs at
32,576/32,768 visits in seed 0 and 32,521/32,768 in seed 1. The historical host
also differs.

The current-row v2/v3 function comparison passes for the control; it supplies
both functions with regenerated **current** rows and cannot establish
historical parity. Both the
[original audit](../fixtures/procedural-qec-v3-negative-20261013/historical-control-audit.json)
and [independent comparison receipt](../fixtures/procedural-qec-v3-negative-20261013/independent-historical-control-comparison.json)
retain the failed requirement. The latter's original SHA-256 is
`7e0160ac9ca5c6241dd8001bbc8775c6dd1c51266755533705debd819d9a087b`.
The two receipts encode schedule commitments differently; their digest
strings should not be equated across encodings.

No ordering metadata, source identity, protocol or numerical artifact was
patched to manufacture parity. A future exact-parity study requires a distinct,
prospectively frozen and reviewed protocol/producer, genuine historical-control
regression and appropriate authorization before outcomes. See the
[required next comparison](PROCEDURAL-QEC-NEGATIVE-RESULT.md#required-next-comparison).
This report does not authorize a replacement protocol, another matrix, or
non-displacing auxiliary supervision.

## What was actually verified

All six assurance checks passed for each cell: closed PROVENANCE custody,
request/source binding, safe tensor structure, scheduled visits/targets,
final predictions/loss, and complete optimizer replay. Complete replay was
performed by the run's native verifier, again by a separate read-only
`--verify-only` invocation, and again during archival. Each starts from the
original native seed, executes all updates, and compares final model and
AdamW tensor payloads plus metadata containing the complete update and
training-diagnostic histories to externally bound original artifacts. Fresh
final scoring also reproduces report losses, predictions and diagnostics.
This is more than loading a checkpoint for inference, but uses the same
native implementation and declared numerical environment.

The [read-only receipt](../fixtures/procedural-qec-v3-negative-20261013/read-only-replay.json)
and [final byte-parity receipt](../fixtures/procedural-qec-v3-negative-20261013/final-byte-parity.json)
record four replayed cells and all **107 run files unchanged**. All expected
cells and the empty worker-error record are retained. The negative training
gate and historical comparison failure remain part of the record; a successful
execution/custody check does not make those scientific gates pass.

The separate standard-library archive check verifies 269 research payload
members (270 ZIP entries including the inventory), their identities, sizes,
member set and transport integrity. Isolated archived-source admission
verified all 156 declared source entries, imports, original request, dataset
and environment equality. An additional full-source TAR.GZ retains all 603
Git-tracked members of tree `061173b9f71e7aa1f5c1c3838e3d1b5e4f07a8ca`.
Independent preservation checks verified every source member against its Git
blob and reconstructed that tree, and verified both ZIP member sets. Those
preservation checks did not execute another optimizer replay.

The source-bound scorer and diagnostics select only admitted train rows;
there were no validation/test-family model calls. The updater's historically
named `validation` collection is a training/development diagnostic view here.
Replay proves reproduction of this declared producer, not independent-host
replication, held-out generalization, mathematical reasoning, geometry benefit,
or historical control preservation.

## Retained evidence and availability

The [bounded public evidence index](../fixtures/procedural-qec-v3-negative-20261013/evidence-index.json)
catalogues unchanged original request, external verification request, complete
summary and small receipts. Its catalogue is documentation, not a replacement
verifier or a newly sealed research bundle. Original model/optimizer tensors,
step histories, predictions, custody and sources remain in the large archive.
The large archives are kept outside the Git tree.

| Original artifact | SHA-256 |
| --- | --- |
| Research ZIP | `65cc25d600e30a35c6ad3daefae6f92979162dfcb38e515397185aa952154d79` |
| Research ZIP inventory | `e8dbdbb0134efc976014dc419054f2888fd3b5e0b02c26bf71c0c71d97c3b3a2` |
| Full-source TAR.GZ | `5b7fd56a20e9f111cdf4d9b278c89de9e23ff3715cfcb2a8d9c2ccfa4ede1d87` |
| Final verification-records ZIP | `e34ca0cc3f7a908036ff21ad9813b8343f9d91b12085e96dc91dc904335a4b4a` |
| Final SHA256SUMS file | `80dbac149fae06dea04dd857db2922c92ddefc17624b842e03af4583135af31a` |
| Original request | `7ee0b351cf737807cf888a66494fd1438714698f2c1b9834ea2073c0ee5c66ff` |
| Original verification request | `6a52d8bcc33dc1c6a85624126e0001525a768f9deb24c8715bf46e9dfe8155b9` |
| Original summary | `c3ca81a4ef2a8aff90ce01ddd4984f4ce555d99eb559f5a2650ca9a6aaa3352b` |

Private owner-held retention of the research ZIP, exact full source, final
verification/control records and final checksums was completed and byte-checked
on 13 October 2026. **There is no public archival download or DOI for these
large artifacts in this record.** Private preservation is not permanent public
archival publication. This local run was not a hosted Actions research run.
No tag, release, Zenodo deposit, changed access permissions or public model
release is represented by this documentation.

Compatibility: additive negative-result reporting and original-byte evidence
copies only. No numerical, protocol, scheduler, source-closure or historical
fixture semantics change. Relevant invariants: TRI-I02/I03/I05/I06/I07/I08,
TRI-I09/I10/I11/I12/I13/I14.
