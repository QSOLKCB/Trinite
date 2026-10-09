# Arithmetic retention with a finite maths curriculum

After merging PR #19, Trent requested the next increment on 2026-10-09.
The corrected maths pilot improved narrow maths counts while reducing earlier
arithmetic training accuracy from 264/264 to 153/264 and 157/264. The next
increment tests explicit arithmetic rehearsal before geometry or scaling.
Affected requirements: TRI-I02, I03, I07, I08, I10, I11, I12, I13 and I14.

## Frozen comparison

The packaged `maths-retention-protocol.json` is the machine contract. Its
identity is `sha256:cf9c5928428cfd1a1dd82f515cdb44b4c10c0e6316b97aa7967f56d416c3af7d`.
Implementation, protocol and review instructions are published before affected
training outcomes. Both arms start from random native initialization. Earlier
models and both complete maths review packets remain retained unchanged.

| Setting | Decision |
| --- | --- |
| Data | Exact admitted PR #19 v2 arithmetic and maths bytes, families, tasks and carriers |
| Cells | Seeds 0 and 1; pooled and rehearsal dense profiles; reversed arm order in seed 1 |
| Pooled control | Repeat PR #19 v2 expanded schedule: one seeded hash-ranked cycle of 264 arithmetic + 1,998 maths training rows |
| Rehearsal | Two independently wrapping streams, each in its original pooled seeded rank; four arithmetic followed by four maths examples in every batch |
| Budget | Same 4,096 updates, batch 8, 32,768 example visits, architecture and AdamW; no warm start or added updates |
| Model | Existing MathsPlan: context 64, two blocks, width 64, four heads, FF 128; 86,528 parameters |
| Optimizer | Existing native answer/EOS mean cross-entropy and AdamW, LR 0.001; geometry loss zero |
| Loss diagnostics | The same unique 2,262 training rows in pooled rank for both arms, every 1,024 updates; validation corpus unused |
| Output diagnostics | Shared prompt-only unconstrained exact answer + EOS, cap 8; final train and reused test splits; initial random test scores reconstructed after training |
| Retention criterion | At least 90% complete-answer/EOS accuracy in each arithmetic training task |
| Maths criterion | At least 90% in each maths training task; descriptive pilot criterion |
| Selection | Final update only; fixed ratio, no ratio sweep, validation/test tuning, early stopping or candidate selection |
| Limits | 480 seconds per numerical invocation; 524,288 scored targets; fixed fresh worker deadline 1,200 seconds |

The rehearsal schedule materializes its single fixed budget of visits; it does
not duplicate corpus items or change admission identities. `MathsPlan` keeps the
existing numerical schema and bounds; the separately bound protocol and cycle
identify the supplied schedule. The diagnostic view is unique training rows,
so rehearsal duplicates do not weight reported diagnostic loss or accuracy.
The native updater still owns all batches, losses, gradients and AdamW steps.

Rehearsal reserves 16,384 visits for each corpus. This changes arithmetic/maths
exposure and task mixing together and reduces maths visits relative to pooled
training. Equal updates and visits do not establish equal scored targets, input
tokens, compute or per-example exposure. Every report retains actual per-corpus
visits, targets, nonpadding input tokens, task visits and per-example visit
histograms, including unvisited examples. No isolated causal explanation is
claimed.

Both training criteria are descriptive diagnostics. Repeated PR #19 test use
is explicitly exploratory development, not a fresh confirmatory benchmark.
Correlated carriers/projections are not independent samples. This protocol does
not assess original folds/composed outputs, held-out rational solving, ternary
format selection, geometry intervention, model scaling or release readiness.
Even a retention pass cannot clear the historical learning gates. Every summary
retains `selected_candidate=null`, `dense_learning_adequate=false` and
`phase5_ready=false`. A new frozen decision must precede another ratio, budget,
corpus or intervention experiment.

## Execution and verification

Use the exact producer checkout and pinned CPU lock in [GETTING_STARTED](GETTING_STARTED.md).

```sh
PYTHONPATH=src python scripts/run_maths_retention.py \
  --freeze-request /tmp/maths-retention-request.json
# Pass the emitted immutable request identity; use a new output directory.
PYTHONPATH=src python scripts/run_maths_retention.py \
  --request /tmp/maths-retention-request.json --request-identity sha256:REQUEST \
  --output /tmp/maths-retention-run
PYTHONPATH=src python scripts/run_maths_retention.py \
  --request /tmp/maths-retention-request.json --request-identity sha256:REQUEST \
  --output /tmp/maths-retention-run --verify-only
```

Requests bind corpus/admission, protocol, complete flat runtime source closure,
fixed launchers, environment and clock. Source additions require a new request;
old packets are verified with their archived producers. The request must be
outside the output directory. All selectors, subprocess argv and interpreter
are fixed; symlink ancestors, overwrite and overlapping request/output fail.
Workers retain failures and logs. Verification calls the current upstream
PROVENANCE verifier, checks the closed artifact index and all root output copies,
restores named safe model/AdamW tensors, validates every scheduled visit/target,
and freshly regenerates output predictions, losses, exposure, thresholds and
summary. Custody checks are separate from learning claims.

Required tests include train/test exclusion, independent stream wrapping, prior
control parity, actual exposure/target accounting, snapshot continuation across
arithmetic wrap, rejected checkpoint/request mutation and the exact per-task
threshold. `scripts/check_research_cpu.py` includes the separate required
`maths_retention` suite. No corpus, tokenizer, model, quantizer, optimizer,
numerical updater or historical producer is changed by this increment.

## Complete paired results

Producer `edbd1baf1fd7ebaadd88bb313276f1b17391b4c8` was published before target
outcomes. Request: `sha256:bbb81e65c84abf0b559d475ada743ead0a0fe97f116119ca652919dd57f90057`.
Summary: `sha256:cadb6263a2a8652bc1c4f17f4a417d3329b3f5b1585f2bee1d7bd1b4fb2603af`.
All four full 4,096-update cells completed and freshly verified, without worker
or pairing errors. Both pooled model/AdamW payloads, complete update/diagnostic
histories and prediction files match PR #19 v2 byte-for-byte on this host.

| Seed | Profile | Arithmetic train | Arithmetic reused test | Maths train | Maths reused test |
| --- | --- | --- | --- | --- | --- |
| 0 | pooled | 153/264 (58.0%) | 0/64 (0.0%) | 1152/1998 (57.7%) | 274/654 (41.9%) |
| 0 | rehearsal | 256/264 (97.0%) | 0/64 (0.0%) | 1030/1998 (51.6%) | 271/654 (41.4%) |
| 1 | pooled | 157/264 (59.5%) | 0/64 (0.0%) | 1263/1998 (63.2%) | 324/654 (49.5%) |
| 1 | rehearsal | 263/264 (99.6%) | 6/64 (9.4%) | 1120/1998 (56.1%) | 308/654 (47.1%) |

Rehearsal clears the predeclared 90% arithmetic criterion in each task and
both seeds; pooled training fails all four arithmetic tasks. All 12 maths
training tasks miss their criterion in every cell. Rehearsal improves arithmetic
training retention while reducing maths training and aggregate reused-test
accuracy in both seeds. Arithmetic generalization remains poor. The maths
majority baseline is 288/654 (44.0%); only seed 1 beats it in either profile.
No schedule is selected and learning adequacy/Phase 5 remain blocked.

| Arithmetic training task (66 examples) | Seed 0 pooled | Seed 0 rehearsal | Seed 1 pooled | Seed 1 rehearsal |
| --- | --- | --- | --- | --- |
| add | 27 | 66 | 34 | 66 |
| divide | 46 | 60 | 41 | 66 |
| multiply | 51 | 65 | 50 | 66 |
| subtract | 29 | 65 | 32 | 65 |

| Maths reused-test task | Examples | Majority | Seed 0 pooled | Seed 0 rehearsal | Seed 1 pooled | Seed 1 rehearsal |
| --- | --- | --- | --- | --- | --- | --- |
| choose | 22 | 8 | 8 | 6 | 4 | 4 |
| determinant | 34 | 18 | 24 | 18 | 22 | 21 |
| dot | 34 | 10 | 18 | 24 | 19 | 21 |
| intersection-count | 28 | 12 | 11 | 2 | 14 | 9 |
| matvec-0 | 102 | 42 | 45 | 52 | 53 | 45 |
| matvec-1 | 102 | 42 | 41 | 50 | 60 | 49 |
| mod-add | 32 | 8 | 12 | 23 | 15 | 8 |
| solve-0 | 102 | 54 | 37 | 35 | 40 | 61 |
| solve-1 | 102 | 54 | 38 | 32 | 51 | 54 |
| union-count | 28 | 12 | 14 | 5 | 12 | 9 |
| vector-add-0 | 34 | 14 | 15 | 11 | 19 | 12 |
| vector-add-1 | 34 | 14 | 11 | 13 | 15 | 15 |

Post-training derived class counts in `analysis.json` keep singular detection
and unique-system solving separate. Across both projections/carriers, unique
test solutions rise from 18/96 to 36/96 in seed 0 and 44/96 to 45/96 in seed 1.
Singular test detection changes from 57/108 to 31/108 and 47/108 to 70/108.
Total solve accuracy is 75→67/204 and 91→115/204; the ERR majority baseline is
108/204. These mixed class outcomes do not establish algebra adequacy, a
causal arithmetic-transfer mechanism or held-out rational solving.

Every cell has zero missing EOS on maths test outputs. Rehearsal has 383/346
wrong answers with EOS, compared with pooled 380/330. Carrier disagreement is
46/327 and 43/327, versus pooled 46/327 and 57/327; both carriers are exact in
123/327 and 139/327, versus 126/327 and 138/327. Freedom from drift is unresolved.

## Actual exposure and resource observations

| Seed | Profile | Arithmetic visits | Maths visits | Arithmetic targets | Maths targets | Nonpadding input tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | pooled | 3823 | 28945 | 9853 | 73825 | 559078 |
| 0 | rehearsal | 16384 | 16384 | 42199 | 41761 | 456676 |
| 1 | pooled | 3818 | 28950 | 9831 | 73849 | 559014 |
| 1 | rehearsal | 16384 | 16384 | 42204 | 41801 | 456750 |

Pooled examples receive 14–15 visits each. Rehearsal arithmetic examples
receive 62–63 and maths examples 8–9 visits. Full per-task visits and per-example
histograms remain in every verified report. The result concerns joint training
from random initialization; sequential checkpoint forgetting is outside this
protocol. Rehearsal changes exposure and batch composition together.

Numerical training wall time was 26.8–30.1 seconds and process peak RSS before
output scoring was 342,232–348,576 KiB. Output scoring, verification and archive
IO are outside that timer. Different targets/input lengths and shared-host
conditions prevent a compute-matched speed claim. Full rehearsal optimizer
replay and independent-host replication remain pending.

## Retained evidence and reproduction

The lossless review packet under `fixtures/maths-retention-v1/` restores 370
files / 81,432,928 bytes: 237 exact producer files, 132 run files and the member
inventory. All producer files match the published producer's GitHub blobs; all
81 request-bound runtime sources are included. The packet retains all four safe
model/AdamW payloads, update histories, complete predictions, admitted corpora,
closed PROVENANCE, worker logs and the standard-library/CPU/upstream check logs.
The independent restored archive matches every original member byte.

Fresh read-only verification reconstructs all scores, losses, exposure and
summary, and preserves every byte of all 132 run files. Fresh archived-producer
verification also passes and preserves all 132 restored run files. Post-training class counts and historical
control parity are derived in `analysis.json`, bound to the complete summary;
they never selected a schedule. The inventory and verification receipts record
the exact identities and scope. Raw weights remain review evidence, not a release.

```sh
cat fixtures/maths-retention-v1/Trinite-maths-retention-v1.zip.part* > /tmp/Trinite-maths-retention-v1.zip
python scripts/restore_foundations_archive.py /tmp/Trinite-maths-retention-v1.zip \
  /tmp/maths-retention-review --archive-identity sha256:d799c429b2942cd1285588404a73163d86a3f6edfdd8c362a8cfd07b2df085ab
cd /tmp/maths-retention-review/producer
cp ../run/request.json /tmp/maths-retention-review-request.json
PYTHONPATH=src python scripts/run_maths_retention.py \
  --request /tmp/maths-retention-review-request.json \
  --request-identity sha256:bbb81e65c84abf0b559d475ada743ead0a0fe97f116119ca652919dd57f90057 \
  --output ../run --verify-only
```

Validation: 115 standard-library checks, all 173 required CPU research checks
and the six standalone rehearsal CPU/schedule/replay checks passed. Frozen
upstream: 51 cases with its existing root-permission skip. Dependency consistency
and changed-document links passed. Hosted CPU and reference checks on the frozen
producer are green across their declared Python versions. Final result-head
checks are reported separately. Full rehearsal optimizer replay, independent
host replication, historical learning gates and release requirements remain
pending; verification success grants no broader learning claim.
