# Balanced rehearsal budget development

Requested after the merge of PR #21 on 2026-10-12. This increment implements the
next [roadmap decision](ROADMAP.md#planned-retention-audit-and-joint-learning-budget-comparison).
Affected requirements: TRI-I02/I03/I05/I07/I08/I11/I12/I13/I14. It changes the
budget recipe and evidence identities, preserving all original v1 and v2 source
bytes, protocols, fixtures and negative results.

## Frozen question and compatibility

Does a larger balanced-rehearsal budget improve finite maths training while
preserving arithmetic training performance?

| Arm | Updates | Arithmetic visits | Maths visits | Seeds |
| --- | --- | --- | --- | --- |
| Control | 4,096 | 16,384 | 16,384 | 0, 1 |
| Candidate | 8,192 | 32,768 | 32,768 | 0, 1 |

Both arms use the same admitted corpora, native dense initialization, 86,528
parameters, AdamW, constant LR 0.001 and independent ranked four-arithmetic /
four-maths streams. Seed 0 runs control first; seed 1 runs candidate first.
Geometry loss is zero. Doubling updates increases both corpora's exposure and
extends the optimizer trajectory. This is a recipe comparison, not an isolated
maths-exposure effect or compute-matched superiority claim.

The separately versioned `trinite.maths-retention-budget-plan.v2` permits up to
8,192 updates, 960 numerical seconds and 1,048,576 scored targets. The frozen
machine protocol lives at `src/trinite/retention_v2/budget/protocol.json`, identity
`sha256:4813eef6d93d8d229cf31624d61d702c77e915a7034b7d994c12145929f0789d`.
Both research arms have the same explicit resource ceilings. Stop conditions
are final update, numerical resource exhaustion, or worker failure; accuracy
cannot stop or select a checkpoint. Loss diagnostics use all unique admitted
training rows every 1,024 updates and at the final update. Every update/history
is retained; the safe model/AdamW checkpoint is final only.

The nested package leaves both prior source scans unchanged. A bounded v1
adapter reuses admitted pooled row construction, native initialization and
optimizer creation, then installs the new typed plan and materialized rehearsal
schedule. It never presents an 8,192-update run as a v1 plan. The authoritative
updater, scorer, strict criteria, admission, tensor reader, exposure counters,
solve-class diagnostics and custody operations are reused. New orchestration
has separate request/checkpoint/report/summary schemas and preserves the frozen
assurance producer rather than changing its globals or protocol.

## Training-only acceptance and evidence

A research candidate passes joint training only if **every** one of the four
arithmetic and twelve maths tasks reaches at least 9/10 complete canonical
answer-plus-EOS accuracy in **both** seeds. Exact task/count maps come from
admission; arithmetic requires 60/66 for each task. Every control and candidate
must complete and verify all six assurance checks, including exact native
model/AdamW/history replay. A complete matrix can report a negative scientific
result. `joint_training_accepted` is separate from `outcome`, and remains false
for smoke, limited inference, incomplete matrices or failed candidate tasks.

Initial and final scoring consume training rows only. No test or validation
model calls occur in this experiment. Unique diagnostics remain separate from
scheduled visits. Per-task counts, majority comparisons, unique/singular solve
counts, carrier/EOS drift, actual visits, targets and input-token exposure remain
visible. Neither a winning seed nor a selected recipe is emitted.

The new profile inherits the [v2 assurance checks](MATHS-RETENTION-V2.md#verification-levels-and-commitments)
and separately supplied experiment/verification identities. Controller failures
retain every expected cell, errors and logs. Each training worker has 2,400
seconds for its complete operation; matrix summary/read-only workers have 9,600
seconds. Nonzero summary exits fail even after writing a completed summary.
Final aggregation rechecks external commitments, actual upstream custody and
root copies. Read-only verification writes no run files.

Smoke uses a separately identified width-16 model, four/eight updates, diagnostic
interval two, cap two, and both carriers of the first training semantic identity
per task. It establishes software integration only. It cannot pass the research
acceptance flag.

## Commands and lossless archival

Acquire the CPU lock using [GETTING_STARTED](GETTING_STARTED.md). Publish the
producer/protocol before research outcomes, then freeze a fresh host-bound request:

```sh
PYTHONPATH=src python scripts/run_maths_retention_budget.py \
  --freeze-request /tmp/budget-request.json
PYTHONPATH=src python scripts/run_maths_retention_budget.py \
  --request /tmp/budget-request.json --request-identity sha256:EXPERIMENT \
  --verification-request /tmp/budget-verification.json \
  --output /tmp/budget-run --tier replay
# Independently retain the emitted verification identity.
PYTHONPATH=src python scripts/run_maths_retention_budget.py --verify-only \
  --request /tmp/budget-request.json --request-identity sha256:EXPERIMENT \
  --verification-request /tmp/budget-verification.json \
  --verification-request-identity sha256:VERIFICATION \
  --output /tmp/budget-run --tier replay > /tmp/budget-audit.json
PYTHONPATH=src python scripts/archive_maths_retention_budget.py \
  --run /tmp/budget-run --output /tmp/budget-evidence.zip \
  --request /tmp/budget-request.json --request-identity sha256:EXPERIMENT \
  --verification-request /tmp/budget-verification.json \
  --verification-request-identity sha256:VERIFICATION
```

The archiver independently replays the complete matrix, compares the retained
summary and external verification-request copy, and exclusively writes a ZIP
outside the run. It preserves every run member, logs, all model/AdamW payloads,
histories, predictions, custody, exact source closure, both external requests
and a fresh replay summary. `inventory.json` lists every other member's byte
count and SHA-256; the emitted inventory/archive digests are retained outside
the ZIP. Current source and external commitments are rechecked, and every ZIP
member must round-trip byte-for-byte. Missing/mutated evidence cannot be archived
as a verified complete run. Paths reject symlink ancestry and overwrites.

This facility preserves this new producer's evidence. It does not reconstruct
the separately reported Ryzen/AVX2 PR #21 audit from a report alone. That audit's
original full artifacts and producer still need independent publication; the
original EPYC/AVX512 archives remain unchanged.

## Published producer source snapshot

The final evidence packet includes `fixtures/retention-budget-v2/producer-source.zip`,
a separate 156-member snapshot of `src/`, `scripts/` and root runtime
metadata from published producer `816cdc1cbaac24c86048d063711891713277d3d5`.
Every member was compared with its published Git blob and round-tripped exactly;
its inventory also records byte counts and SHA-256 identities. This includes
pinned PROVENANCE and QEC code and licence files beyond the direct request source
map. Those donor runtime bytes are bound transitively by the original source pin
and freshly audited at the existing observation boundary. Use the complete
snapshot alongside the evidence archive for offline review; external wheels still
follow the unchanged hash lock. It is not a replacement for the original Ryzen
producer or its missing full audit records.

## Measured training-only result

Producer `816cdc1cbaac24c86048d063711891713277d3d5` was published before the
new research request and outcomes. All four workers completed; fresh verification
passed all six checks in every cell, with no worker or pairing errors. The complete
24,576-update matrix was measured on Intel Xeon Platinum 8573C / AVX512,
Linux 6.18.44, CPython 3.12.14 and the unchanged hash-locked Torch 2.8.0+cpu lane.
These are same-host paired results, not reproduction of the original EPYC or the
independently reported Ryzen counts. Both older producers/results remain intact.

| Seed | Arm | Arithmetic correct / 264 | Maths correct / 1,998 |
| --- | --- | --- | --- |
| 0 | control (4,096 updates) | 257 | 975 |
| 0 | candidate (8,192 updates) | 262 | 1206 |
| 1 | control (4,096 updates) | 257 | 1074 |
| 1 | candidate (8,192 updates) | 263 | 1269 |

Arithmetic passes every task in all four cells. The larger budget improves maths
aggregate counts by 231 and 195 examples, but **all twelve maths tasks fail their
90% criterion in both candidate seeds**. `joint_training_accepted=false`; no
budget, checkpoint or winning seed is selected. This remains incomplete joint
learning, despite the positive aggregate contrast and successful software replay.
No held-out model calls occurred. Some individual tasks regress despite the
aggregate improvement; the full task table retains those negative contrasts.

The exact maths gate table is:

| Task | Examples | Required correct | Control 0 | Candidate 0 | Control 1 | Candidate 1 |
| --- | --- | --- | --- | --- | --- | --- |
| choose | 84 | 76 | 34 | 39 | 31 | 37 |
| determinant | 104 | 94 | 66 | 68 | 67 | 74 |
| dot | 104 | 94 | 72 | 67 | 72 | 73 |
| intersection-count | 76 | 69 | 36 | 52 | 58 | 56 |
| matvec-0 | 312 | 281 | 162 | 190 | 154 | 221 |
| matvec-1 | 312 | 281 | 171 | 156 | 168 | 195 |
| mod-add | 98 | 89 | 33 | 43 | 41 | 40 |
| solve-0 | 312 | 281 | 123 | 210 | 167 | 227 |
| solve-1 | 312 | 281 | 134 | 213 | 188 | 190 |
| union-count | 76 | 69 | 51 | 65 | 39 | 51 |
| vector-add-0 | 104 | 94 | 42 | 53 | 36 | 51 |
| vector-add-1 | 104 | 94 | 51 | 50 | 53 | 54 |

The [retained packet](../fixtures/retention-budget-v2/) contains the separately
held requests, complete replay summary and original producer source snapshot.
The lossless evidence archive and independent member/checksum audit complete
that packet. Exact per-task majority counts, unique/singular solve diagnostics,
carrier/EOS errors, actual visits, targets and input tokens remain in every report.
Archive publication is a deliberate full-evidence exception to the small-fixture
convention; it preserves the completed negative result rather than refreshing any
historical fixture. Verification establishes computational/evidence reproduction,
not fresh capability or learning adequacy.

Experiment request identity:
`sha256:11a740c30a67918e9fa024ec21bee08d1c72dca069911053830e6979a1e3bcbb`.
Independently retained verification request identity:
`sha256:9180f158f8d111e7e1962b290ed673c925a200d60fc6f6bfea3ec3355daefa3c`.

## Status and remaining gates

Implementation and measured training outcomes are reported separately. The joint
training criterion fails in both candidate seeds; a new separately frozen
training-only decision must precede another learning experiment.
A fresh family-level confirmation protocol with independent exact labels,
exposure history and declared singular/unique and rational-solving coverage
must precede capability comparisons. Resplitting known examples is not fresh.
Original fold-sum and composed-output adequacy retain their own blocked gates.
Format selection, geometry intervention, scaling, independent replication, CPU
export and release remain separate decisions. The future recursive scaffold
and RES=RAG measurement proposals remain unchanged.

## Software validation

[CPU conformance run 38239328326](https://github.com/QSOLKCB/Trinite/actions/runs/38239328326)
passed the standard-library and all 209 required numerical/research cases on
Python 3.11, 3.12 and 3.13 at `575352bb7e935afdb3973e53b71d02647c250d72`.
Reference backend conformance also passed. Local CPython 3.12.14 validation passed
122 standard-library checks, 50 frozen upstream cases with the existing
root-permission skip, the corrected joint-gate regression, hash-lock dependency
consistency, changed links and whitespace checks. The new numerical producer
remains exactly the published `816cdc1` source.

The first CI run exposed an alias in a synthetic test fixture: initial and final
scores shared one dictionary, so mutating a final score also changed the paired
initial score. The fixture now copies the initial score independently. This
changed no numerical/evidence producer, protocol, request or research output.
The corrected full CI run is the completed software-assurance evidence; the
interrupted/stale local runner that had already loaded the old fixture is not
reported as a passing full suite.

The published evidence ZIP is 44,642,427 bytes and is retained as eleven ordered
4-MiB-or-smaller lossless parts. Concatenation reproduces the externally recorded
archive SHA-256 exactly; the final packet records per-part checksums. Independent
post-archive inspection verified all 229 inventoried members. The archival audit
left every one of the 129 original run files byte-identical.
