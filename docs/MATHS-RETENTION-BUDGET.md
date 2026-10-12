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

## Status and remaining gates

Implementation and research outcomes are reported separately. Production
execution and archival results are pending until the published producer runs.
A fresh family-level confirmation protocol with independent exact labels,
exposure history and declared singular/unique and rational-solving coverage
must precede capability comparisons. Resplitting known examples is not fresh.
Original fold-sum and composed-output adequacy retain their own blocked gates.
Format selection, geometry intervention, scaling, independent replication, CPU
export and release remain separate decisions. The future recursive scaffold
and RES=RAG measurement proposals remain unchanged.
