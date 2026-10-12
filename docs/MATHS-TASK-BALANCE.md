# Task-balanced maths scheduling

Requested after the merge of PR #23 on 2026-10-12, with learning development
selected as the next roadmap increment. Affected requirements:
TRI-I02/I03/I05/I07/I08/I11/I12/I13/I14. All earlier protocols, source closures,
fixtures and negative results remain unchanged.

## Frozen question

Does distributing maths visits evenly across tasks improve joint arithmetic
and finite maths training at the existing 8,192-update budget?

The prior [budget experiment](MATHS-RETENTION-BUDGET.md) improved aggregate maths
accuracy while every maths task still failed its training gate. Its row cycle
allocates more visits to larger tasks. This new comparison tests a single
predeclared scheduling recipe without searching ratios or consulting held-out
results.

| Setting | Control | Candidate |
| --- | --- | --- |
| Updates per seed | 8,192 | 8,192 |
| Seeds | 0, 1 | 0, 1 |
| Arithmetic/maths visits per update | 4 / 4 | 4 / 4 |
| Maths selection | Existing seeded ranked-row cycle | Round-robin tasks, seeded ranked rows within each task |
| Visits per maths task | Determined by row cycle | 2,731 for first eight tasks; 2,730 for last four |

Tasks cycle lexicographically: choose, determinant, dot, intersection-count,
matvec-0, matvec-1, mod-add, solve-0, solve-1, union-count, vector-add-0,
vector-add-1. The global maths slot selects the task modulo twelve; each task
has its own row cursor. Within-task order filters the existing seed-ranked
unique training view. Arithmetic slots remain byte-for-byte identical in both
arms. Each arm has 32,768 arithmetic and 32,768 maths visits per seed.

Both arms retain the dense 86,528-parameter native model, AdamW, constant
LR 0.001, zero geometry loss, final checkpoint only, and unique-training loss
diagnostics every 1,024 updates and at completion. Each cell has the unchanged
960 numerical-second and 1,048,576 scored-target ceilings. Stop conditions are
the final update, resource exhaustion or worker failure; accuracy never selects
a checkpoint. Seed 0 runs control first; seed 1 runs candidate first.

Task frequencies and ordering change together. Input/target token exposure and
per-example repetitions are measured, not matched. This is a scheduling-recipe
comparison, not a causal estimate of frequency alone or compute-matched
superiority. No learning improvement is assumed.

## Version and evidence contract

The canonical protocol is
`src/trinite/retention_v2/task_balance/protocol.json`, identity
`sha256:1c391549cad55ef72081669f0df495d7a387e379b8a34922ae72251e19a05be2`.
New task-balance v1 request, checkpoint, report, summary and archive schemas
bind the complete historical and new runtime source closures, fixed launchers,
environment, corpora, schedule and unique diagnostic view. The resource plan
reuses `trinite.maths-retention-budget-plan.v2`; its unchanged bounds and
optimizer configuration do not need a new implementation. The separate
protocol and schedule identity specify which admitted rows occupy the plan's
update slots.

The control exactly reproduces the previous budget candidate's schedule,
initialization, updates and diagnostics under the same seed/environment. Both
arms are rerun under the new source-bound request. The new package preserves
historical source scans; it adds no numerical updater, scorer, admission path
or tensor reader. Native optimizer/history replay starts from the declared
seed, using the existing authoritative operations.

The [six v2 assurance checks](MATHS-RETENTION-V2.md#verification-levels-and-commitments)
remain separate. External experiment and artifact-verification identities are
mandatory. Final inference cannot claim complete optimizer replay. Actual
upstream custody verification, exact root copies, tensors, AdamW moments,
every history/diagnostic and predictions are rechecked. Foreign environments
require separate requests; old fingerprints are never patched. Verification
is read-only. All expected cells, failures, logs and failed receipts are retained.

Training-only acceptance requires at least 90% complete canonical
answer-plus-EOS accuracy in **every** four arithmetic and twelve maths tasks in
**both** candidate seeds, with all four cells passing complete replay. Counts
and thresholds come from admission. Per-task majority comparisons, singular/
unique solve diagnostics, carrier/EOS drift and actual exposure remain visible.
A complete verified matrix can have `joint_training_accepted=false`. No winning
seed or recipe is selected. No test or validation model calls occur.

Smoke uses the separately identified width-16 model, eight updates in both
arms, diagnostics every two updates and both carriers of the first training
semantic identity per task. It verifies integration only and cannot satisfy
research acceptance.

## Execution and retention

Use the hash-locked CPU environment in [GETTING_STARTED](GETTING_STARTED.md).
Publish the producer/protocol before freezing a research request or generating
target outcomes. The default profile is research; use `--run-profile smoke`
only when freezing a separately identified integration request.

```sh
PYTHONPATH=src python scripts/run_maths_task_balance.py \
  --freeze-request /tmp/task-balance-request.json
PYTHONPATH=src python scripts/run_maths_task_balance.py \
  --request /tmp/task-balance-request.json --request-identity sha256:EXPERIMENT \
  --verification-request /tmp/task-balance-verification.json \
  --output /tmp/task-balance-run --tier replay
# Independently retain the emitted verification identity.
PYTHONPATH=src python scripts/run_maths_task_balance.py --verify-only \
  --request /tmp/task-balance-request.json --request-identity sha256:EXPERIMENT \
  --verification-request /tmp/task-balance-verification.json \
  --verification-request-identity sha256:VERIFICATION \
  --output /tmp/task-balance-run --tier replay > /tmp/task-balance-audit.json
PYTHONPATH=src python scripts/archive_maths_task_balance.py \
  --run /tmp/task-balance-run --output /tmp/task-balance-evidence.zip \
  --request /tmp/task-balance-request.json --request-identity sha256:EXPERIMENT \
  --verification-request /tmp/task-balance-verification.json \
  --verification-request-identity sha256:VERIFICATION
```

Fresh training workers have 2,400 seconds; matrix summary/read-only workers
have 9,600 seconds. The archiver freshly replays every cell before exclusively
writing a ZIP outside the run. It retains all run members, complete model/AdamW
payloads, histories, predictions, custody, logs, exact source closure, external
requests and a fresh replay summary. Every member has a byte count and digest,
and must round-trip exactly; archive and inventory identities are emitted
separately. No old evidence is refreshed or resealed.

## Status and remaining gates

Implementation and numerical validation are separate from research outcomes.
The new required CPU suite exercises real training, admission, upstream custody,
full replay, external commitment failures, controller errors and lossless
archival. Standard-library tests cover exact scheduling and preservation of the
published budget source. Research results must be recorded separately after
producer publication.

Joint learning remains unproven. Original fold-sum and composed-output adequacy,
fresh family-level confirmation, independent replication, format selection,
geometry intervention, scaling, CPU export and release retain their own gates.
This training-only experiment does not authorize Phase 5 geometry-on training.
