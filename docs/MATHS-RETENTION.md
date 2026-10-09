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

## Outcome status at protocol publication

Full training outcomes are pending. No measured improvement or retention pass
is claimed at this publication point. Successful and failed cells, source,
corpora, final safe checkpoints, histories, predictions, custody and fresh
verification will be retained together and reported here after execution.
