# Training budget development protocol v1

Freeze this protocol before its target outcomes. The canonical byte contract is
[budget-protocol.json](../src/trinite/budget-protocol.json). This increment follows
the failed [convergence sweep](CONVERGENCE.md); its historical protocols and
results stay unchanged.

## Decision and scope

The lower 0.001 rate improved every arithmetic/lattice seed relative to 0.003,
but no rate cleared every training task. Isolate additional update exposure:
nine fresh dense cells, arithmetic/fold/lattice × seeds 0/1/2, constant rate
0.001, batch eight and 4,096 updates (four times the previous budget). Keep the
86,528-parameter architecture, random initialization, training order, AdamW,
clipping, admitted examples, answer/EOS loss and geometry loss zero unchanged.
No schedule, curriculum or model change is bundled with this budget decision.

Record milestones 1,024/2,048/3,072/4,096, each with named model/optimizer tensors,
replay metadata, full consumed-example/target/loss/gradient history, prompt-only
greedy complete-answer/EOS training predictions, per-task 90% descriptive checks,
and preclip/clipping diagnostics. Verify each milestone's predictions and full
training loss against its retained model. The first 1,024 model identity can be
compared with the previous low-rate checkpoint as a numerical prefix check;
that historical comparison is additional evidence, not a new training input.

Use training rows for the inherited numerical loop's validation field as well;
this is a training-loss diagnostic. No validation/test model calls, held-out
stage, quantized-lane comparison, automatic selection or early stopping. Even
if every descriptive training check passes, learning adequacy and Phase 5 stay
blocked until a separately frozen learning/evaluation protocol. Public finite
fixtures cannot establish fresh generalisation.

## Bounds and evidence

A separate plan permits at most 4,096 updates, batch eight, 524,288 scored target
tokens and 480 cumulative measured update seconds. Original RunConfig and
FoundationsPlan construction limits remain unchanged. Each worker is bounded
by 720 seconds including scoring/checkpoints/observer work. The manual workflow
allows 132 minutes: nine × 720 seconds plus 24 minutes retention headroom.
Checkpoints are limited to 32 MiB tensors and 8 MiB metadata each. Retain all
failures/timeouts and nine-cell failed summaries; publish no aggregate groups
unless every cell validates. Manual cancellation or runner loss may interrupt
upload. Source/data/environment/runner identities bind a newly frozen request.

Publish this protocol commit before execution, then freeze a request on its
producer. Retain logs, all milestone checkpoints/predictions, closed upstream
PROVENANCE bundles and a complete independently hash-checked archive. Read-only
verification must leave the evidence unchanged. Custody plus regenerated model
predictions does not establish independent full optimizer replay, mathematical
proof or capability. No historical evidence may be rewritten.
