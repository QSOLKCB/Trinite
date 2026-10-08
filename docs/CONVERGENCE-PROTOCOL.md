# Training convergence development protocol v1

This new development protocol is frozen before its target model outcomes.
The byte contract is [convergence-protocol.json](../src/trinite/convergence-protocol.json).
The old foundations protocol, blocked decision and archived producer are unchanged.

Run 27 dense CPU cells: arithmetic, fold and lattice; seeds 0, 1 and 2;
constant AdamW learning rates 0.001, 0.003 (reference) and 0.006. Every candidate
uses the same 86,528-parameter randomly initialized model, seeded training order,
batch 8, 1,024 updates and 131,072-target cap as foundations v1. Keep the loss,
weight decay, clipping, optimizer, quantizer definitions and geometry loss zero.
Compare learning-rate candidates within each workload/seed; the data,
initialization and consumed examples/targets must match before aggregation.

At fixed updates 256, 512, 768 and 1,024, record unconstrained prompt-only greedy
complete-answer/EOS predictions for **training examples only**, using the prior
workload caps. Record task counts, train-majority baselines, full teacher-forced
training loss, gradients before clipping, clipping frequency and update history.
Use training examples for the inherited numerical loop's `validation` field as
well; in this protocol it is a training-loss diagnostic, not a validation score.
Neither validation nor test examples reach the model. All original labels are
public fixtures, so this is exploratory learning development, not fresh evaluation.

No early stopping, candidate selection, test worker, geometry objective or
quantized-lane comparison is part of this increment. Descriptive 90% per-task
training checks do not unlock held-out evaluation. Retain failed candidates,
regressions and timeouts. Follow with a **separately frozen** learning protocol,
including the candidate/budget decision and evaluation/exposure controls, before
any held-out run. Do not tune the old experiment or overwrite its evidence.

Each worker has 240 seconds including scoring/evidence. Numerical update
invocations share a cumulative 120-second measured envelope across milestones;
score/observer time is excluded and reported separately. A 132-minute manual
workflow covers 27 workers x 240 seconds (108 minutes), plus 24 minutes overhead.
Use immutable source/data/environment-bound requests; pin the runner bytes.
Publish a protocol commit before measurements, freeze a request on the producer
source, and retain final named safetensors/moments, checkpoint metadata, every
prediction, curve, history, logs and actual closed PROVENANCE verification.

Verification checks artifact custody, request/context, final checkpoint,
recorded predictions/scores, schedule, diagnostic history and paired controls.
It does not independently recompute intermediate numerical trajectories or
prove model-generated predictions: an explicit replay is a separate check.
The summary retains all 27 cells and publishes no groups if any cell fails or
pairing is inconsistent. Completion means diagnostic execution and integrity
checks completed; learning adequacy, Phase 5 readiness and release readiness
remain false.
