# Training-only fold-sum exposure protocol

Freeze the [machine protocol](../src/trinite/fold-exposure-protocol.json) before
new model outcomes. This development increment responds to the retained scalar
sum training failure; it does not run the Phase 5 intervention.

Use only the unchanged, independently checked scalar fold corpus and its family
splits. Train nine fresh dense CPU cells: seeds 0, 1, 2 crossed with sum
multiplicities 1, 2, 4. Keep seeded parent order, visiting count once, sum
consecutively the chosen number of times, then squares once for each parent.
Cycle the fixed list for 4,096 updates, batch eight, constant 0.001. Initialization,
model, loss, AdamW, clipping, 524,288-target ceiling and 480-second numerical
invocation bound stay unchanged. Geometry loss remains zero. Multiplicity one
is the historical scalar exposure control. Equal updates do not mean equal task
exposure, scored tokens or measured compute. Changed rows affect training loss
weighting; do not compare weighted losses as if they were the same objective.

Every diagnostic uses training rows only. Score each unique admitted training
example exactly once at the final checkpoint with unconstrained prompt-only
greedy generation, cap four, complete answer and EOS. Each of count, sum and
squares must reach 90% in every seed; retain all counts and all candidates,
including regressions. Correlated projections and carriers are not independent
samples. No early stopping, new data, held-out model calls, outcome-driven
schedules, automatic winner selection or format comparison.

Bind the request to protocol/source/environment, exact corpus/admission and the
previous scalar request, summary and blocked decision. Retain named safe model
and AdamW tensors, complete exposure histories, predictions, diagnostics,
wall/RSS observations and closed PROVENANCE receipts. Fresh verification restores
checkpoints, checks the actual scheduled identities and targets, regenerates
predictions/loss, compares evidence receipts and requires matched initialization
across candidates for each seed. Same-host interrupted replay is software
conformance; full optimizer replay and independent-host replication are separate.

Use fixed fresh workers: 600 seconds per training cell; 1,200 seconds for summary
or read-only verification. A manual CPU workflow allows 150 minutes for the
130-minute train/summary/reverification envelope and retention/setup headroom.
Retain partial outputs and summary failures. Stable local directories are required;
this is not a hostile concurrent filesystem snapshot. Runner loss can interrupt
retention.

Successful training only informs a subsequent explicit, separately frozen
learning decision. Held-out scoring, composed adequacy, format selection,
Phase 5, broader curriculum, scaling and release remain blocked. Preserve all
historical failures and protocols. A software check never closes an empirical gate.
Affected contracts: TRI-I02, I03, I05, I06, I07, I08, I10, I11, I12, I13, I14.
