# Training-only fold-order protocol

Freeze the [machine protocol](../src/trinite/fold-order-protocol.json) and
implementation before full-run outcomes. This follows the negative
[fold-exposure result](FOLD-EXPOSURE.md); it does not start Phase 5.

Compare six fresh dense CPU cells: seeds 0, 1, 2, each with `parent` and `mixed`
ordering. `parent` delegates unchanged scalar seeded parent rank and the
count/sum/squares projection order. `mixed` sorts all unique admitted training
rows by `(sha256(canonical JSON {seed, example_identity}), example_identity)`.
Canonical JSON is the existing UTF-8, sorted-key compact encoding with its final
newline. Reuse the fixed cycle; do not reshuffle at epoch boundaries. Run parent
first for seeds 0 and 2, mixed first for seed 1. This rotation is not full temporal
counterbalancing or independent-host replication.

Use the unchanged scalar fold corpus, admission and family splits. Both profiles
visit each of 3,450 training examples **eight times**: 3,450 updates with batch
eight, rather than the preceding 4,096-update partial cycle. Each task has 1,150
unique examples and 9,200 actual visits. This new shared budget deliberately
removes unequal final-prefix exposure. It is not a like-for-like comparison with
the preceding 4,096-update results. Retain actual visit-count identities, target
totals and schedule identities; fail pairing if exposure or initialization differ.

Keep the model, random initialization, tokenizer, AdamW, constant 0.001 learning
rate, clipping and authoritative numerical loop unchanged. Training diagnostics
occur every 1,150 updates and use training rows only. The existing 524,288-target
and 480-second numerical invocation bounds remain. Geometry loss is zero.
Different order also changes minibatch composition and per-step target-mean loss
weighting; this experiment cannot isolate a causal mechanism for an improvement.
Equal total visits/targets do not imply equal elapsed compute.

Score each unique training example once at the final checkpoint, with existing
prompt-only unconstrained greedy generation, cap four, complete answer and EOS.
Each of count, sum and squares must reach 90% in every seed. Retain both profiles
and every failure. Do not select a winner automatically, stop early, add data,
alter labels or call a held-out model. Correlated projections/carriers are not
independent samples. Training success does not close composed-output adequacy.

Bind requests to protocol, actual source bytes, locked environment, corpus and
admission, and the preceding fold-exposure request/summary identities. Use safe
named model/AdamW tensors; bind replay to the full ordered cycle and actual
history. Fresh verification checks closed PROVENANCE bundles, restores each
checkpoint, regenerates predictions and loss, and compares the complete summary.
It does not independently replay all optimizer updates.

Use fixed fresh workers with 600 seconds per training cell and 1,200 seconds per
summary/read-only verification. The manual workflow permits 120 minutes for the
100-minute worker envelope plus setup/retention. Retain timeouts, nonzero exits,
spawn errors, partial outputs and summary failures. Require stable local paths;
this is not a hostile concurrent filesystem snapshot. Runner loss may interrupt
retention.

Held-out scoring, original composed assessment, format selection, Phase 5,
broader curriculum, scaling and release remain blocked. Any subsequent learning
or evaluation decision requires its own frozen protocol. Preserve historical
protocols and evidence. Affected contracts: TRI-I02, I03, I05, I06, I07, I08, I10,
I11, I12, I13 and I14.
