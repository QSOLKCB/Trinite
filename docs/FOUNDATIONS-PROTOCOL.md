# Foundations learning-adequacy protocol v1

This protocol is frozen before foundations model outcomes. It implements the
first repository-informed increment in ROADMAP, ahead of Phase 5. The machine
contract is [foundations-protocol.json](../src/trinite/foundations-protocol.json).
Its immutable freeze commit and byte identity are recorded by the implementation.
Previous comparison protocol, settings and evidence remain unchanged.

## Data and scope

Generate new minimal symbolic facts, not donor prose or copied software:

- Arithmetic: add/subtract/multiply and exact division over operands -3 through
  3, with reduced signed rational answers and an explicit division-by-zero error.
- Fold: count, sum and sum of squares of lists of length 2–4 with values -2
  through 2.
- Lattice: bounded 3-coordinate addresses, symbolic role/address conversion,
  balanced valid/invalid classification and the fixed 27-cell traversal.

Two minimal carriers are grouped. Arithmetic families are unordered absolute
operand pairs across signs and operations. Fold families are sorted nonzero
absolute multisets across signs, permutations and zero insertion. Lattice
families are base coordinates across roles, traversal, carriers and corruptions.
Rank families by the declared SHA-256 split policy/seed 31: floor(80%) train,
max(1, floor(10%)) validation, remainder test. Retain labels, independent oracle
checks, rendering/parser agreement, source admissions and exposure limitations.
Fixtures are fully exposed conformance/learning data, not a fresh benchmark.

## Fixed learning budget

All 27 workload/seed/lane cells use the same model: context 64, two blocks,
width 64, four heads, feed-forward width 128, 86,528 parameters. Weights start
from random private seeded initialization; no teachers, pretrained weights or
geometry objective. Dense/ternary/four-state retain their existing definitions.

Each cell has 1,024 updates, batch size 8, constant AdamW learning rate 0.003,
validation every 256 updates, at most 131,072 scored answer/EOS tokens and 120
seconds for the update invocation. The new bounded foundations plan shares the
authoritative numerical update loop. The original native training-plan limits
are unchanged. Train order is seeded hash rank/cycle; selection is final update
only. Validation loss chooses nothing.

The worker limit is 240 seconds. A 240-minute manual workflow covers the
conservative 54-worker envelope of 216 minutes plus 24 minutes of setup,
aggregation and upload. Complete failure records and logs remain retained.

## Gates and scoring

All training workers terminate before the test decision. A task's exact answer
includes EOS; unconstrained greedy decoding receives prompt bytes only. Fixed
workload-wide generation caps are arithmetic 5, fold 9 and lattice 9 tokens.
Expected answer length never supplies an item-specific generation hint.

Unlock held-out evaluation for every lane only if all nine dense workload/seed
cells have verified evidence and every task achieves at least 90% training exact
answers. Otherwise retain a blocked decision, score no held-out items, and make
no quality comparison. Record the decision before starting test workers; a
declared decision without matching verified cells is insufficient.

A lane is learning-adequate only if every seed/task has training exact accuracy
at least 90%, held-out exact accuracy at least 50%, and held-out accuracy at
least 10 percentage points above its per-task training-majority-answer baseline.
The baseline uses training answers only, with lexical tie-breaking. Counts and
rational threshold comparisons determine gates; hex floats are descriptive.
Retain all task scores, families, outputs, failed lanes and descriptive paired
seed ranges. Global accuracy cannot conceal a failed task.

Quality margin is quantized-minus-dense family macro >= -0.05. Cost diagnostics
retain training wall ratios <= 1.25 and latent float32 byte ratios <= 1. These
are descriptive shared-host measurements. Useful format selection remains
blocked without adequate dense learning; equal zero scores are insufficient.

## Evidence and outcome discipline

Bind requests to the exact protocol, implementation/runner/dependencies,
admitted data and CPU environment. Retain final safetensors, named optimizer
state, complete histories, predictions, baseline outputs, the unlock decision
and actual closed upstream PROVENANCE verification. Reject altered result copies,
inconsistent snapshots, missing evidence and unmatched pairs. Aggregation writes
a failed/blocked summary rather than publishing partial comparison groups.

No setting or threshold changes after target outcomes. Failure is a valid result
of this increment and keeps Phase 5 pending. Changed tasks, budgets or objectives
require a new frozen protocol; changed implementation requires a new request.
Software conformance, learning observations and scientific capability claims
remain separate. No geometry intervention, quantum execution, packed export,
general reasoning advantage or independent replication is established here.
