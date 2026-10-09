# Scalar curriculum learning

This increment trains the independently verified scalar preparation from
[GENERALIZATION.md](GENERALIZATION.md) under the separately frozen
[scalar protocol](SCALAR-LEARNING-PROTOCOL.md). It keeps arithmetic texts as the
control, splits every fold into count/sum/squares targets and projects traversal
into value/x/y/z, retaining the other lattice tasks and all original family splits.
The new matched experiment has 27 workload/seed/lane cells. Historical composed
learning failures and the descriptive diagnostic packet remain unchanged.

## Implementation and checks

`ScalarPlan` delegates all numerical/resource bounds to the existing budget
profile; only its schema and declared exposure order differ. The 86,528-parameter
CPU research model, random initializer, tokenizer, loss, AdamW and numerical
update loop are unchanged. Parent hash order and fixed sibling order define the
complete repeated training cycle. Arithmetic numerical exposure matches the
parent; scalar targets change exposure versus the historical composed corpus.
Equal updates and a common target-token ceiling do not imply equal exposure to
that earlier corpus or equal training compute between workloads.

All numerical diagnostics during updates use training rows only. Safe checkpoints
use the existing closed named-tensor/optimizer reader, with a new source/request/
protocol/admission/train-only context and complete schedule/progress history.
The upstream observer binds request, admitted corpus, reports, predictions and
checkpoint bytes. Fresh verification regenerates training/test predictions and
losses and compares each retained verification receipt exactly. Retained success
files do not authorize scoring. Every worker terminates before fresh matrix
authorization; failed/missing/unmatched cells or a failed dense training task
block all held-out model calls. False decisions and all partial outputs remain.

The unchanged thresholds apply independently to every scalar task: dense training
90% exact answer/EOS, held-out 50% and ten percentage points above training-majority
answers in every seed. Fold has 3,450 training projections, so the new gate wrapper
validates their aggregate before delegating the unchanged per-task thresholds.
Correlated projections and two carriers are not independent samples. Family
macro and paired quality/time/storage margins are descriptive three-seed measures.
Latent float32 storage is not packed inference savings.

Scalar adequacy has its own `dense_scalar_learning_adequate` flag. Original
`dense_learning_adequate`, format selection, Phase 5 and release stay false even
if every scalar task passes. No original composed-output model call is made;
that assessment requires its own declared protocol before outcomes.

## Run and reverify

Use the [hash-locked CPU environment](GETTING_STARTED.md), then:

```sh
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/run_scalar_learning.py --freeze-request /tmp/scalar-request.json > /tmp/scalar-freeze.json
SCALAR_REQUEST_ID=$(python -c 'import json; print(json.load(open("/tmp/scalar-freeze.json"))["request_identity"])')
python scripts/run_scalar_learning.py --request /tmp/scalar-request.json --request-identity "$SCALAR_REQUEST_ID" --output /tmp/scalar-learning
python scripts/run_scalar_learning.py --verify-only --request /tmp/scalar-request.json --request-identity "$SCALAR_REQUEST_ID" --output /tmp/scalar-learning
```

The new, disjoint output retains the exact request, all final checkpoints and
update histories, training diagnostics/predictions, verified matrix decision,
held-out outputs when eligible, closed bundles, receipts, worker logs and complete
summary. Training workers allow 600 seconds; evaluation and summary workers
allow 1,200 seconds each. `scalar-learning.yml` provides an explicit manual CPU
workflow with 360-minute envelope and unconditional artifact retention. Routine
three-Python CI includes the small scalar conformance suite, not a full matrix.

Requests bind CPU, Python, kernel, packages and all source/admission bytes. A
different fingerprint rejects. Freeze a new request for changed source; do not
rewrite historical environments or regenerate old fixtures to force acceptance.
Replay scope and remaining trusted numerical/filesystem components are explicit.

## Measurements and remaining work

The protocol was frozen at `06edd13c3011028b18231c144b83ef8d92528c74` before
training outcomes. Full matrix measurement and its verified evidence are pending
in this implementation draft. No scalar learning claim is made from unit tests.

Affected contracts: TRI-I02, I03, I05, I06, I07, I08, I10, I11, I12, I13 and I14.
No model, optimizer, original scientific protocol, admission fixture or archived
producer changes. Structure/algebra expansion, new books/prose, geometry
intervention, scaling and release remain conditional later increments.
