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

Here, composed-output assessment means the original fold-tuple and traversal-address
tasks. Unchanged non-traversal lattice controls, including role-to-address, remain
in this admitted corpus as declared in the preparation.

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
training outcomes. Producer `df2e01037979d4268de9443ef86d4d2c39bf1f4b`
completed all 27 training cells under the unchanged frozen request. Dense
arithmetic and lattice passed every training task in all three seeds. Fold's
training results were:

| Dense seed | Count | Sum | Squares |
| --- | ---: | ---: | ---: |
| 0 | 1,150/1,150 | 934/1,150 | 1,147/1,150 |
| 1 | 1,148/1,150 | 868/1,150 | 1,141/1,150 |
| 2 | 1,145/1,150 | 882/1,150 | 1,132/1,150 |

Each task has 1,150 examples. Sum misses the 90% gate in every seed, so the
matrix decision is false and no held-out model predictions are produced.
This is a completed, blocked scientific result; training success on the other
tasks is not a generalization claim. All three lanes and seeds remain in the
summary, including negative results.

The next learning increment requires a separately frozen training-only decision
about fold-sum exposure or budget. No tuning, budget extension or composed-output
assessment is performed after seeing these outcomes.

### Producer lineage and admission correction

After measurement started, commit
`7496237d3c6b0fa353db5a4bc6e491b0b49371a0` tightened fresh request admission:
generated corpus and manifest identities must equal the exact preparation in the
already bound generalization report. A regression rejects either identity's drift
before model construction. The measured request was independently checked to
already match that preparation exactly. This correction changes source binding
and rejection behavior; it changes no data, model, update, exposure or scorer.
The measured request and archived producer retain their original identities.
Current source requires a fresh request; replay this retained packet with its
archived producer rather than relabelling its request.

## Retained evidence and reproduction

The complete packet is retained in [scalar-learning-v1](../fixtures/scalar-learning-v1/inventory.json):
[request](../fixtures/scalar-learning-v1/request.json),
[summary](../fixtures/scalar-learning-v1/summary.json),
[blocked decision](../fixtures/scalar-learning-v1/test-decision.json),
[arithmetic control parity](../fixtures/scalar-learning-v1/arithmetic-control-parity.json)
and [fresh archive verification](../fixtures/scalar-learning-v1/verification.json).
All 27 training bundles, exact verification receipts, safe tensor/AdamW payloads,
4,096-step histories, diagnostics, predictions and worker logs are included.
There were no worker, pairing or aggregate integrity errors. No held-out
predictions exist under the blocked decision.

The eight numbered archive parts restore 858 files and 449,066,128 bytes,
including the exact producer closure and prior admission anchors. The archive
uses lossless ZIP/LZMA with the existing bounded, hash-checking restorer.

- Protocol identity: `sha256:d1f64a577e78d7c38f58f3cf21e7683763869dc7e50e5a7be327ab42e69be670`.
- Measured request: `sha256:f75f368f04963336de3c32f70e3c28592b4ae010f2c6e7c3acd6ab2ac016b70d`.
- Summary: `sha256:60b017fa8cf89a1a9bcc5214d18ac380906565cfff77cd5a279032a2b77e3889`.
- Archive: `sha256:34847d7b746c580867fd2c42468d40b685f7abbbce7d015cc2f5eb4065d8f71d`.

After acquiring the CPU lock, run from the repository root:

```sh
cat fixtures/scalar-learning-v1/Trinite-scalar-learning-v1.zip.part* > /tmp/Trinite-scalar-learning-v1.zip
python scripts/restore_foundations_archive.py /tmp/Trinite-scalar-learning-v1.zip /tmp/scalar-review --archive-identity sha256:34847d7b746c580867fd2c42468d40b685f7abbbce7d015cc2f5eb4065d8f71d
cp /tmp/scalar-review/run/request.json /tmp/scalar-review-request.json
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/scalar-review/producer/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python /tmp/scalar-review/producer/scripts/run_scalar_learning.py --verify-only --request /tmp/scalar-review-request.json --request-identity sha256:f75f368f04963336de3c32f70e3c28592b4ae010f2c6e7c3acd6ab2ac016b70d --output /tmp/scalar-review/run
```

Keep the supplied request outside the output directory; the CLI rejects request/
output overlap. Restoration verifies every transport byte. Numerical verification
requires the exact retained CPU/kernel/Python/package fingerprint; other hosts
must freeze a new request for a fresh experiment. The same-host verification
checked all 59 declared source files against their retained identities and
GitHub producer blobs, checked 112 versioned producer files, and regenerated
checkpoint predictions,
losses, receipts, the decision and summary without changing any run file.
The packet also retains the producer label and six generated package metadata
files, separately identified as non-repository members. They do not substitute
for source checks. Verification did not replay every optimizer update or establish
independent-host replication. Separately, all nine arithmetic controls matched historical bound
model/optimizer bytes, normalized full histories and train diagnostics exactly.
No new runtime speedup or packed-storage benefit is claimed.

The small local scalar suite passed 21 cases; hosted Python 3.11, 3.12 and 3.13
CI passed 101 standard-library, 51 upstream and 143 research checks per version
on admission-guard commit `7496237d3c6b0fa353db5a4bc6e491b0b49371a0`.
The locally root-only upstream permission case was skipped; hosted CI passed it.
Software conformance and evidence integrity do not close the scientific gates.

Affected contracts: TRI-I02, I03, I05, I06, I07, I08, I10, I11, I12, I13 and I14.
No model, optimizer, original scientific protocol, admission fixture or archived
producer changes. Structure/algebra expansion, new books/prose, geometry
intervention, scaling and release remain conditional later increments.
