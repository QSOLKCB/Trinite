# Fold-order development

The next eligible increment after the negative fold-exposure experiment is a
separately frozen [training-only ordering comparison](FOLD-ORDER-PROTOCOL.md).
Six dense CPU cells compare parent-grouped and deterministic mixed-task order
across three seeds. Each example receives eight visits in both profiles, with
3,450 updates at batch eight. This differs explicitly from the preceding
4,096-update experiment, whose incomplete last cycle would confound a reordered
schedule with changed example exposure.

`fold_order_plan.py` delegates existing CPU/resource constraints to BudgetPlan.
`fold_order.py` composes scalar admission, ordering and scoring, native updates,
safe named tensors and the closed upstream observer. It binds a separate request
and checkpoint profile to actual source and ordered-cycle identities. The runner
uses fixed fresh workers and retains all cell/summary failures, including spawn
errors. A new manual CPU workflow retains completed and partial output. Routine
CI requires the bounded fold-order conformance suite.

## Run and reverify

Acquire the [locked CPU environment](GETTING_STARTED.md), then:

```sh
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/run_fold_order.py --freeze-request /tmp/fold-order-request.json > /tmp/fold-order-freeze.json
FOLD_ORDER_ID=$(python -c 'import json; print(json.load(open("/tmp/fold-order-freeze.json"))["request_identity"])')
python scripts/run_fold_order.py --request /tmp/fold-order-request.json --request-identity "$FOLD_ORDER_ID" --output /tmp/fold-order
python scripts/run_fold_order.py --verify-only --request /tmp/fold-order-request.json --request-identity "$FOLD_ORDER_ID" --output /tmp/fold-order
```

Requests and output directories must be disjoint. Freeze a new request for a
changed source or environment. Read-only verification regenerates reports from
retained models; it does not run the optimizer again. Scientific negative
results can be fully verified; a failed worker or unverified cell returns nonzero.

## Status and limits

Implemented and measured under frozen producer commit
`58d94f3d0e9f1363bf43b2cc1f97c4f06e2c44a2`, published before the full run. All
six cells and the complete summary verified with no worker or pairing errors.
Bounded conformance establishes
control parity at the same plan, deterministic training-only permutation, equal
full-run visits/targets, replay, source/admission/cell binding, conservative
aggregation and failure retention. Software conformance alone cannot establish training success.

All final task gates remain 90% exact answer plus EOS, over 1,150 unique examples
per task. Both profiles and all seeds remain visible, including regressions.
Mixed order also changes minibatch composition and per-step target weighting;
results do not identify a causal mechanism. There is no automatic schedule
selection, held-out scoring, composed adequacy claim or Phase 5 unlock. Historical
negative results remain unchanged. Independent-host replication and the manual
hosted full matrix remain separate pending gates.


## Measured training-only results

Correct complete answers plus EOS out of **1,150 unique training examples per
task**; every task requires at least 1,035 correct answers in every seed.

| Seed | Ordering | Count | Sum | Squares | All tasks pass |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | parent | 1148 | 844 | 1144 | No |
| 0 | mixed | 1146 | 650 | 1130 | No |
| 1 | parent | 1150 | 731 | 1140 | No |
| 1 | mixed | 1143 | 945 | 1125 | No |
| 2 | parent | 1143 | 778 | 1118 | No |
| 2 | mixed | 1150 | 654 | 1140 | No |

Every cell fails sum while passing count and squares. Relative to parent grouping,
mixed order changes sum counts by −194, +214 and −124 for seeds 0, 1 and 2.
The mixed direction across seeds does not establish a general improvement.
No candidate clears all training gates and no schedule is selected. Any subsequent
budget, schedule or curriculum decision needs a separately frozen training-only
protocol before its outcomes; held-out evaluation and Phase 5 remain blocked.

All six actual histories contain 27,600 example visits, exactly eight per example,
and 62,784 scored targets. Visit-count identities agree across every seed/profile.
Initializations match within each seed. This establishes the declared exposure
control, not identical minibatches, token weighting, compute time or generalization.
Some early cells overlapped local conformance checks; wall/RSS observations are
descriptive and must not be used as a paired speed comparison.

Local validation passed 109 standard-library checks and all 163 required CPU
research checks, including eleven new fold-order cases. The frozen upstream suite
ran 51 checks with its existing unprivileged-permission skip under root; dependency
consistency and relative file links passed. The expanded research suite took
519 seconds locally, so its CI job now permits 15 minutes including locked
dependency acquisition; every required check remains enabled.


## Retained evidence and reproduction

The [review inventory](../fixtures/fold-order-v1/inventory.json) identifies the
lossless six-part packet, all six full checkpoints, optimizer tensors, update
histories, predictions, worker logs, closed PROVENANCE bundles and exact producing
source. Direct records include the [request](../fixtures/fold-order-v1/request.json),
[summary](../fixtures/fold-order-v1/summary.json),
[source check](../fixtures/fold-order-v1/source-check.json),
[read-only verification](../fixtures/fold-order-v1/read-only-verification.json),
[run inventory](../fixtures/fold-order-v1/run-members.json) and
[archive restoration](../fixtures/fold-order-v1/archive-restoration.json).

The packet is 21,647,259 bytes and restores
315 files / 115,030,892 bytes. All
149 retained producer files match published GitHub blobs, and all 63 request-bound
source artifacts match that producer. Fresh read-only verification regenerated
all six reports and the complete summary while leaving all 166 existing run
files unchanged. Archive restoration matches original run and producer bytes.
This packet is a separately identified review transport exception to the
small-fixture convention, not an evolving training directory in git.

After acquiring the [CPU lock](GETTING_STARTED.md), verify with the exact retained
Python/kernel/CPU/package fingerprint. Other environments need a fresh request.

```sh
cat fixtures/fold-order-v1/Trinite-fold-order-v1.zip.part* > /tmp/Trinite-fold-order-v1.zip
python scripts/restore_foundations_archive.py /tmp/Trinite-fold-order-v1.zip /tmp/fold-order-review --archive-identity sha256:cc3a93592339c7689373d49afdd01b575047e39af5399586c4a5bfbdd682d4e7
cp /tmp/fold-order-review/run/request.json /tmp/fold-order-review-request.json
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/fold-order-review/producer/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python /tmp/fold-order-review/producer/scripts/run_fold_order.py --verify-only --request /tmp/fold-order-review-request.json --request-identity sha256:c26be205038b8ebd9b648bad60ed335bc241544c3e0b821c4d638e3ac22c54d7 --output /tmp/fold-order-review/run
```

This verification restores final checkpoints and regenerates predictions/losses;
it does not replay every optimizer update or establish independent replication.
Full optimizer replay, independent-host replication and the manual hosted matrix
remain pending. No generalization, causal mechanism or speedup is claimed.
