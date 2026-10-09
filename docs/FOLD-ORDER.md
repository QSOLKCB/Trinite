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

Implemented; full six-cell measurement is pending. Bounded conformance establishes
control parity at the same plan, deterministic training-only permutation, equal
full-run visits/targets, replay, source/admission/cell binding, conservative
aggregation and failure retention. It cannot establish training success.

All final task gates remain 90% exact answer plus EOS, over 1,150 unique examples
per task. Both profiles and all seeds remain visible, including regressions.
Mixed order also changes minibatch composition and per-step target weighting;
results do not identify a causal mechanism. There is no automatic schedule
selection, held-out scoring, composed adequacy claim or Phase 5 unlock. Historical
negative results remain unchanged. Independent-host replication and the manual
hosted full matrix remain separate pending gates.
