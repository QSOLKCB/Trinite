# Fold-sum exposure development

The next roadmap increment develops the exposed sum training floor under a
separately frozen [training-only protocol](FOLD-EXPOSURE-PROTOCOL.md). The previous
[scalar experiment](SCALAR-LEARNING.md) remains a negative result: all three dense
sum cells missed the 90% training gate. This increment retains the same data and
model and compares fixed 1×, 2× and 4× sum exposure in three dense seeds.

`fold_exposure_plan.py` delegates existing resource and numerical constraints to
BudgetPlan. `fold_exposure.py` composes unchanged scalar ordering, admission,
native updates/scoring, safe named tensors and the closed upstream observer.
`scripts/run_fold_exposure.py` launches only fixed fresh training and verification
workers. There is no held-out worker. Each unique train example is scored once,
regardless of its exposure multiplicity. Checkpoints bind the new source/request/
protocol context and full schedule. Fresh verification checks evidence copies and
regenerates training predictions and weighted loss. All candidates stay visible;
none is selected automatically. The 1× control has a numerical parity fixture
against the scalar implementation.

## Run and reverify

Use the [locked CPU environment](GETTING_STARTED.md):

```sh
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/run_fold_exposure.py --freeze-request /tmp/fold-request.json > /tmp/fold-freeze.json
FOLD_REQUEST_ID=$(python -c 'import json; print(json.load(open("/tmp/fold-freeze.json"))["request_identity"])')
python scripts/run_fold_exposure.py --request /tmp/fold-request.json --request-identity "$FOLD_REQUEST_ID" --output /tmp/fold-exposure
python scripts/run_fold_exposure.py --verify-only --request /tmp/fold-request.json --request-identity "$FOLD_REQUEST_ID" --output /tmp/fold-exposure
```

Fresh outputs must be disjoint from the immutable request. Different source or
locked environment requires a fresh request. The manual `fold-exposure.yml`
workflow retains completed/failed/partial outcomes; routine CPU CI runs bounded
conformance cases instead of the full experiment.

## Status and limits

Implemented; empirical measurement pending. Freeze the protocol/source commit
before the full experiment and retain negative and failed cells. Sum improvement
can come with count/squares regressions; all three task gates remain unchanged.
A successful training schedule requires a separately frozen learning protocol;
it cannot authorize held-out calls, composed assessment, geometry intervention,
format selection, scaling or release. No model, optimizer, tokenizer, admission
fixture or historical protocol is changed.
