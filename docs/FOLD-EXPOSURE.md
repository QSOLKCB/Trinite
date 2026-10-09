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

Implemented and measured under protocol/producer commit
`42c256e583f447b8ddddc88664dc300585e38ded`, published before the full
experiment. All nine workers completed and all closed bundles, checkpoints,
predictions, losses and initialization pairings freshly verified without worker
or pairing errors. Every candidate fails at least one training task in every seed. Sum improvement
can come with count/squares regressions; all three task gates remain unchanged.
A successful training schedule requires a separately frozen learning protocol;
it cannot authorize held-out calls, composed assessment, geometry intervention,
format selection, scaling or release. No model, optimizer, tokenizer, admission
fixture or historical protocol is changed.

## Measured training-only results

Each entry below is correct complete answer plus EOS out of **1,150 unique
training examples per task**. The unchanged 90% gate requires at least 1,035.
Exposure repeats never enlarge these denominators.

| Seed | Sum multiplicity | Count | Sum | Squares | All tasks pass |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 1× | 1150 | 934 | 1147 | No |
| 0 | 2× | 1148 | 794 | 1142 | No |
| 0 | 4× | 1077 | 412 | 779 | No |
| 1 | 1× | 1148 | 868 | 1141 | No |
| 1 | 2× | 1134 | 764 | 1085 | No |
| 1 | 4× | 1048 | 425 | 670 | No |
| 2 | 1× | 1145 | 882 | 1132 | No |
| 2 | 2× | 1148 | 699 | 1085 | No |
| 2 | 4× | 1080 | 461 | 961 | No |

The 2× and 4× schedules have lower sum accuracy than the 1× control in all
three seeds. All 4× squares cells also fall below the gate. These are descriptive
training outcomes for the declared consecutive-repeat schedules; they do not
identify a causal failure mechanism or justify a general claim about oversampling.
Weighted exposure losses are different objectives and are not a like-for-like
quality measure. No new learning schedule is selected. The next increment needs
a separately frozen training-budget/schedule or curriculum decision; held-out
and original composed assessment, format selection and Phase 5 stay blocked.

All three 1× controls match the earlier scalar run exactly for full model/AdamW
tensor payloads, complete histories/diagnostics and unique training predictions.
The prior closed control bundles were freshly checked before comparison. This is
same-host byte parity, not independent-host reproduction. The bounded interrupted
replay fixture also passes; this does not mean all nine optimizer histories were
independently reexecuted.

## Retained evidence and reproduction

The [review inventory](../fixtures/fold-exposure-v1/inventory.json) identifies the
lossless packet, all nine full checkpoints/update histories/predictions, closed
PROVENANCE bundles and receipts, worker logs and exact producing source. Direct
records include the [request](../fixtures/fold-exposure-v1/request.json),
[summary](../fixtures/fold-exposure-v1/summary.json),
[full control parity](../fixtures/fold-exposure-v1/control-parity.json),
[producer source check](../fixtures/fold-exposure-v1/source-check.json) and
[fresh read-only verification](../fixtures/fold-exposure-v1/read-only-verification.json).
This is a separately identified review transport exception to the small-fixture
convention; the run is not an evolving training directory in git.

The producer source check compares 135 retained files against the published
GitHub commit's blob identities. Its separate producer label is explicitly
identified as non-repository metadata. Requests bind 63 actual source artifacts.
The four archive parts restore 386 files and 188,259,925 bytes. Fresh read-only
verification leaves all 249 existing run files unchanged and regenerates model
predictions/losses and the complete summary;
it does not replay every optimizer update or establish independent replication.
Other environments must freeze a new request instead of relabelling this run.

After acquiring the [CPU lock](GETTING_STARTED.md), use the commands below with
the exact retained Python/kernel/CPU/package fingerprint:

```sh
cat fixtures/fold-exposure-v1/Trinite-fold-exposure-v1.zip.part* > /tmp/Trinite-fold-exposure-v1.zip
python scripts/restore_foundations_archive.py /tmp/Trinite-fold-exposure-v1.zip /tmp/fold-review --archive-identity sha256:6c94ef2fe471c460d6d958306122c11ad310d6964ecfb74fefcf85d43b0da6d6
cp /tmp/fold-review/run/request.json /tmp/fold-review-request.json
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/fold-review/producer/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python /tmp/fold-review/producer/scripts/run_fold_exposure.py --verify-only --request /tmp/fold-review-request.json --request-identity sha256:88838690d805324ab8fb445b454f82f1e122ef89afba88276ac50d55080babc1 --output /tmp/fold-review/run
```

Local verification passed 109 standard-library and 151 required CPU research
checks (including eight new fold-exposure cases). The frozen upstream suite
completed 51 checks with its existing root-permission skip; hosted unprivileged
CI checks that case. All nine hosted Python 3.11–3.13 CPU and reference CI jobs pass on the frozen
implementation commit. A separate manual hosted full matrix and independent-host
replication remain pending. No new speed, packed savings or generalization claim.
