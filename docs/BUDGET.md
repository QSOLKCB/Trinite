# Training-only budget development

Status: implementation of the next post-convergence increment. The [protocol](BUDGET-PROTOCOL.md)
was frozen at `0510c28d682b4d75824d9de25a17841e17e4fd79` before target outcomes;
byte identity `sha256:9e75f2f3642970e7ca56e81b65174d5f2794e07a008379cd0d621363c58d5087`.
This isolates four-times-longer exposure at constant 0.001 in nine dense cells.
It does not change the numerical model, native update loop, data or historical
protocols/results. Original plan construction limits stay unchanged.

`budget_plan.py` explicitly validates the larger bounded resources and delegates
unchanged settings to the foundations plan validator. `budget.py` composes
admitted foundations inputs, shared train-only scoring, native AdamW updates,
checkpoint progress/tensor laws and closed PROVENANCE observation.
`budget_checkpoint.py` binds its distinct profile, train-only view, full plan,
request, source, data and environment. The shared named-tensor reader is factored
from the existing comparison restore into `restore_tensors`; comparison and
foundations restoration retain their existing schemas and size bounds.
No second model, optimizer, dataset generator or tensor inventory is introduced.

Every fixed milestone retains tensors/moments, safe metadata, consumed examples,
training predictions, losses and preclip/clipping diagnostics. Verification
restores **each** milestone, regenerates greedy train predictions and train loss,
and checks its history is the final history's prefix. This is stronger than
custody-only intermediate receipts; it is still not an independent full optimizer
trajectory replay. Short interrupted replay and prior-low-rate numerical prefix
checks are conformance tests. Descriptive training checks cannot unlock held-out
scoring, certify learning adequacy or select a candidate automatically.

## Execute and review

Use the [hash-locked CPU environment](GETTING_STARTED.md). These explicit full
runs are separate from PR CI and never evaluate held-out examples.

```sh
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/run_budget.py --freeze-request /tmp/budget-request.json > /tmp/budget-freeze.json
BUDGET_REQUEST_ID=$(python -c 'import json; print(json.load(open("/tmp/budget-freeze.json"))["request_identity"])')
python scripts/run_budget.py --request /tmp/budget-request.json --request-identity "$BUDGET_REQUEST_ID" --output /tmp/budget
python scripts/run_budget.py --verify-only --request /tmp/budget-request.json --request-identity "$BUDGET_REQUEST_ID" --output /tmp/budget
```

Requests reject changed source/environment/admission/protocol. Source receipts
check actual runner bytes, not just a constant. The runner fixes interpreter,
script and closed cell selectors; paths are separate argv entries, shell is
false and no caller-provided executable is accepted. Nine workers have bounded
720-second timeouts; all failures/logs remain and summaries retain nine records.
Aggregation is transactional. Read-only verification compares the retained
canonical summary without writing to the run. The manual workflow always
attempts retention; runner loss/cancellation may still prevent upload.

Required checks: root/upstream, existing CPU/comparison/foundations/convergence,
and `python -m unittest discover -s tests/budget -v` on Python 3.11–3.13. New
checks cover exact interrupted replay, old numerical prefixes, checkpoint
profile/data-view rejection, real closed evidence corruption, admission binding,
custody-valid false predictions, positive/failed/mismatched summaries, timeouts
and fixed subprocess/resource boundaries.

Affected requirements: TRI-I02/03/05/06/07/08/11/12/13/14. Native numerical
semantics, old frozen fixture identities and old evidence remain preserved.
The checkpoint reader refactor changes current source receipts; review historical
requests using their archived producer and freeze a new request for current runs.
Game theory and QEC/ETQ/UFT-ID research are separate curriculum/research additions;
none enters this frozen budget experiment.
